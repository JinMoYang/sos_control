package com.example.activeperception.acquire

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Shared-spec LCG (NeuralAeNetTest / sense/nae_ref_check.py / nae_trainer_ref.py):
 * state' = state*A + C mod 2^64, u = top 24 bits / 2^24. Drives init + shuffles so a
 * fixed seed reproduces bit-identical streams across Kotlin and the numpy reference
 * (24-bit uniforms are exact in float32).
 */
internal class NaeLcg(var state: Long) {
    fun nextU(): Double {
        state = state * 6364136223846793005L + 1442695040888963407L
        return (state ushr 40).toDouble() / (1 shl 24).toDouble()
    }
}

/**
 * On-device trainer for the Neural-AE hist_scalar net: torch-free mirror of
 * sense/nae_train_app.py train() (lines 248-312), producing nae-bin-v1 bytes that
 * NeuralAeNet loads. Semantics copied exactly:
 *  - loss = MSE on the SQUASHED output (model applies Eq. 4: 2*(sigmoid-0.5)*ln 10)
 *    vs the clamped dlog-e target - torch's `mse_loss(model(X), y)`
 *  - Adam, torch defaults (beta 0.9/0.999, eps 1e-8), per-epoch shuffled minibatches,
 *    per-batch mean loss, train loss reported as the sample-weighted batch average
 *  - val MSE on the whole split each epoch; snapshot best-val weights when improved
 *    by > 1e-6, stop after 15 stale epochs, return the snapshot (python's rule)
 *  - init mirrors torch build_model(): conv/linear weights AND biases
 *    U(+-1/sqrt(fan_in)) (kaiming-uniform default), head weight+bias zero
 *    (neural_ae_app.py's v2 "start at no change" fix)
 * Forward math is in lockstep with NeuralAeNet.predictLogU (loop-order identical:
 * convFwd = NeuralAeNet.conv lines 69-84, fcFwd = fc lines 86-95, squash line 65),
 * so the trainer's own forward and the loaded net agree bit-for-bit. Storage is
 * float32 (like the .bin), all accumulation float64. Backward is manual and was
 * gradient-checked against central differences in sense/nae_trainer_ref.py before
 * this port (worst floored rel err 8.5e-6 in f64; the NaeTrainerTest scenario
 * rehearses at 8.3e-5).
 *
 * Deterministic for a fixed seed and sample order; no threads, no Android.
 */
class NaeTrainer(private val seed: Long = 42L) {

    /** One training sample: 59*256 hist (row-major, NaeFeatures layout) and target
     *  log-u in [-ln10, ln10] (clamped again defensively at train time). */
    class Sample(val hist: FloatArray, val targetLogU: Double) {
        init {
            require(hist.size == NeuralAeNet.N_HIST * NeuralAeNet.N_BINS) {
                "hist size ${hist.size} != ${NeuralAeNet.N_HIST * NeuralAeNet.N_BINS}"
            }
        }
    }

    /**
     * Train from scratch (initWeights == null) or continue from a nae-bin-v1 blob.
     * Returns nae-bin-v1 bytes readable by NeuralAeNet; when fine-tuning, the source
     * blob's JSON header (grid metadata) is carried over verbatim. [onEpoch] gets
     * (epoch, trainLoss, valLoss) per epoch; valLoss = NaN when valFraction <= 0
     * (then there is also no early stop - final weights are returned).
     */
    fun train(samples: List<Sample>, valFraction: Double = 0.2, epochs: Int = 60,
              batchSize: Int = 32, lr: Double = 1e-3, initWeights: ByteArray? = null,
              onEpoch: (Int, Double, Double) -> Unit = { _, _, _ -> }): ByteArray {
        require(samples.isNotEmpty()) { "no samples" }
        require(epochs >= 1 && batchSize >= 1) { "bad epochs/batchSize" }
        val rng = NaeLcg(seed)
        val p = if (initWeights != null) decode(initWeights) else initParams(rng)
        val header = if (initWeights != null) headerOf(initWeights) else DEFAULT_HEADER

        val n = samples.size
        val y = DoubleArray(n) { samples[it].targetLogU.coerceIn(-LN10, LN10) }
        // split: shuffled index order, first nVal held out (nae_train_app's
        // order[:n_val], here over samples - the app trains from one merged pool)
        val order = IntArray(n) { it }
        shuffle(order, rng)
        val nVal = if (valFraction <= 0.0) 0
                   else max(1, (valFraction * n).roundToInt()).coerceAtMost(n - 1)
        val valIdx = order.copyOfRange(0, nVal)
        val trIdx = order.copyOfRange(nVal, n)

        val m = Array(p.size) { DoubleArray(p[it].size) }   // Adam 1st/2nd moments
        val v = Array(p.size) { DoubleArray(p[it].size) }
        var t = 0
        var best = Double.MAX_VALUE
        var bestP: Array<FloatArray>? = null
        var bad = 0
        for (ep in 0 until epochs) {
            shuffle(trIdx, rng)                             // torch.randperm mirror
            var tot = 0.0
            var i0 = 0
            while (i0 < trIdx.size) {
                val i1 = min(i0 + batchSize, trIdx.size)
                val bh = Array(i1 - i0) { samples[trIdx[i0 + it]].hist }
                val by = DoubleArray(i1 - i0) { y[trIdx[i0 + it]] }
                val (loss, g) = lossAndGrad(p, bh, by)
                tot += loss * (i1 - i0)                     // pre-step loss, like torch
                t++
                adamStep(p, g, m, v, t, lr)
                i0 = i1
            }
            val trainLoss = tot / trIdx.size
            val valLoss = if (nVal > 0) {
                val vh = Array(nVal) { samples[valIdx[it]].hist }
                val vy = DoubleArray(nVal) { y[valIdx[it]] }
                loss(p, vh, vy)
            } else Double.NaN
            onEpoch(ep, trainLoss, valLoss)
            if (nVal > 0) {
                if (valLoss < best - MIN_DELTA) {
                    best = valLoss; bad = 0
                    bestP = Array(p.size) { p[it].copyOf() }
                } else if (++bad >= PATIENCE) break
            }
        }
        return encode(bestP ?: p, header)
    }

    // -- forward ------------------------------------------------------------
    // Activations reused across samples; a3 doubles as the flat fc1 input
    // (c*4+t == torch flatten(1), same as NeuralAeNet line 60-61).
    private class Acts {
        val x0 = DoubleArray(59 * 256)
        val a1 = DoubleArray(64 * 64)
        val a2 = DoubleArray(64 * 16)
        val a3 = DoubleArray(64 * 4)
        val a4 = DoubleArray(128)
        val z5 = DoubleArray(64)
        val a6 = DoubleArray(64)
        var s = 0.0
        var out = 0.0
    }

    /** conv1d k=4 s=4 + relu. Loop order identical to NeuralAeNet.conv (69-84). */
    private fun convFwd(x: DoubleArray, inC: Int, inL: Int,
                        w: FloatArray, b: FloatArray, yOut: DoubleArray) {
        val outL = inL / 4                                  // stride == kernel
        for (o in 0 until 64) {
            for (j in 0 until outL) {
                var acc = b[o].toDouble()
                for (i in 0 until inC) {
                    val base = i * inL + j * 4
                    val wb = (o * inC + i) * 4
                    for (k in 0 until 4) acc += w[wb + k] * x[base + k]
                }
                yOut[o * outL + j] = if (acc > 0.0) acc else 0.0
            }
        }
    }

    /** fc (+ optional relu). Loop order identical to NeuralAeNet.fc (86-95). */
    private fun fcFwd(x: DoubleArray, w: FloatArray, b: FloatArray,
                      nOut: Int, nIn: Int, relu: Boolean, yOut: DoubleArray) {
        for (o in 0 until nOut) {
            var acc = b[o].toDouble()
            for (i in 0 until nIn) acc += w[o * nIn + i] * x[i]
            yOut[o] = if (relu && acc < 0.0) 0.0 else acc
        }
    }

    /** Full forward, keeping activations. Mirrors predictLogU (55-66) exactly. */
    private fun forwardActs(p: Array<FloatArray>, hist: FloatArray, a: Acts) {
        for (i in hist.indices) a.x0[i] = hist[i].toDouble()
        convFwd(a.x0, 59, 256, p[0], p[1], a.a1)            // -> [64,64]
        convFwd(a.a1, 64, 64, p[2], p[3], a.a2)             // -> [64,16]
        convFwd(a.a2, 64, 16, p[4], p[5], a.a3)             // -> [64,4]
        fcFwd(a.a3, p[6], p[7], 128, 256, true, a.a4)
        fcFwd(a.a4, p[8], p[9], 64, 128, false, a.z5)
        fcFwd(a.z5, p[10], p[11], 64, 64, true, a.a6)       // joint
        var s = p[13][0].toDouble()                         // head, nOut = 1
        for (i in 0 until 64) s += p[12][i] * a.a6[i]
        a.s = s
        a.out = 2.0 * (1.0 / (1.0 + exp(-s)) - 0.5) * LN10  // Eq. 4 squash (line 65)
    }

    /** Trainer-side predictLogU; bit-identical to NeuralAeNet on the same floats. */
    internal fun forwardLogU(p: Array<FloatArray>, hist: FloatArray): Double {
        val a = Acts()
        forwardActs(p, hist, a)
        return a.out
    }

    /** Batch-mean MSE of the squashed output vs targets (no gradients). */
    internal fun loss(p: Array<FloatArray>, hists: Array<FloatArray>, y: DoubleArray): Double {
        val a = Acts()
        var acc = 0.0
        for (b in hists.indices) {
            forwardActs(p, hists[b], a)
            val e = a.out - y[b]
            acc += e * e
        }
        return acc / hists.size
    }

    // -- backward -----------------------------------------------------------
    // Verified against central differences in sense/nae_trainer_ref.py; relu grad
    // mask = (activation > 0), 0 at pre==0 like torch. Grads are float64 sums over
    // the batch of d(mean sq err)/d(param).

    /** fc backward: dY [nOut] (already relu-masked) -> accumulate gw/gb, return dX. */
    private fun fcBwd(dY: DoubleArray, x: DoubleArray, w: FloatArray,
                      gw: DoubleArray, gb: DoubleArray, nOut: Int, nIn: Int): DoubleArray {
        val dX = DoubleArray(nIn)
        for (o in 0 until nOut) {
            val dy = dY[o]
            if (dy == 0.0) continue
            gb[o] += dy
            val row = o * nIn
            for (i in 0 until nIn) {
                gw[row + i] += dy * x[i]
                dX[i] += dy * w[row + i]
            }
        }
        return dX
    }

    /** conv1d k4 s4 backward: window j touches only x[i, 4j..4j+3] (stride==kernel),
     *  so dX slices are disjoint per j. dPre already relu-masked. */
    private fun convBwd(dPre: DoubleArray, x: DoubleArray, w: FloatArray,
                        gw: DoubleArray, gb: DoubleArray,
                        inC: Int, inL: Int, needDx: Boolean): DoubleArray? {
        val outL = inL / 4
        val dX = if (needDx) DoubleArray(inC * inL) else null
        for (o in 0 until 64) {
            for (j in 0 until outL) {
                val gj = dPre[o * outL + j]
                if (gj == 0.0) continue
                gb[o] += gj
                for (i in 0 until inC) {
                    val base = i * inL + j * 4
                    val wb = (o * inC + i) * 4
                    for (k in 0 until 4) {
                        gw[wb + k] += gj * x[base + k]
                        if (dX != null) dX[base + k] += gj * w[wb + k]
                    }
                }
            }
        }
        return dX
    }

    private fun mask(d: DoubleArray, act: DoubleArray) {
        for (i in d.indices) if (act[i] <= 0.0) d[i] = 0.0
    }

    /** Batch loss + gradients for every tensor, nae-bin-v1 order. */
    internal fun lossAndGrad(p: Array<FloatArray>, hists: Array<FloatArray>,
                             y: DoubleArray): Pair<Double, Array<DoubleArray>> {
        val g = Array(NeuralAeNet.LAYERS.size) { DoubleArray(NeuralAeNet.LAYERS[it].second) }
        val bSize = hists.size
        val a = Acts()
        var loss = 0.0
        for (b in 0 until bSize) {
            forwardActs(p, hists[b], a)
            val err = a.out - y[b]
            loss += err * err
            val dOut = 2.0 * err / bSize                    // d(mean sq err)/d out_b
            val sig = 1.0 / (1.0 + exp(-a.s))
            val dS = dOut * 2.0 * LN10 * sig * (1.0 - sig)  // through Eq. 4 squash
            val da6 = DoubleArray(64)                       // head [1,64]
            for (i in 0 until 64) {
                g[12][i] += dS * a.a6[i]
                da6[i] = dS * p[12][i]
            }
            g[13][0] += dS
            mask(da6, a.a6)                                 // joint relu
            val dz5 = fcBwd(da6, a.z5, p[10], g[10], g[11], 64, 64)
            val da4 = fcBwd(dz5, a.a4, p[8], g[8], g[9], 64, 128)  // fc2, no relu
            mask(da4, a.a4)                                 // fc1 relu
            val da3 = fcBwd(da4, a.a3, p[6], g[6], g[7], 128, 256)
            mask(da3, a.a3)                                 // conv relu chain
            val da2 = convBwd(da3, a.a2, p[4], g[4], g[5], 64, 16, true)!!
            mask(da2, a.a2)
            val da1 = convBwd(da2, a.a1, p[2], g[2], g[3], 64, 64, true)!!
            mask(da1, a.a1)
            convBwd(da1, a.x0, p[0], g[0], g[1], 59, 256, false)   // input: no grad
        }
        return (loss / bSize) to g
    }

    // -- optimizer / init / shuffle ------------------------------------------

    /** torch.optim.Adam defaults with bias correction; float64 state, params
     *  re-stored float32 (the .bin's precision). */
    private fun adamStep(p: Array<FloatArray>, g: Array<DoubleArray>,
                         m: Array<DoubleArray>, v: Array<DoubleArray>, t: Int, lr: Double) {
        val bc1 = 1.0 - BETA1.pow(t)
        val bc2 = 1.0 - BETA2.pow(t)
        for (li in p.indices) {
            val pt = p[li]; val gt = g[li]; val mt = m[li]; val vt = v[li]
            for (i in pt.indices) {
                val gi = gt[i]
                mt[i] = BETA1 * mt[i] + (1.0 - BETA1) * gi
                vt[i] = BETA2 * vt[i] + (1.0 - BETA2) * gi * gi
                pt[i] = (pt[i] - lr * (mt[i] / bc1) / (sqrt(vt[i] / bc2) + EPS)).toFloat()
            }
        }
    }

    /** Fresh weights, nae-bin-v1 tensor order. Sequential rng draws per tensor;
     *  head tensors are zero and consume no draws (numpy ref mirrors this). */
    internal fun initParams(rng: NaeLcg): Array<FloatArray> =
        Array(NeuralAeNet.LAYERS.size) { li ->
            val nFloats = NeuralAeNet.LAYERS[li].second
            val fanIn = FAN_IN[li]
            if (fanIn < 0) FloatArray(nFloats)              // zero head
            else {
                val bound = 1.0 / sqrt(fanIn.toDouble())
                FloatArray(nFloats) { ((rng.nextU() * 2.0 - 1.0) * bound).toFloat() }
            }
        }

    /** Fisher-Yates with the shared LCG: j = floor(u*(i+1)), clamped. */
    private fun shuffle(a: IntArray, rng: NaeLcg) {
        for (i in a.size - 1 downTo 1) {
            val j = (rng.nextU() * (i + 1)).toInt().coerceAtMost(i)
            val tmp = a[i]; a[i] = a[j]; a[j] = tmp
        }
    }

    // -- nae-bin-v1 io -------------------------------------------------------

    /** Parse a nae-bin-v1 blob into flat tensors (same checks as NeuralAeNet 33-51,
     *  minus the per-name scan - the loader is the authority on that). */
    internal fun decode(bytes: ByteArray): Array<FloatArray> {
        require(bytes.size >= 4) { "truncated .bin" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val headerLen = bb.int
        require(headerLen in 2..bytes.size - 4) { "bad header length $headerLen" }
        val header = ByteArray(headerLen).also { bb.get(it) }.toString(Charsets.UTF_8)
        require(header.contains("nae-bin-v1")) { "not a nae-bin-v1 file" }
        require(bb.remaining() == NeuralAeNet.TOTAL_FLOATS * 4) {
            "weight blob ${bb.remaining()} bytes, expected ${NeuralAeNet.TOTAL_FLOATS * 4}"
        }
        return Array(NeuralAeNet.LAYERS.size) { li ->
            FloatArray(NeuralAeNet.LAYERS[li].second) { bb.float }
        }
    }

    /** uint32-LE header length + header + tensors as LE float32 in fixed order
     *  (nae_train_app.export_bin's layout). */
    internal fun encode(p: Array<FloatArray>, header: ByteArray): ByteArray {
        val bb = ByteBuffer.allocate(4 + header.size + NeuralAeNet.TOTAL_FLOATS * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        bb.putInt(header.size)
        bb.put(header)
        for (tensor in p) for (f in tensor) bb.putFloat(f)
        return bb.array()
    }

    private fun headerOf(bytes: ByteArray): ByteArray {
        val len = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).int
        return bytes.copyOfRange(4, 4 + len)
    }

    companion object {
        private val LN10 = ln(10.0)                         // Eq. 4 bound, M = 10
        private const val BETA1 = 0.9                       // torch Adam defaults
        private const val BETA2 = 0.999
        private const val EPS = 1e-8
        private const val PATIENCE = 15                     // nae_train_app defaults
        private const val MIN_DELTA = 1e-6

        /** fan_in per nae-bin-v1 tensor (bias shares its weight's fan_in), for the
         *  torch kaiming-uniform default bound 1/sqrt(fan_in); -1 = head zero-init. */
        private val FAN_IN = intArrayOf(
            236, 236,       // conv0: 59*4
            256, 256,       // conv2: 64*4
            256, 256,       // conv4: 64*4
            256, 256,       // fc0
            128, 128,       // fc2
            64, 64,         // joint
            -1, -1)         // head: zeros

        /** Exact header nae_train_app.export_bin writes for the app 3x3 grid
         *  (byte-identical to NeuralAeNetTest's embedded copy). */
        internal val DEFAULT_HEADER = (
            """{"format": "nae-bin-v1", "arch": "hist_scalar", "m_exp": 10.0, """ +
            """"grid": {"gains": [100, 200, 400], "exposures_us": [16000, 32000, 64000]}, """ +
            """"layers": [{"name": "hist.conv.0.weight", "shape": [64, 59, 4]}, """ +
            """{"name": "hist.conv.0.bias", "shape": [64]}, """ +
            """{"name": "hist.conv.2.weight", "shape": [64, 64, 4]}, """ +
            """{"name": "hist.conv.2.bias", "shape": [64]}, """ +
            """{"name": "hist.conv.4.weight", "shape": [64, 64, 4]}, """ +
            """{"name": "hist.conv.4.bias", "shape": [64]}, """ +
            """{"name": "hist.fc.0.weight", "shape": [128, 256]}, """ +
            """{"name": "hist.fc.0.bias", "shape": [128]}, """ +
            """{"name": "hist.fc.2.weight", "shape": [64, 128]}, """ +
            """{"name": "hist.fc.2.bias", "shape": [64]}, """ +
            """{"name": "joint.0.weight", "shape": [64, 64]}, """ +
            """{"name": "joint.0.bias", "shape": [64]}, """ +
            """{"name": "head.weight", "shape": [1, 64]}, """ +
            """{"name": "head.bias", "shape": [1]}]}""").toByteArray(Charsets.UTF_8)
    }
}
