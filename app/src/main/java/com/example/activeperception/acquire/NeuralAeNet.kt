package com.example.activeperception.acquire

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.min

/**
 * Dependency-free inference for the Neural-AE-style exposure predictor (Onzon et al.,
 * CVPR 2021; hist-only scalar variant, trained by sense/nae_train_app.py). Pure Kotlin,
 * no TFLite/Android: conv1d(59->64,k4,s4)+relu x3 -> fc(256->128)+relu -> fc(128->64)
 * -> joint fc(64->64)+relu -> head fc(64->1), output 2*(sigmoid-0.5)*ln(10) = log of the
 * multiplicative exposure change u in [1/10, 10] (their Eq. 4, M=10).
 *
 * Weight file = nae-bin-v1 (spec in nae_train_app.py's docstring): uint32 LE header
 * length, UTF-8 JSON header, then per layer in fixed order the tensor as LE float32,
 * C row-major, PyTorch layout (conv [outC,inC,k], linear [out,in]). The architecture
 * fixes every shape, so the header is only sanity-checked (format tag + layer names in
 * order), not structurally parsed - keeps this file free of any JSON library.
 *
 * Math is float32 weights with Double accumulators, mirroring sense/nae_ref_check.py's
 * loop reference; goldens in NeuralAeNetTest assert lockstep at 1e-4.
 */
class NeuralAeNet(weights: ByteArray) {

    private val layers: Array<FloatArray>

    init {
        require(weights.size >= 4) { "truncated .bin" }
        val bb = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN)
        val headerLen = bb.int
        require(headerLen in 2..weights.size - 4) { "bad header length $headerLen" }
        val header = ByteArray(headerLen).also { bb.get(it) }.toString(Charsets.UTF_8)
        require(header.contains("nae-bin-v1")) { "not a nae-bin-v1 file" }
        var at = 0
        for ((name, _) in LAYERS) {
            at = header.indexOf("\"$name\"", at)
            require(at >= 0) { "header missing layer $name (or out of order)" }
        }
        require(bb.remaining() == TOTAL_FLOATS * 4) {
            "weight blob ${bb.remaining()} bytes, expected ${TOTAL_FLOATS * 4}"
        }
        layers = Array(LAYERS.size) { li ->
            FloatArray(LAYERS[li].second) { bb.float }
        }
    }

    /** hist = 59 multi-scale histograms row-major (hist[c*256+t], NaeFeatures layout)
     *  -> log u, bounded to [-ln 10, ln 10]. */
    fun predictLogU(hist: FloatArray): Double {
        require(hist.size == N_HIST * N_BINS) { "hist size ${hist.size} != ${N_HIST * N_BINS}" }
        var x = DoubleArray(hist.size) { hist[it].toDouble() }
        x = conv(x, N_HIST, N_BINS, layers[0], layers[1])       // -> [64,64]
        x = conv(x, 64, 64, layers[2], layers[3])               // -> [64,16]
        x = conv(x, 64, 16, layers[4], layers[5])               // -> [64,4], flat c*4+t
        var z = fc(x, layers[6], layers[7], 128, 256, relu = true)
        z = fc(z, layers[8], layers[9], 64, 128, relu = false)
        z = fc(z, layers[10], layers[11], 64, 64, relu = true)  // joint
        val s = fc(z, layers[12], layers[13], 1, 64, relu = false)[0]
        return 2.0 * (1.0 / (1.0 + exp(-s)) - 0.5) * ln(10.0)
    }

    /** conv1d k=4 s=4 + relu; x flat [inC*inL], w [64*inC*4] (torch [outC,inC,k]). */
    private fun conv(x: DoubleArray, inC: Int, inL: Int, w: FloatArray, b: FloatArray): DoubleArray {
        val outL = (inL - 4) / 4 + 1
        val y = DoubleArray(64 * outL)
        for (o in 0 until 64) {
            for (j in 0 until outL) {
                var acc = b[o].toDouble()
                for (i in 0 until inC) {
                    val base = i * inL + j * 4
                    val wb = (o * inC + i) * 4
                    for (k in 0 until 4) acc += w[wb + k] * x[base + k]
                }
                y[o * outL + j] = if (acc > 0.0) acc else 0.0
            }
        }
        return y
    }

    private fun fc(x: DoubleArray, w: FloatArray, b: FloatArray,
                   nOut: Int, nIn: Int, relu: Boolean): DoubleArray {
        val y = DoubleArray(nOut)
        for (o in 0 until nOut) {
            var acc = b[o].toDouble()
            for (i in 0 until nIn) acc += w[o * nIn + i] * x[i]
            y[o] = if (relu && acc < 0.0) 0.0 else acc
        }
        return y
    }

    companion object {
        const val N_HIST = 59
        const val N_BINS = 256

        /** nae-bin-v1 layer order (name, float count). Shapes are fixed by the net. */
        val LAYERS = listOf(
            "hist.conv.0.weight" to 64 * 59 * 4, "hist.conv.0.bias" to 64,
            "hist.conv.2.weight" to 64 * 64 * 4, "hist.conv.2.bias" to 64,
            "hist.conv.4.weight" to 64 * 64 * 4, "hist.conv.4.bias" to 64,
            "hist.fc.0.weight" to 128 * 256, "hist.fc.0.bias" to 128,
            "hist.fc.2.weight" to 64 * 128, "hist.fc.2.bias" to 64,
            "joint.0.weight" to 64 * 64, "joint.0.bias" to 64,
            "head.weight" to 64, "head.bias" to 1)

        val TOTAL_FLOATS = LAYERS.sumOf { it.second }           // 93441
    }
}

/**
 * Multi-scale histogram features, SHARED SPEC with nae_train_app.multiscale_hist -
 * the two must stay in lockstep (goldens in NeuralAeNetTest + sense/nae_ref_check.py):
 *  - luma plane row-major lum[y*w+x], values in [0,255]
 *  - 59 rows: row 0 = 1x1 global; rows 1..9 = 3x3 row-major (ty*3+tx); rows 10..58 =
 *    7x7 row-major; tile (ty,tx) of nxn covers rows [ty*h/n, (ty+1)*h/n) and cols
 *    [tx*w/n, (tx+1)*w/n) with integer floor division (needs w >= 7, h >= 7)
 *  - bin = clamp(floor(luma), 0, 255)
 *  - value = (count / tilePixels) computed in Double, stored as Float
 * The 1+9+49 tiling is our reading of Onzon's 59 multi-scale histograms (ASSUMED -
 * see nae_train_app.py). Trainer-side luma is BT.601 of the saved JPEG; feed the
 * matching BT.601 luma of the formed candidate here.
 */
object NaeFeatures {
    private val TILINGS = intArrayOf(1, 3, 7)

    fun multiScaleHist(lum: FloatArray, w: Int, h: Int): FloatArray {
        require(lum.size == w * h) { "lum size ${lum.size} != $w*$h" }
        require(w >= 7 && h >= 7) { "image $w x $h smaller than 7x7 tiling" }
        val out = FloatArray(NeuralAeNet.N_HIST * NeuralAeNet.N_BINS)
        val counts = IntArray(NeuralAeNet.N_BINS)
        var row = 0
        for (n in TILINGS) {
            for (ty in 0 until n) {
                val y0 = ty * h / n; val y1 = (ty + 1) * h / n
                for (tx in 0 until n) {
                    val x0 = tx * w / n; val x1 = (tx + 1) * w / n
                    java.util.Arrays.fill(counts, 0)
                    for (y in y0 until y1) {
                        val base = y * w
                        for (x in x0 until x1) {
                            val b = floor(lum[base + x].toDouble()).toInt()
                            counts[if (b < 0) 0 else if (b > 255) 255 else b]++
                        }
                    }
                    val npix = ((y1 - y0) * (x1 - x0)).toDouble()
                    val off = row * NeuralAeNet.N_BINS
                    for (b in 0 until NeuralAeNet.N_BINS)
                        out[off + b] = (counts[b] / npix).toFloat()
                    row++
                }
            }
        }
        return out
    }
}

/**
 * Grid-side helpers for acting on predictLogU, ported from sense/neural_ae_app.py
 * (GridSpec): exposure convention e = (iso/100) * shutterSeconds, cell id =
 * gainIdx * nShutter + shutterIdx (acquire/Grid.kt). Grid-parameterized - pass the
 * active Grid's gains/exposuresUs (any nIso x nShutter).
 */
object NaeSnap {
    /** Exposure value of a cell. */
    fun cellE(cell: Int, gains: IntArray, exposuresUs: IntArray): Double {
        val nSh = exposuresUs.size
        return (gains[cell / nSh] / 100.0) * (exposuresUs[cell % nSh] * 1e-6)
    }

    /** Eq. 6 shutter-priority split of target exposure e, snapped to the grid ->
     *  cell id. T_MAX = the grid's longest shutter (sim used 15 ms on its 5x5). */
    fun splitAndSnap(e: Double, gains: IntArray, exposuresUs: IntArray): Int {
        var tMaxUs = exposuresUs[0]
        for (u in exposuresUs) if (u > tMaxUs) tMaxUs = u
        val tMax = tMaxUs * 1e-6
        val t = min(e, tMax)
        val k = max(1.0, e / tMax)
        var sh = 0; var bestSh = Double.MAX_VALUE
        for (j in exposuresUs.indices) {
            val d = abs(log2(exposuresUs[j] * 1e-6) - log2(max(t, 1e-9)))
            if (d < bestSh) { bestSh = d; sh = j }
        }
        var iso = 0; var bestIso = Double.MAX_VALUE
        for (i in gains.indices) {
            val d = abs(log2(gains[i] / 100.0) - log2(k))
            if (d < bestIso) { bestIso = d; iso = i }
        }
        return iso * exposuresUs.size + sh
    }

    /** Eq. 5 EMA in log-exposure, mu = 0.9. */
    fun emaLogE(prevLogE: Double, newLogE: Double, mu: Double = 0.9): Double =
        mu * prevLogE + (1.0 - mu) * newLogE
}
