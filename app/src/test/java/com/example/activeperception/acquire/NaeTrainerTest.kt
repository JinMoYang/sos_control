package com.example.activeperception.acquire

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure JVM tests for NaeTrainer. Every scenario here was rehearsed bit-for-bit in
 * sense/nae_trainer_ref.py (same shared-spec LCG streams, same float32 storage):
 *  - the backward pass was verified there against float64 central differences
 *    (worst floored rel err 8.5e-6) before this port existed;
 *  - h = 2e-6 is the FD step the ref's scan found to cross NO relu kink for the
 *    fixed Lcg(11) stream (larger steps do, which breaks FD, not the gradient) -
 *    ref worst floored rel err 8.3e-5, so the 1e-3 gate has ~12x margin;
 *  - the overfit run reaches ~3e-8 by epoch 200 in the ref, ~4 decades under the
 *    1e-3 gate, absorbing any Kotlin-vs-numpy summation-order drift.
 */
class NaeTrainerTest {

    private fun sample(gen: NaeLcg): NaeTrainer.Sample {
        // draw order (hist floats, then target) mirrors the ref's fixtures
        val hist = FloatArray(NeuralAeNet.N_HIST * NeuralAeNet.N_BINS) { gen.nextU().toFloat() }
        return NaeTrainer.Sample(hist, gen.nextU() * 3.0 - 1.5)
    }

    /** Central-difference check of every tensor's analytic gradient. */
    @Test fun gradientsMatchCentralDifferences() {
        val gen = NaeLcg(11L)
        val trainer = NaeTrainer()
        val params = Array(NeuralAeNet.LAYERS.size) { li ->
            FloatArray(NeuralAeNet.LAYERS[li].second) { ((gen.nextU() - 0.5) * 0.1).toFloat() }
        }
        val hists = Array(2) { FloatArray(59 * 256) { gen.nextU().toFloat() } }
        val y = DoubleArray(2) { gen.nextU() * 3.0 - 1.5 }
        val (_, grads) = trainer.lossAndGrad(params, hists, y)
        val h = 2e-6                        // kink-free step for this exact stream
        for (li in params.indices) {
            val p = params[li]
            repeat(20) {
                val idx = (gen.nextU() * p.size).toInt().coerceAtMost(p.size - 1)
                val orig = p[idx]
                val hi = (orig.toDouble() + h).toFloat()
                val lo = (orig.toDouble() - h).toFloat()
                p[idx] = hi
                val l1 = trainer.loss(params, hists, y)
                p[idx] = lo
                val l2 = trainer.loss(params, hists, y)
                p[idx] = orig
                // divide by the ACTUAL float32 step, killing storage-rounding error
                val fd = (l1 - l2) / (hi.toDouble() - lo.toDouble())
                val g = grads[li][idx]
                // floor keeps near-zero grads (FD noise-dominated) judged on abs terms
                val rel = abs(g - fd) / max(max(abs(g), abs(fd)), 1e-6)
                assertTrue("tensor $li idx $idx: analytic=$g fd=$fd rel=$rel", rel < 1e-3)
            }
        }
    }

    /** Whole loop learns: 8 samples memorized to < 1e-3 (ref: 3e-8 at epoch 200). */
    @Test fun overfitsEightSamples() {
        val gen = NaeLcg(5L)
        val samples = List(8) { sample(gen) }
        var lastTrain = Double.MAX_VALUE
        var lastVal = 0.0
        val bytes = NaeTrainer(42L).train(
            samples, valFraction = 0.0, epochs = 200, batchSize = 8, lr = 1e-3,
            onEpoch = { _, tr, vl -> lastTrain = tr; lastVal = vl })
        assertTrue("last train loss $lastTrain", lastTrain < 1e-3)
        assertTrue(lastVal.isNaN())                          // no val split -> NaN
        // the RETURNED weights fit too (covers encode -> decode -> forward)
        val trainer = NaeTrainer(42L)
        val p = trainer.decode(bytes)
        val fit = trainer.loss(p, Array(8) { samples[it].hist },
                               DoubleArray(8) { samples[it].targetLogU })
        assertTrue("returned-weights fit $fit", fit < 1e-3)
    }

    /** Trained bytes load in NeuralAeNet and agree with the trainer's own forward;
     *  a fine-tune pass from those bytes stays loadable (header carried over). */
    @Test fun trainedBytesRoundTripThroughNeuralAeNet() {
        val gen = NaeLcg(3L)
        val samples = List(6) { sample(gen) }
        val trainer = NaeTrainer(42L)
        var epochsSeen = 0
        val bytes = trainer.train(samples, valFraction = 0.2, epochs = 3, batchSize = 4,
            onEpoch = { _, tr, vl ->
                epochsSeen++
                assertTrue(tr.isFinite() && vl.isFinite())   // val split active
            })
        assertEquals(3, epochsSeen)
        val net = NeuralAeNet(bytes)                         // loader accepts the blob
        val p = trainer.decode(bytes)
        repeat(3) {
            val hist = FloatArray(59 * 256) { gen.nextU().toFloat() }
            val own = trainer.forwardLogU(p, hist)
            assertEquals(own, net.predictLogU(hist), 1e-5)
            assertTrue(abs(own) <= ln(10.0))                 // Eq. 4 bound, M=10
        }
        // continue-from-bin path: decode, 1 epoch, re-encode with the same header
        val bytes2 = trainer.train(samples, valFraction = 0.0, epochs = 1,
                                   batchSize = 4, initWeights = bytes)
        val net2 = NeuralAeNet(bytes2)
        val hist = FloatArray(59 * 256) { gen.nextU().toFloat() }
        assertTrue(abs(net2.predictLogU(hist)) <= ln(10.0))
    }
}
