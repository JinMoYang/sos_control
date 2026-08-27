package com.example.activeperception.acquire

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure JVM tests for NeuralAeNet / NaeFeatures / NaeSnap against numpy goldens from
 * sense/nae_ref_check.py. Weights and inputs come from the shared 64-bit LCG defined
 * there (Kotlin Long arithmetic wraps mod 2^64, matching the reference's masking), so
 * only the expected outputs are embedded. Kotlin-side hist correctness is asserted by
 * these shared-spec vectors - the reference proved its loop mirror bit-identical to
 * the trainer's vectorized builder, and this test pins the Kotlin port to the same
 * numbers.
 */
class NeuralAeNetTest {

    /** Shared LCG spec (see nae_ref_check.py): state' = state*A + C; u = top24/2^24. */
    private class Lcg(seed: Long) {
        var state = seed
        fun nextU(): Double {
            state = state * 6364136223846793005L + 1442695040888963407L
            return (state ushr 40).toDouble() / (1 shl 24).toDouble()
        }
        fun weight(): Float = ((nextU() - 0.5) * 0.1).toFloat()
        fun hist(): Float = nextU().toFloat()
        fun luma(): Float = floor(nextU() * 256.0).coerceAtMost(255.0).toFloat()
    }

    /** Exact header written by nae_train_app.export_bin for the app 3x3 grid. */
    private val header = ("""{"format": "nae-bin-v1", "arch": "hist_scalar", "m_exp": 10.0, """ +
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

    private fun bin(fill: () -> Float): ByteArray {
        val bb = ByteBuffer.allocate(4 + header.size + NeuralAeNet.TOTAL_FLOATS * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        bb.putInt(header.size); bb.put(header)
        for ((_, count) in NeuralAeNet.LAYERS) repeat(count) { bb.putFloat(fill()) }
        return bb.array()
    }

    private fun golden(seed: Long, expected: Double) {
        val gen = Lcg(seed)
        val net = NeuralAeNet(bin { gen.weight() })          // weights first, then hist,
        val hist = FloatArray(NeuralAeNet.N_HIST * NeuralAeNet.N_BINS) { gen.hist() }
        val out = net.predictLogU(hist)                      // same stream as the ref
        assertEquals(expected, out, 1e-4)
        assertTrue(abs(out) <= ln(10.0))                     // Eq. 4 bound, M=10
    }

    @Test fun predictLogUMatchesNumpySeed42() = golden(42L, -0.010163148930)

    @Test fun predictLogUMatchesNumpySeed1234() = golden(1234L, -0.044071906837)

    @Test fun zeroWeightsPredictNoChange() {
        val net = NeuralAeNet(bin { 0f })                    // sigmoid(0) -> log u = 0
        val gen = Lcg(7L)
        assertEquals(0.0, net.predictLogU(FloatArray(59 * 256) { gen.hist() }), 1e-12)
    }

    @Test(expected = IllegalArgumentException::class)
    fun truncatedBlobRejected() {
        NeuralAeNet(bin { 0f }.copyOf(5000))
    }

    @Test fun multiScaleHistMatchesNumpyGoldens() {
        val w = 37; val h = 23                               // non-divisible by 3 and 7
        val gen = Lcg(99L)
        val lum = FloatArray(w * h) { gen.luma() }
        val hist = NaeFeatures.multiScaleHist(lum, w, h)
        assertEquals(59 * 256, hist.size)
        var checksum = 0.0
        for (i in hist.indices) checksum += hist[i].toDouble() * ((i % 31) + 1)
        assertEquals(929.122956335079, checksum, 1e-9)       // nae_ref_check.py goldens
        assertEquals(0.010575792752, hist[114].toDouble(), 1e-9)    // row 0 bin 114
        assertEquals(0.031250000000, hist[1376].toDouble(), 1e-9)   // row 5 bin 96
        assertEquals(0.055555555969, hist[7689].toDouble(), 1e-9)   // row 30 bin 9
        assertEquals(0.041666667908, hist[14849].toDouble(), 1e-9)  // row 58 bin 1
        for (row in 0 until 59) {                            // each tile hist normalized
            var s = 0.0
            for (b in 0 until 256) s += hist[row * 256 + b]
            assertEquals(1.0, s, 1e-5)
        }
    }

    @Test fun snapHelpersMatchPythonGridSpec() {
        val gains = intArrayOf(100, 200, 400)
        val expUs = intArrayOf(16000, 32000, 64000)
        val e = doubleArrayOf(0.016, 0.032, 0.064, 0.032, 0.064, 0.128, 0.064, 0.128, 0.256)
        val snap = intArrayOf(0, 1, 2, 1, 2, 5, 2, 5, 8)     // shutter-priority reachable set
        for (c in 0 until 9) {
            assertEquals(e[c], NaeSnap.cellE(c, gains, expUs), 1e-12)
            assertEquals(snap[c], NaeSnap.splitAndSnap(NaeSnap.cellE(c, gains, expUs), gains, expUs))
        }
        assertEquals(8, NaeSnap.splitAndSnap(10.0, gains, expUs))
        assertEquals(0, NaeSnap.splitAndSnap(1e-5, gains, expUs))
        assertEquals(5, NaeSnap.splitAndSnap(0.1, gains, expUs))
        assertEquals(0.1, NaeSnap.emaLogE(0.0, 1.0), 1e-12)  // Eq. 5, mu = 0.9
    }
}
