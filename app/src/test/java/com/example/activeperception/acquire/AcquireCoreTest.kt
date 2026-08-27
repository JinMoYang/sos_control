package com.example.activeperception.acquire

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-logic tests for the acquire-and-select core. No Android deps, run on the JVM
 * (`./gradlew testDebugUnitTest`). Mirrors the kotlinc-verified reference checks.
 */
class AcquireCoreTest {

    private class IntSource(val grid: Grid) : CandidateSource<Int> {
        override fun render(cells: IntArray): List<Int> = cells.map { grid.indices(it).first }
    }
    private class ConfDetector : Detector<Int> {
        override fun detectBatch(images: List<Int>) =
            images.map { listOf(Detection(floatArrayOf(0f, 0f, 1f, 1f), (it + 1).toFloat(), 2)) }
    }
    private class EmptyDetector : Detector<Int> {
        override fun detectBatch(images: List<Int>) = images.map { emptyList<Detection>() }
    }

    @Test fun gridRatiosAndBurst() {
        val g = REAL_4x4
        assertEquals(4, g.nGain); assertEquals(4, g.nShutter)
        assertEquals(listOf(1.0, 2.0, 4.0, 8.0), (0 until 4).map { g.gainRatio(it) })
        assertEquals(listOf(1, 2, 4, 8), (0 until 4).map { g.burstN(it) })
        assertEquals(8, g.maxBurst)
        for (gi in 0 until g.nGain) for (sj in 0 until g.nShutter)
            assertEquals(gi to sj, g.indices(g.cell(gi, sj)))
    }

    @Test fun planSchedule() {
        val g = REAL_4x4
        val anchor = g.cell(2, 1)
        val gainOnly = plan(1, anchor, g, 5)
        assertEquals(g.nGain, gainOnly.size)
        assertTrue(gainOnly.all { g.indices(it).second == 1 })
        assertEquals(listOf(0, 1, 2, 3), gainOnly.map { g.indices(it).first }.sorted())
        assertEquals(g.nGain * g.nShutter, plan(5, anchor, g, 5).size)
        assertEquals(16, plan(0, g.cell(0, 0), g, 5).size)   // t=0 is a burst frame
    }

    @Test fun controllerArgmaxAndAnchor() {
        val g = REAL_4x4
        val ctrl = AcquireSelectController(IntSource(g), ConfDetector(), g, 5, g.cell(0, 0))
        val r = ctrl.step()
        var amax = 0; for (i in r.scores.indices) if (r.scores[i] > r.scores[amax]) amax = i
        assertEquals(amax, r.chosen)
        assertEquals(g.nGain - 1, g.indices(r.cell).first)   // brightest = highest gain
        assertEquals(r.cell, ctrl.anchor)
        assertEquals(1, ctrl.t)
    }

    @Test fun sumConfUsesSelectThreshold() {
        fun d(c: Float) = Detection(floatArrayOf(0f, 0f, 1f, 1f), c, 2)
        assertEquals(0.3, sumConf(listOf(d(0.1f), d(0.3f), d(0.05f)), 0.25f), 1e-6)  // tail dropped
        assertEquals(0.0, sumConf(listOf(d(0.1f), d(0.1f)), 0.25f), 1e-9)            // all tail
    }

    @Test fun controllerEmptyHoldsAnchor() {
        val g = REAL_4x4
        val start = g.cell(2, 2)
        val ctrl = AcquireSelectController(IntSource(g), EmptyDetector(), g, 5, start)
        val r = ctrl.step()
        assertEquals(start, ctrl.anchor)                     // did NOT drift to lowest gain
        assertTrue(r.detections.isEmpty()); assertEquals(start, r.cell)
    }

    @Test fun formationRegainBurstSrgb() {
        val a = Formation.formCandidate(listOf(intArrayOf(100)), 1, 1.0, 1023.0)
        val b = Formation.formCandidate(listOf(intArrayOf(100)), 1, 2.0, 1023.0)
        assertTrue(b[0] > a[0])                              // re-gain x2 brighter
        val summed = Formation.formCandidate(listOf(intArrayOf(30), intArrayOf(70)), 2, 1.0, 1023.0)
        val direct = Formation.formCandidate(listOf(intArrayOf(100)), 1, 1.0, 1023.0)
        assertEquals(direct[0], summed[0])                  // burst sum == direct
        assertEquals(0, Formation.srgbU8(0.0, 1023.0))
        assertEquals(255, Formation.srgbU8(1023.0, 1023.0))
        val expMid = 1.055 * Math.pow(0.5, 1.0 / 2.4) - 0.055
        assertTrue(Math.abs(Formation.linearToSrgb(0.5) - expMid) < 1e-9)
    }

    @Test fun demosaicAndPack() {
        val w = 4; val h = 4
        val bayer = IntArray(w * h) { i ->
            val x = i % w; val y = i / w
            when { y % 2 == 0 && x % 2 == 0 -> 400; y % 2 == 1 && x % 2 == 1 -> 100; else -> 200 }
        }
        val (r, gg, b) = Formation.demosaic2x2(bayer, w, h, "RGGB").let { Triple(it[0], it[1], it[2]) }
        assertTrue(r.all { it == 400 } && gg.all { it == 200 } && b.all { it == 100 })
        val argb = Formation.packCandidateArgb(r, gg, b, 1.0, 1023.0)
        assertEquals(0xFF, (argb[0] ushr 24) and 0xFF)
        assertTrue(((argb[0] ushr 16) and 0xFF) > ((argb[0] ushr 8) and 0xFF))
    }

    @Test fun whiteBalanceAndColorMatrixAreAppliedBeforeSrgb() {
        val corrected = Formation.correctLinearRgb(
            100.0, 200.0, 50.0, 2.0,
            floatArrayOf(2f, 1f, 4f),
            doubleArrayOf(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )
        // WB+gain = [400,400,400], then CCM swaps R/G (still neutral).
        assertEquals(400.0, corrected[0], 1e-9)
        assertEquals(400.0, corrected[1], 1e-9)
        assertEquals(400.0, corrected[2], 1e-9)
    }

    @Test fun sensorOrientationRotatesPixelsAndDimensions() {
        // Source rows: [1 2 3] / [4 5 6]. Clockwise 90 => [4 1] / [5 2] / [6 3].
        val rotated = Formation.orientArgb(intArrayOf(1, 2, 3, 4, 5, 6), 3, 2, 90)
        assertEquals(2, rotated.width)
        assertEquals(3, rotated.height)
        assertEquals(listOf(4, 1, 5, 2, 6, 3), rotated.pixels.toList())
    }
}
