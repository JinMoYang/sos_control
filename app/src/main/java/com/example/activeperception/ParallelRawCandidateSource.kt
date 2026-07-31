package com.example.activeperception

import android.graphics.Bitmap
import com.example.activeperception.acquire.CandidateSource
import com.example.activeperception.acquire.Formation
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.RawCapturer
import com.example.activeperception.acquire.RawFrame
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Drop-in for [com.example.activeperception.acquire.RawCandidateSource], bit-equivalent but
 * much faster. Two changes: an sRGB lookup table replacing the per-pixel `pow(x, 1/2.4)`,
 * and a worker pool splitting the independent per-row and per-cell work.
 *
 * 4 threads is the measured sweet spot on Snapdragon 8 Elite before memory bandwidth
 * dominates. The pure math in `acquire/Formation.kt` is left untouched.
 */
class ParallelRawCandidateSource(
    private val grid: Grid,
    private val capturer: RawCapturer,
    nThreads: Int = 4
) : CandidateSource<Bitmap> {

    private val pool = Executors.newFixedThreadPool(nThreads)
    @Volatile private var srgbLut: IntArray = IntArray(0)
    @Volatile private var lutMaxDn: Double = -1.0

    /** sRGB LUT for the given white level, rebuilt only when it changes. Values fed to it are
     *  already clipped to [0, maxDn] by the caller, so the table needs no headroom. */
    private fun lutFor(maxDn: Double): IntArray {
        if (lutMaxDn != maxDn) {
            val n = maxDn.toInt() + 1
            srgbLut = IntArray(n) { Formation.srgbU8(it.toDouble(), maxDn) }
            lutMaxDn = maxDn
        }
        return srgbLut
    }

    override fun render(cells: IntArray): List<Bitmap> {
        val rows = cells.map { grid.indices(it).second }.toSortedSet()
        val frames: List<RawFrame> = if (rows.size == 1) {
            capturer.capture(grid.exposuresUs[rows.first()], grid.baseGain, 1)
        } else {
            capturer.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        }
        require(frames.isNotEmpty()) { "RawCapturer returned no frames" }
        return formAllCells(frames, cells)
    }

    /** Form bitmaps from already-captured frames, so callers that want to time formation
     *  alone (Bench, Verify) can keep capture out of the measurement. Always returns exactly
     *  `cells.size` bitmaps, in order — callers index detections against `cells`. */
    fun formAllCells(frames: List<RawFrame>, cells: IntArray): List<Bitmap> {
        val avail = frames.size
        val w = frames[0].width; val h = frames[0].height
        val maxDn = frames[0].maxDn; val cfa = frames[0].cfaPattern
        val lut = lutFor(maxDn)
        val lutMax = lut.size - 1

        // Per shutter row: burst-sum + demosaic. This is the expensive stage and it scales
        // with the number of DISTINCT rows in `cells`, not with cells.size.
        val rows = cells.map { grid.indices(it).second }.toSortedSet()
        val rowFutures = rows.map { sj ->
            pool.submit<Pair<Int, Array<IntArray>>> {
                val n = minOf(grid.burstN(sj), avail)
                val summed = IntArray(frames[0].bayer.size)
                for (k in 0 until n) {
                    val bk = frames[k].bayer
                    for (p in summed.indices) summed[p] += bk[p]
                }
                sj to Formation.demosaic2x2(summed, w, h, cfa)
            }
        }
        val rgbByRow = HashMap<Int, Array<IntArray>>()
        for (f in rowFutures) { val (sj, rgb) = f.get(); rgbByRow[sj] = rgb }

        // Per cell: digital re-gain + sRGB encode + ARGB pack.
        val ow = w / 2; val oh = h / 2
        val cellFutures = cells.map { c ->
            pool.submit<IntArray> {
                val (gi, sj) = grid.indices(c)
                val rgb = rgbByRow[sj]!!
                packArgbWithLut(rgb[0], rgb[1], rgb[2], grid.gainRatio(gi), lut, lutMax)
            }
        }
        return cellFutures.map {
            Bitmap.createBitmap(it.get(), ow, oh, Bitmap.Config.ARGB_8888)
        }
    }

    /** Equivalent to `Formation.packCandidateArgb`, minus the per-pixel `Math.pow`. */
    private fun packArgbWithLut(
        r: IntArray, g: IntArray, b: IntArray, gainRatio: Double,
        lut: IntArray, lutMax: Int
    ): IntArray {
        val out = IntArray(r.size)
        for (p in r.indices) {
            val ri = (r[p] * gainRatio).toInt().coerceIn(0, lutMax)
            val gi = (g[p] * gainRatio).toInt().coerceIn(0, lutMax)
            val bi = (b[p] * gainRatio).toInt().coerceIn(0, lutMax)
            out[p] = (0xFF shl 24) or (lut[ri] shl 16) or (lut[gi] shl 8) or lut[bi]
        }
        return out
    }

    fun shutdown() {
        pool.shutdown()
        pool.awaitTermination(2, TimeUnit.SECONDS)
    }
}
