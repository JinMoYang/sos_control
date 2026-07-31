package com.example.activeperception.acquire

import android.graphics.Bitmap

/**
 * One linear RAW frame: black-level-subtracted Bayer plane plus geometry / CFA / white level.
 * `bayer.size == width * height`, values in [0, maxDn].
 */
class RawFrame(
    val bayer: IntArray,
    val width: Int,
    val height: Int,
    val cfaPattern: String,
    val maxDn: Double
)

/** Captures [nBurst] frames back-to-back at the given exposure/ISO with AE/AWB off. */
interface RawCapturer {
    fun capture(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame>
}

/**
 * Reference implementation of the digital re-gain path: capture once at the base ISO, form
 * the gain column by digital scaling; on a probe step sum the fastest-exposure burst to
 * realize the shutter rows first.
 *
 * Superseded on device by [com.example.activeperception.ParallelRawCandidateSource], which
 * is bit-equivalent but LUT-accelerated and parallel. Kept as the readable spec of the math.
 */
class RawCandidateSource(
    private val grid: Grid,
    private val capturer: RawCapturer
) : CandidateSource<Bitmap> {

    override fun render(cells: IntArray): List<Bitmap> {
        val rows = cells.map { grid.indices(it).second }.toSortedSet()
        val frames: List<RawFrame> = if (rows.size == 1) {
            capturer.capture(grid.exposuresUs[rows.first()], grid.baseGain, 1)
        } else {
            capturer.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        }
        require(frames.isNotEmpty()) { "RawCapturer returned no frames" }
        val avail = frames.size
        val w = frames[0].width; val h = frames[0].height
        val maxDn = frames[0].maxDn; val cfa = frames[0].cfaPattern

        // One burst-sum + demosaic per shutter row present in `cells`.
        val rgbByRow = HashMap<Int, Array<IntArray>>()
        for (sj in rows) {
            val n = minOf(grid.burstN(sj), avail)
            val summed = IntArray(frames[0].bayer.size)
            for (k in 0 until n) {
                val bk = frames[k].bayer
                for (p in summed.indices) summed[p] += bk[p]
            }
            rgbByRow[sj] = Formation.demosaic2x2(summed, w, h, cfa)
        }

        val ow = w / 2; val oh = h / 2
        return cells.map { c ->
            val (gi, sj) = grid.indices(c)
            val rgb = rgbByRow[sj]!!
            val argb = Formation.packCandidateArgb(rgb[0], rgb[1], rgb[2], grid.gainRatio(gi), maxDn)
            Bitmap.createBitmap(argb, ow, oh, Bitmap.Config.ARGB_8888)
        }
    }
}
