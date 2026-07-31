package com.example.activeperception.acquire

/**
 * Per-pixel image formation: burst-sum -> digital re-gain -> sRGB. Pure math, kept out of
 * the Android sources so it stays unit-testable. Demosaic output feeds Bitmap packing.
 */
object Formation {

    /** IEC 61966-2-1 sRGB OETF (linear [0,1] -> sRGB [0,1]). */
    fun linearToSrgb(x: Double): Double {
        val c = if (x < 0.0) 0.0 else x
        return if (c <= 0.0031308) 12.92 * c else 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055
    }

    /** One gained linear ADU value -> 8-bit sRGB code, clipped to [0, maxDn]. */
    fun srgbU8(linearAdu: Double, maxDn: Double): Int {
        val lin = (if (linearAdu < 0.0) 0.0 else if (linearAdu > maxDn) maxDn else linearAdu) / maxDn
        val s = linearToSrgb(lin)
        return Math.round((if (s < 0.0) 0.0 else if (s > 1.0) 1.0 else s) * 255.0).toInt()
    }

    /**
     * Form one candidate channel: sum the first [nSum] burst planes, apply [gainRatio],
     * clip, encode. [frames] are linear ADU planes, all captured at the base gain.
     */
    fun formCandidate(frames: List<IntArray>, nSum: Int, gainRatio: Double, maxDn: Double): IntArray {
        val n = frames[0].size
        val out = IntArray(n)
        for (p in 0 until n) {
            var s = 0.0
            for (k in 0 until nSum) s += frames[k][p].toDouble()
            out[p] = srgbU8(s * gainRatio, maxDn)
        }
        return out
    }

    /**
     * 2×2-block demosaic of a linear Bayer plane -> half-resolution linear [R,G,B].
     * [pattern] is the 4-char CFA of the top-left 2×2, row-major ("RGGB", "BGGR", ...),
     * from SENSOR_INFO_COLOR_FILTER_ARRANGEMENT. Block averaging rather than full-res
     * interpolation — the detector downsamples to 640 anyway.
     */
    fun demosaic2x2(bayer: IntArray, width: Int, height: Int, pattern: String): Array<IntArray> {
        val ow = width / 2
        val oh = height / 2
        val r = IntArray(ow * oh); val g = IntArray(ow * oh); val b = IntArray(ow * oh)
        val site = intArrayOf(0, 1, width, width + 1)   // (0,0),(0,1),(1,0),(1,1)
        for (by in 0 until oh) {
            for (bx in 0 until ow) {
                val base = (by * 2) * width + (bx * 2)
                var rv = 0; var gv = 0; var gn = 0; var bv = 0
                for (k in 0 until 4) {
                    val v = bayer[base + site[k]]
                    when (pattern[k]) {
                        'R' -> rv = v
                        'B' -> bv = v
                        else -> { gv += v; gn++ }
                    }
                }
                val o = by * ow + bx
                r[o] = rv; g[o] = if (gn > 0) gv / gn else gv; b[o] = bv
            }
        }
        return arrayOf(r, g, b)
    }

    /** Re-gain + sRGB-encode linear [R,G,B] planes into ARGB_8888 ints. */
    fun packCandidateArgb(r: IntArray, g: IntArray, b: IntArray,
                          gainRatio: Double, maxDn: Double): IntArray {
        val out = IntArray(r.size)
        for (p in r.indices) {
            val rr = srgbU8(r[p] * gainRatio, maxDn)
            val gg = srgbU8(g[p] * gainRatio, maxDn)
            val bb = srgbU8(b[p] * gainRatio, maxDn)
            out[p] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
        }
        return out
    }
}
