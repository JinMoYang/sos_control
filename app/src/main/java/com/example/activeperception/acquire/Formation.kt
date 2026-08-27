package com.example.activeperception.acquire

/**
 * Per-pixel image formation: burst-sum -> digital re-gain -> sRGB. Pure math, kept out of
 * the Android sources so it stays unit-testable. Demosaic output feeds Bitmap packing.
 */
object Formation {

    data class OrientedArgb(val pixels: IntArray, val width: Int, val height: Int)

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

    /**
     * Re-gain, white-balance, transform camera RGB to linear sRGB, then apply the sRGB OETF.
     * [whiteBalance] is R/G/B and [cameraToSrgb] is a row-major 3x3 matrix. Defaults preserve
     * the original minimal-ISP path for reference tests and non-Camera2 callers.
     */
    fun packCandidateArgb(r: IntArray, g: IntArray, b: IntArray,
                          gainRatio: Double, maxDn: Double,
                          whiteBalance: FloatArray = floatArrayOf(1f, 1f, 1f),
                          cameraToSrgb: DoubleArray = IDENTITY_3X3): IntArray {
        val out = IntArray(r.size)
        for (p in r.indices) {
            val wr = r[p] * gainRatio * whiteBalance[0]
            val wg = g[p] * gainRatio * whiteBalance[1]
            val wb = b[p] * gainRatio * whiteBalance[2]
            val rr = srgbU8(cameraToSrgb[0] * wr + cameraToSrgb[1] * wg + cameraToSrgb[2] * wb, maxDn)
            val gg = srgbU8(cameraToSrgb[3] * wr + cameraToSrgb[4] * wg + cameraToSrgb[5] * wb, maxDn)
            val bb = srgbU8(cameraToSrgb[6] * wr + cameraToSrgb[7] * wg + cameraToSrgb[8] * wb, maxDn)
            out[p] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
        }
        return out
    }

    fun correctLinearRgb(r: Double, g: Double, b: Double, gainRatio: Double,
                         whiteBalance: FloatArray, cameraToSrgb: DoubleArray): DoubleArray {
        require(whiteBalance.size == 3 && cameraToSrgb.size == 9)
        val wr = r * gainRatio * whiteBalance[0]
        val wg = g * gainRatio * whiteBalance[1]
        val wb = b * gainRatio * whiteBalance[2]
        return doubleArrayOf(
            cameraToSrgb[0] * wr + cameraToSrgb[1] * wg + cameraToSrgb[2] * wb,
            cameraToSrgb[3] * wr + cameraToSrgb[4] * wg + cameraToSrgb[5] * wb,
            cameraToSrgb[6] * wr + cameraToSrgb[7] * wg + cameraToSrgb[8] * wb
        )
    }

    /** Rotate the sensor raster clockwise before display, detection, and JPEG logging. */
    fun orientArgb(pixels: IntArray, width: Int, height: Int,
                   clockwiseDegrees: Int): OrientedArgb {
        require(pixels.size == width * height)
        return when ((clockwiseDegrees % 360 + 360) % 360) {
            0 -> OrientedArgb(pixels, width, height)
            90 -> {
                val out = IntArray(pixels.size)
                for (y in 0 until height) for (x in 0 until width) {
                    val nx = height - 1 - y
                    val ny = x
                    out[ny * height + nx] = pixels[y * width + x]
                }
                OrientedArgb(out, height, width)
            }
            180 -> {
                val out = IntArray(pixels.size)
                for (i in pixels.indices) out[pixels.lastIndex - i] = pixels[i]
                OrientedArgb(out, width, height)
            }
            270 -> {
                val out = IntArray(pixels.size)
                for (y in 0 until height) for (x in 0 until width) {
                    val nx = y
                    val ny = width - 1 - x
                    out[ny * height + nx] = pixels[y * width + x]
                }
                OrientedArgb(out, height, width)
            }
            else -> error("Only 0/90/180/270 degree RAW orientation is supported")
        }
    }

    val IDENTITY_3X3 = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
}
