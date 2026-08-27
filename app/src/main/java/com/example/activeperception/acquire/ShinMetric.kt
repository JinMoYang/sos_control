package com.example.activeperception.acquire

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log2
import kotlin.math.sqrt

/**
 * Shin et al. IROS 2019 noise-aware image quality metric, ported from
 * Utils_Metric/Metric_Our.m (github.com/UkcheolShin/Noise-AwareCameraExposureControl,
 * master @ a264e63):
 *
 *   f = Alpha*Kg*M_grad + (1-Alpha)*Ke*H(I) - Beta*sigma_noise
 *
 * M_grad: Shim IROS'14 log-mapped normalized Sobel magnitude, 10x10 grid block
 * means, mean/std. H(I): 256-bin Shannon entropy. sigma_noise: Immerkaer CVIU'96
 * estimator restricted to a homogeneous (bottom-p gradient) AND unsaturated
 * [T_lower, T_upper] mask, whole-image fallback when the mask is near-empty.
 *
 * Single-plane port: the MATLAB runs noise per RGB channel and grad/entropy on
 * rgb2gray; on-device we only have the BT.601 luma plane, so all terms use it.
 * Oracle: sense/shin_metric_ref.py (NumPy mirror, generates the test goldens).
 * Arithmetic is kept exactly mirrorable: Sobel magnitude^2 stays as the exact
 * integer gx^2+gy^2 so mask thresholds cannot drift between ports.
 */
object ShinMetric {

    // Metric_Our.m hyperparameters, verbatim.
    private const val ALPHA = 0.4
    private const val BETA = 0.4
    private const val KG = 2.0             // gradient scaling, "empirically decided"
    private const val LAMBDA = 1.0e3       // Shim IROS'14 mapping
    private const val GAMMA = 0.06
    private const val KE = 0.125           // 1/8
    private const val P_HOMOG = 0.10       // bottom-p gradient percentile -> homogeneous mask
    private const val T_LOWER = 15         // unsaturated band
    private const val T_UPPER = 235
    private const val GRID = 10            // 10x10 grid stats on the mapped gradient

    // imgradient normalizer: max Sobel response is 4*255 per axis.
    private val GNORM_DEN = sqrt(16.0 * 255 * 255 + 16.0 * 255 * 255)

    /** Shin IROS'19 f(I) on a row-major luminance plane (0..255 floats). Higher = better. */
    fun score(lum: FloatArray, w: Int, h: Int): Double {
        require(lum.size == w * h) { "lum size ${lum.size} != ${w}x$h" }
        require(w >= GRID && h >= GRID) { "plane smaller than ${GRID}x$GRID grid" }
        // uint8 read (MATLAB imread): round half up, clamp 0..255.
        val img = IntArray(w * h) {
            floor(lum[it].toDouble() + 0.5).toInt().coerceIn(0, 255)
        }
        val g2 = sobelG2(img, w, h)
        return ALPHA * KG * gradTerm(g2, w, h) +
            (1 - ALPHA) * KE * entropyTerm(img) -
            BETA * noiseTerm(img, g2, w, h)
    }

    /** Mean intensity 0..255 — the NM initial-simplex input (Alg. 1). */
    fun meanIntensity(lum: FloatArray): Double {
        var s = 0.0
        for (v in lum) s += v
        return s / lum.size
    }

    /**
     * Adapter: ARGB pixels (as from Bitmap.getPixels) -> luminance plane via
     * BT.601 ints/1000, matching MeasurementController.meanLumaRatio's weights.
     */
    fun lumFromArgb(px: IntArray, w: Int, h: Int): FloatArray {
        require(px.size == w * h) { "px size ${px.size} != ${w}x$h" }
        return FloatArray(px.size) { i ->
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            ((r * 299 + g * 587 + b * 114) / 1000).coerceIn(0, 255).toFloat()
        }
    }

    /**
     * imgradient(I,'Sobel') magnitude^2: replicate-padded correlation with
     * Gx = [1 0 -1; 2 0 -2; 1 0 -1] and its transpose. gx,gy are ints, so
     * gx^2+gy^2 is exact in double.
     */
    private fun sobelG2(img: IntArray, w: Int, h: Int): DoubleArray {
        val g2 = DoubleArray(w * h)
        for (y in 0 until h) {
            val ym = (y - 1).coerceAtLeast(0) * w
            val y0 = y * w
            val yp = (y + 1).coerceAtMost(h - 1) * w
            for (x in 0 until w) {
                val xm = (x - 1).coerceAtLeast(0)
                val xp = (x + 1).coerceAtMost(w - 1)
                val gx = img[ym + xm] + 2 * img[y0 + xm] + img[yp + xm] -
                    img[ym + xp] - 2 * img[y0 + xp] - img[yp + xp]
                val gy = img[ym + xm] + 2 * img[ym + x] + img[ym + xp] -
                    img[yp + xm] - 2 * img[yp + x] - img[yp + xp]
                g2[y0 + x] = (gx * gx + gy * gy).toDouble()
            }
        }
        return g2
    }

    /**
     * Immerkaer sigma on the homogeneous-and-unsaturated mask. Laplacian is
     * conv2(I, [1 -2 1; -2 4 -2; 1 -2 1]) cropped to 'same', i.e. ZERO padding
     * (border bleed kept — the MATLAB keeps it too).
     */
    private fun noiseTerm(img: IntArray, g2: DoubleArray, w: Int, h: Int): Double {
        val n = w * h
        // Gth = sorted(grad^2)[int32(n*p)], 1-based with round-half-up.
        val sorted = g2.copyOf().also { it.sort() }
        val idx = floor(n * P_HOMOG + 0.5).toInt().coerceIn(1, n)
        val gth = sorted[idx - 1]

        var ns = 0
        var sumAbsMasked = 0.0
        var sumAbsAll = 0.0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val i = y * w + x
                // zero-padded 3x3: out-of-bounds neighbors contribute 0
                var lap = 4 * img[i]
                if (y > 0) lap -= 2 * img[i - w]
                if (y < h - 1) lap -= 2 * img[i + w]
                if (x > 0) lap -= 2 * img[i - 1]
                if (x < w - 1) lap -= 2 * img[i + 1]
                if (y > 0 && x > 0) lap += img[i - w - 1]
                if (y > 0 && x < w - 1) lap += img[i - w + 1]
                if (y < h - 1 && x > 0) lap += img[i + w - 1]
                if (y < h - 1 && x < w - 1) lap += img[i + w + 1]
                val a = abs(lap).toDouble()
                sumAbsAll += a
                if (g2[i] <= gth && img[i] in T_LOWER..T_UPPER) {
                    ns++
                    sumAbsMasked += a
                }
            }
        }
        // "no reliable regions": fall back to the original Immerkaer '96 form.
        if (ns < n * 1e-4) {
            return sqrt(PI / 2) * sumAbsAll / (6.0 * (w - 2) * (h - 2))
        }
        return sqrt(PI / 2) * sumAbsMasked / (6.0 * ns)
    }

    /**
     * Shim IROS'14 mapped gradient, GRID x GRID block means, mean/std (sample
     * std, N-1). Block bounds are proportional — identical to the MATLAB's
     * H/num blocks whenever GRID divides H and W (the only case it runs).
     * Flat image: std = 0 -> 0 (the MATLAB yields NaN 0/0; see caveat).
     */
    private fun gradTerm(g2: DoubleArray, w: Int, h: Int): Double {
        val ng = ln(LAMBDA * (1 - GAMMA) + 1)
        val cells = DoubleArray(GRID * GRID)
        for (i in 0 until GRID) {
            val r0 = h * i / GRID
            val r1 = h * (i + 1) / GRID
            for (j in 0 until GRID) {
                val c0 = w * j / GRID
                val c1 = w * (j + 1) / GRID
                var s = 0.0
                for (y in r0 until r1) {
                    for (x in c0 until c1) {
                        val gnorm = sqrt(g2[y * w + x]) / GNORM_DEN
                        if (gnorm >= GAMMA) s += ln(LAMBDA * (gnorm - GAMMA) + 1) / ng
                    }
                }
                cells[i * GRID + j] = s / ((r1 - r0) * (c1 - c0))
            }
        }
        var mean = 0.0
        for (c in cells) mean += c
        mean /= cells.size
        var varSum = 0.0
        for (c in cells) varSum += (c - mean) * (c - mean)
        val std = sqrt(varSum / (cells.size - 1))
        return if (std == 0.0) 0.0 else mean / std
    }

    /** MATLAB entropy(): 256-bin histogram, -sum p*log2(p) over nonzero bins. */
    private fun entropyTerm(img: IntArray): Double {
        val hist = IntArray(256)
        for (v in img) hist[v]++
        val n = img.size.toDouble()
        var e = 0.0
        for (c in hist) {
            if (c > 0) {
                val p = c / n
                e -= p * log2(p)
            }
        }
        return e
    }
}
