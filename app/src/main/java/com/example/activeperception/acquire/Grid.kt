package com.example.activeperception.acquire

/**
 * Action grid: gain axis (digital re-gain) × shutter axis (burst sum).
 * Cell id = `gainIdx * nShutter + shutterIdx`. Pure logic, no Android deps.
 *
 * [digitalBoost] is folded into [gainRatio], so it brightens the VIRTUAL formation path
 * (Proposed / AE_quant / Fixed) without touching physical captures, which send `gains[gi]`
 * straight to SENSOR_SENSITIVITY. Pixel-wise virtual-vs-physical comparison must divide
 * the virtual side by it; detection counts are unaffected.
 */
class Grid(
    val gains: IntArray,
    val exposuresUs: IntArray,
    /** Mutable so the UI can change it without rebuilding the Grid; read on every formation. */
    var digitalBoost: Double = 1.0
) {
    val nGain: Int get() = gains.size
    val nShutter: Int get() = exposuresUs.size
    val baseGain: Int get() = gains[0]
    val fastestExposureUs: Int get() = exposuresUs[0]

    /** Digital multiplier for a gain index, relative to the base (capture) gain. */
    fun gainRatio(gainIdx: Int): Double =
        (gains[gainIdx].toDouble() / gains[0]) * digitalBoost

    /** Short sub-frames to sum to realize this shutter row. */
    fun burstN(shutterIdx: Int): Int =
        Math.round(exposuresUs[shutterIdx].toDouble() / exposuresUs[0]).toInt()

    val maxBurst: Int get() = burstN(nShutter - 1)

    fun cell(gainIdx: Int, shutterIdx: Int): Int = gainIdx * nShutter + shutterIdx
    fun indices(cell: Int): Pair<Int, Int> = Pair(cell / nShutter, cell % nShutter)
}

/*
 * Presets. Size `gains` from SENSOR_INFO_SENSITIVITY_RANGE and `exposuresUs` from
 * SENSOR_INFO_EXPOSURE_TIME_RANGE (both logged by RawSensorCapturer.open()). Exposure
 * ratios fix the burst counts: 1:2:4 -> 1,2,4 sub-frames.
 *
 * Device note (S25): iso_diag shows RAW pixels scaling cleanly with ISO up to ~400 and
 * flat beyond it — the analog gain caps there and higher ISOs apply digital gain that
 * bypasses the RAW path. Grids whose gain axis stays <= 400 are the ones where Verify
 * actually exercises distinct analog amplification.
 */

/** 4×4, ISO 100-800 × 4-32ms. maxBurst = 8. */
val REAL_4x4 = Grid(
    gains = intArrayOf(100, 200, 400, 800),
    exposuresUs = intArrayOf(4000, 8000, 16000, 32000)
)

/** 3×3 for dim indoor: drops REAL_4x4's darkest cell. maxBurst = 4. */
val REAL_3x3 = Grid(
    gains = intArrayOf(200, 400, 800),
    exposuresUs = intArrayOf(8000, 16000, 32000)
)

/** 3×3, one stop over [REAL_3x3]. Costs ~6 dB noise floor at base. */
val REAL_3x3_BRIGHT = Grid(
    gains = intArrayOf(400, 800, 1600),
    exposuresUs = intArrayOf(8000, 16000, 32000)
)

/** 3×3 for night: +3 stops over [REAL_3x3] at the brightest cell. 64ms blurs hand-held. */
val REAL_3x3_NIGHT = Grid(
    gains = intArrayOf(400, 800, 1600),
    exposuresUs = intArrayOf(16000, 32000, 64000)
)

/** 3×3 for daylight, entirely inside the analog-gain zone. */
val REAL_3x3_DAYLIGHT = Grid(
    gains = intArrayOf(25, 100, 400),
    exposuresUs = intArrayOf(4000, 8000, 16000)
)

/** 4×3 spanning 64× ISO, to cover the analog zone and the plateau in one pass. */
val REAL_4x3_WIDE = Grid(
    gains = intArrayOf(25, 100, 400, 1600),
    exposuresUs = intArrayOf(4000, 8000, 16000)
)

/**
 * 3×3 for indoor dim scenes — the default. The gain axis stays inside the analog zone so
 * Verify's per-cell physical captures exercise real amplification; [digitalBoost] = 2.0
 * then brightens the virtual path enough for the detector to see anything.
 */
val REAL_3x3_INDOOR = Grid(
    gains = intArrayOf(100, 200, 400),
    exposuresUs = intArrayOf(16000, 32000, 64000),
    digitalBoost = 2.0
)

/**
 * RayNeo X3 Pro production grid validated on camera 0.
 *
 * The 32 ms base exposure nearly fills the measured 33.329 ms frame period. Summing the
 * centered 1/2/4-frame windows therefore produces useful 32/64/128 ms virtual shutters
 * without the large temporal holes observed with a 16 ms base. ISO is kept at the verified
 * 100-400 range and the gain axis is formed digitally from the same linear RAW data.
 */
val RAYNEO_X3_PRO_3x3 = Grid(
    gains = intArrayOf(100, 200, 400),
    exposuresUs = intArrayOf(32_000, 64_000, 128_000),
    digitalBoost = 1.0
)
