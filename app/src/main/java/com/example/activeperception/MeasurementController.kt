package com.example.activeperception

import android.graphics.Bitmap
import android.os.Build
import com.example.activeperception.acquire.StepResult
import com.example.activeperception.acquire.plan as acquirePlan
import com.example.activeperception.acquire.CandidateSource
import com.example.activeperception.acquire.Detection
import com.example.activeperception.acquire.Detector
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.RawFrame
import com.example.activeperception.acquire.RegimeClassifier
import org.json.JSONArray
import org.json.JSONObject

/**
 * Tie-break used by Proposed when sum_conf is 0 on every candidate.
 *
 * ENTROPY        Shannon entropy of the luminance histogram. Brightness-biased: in dim
 *                scenes it picks the brightest cell regardless of blur.
 * LAPLACIAN_VAR  Variance of the 5-point Laplacian. Penalizes blur, but Lap² still scales
 *                with absolute brightness.
 * TENENGRAD_NORM Σ|∇I|² / Σ I² via Sobel. Brightness-invariant since both sums scale
 *                identically, and blur still costs gradient per unit brightness.
 * CRETE_ROFFET   Crete et al. 2007 perceptual blur metric.
 * SAFE_CELL      No image metric at all. With no detection anywhere the SNR is too low for
 *                a metric to discriminate, so pick by brightness target on the motion-safe
 *                shutter row instead.
 */
enum class FallbackMetric { ENTROPY, LAPLACIAN_VAR, TENENGRAD_NORM, CRETE_ROFFET, SAFE_CELL }

/**
 * Who chooses (iso, exposure) in the AE / AE_quant modes.
 * PHONE is the HAL's CONTROL_AE_MODE_ON — vendor-tuned and closed. CUSTOM_BRIGHTNESS is
 * [CustomAeBrightness], which is deterministic and reproducible. Quantization in AE_quant
 * is identical either way; only the upstream decision differs.
 */
enum class AeStrategy { PHONE, CUSTOM_BRIGHTNESS }

fun AeStrategy.tag(): String = when (this) {
    AeStrategy.PHONE -> "phone"
    AeStrategy.CUSTOM_BRIGHTNESS -> "custom"
}

/** Short run-dir suffix. */
fun FallbackMetric.tag(): String = when (this) {
    FallbackMetric.ENTROPY -> "ent"
    FallbackMetric.LAPLACIAN_VAR -> "lap"
    FallbackMetric.TENENGRAD_NORM -> "ten"
    FallbackMetric.CRETE_ROFFET -> "cre"
    FallbackMetric.SAFE_CELL -> "safe"
}
/** Status-tag display char. */
fun FallbackMetric.shortTag(): String = when (this) {
    FallbackMetric.ENTROPY -> "E"
    FallbackMetric.LAPLACIAN_VAR -> "L"
    FallbackMetric.TENENGRAD_NORM -> "T"
    FallbackMetric.CRETE_ROFFET -> "C"
    FallbackMetric.SAFE_CELL -> "S"
}
/** Value written into the candidates.csv `tie_break` column when this strategy was used. */
fun FallbackMetric.tieBreakName(): String = when (this) {
    FallbackMetric.ENTROPY -> "entropy"
    FallbackMetric.LAPLACIAN_VAR -> "laplacian"
    FallbackMetric.TENENGRAD_NORM -> "tenengrad"
    FallbackMetric.CRETE_ROFFET -> "crete_roffet"
    FallbackMetric.SAFE_CELL -> "safe_cell"
}

/**
 * Drives every measurement mode (Fixed | AE | AE_quant | Proposed | Verify | VerifyProbe |
 * Bench | IsoDiag) over the RAW capturer, the detector, and the acquire core. Runs on a
 * worker thread — capture and inference both block.
 *
 * Conventions shared by all modes:
 *  - the detector floors at conf 0.01, keeping a tail for the offload signal
 *  - selection re-thresholds at [selectConf]
 *  - the same on-device YOLOv8n runs everywhere
 *
 * Per-run file layout is in [MeasurementLogger].
 */
class MeasurementController(
    private val raw: RawSensorCapturer,
    private val detector: Detector<Bitmap>,
    private val grid: Grid,
    private val sensors: SensorDataManager,
    private val logger: MeasurementLogger,
    /** Operating confidence cutoff, above the detector's 0.01 decode floor. Applied to
     *  sum_conf in frames.csv and to Proposed's selection. Set from the UI, recorded in
     *  manifest as detector.select_conf_threshold. */
    private val selectConf: Float = 0.25f
) {
    @Volatile var running = false; private set
    fun stop() { running = false }

    /** The canonical per-pass frame index: the `f` that names img/frame_XXXX.jpg and fills
     *  frames.csv `idx`. Published by [logFrame] just before each onFrame callback so
     *  external listeners (OffloadClient) can tag their work with a joinable id. */
    @Volatile var lastFrameIdx: Int = -1; private set

    /** Per-pass totals for the post-run banner. Vehicle classes only — the detector already
     *  filters. The at-floor count includes the 0.01 tail, which is mostly noise. */
    @Volatile var detectionTotalAboveThresh: Int = 0; private set
    @Volatile var detectionTotalAtFloor: Int = 0; private set
    @Volatile var totalFramesLogged: Int = 0; private set

    /** Mean luma target for SAFE_CELL, as a fraction of white. 0.50 is the AE mid-gray
     *  reference: the linear midpoint of 8-bit display space, ≈ 18% gray after sRGB gamma. */
    private val SAFE_TARGET_RATIO = 0.50

    private val regime = RegimeClassifier()
    private var passStartMs: Long = 0L

    private fun now() = System.nanoTime()
    private fun ms(t0: Long) = (now() - t0) / 1e6

    // ---------- manifest + per-frame helpers shared by all modes ----------

    private fun headers() = listOf(
        "idx", "ts", "frame_number", "time_since_start", "lap", "heading_angle",
        "regime", "yaw_rate", "lux", "accel",
        "method", "chosen_cell", "gain", "exposure_us",
        "iso_req", "iso_applied", "exp_req", "exp_applied",
        "K", "is_burst", "formation_ms", "infer_ms", "total_ms",
        "n_det", "sum_conf", "img_path",
        // Proposed only: which path picked the cell — conf | safe_cell | entropy |
        // laplacian | tenengrad | crete_roffet. Blank elsewhere.
        "tie_break",
        // AE_quant only: AE's continuous choice before quantization. The gain/exposure_us
        // columns above carry the quantized cell that was actually captured. Blank elsewhere.
        "ae_iso", "ae_exp_us"
    )

    /** [methodParams] carries the mode's own configuration so a run variant is identifiable
     *  from the manifest alone, without parsing the run directory name. Nested so the
     *  top-level schema stays stable as modes come and go. */
    private fun writeManifest(method: String, isGtReference: Boolean,
                              captureWidth: Int, captureHeight: Int,
                              methodParams: JSONObject? = null) {
        val m = JSONObject()
        m.put("method", method)
        m.put("is_gt_reference", isGtReference)
        m.put("pass_start_ts", System.currentTimeMillis())
        m.put("device_model", Build.MODEL)
        m.put("os_sdk", Build.VERSION.SDK_INT)
        if (methodParams != null) m.put("method_params", methodParams)
        m.put("grid", JSONObject().apply {
            put("gains", JSONArray(grid.gains.toTypedArray()))
            put("exposures_us", JSONArray(grid.exposuresUs.toTypedArray()))
            put("n_gain", grid.nGain); put("n_shutter", grid.nShutter)
            // Uniform digital multiplier folded into gainRatio; see Grid.
            put("digital_boost", grid.digitalBoost)
        })
        m.put("capture_resolution", JSONObject().apply {
            put("width", captureWidth); put("height", captureHeight)
            put("downsampled_in_decode", true)
            // The RAW size above is NOT the space the logged boxes live in — decode()
            // block-skips 2× and demosaic2x2 halves again. Every xyxy in dets.jsonl and
            // candidate_dets.jsonl is in the detector_input space recorded here.
            put("detector_input_width", raw.bitmapWidth)
            put("detector_input_height", raw.bitmapHeight)
        })
        m.put("cfa_pattern", raw.cfaPattern)
        m.put("white_level", raw.maxDn)
        m.put("sync_max_latency", raw.syncMaxLatency)
        m.put("timestamp_source", raw.timestampSource)
        // Mirrors the TfliteYoloDetector defaults the Activity constructs it with. Boxes in
        // dets.jsonl are post-NMS, post-class-filter, post-conf_floor.
        m.put("detector", JSONObject().apply {
            put("model", "yolov8n_640_fp16")
            put("img_size", 640)
            put("conf_floor", 0.01)
            put("iou_thresh", 0.45)
            put("max_det_per_frame", 100)
            put("allowed_classes", JSONArray(intArrayOf(2, 3, 5, 7).toTypedArray()))
            put("num_classes", 80)
            put("select_conf_threshold", selectConf)
        })
        logger.manifest(m)
    }

    private fun sumConf(dets: List<Detection>, sel: Float = selectConf) =
        dets.sumOf { if (it.confidence >= sel) it.confidence.toDouble() else 0.0 }

    private fun detJson(d: Detection): JSONObject {
        val xy = JSONArray()
        for (v in d.xyxy) xy.put(v.toDouble())
        return JSONObject().apply {
            put("xyxy", xy); put("conf", d.confidence.toDouble()); put("cls", d.classId)
        }
    }

    /** Full detection list including the sub-threshold tail. `method` keeps rows separable
     *  when a mode writes more than one per frame. */
    private fun writeDets(frame: Int, method: String, cell: Int, dets: List<Detection>) {
        val arr = JSONArray()
        for (d in dets) arr.put(detJson(d))
        logger.jsonl("dets", JSONObject().apply {
            put("frame", frame); put("method", method); put("cell", cell); put("dets", arr)
        })
    }

    /** Snap a continuous (iso, expUs) onto the discrete action set, each axis independently,
     *  by smallest |ln(target / grid)|. Log distance because the axes are geometric. */
    private fun quantizeToGrid(iso: Int, expUs: Int): Pair<Int, Int> {
        val isoSafe = iso.coerceAtLeast(1).toDouble()
        val expSafe = expUs.coerceAtLeast(1).toDouble()
        val gi = grid.gains.indices.minByOrNull {
            kotlin.math.abs(kotlin.math.ln(isoSafe / grid.gains[it]))
        } ?: 0
        val sj = grid.exposuresUs.indices.minByOrNull {
            kotlin.math.abs(kotlin.math.ln(expSafe / grid.exposuresUs[it]))
        } ?: 0
        return Pair(gi, sj)
    }

    /** One imu.csv row per sensor sample, tagged by `source`, carrying only that sensor's
     *  channels and its own ns timestamp. Merging channels into one row would attach a stale
     *  timestamp to whichever sensor did not fire, which offline orientation fusion can't undo. */
    private fun wireImuLog() {
        logger.csv("imu", listOf("ts", "source", "ax", "ay", "az", "gx", "gy", "gz", "lux"))
        sensors.onAccelSample = { ts, ax, ay, az ->
            logger.row("imu", listOf(ts, "accel", ax, ay, az, "", "", "", ""))
        }
        sensors.onGyroSample = { ts, gx, gy, gz ->
            logger.row("imu", listOf(ts, "gyro", "", "", "", gx, gy, gz, ""))
        }
        sensors.onLuxSample = { ts, lux ->
            logger.row("imu", listOf(ts, "light", "", "", "", "", "", "", lux))
        }
    }

    private fun unwireImuLog() {
        sensors.onAccelSample = null
        sensors.onGyroSample = null
        sensors.onLuxSample = null
    }

    /**
     * Shared per-frame row writer.
     *
     * Two different ISO conventions land in the same row on purpose: `gain` is EFFECTIVE ISO
     * (physical × digitalBoost, matching the saved bitmap), while `iso_req` / `iso_applied`
     * stay physical so the camera metadata is faithful. digital_boost is in the manifest, so
     * physical = effective / boost.
     *
     * Note that `iso_req` / `exp_req` describe the capture that produced the pixels, which in
     * Proposed is the base capture — not the virtual cell named by `gain` / `exposure_us`.
     */
    private fun logFrame(
        idx: Int, method: String, chosenCell: Int, gainVal: Int, expUs: Int,
        k: Int, isBurst: Boolean, formationMs: Double, inferMs: Double, totalMs: Double,
        dets: List<Detection>, imgPath: String, tieBreak: String = "",
        aeIso: Int = -1, aeExpUs: Int = -1
    ) {
        // Published before the mode's onFrame callback fires, so listeners see the matching id.
        lastFrameIdx = idx
        detectionTotalAtFloor += dets.size
        detectionTotalAboveThresh += dets.count { it.confidence >= selectConf }
        totalFramesLogged += 1
        val m = raw.lastMeta.firstOrNull()
        val aeIsoCol = if (aeIso < 0) "" else aeIso.toString()
        val aeExpCol = if (aeExpUs < 0) "" else aeExpUs.toString()
        logger.row("frames", listOf(
            idx, m?.timestamp ?: -1L, m?.frameNumber ?: -1L,
            "%.3f".format((System.currentTimeMillis() - passStartMs) / 1000.0),
            sensors.getCurrentLap(), "%.3f".format(sensors.currentHeadingAngle),
            regime.update(sensors.currentYawRate),
            sensors.currentYawRate, sensors.currentLux, sensors.currentAccel,
            method, chosenCell, effIso(gainVal), expUs,
            m?.requestedIso ?: -1, m?.appliedIso ?: -1,
            m?.requestedExpUs ?: -1L, m?.appliedExpUs ?: -1L,
            k, if (isBurst) 1 else 0,
            "%.1f".format(formationMs), "%.1f".format(inferMs), "%.1f".format(totalMs),
            dets.size, "%.3f".format(sumConf(dets)), imgPath, tieBreak,
            aeIsoCol, aeExpCol
        ))
        writeDets(idx, method, chosenCell, dets)
    }

    // ---------- Fixed ----------

    /** Every frame captured at one physical (gain, exposure) cell, AE off. */
    fun runFixed(gainIdx: Int, shutterIdx: Int, maxFrames: Int, isGtReference: Boolean,
                 onStatus: (String) -> Unit,
                 onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val gainVal = grid.gains[gainIdx]; val expUs = grid.exposuresUs[shutterIdx]
        val cell = grid.cell(gainIdx, shutterIdx)
        startPass("fixed_g${gainVal}_e${expUs}", isGtReference, methodParams = JSONObject()
            .put("gain_idx", gainIdx).put("shutter_idx", shutterIdx).put("cell", cell))
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now()
            val frames = raw.capture(expUs, gainVal, 1)
            if (frames.isEmpty()) break
            val tf = now()
            val bmp = bitmapFromRaw(frames[0])
            val formMs = ms(tf)
            val ti = now(); val dets = detector.detectBatch(listOf(bmp))[0]; val infMs = ms(ti)
            val totalMs = ms(t0)
            val path = logger.saveJpeg("${frameName(f)}_${isoExpTag(effIso(gainVal), expUs)}", bmp)
            logFrame(f, "fixed", cell, gainVal, expUs, 1, false, formMs, infMs, totalMs, dets, path)
            onFrame(bmp, dets, effIso(gainVal), expUs)
            onStatus("Fixed f=$f g=${effIso(gainVal)} exp=${expUs}us ndet=${dets.size} ${"%.0f".format(totalMs)}ms")
            f++
        }
        endPass()
    }

    // ---------- AE ----------

    /** Every frame captured under [aeStrategy], logging whichever (iso, exp) it chose. */
    fun runAe(maxFrames: Int, isGtReference: Boolean,
              aeStrategy: AeStrategy,
              onStatus: (String) -> Unit,
              onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val customAe = if (aeStrategy == AeStrategy.CUSTOM_BRIGHTNESS) CustomAeBrightness() else null
        startPass("ae_${aeStrategy.tag()}", isGtReference, methodParams = JSONObject()
            .put("ae_strategy", aeStrategy.tag())
            .apply { if (customAe != null) put("custom_ae", customAe.toJson()) })
        // Custom AE seed; replaced every frame by the brightness feedback below.
        var nextIso = grid.baseGain
        var nextExpUs = grid.fastestExposureUs
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now()
            val frames = when (aeStrategy) {
                AeStrategy.PHONE -> raw.captureAe(1)
                AeStrategy.CUSTOM_BRIGHTNESS -> raw.capture(nextExpUs, nextIso, 1)
            }
            if (frames.isEmpty()) break
            val tf = now()
            val bmp = bitmapFromRaw(frames[0])
            val formMs = ms(tf)
            val ti = now(); val dets = detector.detectBatch(listOf(bmp))[0]; val infMs = ms(ti)
            val totalMs = ms(t0)
            val gainVal: Int; val expUs: Int
            when (aeStrategy) {
                AeStrategy.PHONE -> {
                    val applied = raw.lastMeta.firstOrNull()
                    gainVal = applied?.appliedIso ?: -1
                    expUs = (applied?.appliedExpUs ?: -1L).toInt()
                }
                AeStrategy.CUSTOM_BRIGHTNESS -> {
                    gainVal = nextIso
                    expUs = nextExpUs
                }
            }
            val path = logger.saveJpeg("${frameName(f)}_${isoExpTag(effIso(gainVal), expUs)}", bmp)
            logFrame(f, "ae", -1, gainVal, expUs, 1, false, formMs, infMs, totalMs, dets, path)
            onFrame(bmp, dets, effIso(gainVal), expUs)
            onStatus("AE[${aeStrategy.tag()}] f=$f iso=${effIso(gainVal)} exp=${expUs}us ndet=${dets.size} ${"%.0f".format(totalMs)}ms")
            if (customAe != null) {
                val ratio = meanRawRatio(frames[0])
                val (newIso, newExp) = customAe.next(gainVal, expUs, ratio)
                nextIso = newIso; nextExpUs = newExp
            }
            f++
        }
        endPass()
    }

    // ---------- AE_quant (AE restricted to the grid's action set) ----------

    /** AE picks (iso, exp), that choice is quantized to the nearest grid cell, and only the
     *  quantized cell is physically captured and detected on. AE's unquantized choice is kept
     *  as metadata in frames.csv (`ae_iso`, `ae_exp_us`).
     *
     *  With PHONE strategy this alternates AE-on and manual captures, which disturbs AE's
     *  convergence further than [runAe] already does. */
    fun runAeQuant(maxFrames: Int, isGtReference: Boolean,
                   aeStrategy: AeStrategy,
                   onStatus: (String) -> Unit,
                   onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val customAe = if (aeStrategy == AeStrategy.CUSTOM_BRIGHTNESS) CustomAeBrightness() else null
        startPass("ae_paired_${aeStrategy.tag()}", isGtReference, methodParams = JSONObject()
            .put("ae_strategy", aeStrategy.tag())
            .apply { if (customAe != null) put("custom_ae", customAe.toJson()) })
        var nextIso = grid.baseGain
        var nextExpUs = grid.fastestExposureUs
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now()
            // Phone AE only reveals its choice through a real capture; Custom AE computes it.
            val aeIso: Int; val aeExpUs: Int
            val aeRawForFeedback: RawFrame?
            when (aeStrategy) {
                AeStrategy.PHONE -> {
                    val aeFrames = raw.captureAe(1)
                    if (aeFrames.isEmpty()) break
                    val aeMeta = raw.lastMeta.firstOrNull()
                    aeIso = aeMeta?.appliedIso ?: -1
                    aeExpUs = (aeMeta?.appliedExpUs ?: -1L).toInt()
                    aeRawForFeedback = aeFrames[0]
                }
                AeStrategy.CUSTOM_BRIGHTNESS -> {
                    aeIso = nextIso; aeExpUs = nextExpUs
                    aeRawForFeedback = null   // feedback comes from the quantized frame instead
                }
            }

            if (!running) break

            val (qGi, qSj) = quantizeToGrid(aeIso, aeExpUs)
            val qGain = grid.gains[qGi]; val qExp = grid.exposuresUs[qSj]; val qCell = grid.cell(qGi, qSj)
            val qFrames = raw.capture(qExp, qGain, 1)
            if (qFrames.isEmpty()) break
            val tfQ = now()
            val bmpQ = bitmapFromRaw(qFrames[0])
            val formQMs = ms(tfQ)
            val tiQ = now(); val detsQ = detector.detectBatch(listOf(bmpQ))[0]; val infQMs = ms(tiQ)
            val totalMs = ms(t0)
            val pathQ = logger.saveJpeg("frame_%04d_aequant_%s".format(f, isoExpTag(effIso(qGain), qExp)), bmpQ)
            logFrame(f, "ae_quant", qCell, qGain, qExp, 1, false, formQMs, infQMs, totalMs,
                detsQ, pathQ, aeIso = aeIso, aeExpUs = aeExpUs)

            onFrame(bmpQ, detsQ, effIso(qGain), qExp)
            onStatus("AE_quant[${aeStrategy.tag()}] f=$f AE→(iso=${effIso(aeIso)}, exp=${aeExpUs}us) → cell=$qCell ${"%.0f".format(totalMs)}ms")
            // Prefer AE's own capture for brightness feedback — it is closer to the exposure
            // AE actually asked for than the quantized one is.
            if (customAe != null) {
                val src = aeRawForFeedback ?: qFrames[0]
                val ratio = meanRawRatio(src)
                val (newIso, newExp) = customAe.next(aeIso, aeExpUs, ratio)
                nextIso = newIso; nextExpUs = newExp
            }
            f++
        }
        endPass()
    }

    // ---------- Proposed (acquire-and-select) ----------

    /** Keeps the rendered bitmaps and the render time visible without changing acquire/. */
    private class RecordingSource(private val inner: CandidateSource<Bitmap>) : CandidateSource<Bitmap> {
        var lastImages: List<Bitmap> = emptyList(); private set
        var lastRenderMs: Double = 0.0; private set
        override fun render(cells: IntArray): List<Bitmap> {
            val t0 = System.nanoTime()
            val imgs = inner.render(cells)
            lastRenderMs = (System.nanoTime() - t0) / 1e6
            lastImages = imgs
            return imgs
        }
    }
    private class TimingDetector(private val inner: Detector<Bitmap>) : Detector<Bitmap> {
        var lastInferMs: Double = 0.0; private set
        override fun detectBatch(images: List<Bitmap>): List<List<Detection>> {
            val t0 = System.nanoTime()
            val out = inner.detectBatch(images)
            lastInferMs = (System.nanoTime() - t0) / 1e6
            return out
        }
    }

    /** Lazy so it is not spun up unless a fallback metric is actually used. */
    private val fallbackPool by lazy { java.util.concurrent.Executors.newFixedThreadPool(4) }

    /** Per-candidate scores in input order. SAFE_CELL scores nothing — the caller picks by
     *  brightness target instead — so it returns zeros. */
    private fun fallbackScores(images: List<Bitmap>, metric: FallbackMetric): DoubleArray {
        if (metric == FallbackMetric.SAFE_CELL) return DoubleArray(images.size)
        val fn: (Bitmap) -> Double = when (metric) {
            FallbackMetric.ENTROPY -> ::pixelEntropy
            FallbackMetric.LAPLACIAN_VAR -> ::laplacianVariance
            FallbackMetric.TENENGRAD_NORM -> ::tenengradNorm
            FallbackMetric.CRETE_ROFFET -> ::creteRoffet
            FallbackMetric.SAFE_CELL -> error("unreachable")
        }
        if (images.size <= 1) return DoubleArray(images.size) { fn(images[it]) }
        val futures = images.map { bmp -> fallbackPool.submit<Double> { fn(bmp) } }
        return DoubleArray(images.size) { futures[it].get() }
    }

    /** Shannon entropy of the luminance histogram, 0..8 bits. See [FallbackMetric]. */
    private fun pixelEntropy(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val hist = IntArray(256)
        for (p in px) {
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            val lum = ((r * 299 + g * 587 + b * 114) / 1000).coerceIn(0, 255)
            hist[lum]++
        }
        val total = px.size.toDouble()
        val ln2 = Math.log(2.0)
        var ent = 0.0
        for (c in hist) if (c > 0) { val q = c / total; ent -= q * (Math.log(q) / ln2) }
        return ent
    }

    /** Variance of the 5-point Laplacian of luminance (Pech-Pacheco et al.; Krotkov
     *  autofocus). See [FallbackMetric]. Single pass: var = E[X²] - E[X]². */
    private fun laplacianVariance(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        // Luminance precomputed so the stencil doesn't unpack RGB 4× per inner pixel.
        val lum = IntArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            lum[i] = (r * 299 + g * 587 + b * 114) / 1000
        }
        var sum = 0.0; var sumSq = 0.0; var n = 0
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val c = lum[row + x]
                val l = lum[row + x - 1]; val rr = lum[row + x + 1]
                val u = lum[row - w + x]; val d = lum[row + w + x]
                val v = (l + rr + u + d - 4 * c).toDouble()
                sum += v; sumSq += v * v; n++
            }
        }
        if (n == 0) return 0.0
        val mean = sum / n
        return sumSq / n - mean * mean
    }

    /** Normalized Tenengrad: Σ|∇I|² / Σ I² via Sobel 3×3. See [FallbackMetric]. */
    private fun tenengradNorm(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val lum = IntArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            lum[i] = (r * 299 + g * 587 + b * 114) / 1000
        }
        var sumG2 = 0.0; var sumI2 = 0.0
        for (y in 1 until h - 1) {
            val row = y * w
            val rowU = row - w; val rowD = row + w
            for (x in 1 until w - 1) {
                // Sobel 3×3, Gx = [-1 0 1; -2 0 2; -1 0 1], Gy transposed.
                val gx = -lum[rowU + x - 1] + lum[rowU + x + 1] +
                         -2 * lum[row  + x - 1] + 2 * lum[row  + x + 1] +
                         -lum[rowD + x - 1] + lum[rowD + x + 1]
                val gy = -lum[rowU + x - 1] - 2 * lum[rowU + x] - lum[rowU + x + 1] +
                          lum[rowD + x - 1] + 2 * lum[rowD + x] + lum[rowD + x + 1]
                sumG2 += (gx * gx + gy * gy).toDouble()
                val v = lum[row + x].toDouble()
                sumI2 += v * v
            }
        }
        return if (sumI2 == 0.0) 0.0 else sumG2 / sumI2
    }

    /** Crete-Roffet 2007 no-reference blur metric, returned as SHARPNESS so that, like every
     *  other metric here, higher is better. Re-blurs with a 9-tap separable box filter and
     *  measures the gradient lost: a sharp image loses a lot, an already-blurred one little.
     *  Sliding-sum box blur, so each pass is O(N) regardless of radius. */
    private fun creteRoffet(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        if (w < 3 || h < 3) return 0.0
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val I = FloatArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            I[i] = ((r * 299 + g * 587 + b * 114) / 1000).toFloat()
        }
        val radius = 4   // 9-tap box (Crete-Roffet paper)
        val tmp = FloatArray(w * h)
        val B = FloatArray(w * h)
        // Horizontal box blur (sliding sum, edge-clamped count)
        for (y in 0 until h) {
            val rowOff = y * w
            var sum = 0f
            val initEnd = minOf(radius, w - 1)
            for (k in 0..initEnd) sum += I[rowOff + k]
            var count = initEnd + 1
            tmp[rowOff] = sum / count
            for (x in 1 until w) {
                val addIdx = x + radius; val rmvIdx = x - radius - 1
                if (addIdx < w) { sum += I[rowOff + addIdx]; count++ }
                if (rmvIdx >= 0) { sum -= I[rowOff + rmvIdx]; count-- }
                tmp[rowOff + x] = sum / count
            }
        }
        // Vertical box blur
        for (x in 0 until w) {
            var sum = 0f
            val initEnd = minOf(radius, h - 1)
            for (k in 0..initEnd) sum += tmp[k * w + x]
            var count = initEnd + 1
            B[x] = sum / count
            for (y in 1 until h) {
                val addIdx = y + radius; val rmvIdx = y - radius - 1
                if (addIdx < h) { sum += tmp[addIdx * w + x]; count++ }
                if (rmvIdx >= 0) { sum -= tmp[rmvIdx * w + x]; count-- }
                B[y * w + x] = sum / count
            }
        }
        // Horizontal and vertical diff comparison
        var sumDh = 0.0; var sumVh = 0.0
        for (y in 0 until h) {
            val rowOff = y * w
            for (x in 1 until w) {
                val dh = Math.abs(I[rowOff + x] - I[rowOff + x - 1])
                val dhB = Math.abs(B[rowOff + x] - B[rowOff + x - 1])
                sumDh += dh; sumVh += maxOf(0f, dh - dhB)
            }
        }
        var sumDv = 0.0; var sumVv = 0.0
        for (y in 1 until h) {
            val rowOff = y * w; val rowOffU = rowOff - w
            for (x in 0 until w) {
                val dv = Math.abs(I[rowOff + x] - I[rowOffU + x])
                val dvB = Math.abs(B[rowOff + x] - B[rowOffU + x])
                sumDv += dv; sumVv += maxOf(0f, dv - dvB)
            }
        }
        // ratio = sumV/sumD ∈ [0,1] is the fraction of gradient lost to re-blurring, so it
        // is ≈1 for a sharp image and ≈0 for a blurred one. The paper's blur_F is
        // max(1-ratioH, 1-ratioV), hence sharpness = 1 - blur_F = min(ratioH, ratioV).
        val ratioH = if (sumDh > 0) sumVh / sumDh else 0.0
        val ratioV = if (sumDv > 0) sumVv / sumDv else 0.0
        return minOf(ratioH, ratioV)
    }

    /**
     * The deployed acquire-select loop. Same as `AcquireSelectController` except for the
     * all-zero case: instead of holding the anchor when no candidate has any detection, it
     * picks by [metric] and anchors there, turning a dead step into exploration. That matters
     * on cold start and through brief occlusions.
     *
     * `acquire/AcquireSelectController` is deliberately left untouched as the reference.
     */
    private inner class EntropyFallbackController(
        private val source: CandidateSource<Bitmap>,
        private val detector: Detector<Bitmap>,
        private val period: Int,
        initAnchor: Int,
        private val metric: FallbackMetric = FallbackMetric.ENTROPY,
        private val selectConf: Float = this@MeasurementController.selectConf
    ) {
        var anchor: Int = initAnchor; private set
        var t: Int = 0; private set
        /** Empty when the conf path picked the cell. */
        var lastFallbackScores: DoubleArray = DoubleArray(0); private set
        var lastUsedFallback: Boolean = false; private set
        /** Every candidate's detections, indexed parallel to StepResult.cells, so the caller
         *  can log all of them and not just the chosen one. */
        var lastAllDets: List<List<Detection>> = emptyList(); private set

        fun step(): StepResult {
            val cells = acquirePlan(t, anchor, grid, period)
            val images = source.render(cells)
            val dets = detector.detectBatch(images)
            lastAllDets = dets
            val confScores = DoubleArray(dets.size) { i ->
                dets[i].sumOf { if (it.confidence >= selectConf) it.confidence.toDouble() else 0.0 }
            }
            var chosen = 0
            for (i in confScores.indices) if (confScores[i] > confScores[chosen]) chosen = i

            lastFallbackScores = DoubleArray(0); lastUsedFallback = false
            if (confScores[chosen] == 0.0) {
                lastUsedFallback = true
                if (metric == FallbackMetric.SAFE_CELL) {
                    // Among shutter-row-0 cells (no motion-blur risk), take the one whose mean
                    // luma is nearest the target in log distance. That adapts: dim scenes land
                    // on high gain, bright ones on low gain instead of saturating.
                    val target = SAFE_TARGET_RATIO
                    var pick = -1; var bestDist = Double.MAX_VALUE
                    for (i in cells.indices) {
                        if (grid.indices(cells[i]).second != 0) continue
                        val mean = meanLumaRatio(images[i])
                        val d = kotlin.math.abs(kotlin.math.ln((mean + 1e-6) / target))
                        if (d < bestDist) { bestDist = d; pick = i }
                    }
                    if (pick < 0) {
                        // Off-probe step anchored off row 0, so no row-0 cell is a candidate.
                        // Fall back to the highest gain available; the next probe step brings
                        // row 0 back into range.
                        var bestGi = -1
                        for (i in cells.indices) {
                            val gi = grid.indices(cells[i]).first
                            if (gi > bestGi) { bestGi = gi; pick = i }
                        }
                        if (pick < 0) pick = 0
                    }
                    chosen = pick
                } else {
                    val scores = fallbackScores(images, metric)
                    lastFallbackScores = scores
                    chosen = 0
                    for (i in scores.indices) if (scores[i] > scores[chosen]) chosen = i
                }
            }

            anchor = cells[chosen]
            t += 1
            return StepResult(cells[chosen], dets[chosen], confScores, cells, chosen)
        }
    }

    /** Live acquire-select controller. Writes frames.csv + dets.jsonl + candidates.csv + img/.
     *  [fallback] selects which tie-break metric is used when sum_conf=0 across all cells.
     *  Candidate cells are always formed by the virtual path (one base capture → burst-sum
     *  + digital re-gain); per-cell physical probing lives in [runVerify] only. */
    fun runProposed(period: Int, maxFrames: Int, isGtReference: Boolean,
                    fallback: FallbackMetric,
                    /** Saves a bitmap for every candidate, not just the chosen one — useful for
                     *  visual inspection, but roughly +50% wall-clock on probe-heavy runs.
                     *  candidate_dets.jsonl records all per-cell boxes either way. */
                    saveAllCandidates: Boolean = false,
                    onStatus: (String) -> Unit,
                    onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        val initAnchor = grid.cell(grid.nGain - 1, 0)
        startPass("proposed_p${period}_${fallback.tag()}", isGtReference,
            methodParams = JSONObject()
                .put("period", period)
                .put("fallback_metric", fallback.tieBreakName())
                .put("init_anchor_cell", initAnchor)
                .put("save_all_candidates", saveAllCandidates))
        // fallback_score is blank unless the fallback fired; tie_break names the path that
        // picked the cell.
        logger.csv("candidates", listOf("frame", "cell", "gain", "exposure_us",
            "sum_conf", "fallback_score", "chosen", "tie_break"))
        val source = RecordingSource(ParallelRawCandidateSource(grid, raw))
        val timing = TimingDetector(detector)
        // Anchor starts at (max gain, fastest shutter) so the very first preview is visible
        // whatever the lighting.
        val ctrl = EntropyFallbackController(source, timing, period, initAnchor, fallback)
        val fbName = fallback.tieBreakName()
        val fbShort = fallback.shortTag()
        var f = 0
        while (running && f < maxFrames) {
            val t0 = now(); val r = ctrl.step(); val totalMs = ms(t0)
            val isBurst = r.cells.size > grid.nGain
            val tieBreak = if (ctrl.lastUsedFallback) fbName else "conf"
            val fbScores = ctrl.lastFallbackScores
            val allDets = ctrl.lastAllDets
            var chosenPath = ""
            for (i in r.cells.indices) {
                val c = r.cells[i]; val (gi, sj) = grid.indices(c)
                val gainEff = effIso(grid.gains[gi]); val expU = grid.exposuresUs[sj]
                val scoreCell = if (i < fbScores.size) "%.3f".format(fbScores[i]) else ""
                logger.row("candidates", listOf(
                    f, c, gainEff, expU,
                    "%.4f".format(r.scores[i]), scoreCell,
                    if (r.cells[i] == r.cell) 1 else 0, tieBreak))
                val candBmp = source.lastImages.getOrNull(i) ?: continue
                val isChosen = (i == r.chosen)
                if (saveAllCandidates || isChosen) {
                    val candPath = logger.saveJpeg(
                        "${frameName(f)}_cell${c}_${isoExpTag(gainEff, expU)}", candBmp)
                    if (isChosen) chosenPath = candPath
                }
                // Joinable with frames.csv on `frame` and candidates.csv on (frame, cell).
                val detsArr = JSONArray()
                val cellDets = if (i < allDets.size) allDets[i] else emptyList()
                for (d in cellDets) detsArr.put(detJson(d))
                logger.jsonl("candidate_dets", JSONObject().apply {
                    put("frame", f); put("cell", c); put("chosen", isChosen)
                    put("gain", gainEff); put("exposure_us", expU)
                    put("sum_conf", r.scores[i])
                    put("tie_break", if (isChosen) tieBreak else "")
                    put("dets", detsArr)
                })
            }
            val (gi, sj) = grid.indices(r.cell)
            val gainVal = grid.gains[gi]; val expUs = grid.exposuresUs[sj]
            val bmp = source.lastImages.getOrNull(r.chosen)
            logFrame(f, "proposed", r.cell, gainVal, expUs,
                r.cells.size, isBurst, source.lastRenderMs, timing.lastInferMs, totalMs,
                r.detections, chosenPath, tieBreak)
            if (bmp != null) onFrame(bmp, r.detections, effIso(gainVal), expUs)
            val tag = if (ctrl.lastUsedFallback) " [$fbShort]" else ""
            onStatus("Proposed f=$f cell=${r.cell}$tag ndet=${r.detections.size} ${"%.0f".format(totalMs)}ms (form=${"%.0f".format(source.lastRenderMs)}, inf=${"%.0f".format(timing.lastInferMs)})")
            f++
        }
        endPass()
    }

    // ---------- Verify: physical half of the probing-realism pair ----------

    /** Physically visits every cell twice on a static scene, saving the RAW, a formed JPEG,
     *  and the detections for each. Pair with [runVerifyProbe] on the same scene to compare
     *  physically-visited cells against virtually-formed ones.
     *
     *  Both passes exist so the physical-vs-physical difference gives a null baseline for
     *  the physical-vs-virtual comparison. The detector runs here so the comparison can be
     *  made from the CSVs alone, without a round trip through offline Python. */
    fun runVerify(isGtReference: Boolean, onStatus: (String) -> Unit,
                  onFrame: (Bitmap, List<Detection>, Int /*iso*/, Int /*expUs*/) -> Unit = { _, _, _, _ -> }) {
        startPass("verify", isGtReference, withFrames = false)
        logger.csv("verify", listOf(
            "ts", "scene", "pass", "cell", "gain", "iso_applied", "exp_applied",
            "black", "white", "raw_path", "img_path", "n_det", "sum_conf"))
        val base = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (base.isNotEmpty()) {
            val p = logger.saveRaw("verify_simbase_${isoExpTag(effIso(grid.baseGain), grid.fastestExposureUs)}", base[0].bayer)
            val m = raw.lastMeta.firstOrNull()
            logger.row("verify", listOf(System.currentTimeMillis(), "static", "sim_base", -1,
                effIso(grid.baseGain), m?.appliedIso ?: -1, m?.appliedExpUs ?: -1,
                base[0].maxDn.toInt(), base[0].maxDn.toInt(), p, "", "", ""))
            val brightCell = grid.cell(grid.nGain - 1, grid.nShutter - 1)
            runCatching {
                val bmp = parallelFormation.formAllCells(base, intArrayOf(brightCell))[0]
                onFrame(bmp, emptyList(), effIso(grid.gains.last()), grid.exposuresUs.last())
            }
        }
        for (pass in listOf("phys1", "phys2")) {
            for (gi in 0 until grid.nGain) for (sj in 0 until grid.nShutter) {
                if (!running) { endPass(); return }
                val cell = grid.cell(gi, sj)
                val frames = raw.capture(grid.exposuresUs[sj], grid.gains[gi], 1)
                if (frames.isEmpty()) continue
                val m = raw.lastMeta.firstOrNull()
                // Effective ISO in the filename for consistency with the JPEG's content;
                // the physical value is recoverable via the manifest's digital_boost.
                val tag = isoExpTag(effIso(grid.gains[gi]), grid.exposuresUs[sj])
                val rawPath = logger.saveRaw("verify_${pass}_${cell}_$tag", frames[0].bayer)
                val bmp = bitmapFromRaw(frames[0])
                val dets = detector.detectBatch(listOf(bmp))[0]
                val imgPath = logger.saveJpeg("verify_${pass}_${cell}_$tag", bmp)
                logger.row("verify", listOf(System.currentTimeMillis(), "static", pass, cell,
                    effIso(grid.gains[gi]), m?.appliedIso ?: -1, m?.appliedExpUs ?: -1,
                    frames[0].maxDn.toInt(), frames[0].maxDn.toInt(),
                    rawPath, imgPath, dets.size, "%.3f".format(sumConf(dets))))
                // Per-box detail for the IoU comparison against runVerifyProbe.
                val detsArr = JSONArray()
                for (d in dets) detsArr.put(detJson(d))
                logger.jsonl("verify_dets", JSONObject().apply {
                    put("method", "verify_$pass"); put("scene", "static")
                    put("cell", cell)
                    put("gain", effIso(grid.gains[gi])); put("exposure_us", grid.exposuresUs[sj])
                    put("dets", detsArr)
                })
                runCatching { onFrame(bmp, dets, effIso(grid.gains[gi]), grid.exposuresUs[sj]) }
                onStatus("Verify $pass cell=$cell ndet=${dets.size}")
            }
        }
        endPass()
    }

    // ---------- VerifyProbe: virtual half of the probing-realism pair ----------

    /** The controller's probe step in isolation: one burst at (baseGain, fastestExposure),
     *  all N cells formed virtually, one batched inference, then a row per cell.
     *
     *  Pair with [runVerify] on the same static scene. The hypothesis is that per-cell pixel
     *  and detector statistics here match the physically-visited ones. */
    fun runVerifyProbe(isGtReference: Boolean, onStatus: (String) -> Unit,
                       onFrame: (Bitmap, List<Detection>, Int, Int) -> Unit = { _, _, _, _ -> }) {
        startPass("verifyprobe", isGtReference, withFrames = false)
        logger.csv("probe", listOf(
            "ts", "cell", "gain", "exposure_us",
            "formation_ms_total", "infer_ms_total", "n_det", "sum_conf", "img_path"))
        val burst = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (burst.isEmpty()) { endPass(); return }
        // Every burst RAW is saved so offline tools can reproduce the formation independently.
        val burstTag = isoExpTag(effIso(grid.baseGain), grid.fastestExposureUs)
        burst.forEachIndexed { k, f -> logger.saveRaw("probe_burst_${k}_$burstTag", f.bayer) }

        val allCells = IntArray(grid.nGain * grid.nShutter) { it }
        val tf = now()
        val bitmaps = parallelFormation.formAllCells(burst, allCells)
        val formMs = ms(tf)
        val ti = now()
        val detsAll = detector.detectBatch(bitmaps)
        val inferMs = ms(ti)
        val nowMs = System.currentTimeMillis()
        for (i in allCells.indices) {
            if (!running) break
            val c = allCells[i]; val (gi, sj) = grid.indices(c)
            val dets = detsAll[i]
            val path = logger.saveJpeg("probe_cell${c}_${isoExpTag(effIso(grid.gains[gi]), grid.exposuresUs[sj])}", bitmaps[i])
            logger.row("probe", listOf(nowMs, c, effIso(grid.gains[gi]), grid.exposuresUs[sj],
                "%.1f".format(formMs), "%.1f".format(inferMs),
                dets.size, "%.3f".format(sumConf(dets)), path))
            // Per-box detail for the IoU comparison against runVerify.
            val detsArr = JSONArray()
            for (d in dets) detsArr.put(detJson(d))
            logger.jsonl("probe_dets", JSONObject().apply {
                put("method", "verifyprobe"); put("cell", c)
                put("gain", effIso(grid.gains[gi])); put("exposure_us", grid.exposuresUs[sj])
                put("dets", detsArr)
            })
            runCatching { onFrame(bitmaps[i], dets, effIso(grid.gains[gi]), grid.exposuresUs[sj]) }
            onStatus("Probe cell=$c ndet=${dets.size} sumConf=${"%.2f".format(sumConf(dets))}")
        }
        endPass()
    }

    // ---------- ISO diagnostic ----------

    /** Sweeps ISO at one fixed exposure and logs mean/std of the black-subtracted RAW pixels,
     *  answering whether the sensor really amplifies when SENSOR_SENSITIVITY changes. The
     *  128× span is far too large to confuse with noise, and std should scale too, since
     *  analog gain amplifies read noise along with signal. Where mean stops scaling is the
     *  analog ceiling — beyond it the HAL is applying digital gain that bypasses RAW. */
    fun runIsoDiag(exposureUs: Int = 16000, framesPerIso: Int = 5,
                   isoList: List<Int> = listOf(25, 100, 400, 800, 1600, 3200),
                   onStatus: (String) -> Unit) {
        writeManifest("iso_diag", isGtReference = false, raw.captureWidth, raw.captureHeight)
        logger.csv("iso_diag", listOf("iso_req", "iso_applied", "exp_req_us", "exp_applied_us",
            "frame_idx", "raw_mean", "raw_std", "raw_min", "raw_max", "black_subtracted"))
        running = true
        for (iso in isoList) {
            if (!running) break
            for (idx in 0 until framesPerIso) {
                if (!running) break
                val frames = raw.capture(exposureUs, iso, 1)
                if (frames.isEmpty()) continue
                val meta = raw.lastMeta.firstOrNull()
                val pixels = frames[0].bayer
                val mean = pixels.average()
                val variance = pixels.sumOf { (it - mean) * (it - mean) } / pixels.size
                val std = kotlin.math.sqrt(variance)
                var lo = Int.MAX_VALUE; var hi = Int.MIN_VALUE
                for (p in pixels) { if (p < lo) lo = p; if (p > hi) hi = p }
                logger.row("iso_diag", listOf(
                    iso, meta?.appliedIso ?: -1,
                    exposureUs, (meta?.appliedExpUs ?: -1L),
                    idx, "%.2f".format(mean), "%.2f".format(std), lo, hi, meta?.black ?: -1))
                onStatus("ISO $iso (applied ${meta?.appliedIso}): mean=${"%.0f".format(mean)} std=${"%.0f".format(std)}")
            }
        }
        logger.flush()
        onStatus("ISO diag done — see iso_diag.csv")
    }

    // ---------- Bench (latency vs K) ----------

    /** Formation, inference, and fallback-metric cost at each K the controller actually uses,
     *  followed by a sensor-control lag sweep.
     *
     *  `formation_ms` here excludes capture, since the frames are taken once up front and
     *  reused. The identically-named column in frames.csv DOES include capture — the two are
     *  not comparable. */
    fun runBench(onStatus: (String) -> Unit) {
        writeManifest("bench", isGtReference = false, captureWidth = -1, captureHeight = -1)
        logger.csv("bench", listOf("k", "imgsz", "quant", "batch_mode",
            "formation_ms", "infer_ms_p50", "infer_ms_p95",
            "entropy_ms_p50", "entropy_ms_p95",
            "laplacian_ms_p50", "laplacian_ms_p95",
            "tenengrad_ms_p50", "tenengrad_ms_p95",
            "crete_ms_p50", "crete_ms_p95"))
        running = true
        val frames = raw.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        if (frames.isEmpty()) { logger.flush(); return }
        // Candidate sets must mirror what plan() emits, NOT plain 0..K-1. The two axes cost
        // very different amounts in formAllCells — the shutter axis needs a burst-sum and a
        // demosaic per ROW, the gain axis is only a per-cell multiply — and since
        // cell = gi*nShutter + sj, numbering 0..K-1 walks the expensive axis.
        val cellSets = listOf(
            intArrayOf(grid.cell(0, 0)),                    // K=1     single cell
            IntArray(grid.nGain) { grid.cell(it, 0) },      // K=nGain off-probe: gain COLUMN
            IntArray(grid.nGain * grid.nShutter) { it }     // K=N     probe step: full grid
        ).distinctBy { it.size }
        for (cells in cellSets) {
            if (!running) break
            val k = cells.size
            val tf = now(); val imgs = formCells(frames, cells); val fms = ms(tf)
            val infers = ArrayList<Double>()
            repeat(5) { val ti = now(); detector.detectBatch(imgs); infers.add(ms(ti)) }
            // Same call shape as the Proposed tie-breaker, so these are the real cost added to
            // a step where the fallback fires. All four are timed to compare them side by side.
            fun timeMetric(metric: FallbackMetric): Pair<Double, Double> {
                val times = ArrayList<Double>(5)
                repeat(5) { val t = now(); fallbackScores(imgs, metric); times.add(ms(t)) }
                times.sort()
                val p50 = times[times.size / 2]
                val p95 = times[(times.size * 95 / 100).coerceAtMost(times.size - 1)]
                return p50 to p95
            }
            val (entP50, entP95) = timeMetric(FallbackMetric.ENTROPY)
            val (lapP50, lapP95) = timeMetric(FallbackMetric.LAPLACIAN_VAR)
            val (tenP50, tenP95) = timeMetric(FallbackMetric.TENENGRAD_NORM)
            val (creP50, creP95) = timeMetric(FallbackMetric.CRETE_ROFFET)
            infers.sort()
            val infP50 = infers[infers.size / 2]
            val infP95 = infers[(infers.size * 95 / 100).coerceAtMost(infers.size - 1)]
            logger.row("bench", listOf(k, 640, "fp16", "loop",
                "%.1f".format(fms), "%.1f".format(infP50), "%.1f".format(infP95),
                "%.1f".format(entP50), "%.1f".format(entP95),
                "%.1f".format(lapP50), "%.1f".format(lapP95),
                "%.1f".format(tenP50), "%.1f".format(tenP95),
                "%.1f".format(creP50), "%.1f".format(creP95)))
            onStatus("Bench K=$k infer=${"%.0f".format(infP50)}ms ent=${"%.0f".format(entP50)} lap=${"%.0f".format(lapP50)} ten=${"%.0f".format(tenP50)} cre=${"%.0f".format(creP50)}")
        }

        // Sensor control lag: cycle ISO across the grid and record wall-clock per capture,
        // whether the first result after each change already carries the requested value
        // (an empirical check on SYNC_MAX_LATENCY), and sensor timestamps for clock alignment.
        logger.csv("lag", listOf("step", "iso_req", "iso_applied",
            "exp_req_us", "exp_applied_us", "frame_number", "sensor_ts_ns", "wall_ms"))
        val gainsSorted = grid.gains.toList().distinct().sorted()
        val lo = gainsSorted.first()
        val mid = gainsSorted[gainsSorted.size / 2]
        val hi = gainsSorted.last()
        // Covers every pairwise transition plus a repeated value, which gives the baseline
        // pipeline time with no setting change at all.
        val isoSeq = listOf(lo, hi, lo, mid, hi, mid, lo, lo)
        val fixedExpUs = grid.exposuresUs[0]
        for ((step, iso) in isoSeq.withIndex()) {
            if (!running) break
            val tNs = System.nanoTime()
            val frames = raw.capture(fixedExpUs, iso, 1)
            val wallMs = (System.nanoTime() - tNs) / 1e6
            if (frames.isEmpty()) continue
            val m = raw.lastMeta.firstOrNull()
            logger.row("lag", listOf(step, iso, m?.appliedIso ?: -1,
                fixedExpUs, m?.appliedExpUs ?: -1L,
                m?.frameNumber ?: -1L, m?.timestamp ?: -1L,
                "%.1f".format(wallMs)))
            onStatus("Lag step=$step iso=$iso applied=${m?.appliedIso} ${"%.0f".format(wallMs)}ms")
        }
        logger.flush()
    }

    // ---------- shared helpers ----------

    /** Writes the manifest, opens the shared logs, and arms [running]. Modes with their own
     *  row schema pass `withFrames = false`. */
    private fun startPass(method: String, isGtReference: Boolean,
                          withFrames: Boolean = true,
                          methodParams: JSONObject? = null) {
        passStartMs = System.currentTimeMillis()
        val w = if (raw.captureWidth > 0) raw.captureWidth else -1
        val h = if (raw.captureHeight > 0) raw.captureHeight else -1
        writeManifest(method, isGtReference, w, h, methodParams)
        if (withFrames) logger.csv("frames", headers())
        wireImuLog()
        detectionTotalAboveThresh = 0
        detectionTotalAtFloor = 0
        totalFramesLogged = 0
        running = true
    }

    private fun endPass() {
        unwireImuLog()
        logger.flush()
        // Written so the run's totals can be cited from the logs alone.
        runCatching {
            val summary = JSONObject()
                .put("total_frames_logged", totalFramesLogged)
                .put("total_dets_above_threshold", detectionTotalAboveThresh)
                .put("total_dets_at_floor", detectionTotalAtFloor)
                .put("select_conf_threshold", selectConf)
                .put("detector_conf_floor", 0.01)
                .put("ts_ms", System.currentTimeMillis())
            java.io.File(logger.dir, "summary.json").writeText(summary.toString(2))
        }
    }

    private fun frameName(idx: Int) = "frame_%04d".format(idx)

    /** Mean luma of a formed bitmap as a fraction of white, for SAFE_CELL's brightness target. */
    private fun meanLumaRatio(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val step = (px.size / 100_000).coerceAtLeast(1)
        var sum = 0L; var count = 0
        var i = 0
        while (i < px.size) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            sum += (r * 299 + g * 587 + b * 114) / 1000
            count++
            i += step
        }
        return if (count == 0) 0.0 else (sum.toDouble() / count) / 255.0
    }

    /** Mean RAW value as a fraction of white, for Custom AE's brightness feedback. */
    private fun meanRawRatio(frame: RawFrame): Double {
        val pixels = frame.bayer
        val step = (pixels.size / 100_000).coerceAtLeast(1)
        var sum = 0L; var count = 0
        var i = 0
        while (i < pixels.size) { sum += pixels[i].toLong(); count++; i += step }
        return if (count == 0 || frame.maxDn <= 0) 0.0
        else (sum.toDouble() / count) / frame.maxDn
    }

    /** Closed-loop AE driving mean brightness toward [targetRatio]. ISO absorbs the correction
     *  first and exposure only grows once ISO clamps, which keeps motion blur down. Unlike
     *  phone AE it is fully determined by the brightness sequence, so runs are reproducible. */
    private class CustomAeBrightness(
        val targetRatio: Double = 0.40,    // bright but short of clipping
        val isoMin: Int = 100,
        val isoMax: Int = 1600,
        val expMinUs: Int = 1000,
        val expMaxUs: Int = 32_000,        // handheld motion-blur cap
        val maxStep: Double = 4.0          // per-call scale limit, damps dim -> bright swings
    ) {
        fun next(currentIso: Int, currentExpUs: Int, lastRatio: Double): Pair<Int, Int> {
            if (lastRatio < 0.001) return Pair(currentIso, currentExpUs)  // no signal — hold
            val scale = (targetRatio / lastRatio).coerceIn(1.0 / maxStep, maxStep)
            val newIso = (currentIso * scale).toInt().coerceIn(isoMin, isoMax)
            val resScale = scale * currentIso / newIso       // what ISO could not absorb
            val newExp = (currentExpUs * resScale).toInt().coerceIn(expMinUs, expMaxUs)
            return Pair(newIso, newExp)
        }

        fun toJson(): JSONObject = JSONObject()
            .put("target_ratio", targetRatio)
            .put("iso_min", isoMin).put("iso_max", isoMax)
            .put("exp_min_us", expMinUs).put("exp_max_us", expMaxUs)
            .put("max_step", maxStep)
    }

    /** ISO + exposure tag for filenames. Milliseconds when the exposure is whole, microseconds
     *  otherwise, so it is always lossless. No dots — they confuse extension parsing. */
    private fun isoExpTag(iso: Int, expUs: Int): String =
        if (expUs % 1000 == 0) "iso${iso}_exp${expUs / 1000}ms"
        else "iso${iso}_exp${expUs}us"

    /** Physical ISO × digitalBoost. Every bitmap goes through a formation path that applies
     *  the boost, so this is the number that matches what a JPEG actually looks like, and it
     *  is what JPEG filenames and the cell-grid UI use. RAW files keep physical ISO. */
    private fun effIso(iso: Int): Int = (iso * grid.digitalBoost).toInt()

    /** One bitmap from one capture that was already taken at the requested cell, so no digital
     *  re-gain beyond digitalBoost. Goes through the parallel path for the LUT and worker pool. */
    private fun bitmapFromRaw(frame: RawFrame): Bitmap =
        parallelFormation.formAllCells(listOf(frame), intArrayOf(grid.cell(0, 0)))[0]

    /** Shared formation path for the modes that capture separately from forming. Same
     *  implementation Proposed uses, so Bench measures the deployed cost. */
    private val parallelFormation by lazy { ParallelRawCandidateSource(grid, raw) }
    private fun formCells(frames: List<RawFrame>, cells: IntArray): List<Bitmap> =
        parallelFormation.formAllCells(frames, cells)
}
