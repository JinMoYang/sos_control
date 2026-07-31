package com.example.activeperception

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import com.example.activeperception.acquire.RawCapturer
import com.example.activeperception.acquire.RawFrame
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Camera2 RAW_SENSOR capturer implementing [RawCapturer]. Opens its own session, captures
 * `nBurst` RAW16 frames at a manual (exposure, ISO) with AE off, decodes to black-level-
 * subtracted linear Bayer, and records applied-vs-requested settings per frame in [lastMeta].
 *
 * Requires CAMERA permission (granted by the Activity). All blocking calls are bounded.
 */
class RawSensorCapturer(
    private val context: Context
) : RawCapturer {

    companion object { private const val TAG = "RawSensorCapturer" }

    data class RawMeta(
        val requestedIso: Int, val appliedIso: Int,
        val requestedExpUs: Long, val appliedExpUs: Long,
        val frameNumber: Long, val timestamp: Long,
        val black: Int, val white: Int
    )

    /** Per-frame metadata of the LAST capture() call, for actuation/settling logging. */
    @Volatile var lastMeta: List<RawMeta> = emptyList(); private set
    var cfaPattern: String = "RGGB"; private set
    var maxDn: Double = 1023.0; private set
    var syncMaxLatency: Int = CameraMetadata.SYNC_MAX_LATENCY_UNKNOWN; private set
    /** Original RAW dimensions, before [decode] downsamples. */
    var captureWidth: Int = 0; private set
    var captureHeight: Int = 0; private set
    /** Dimensions of the Bitmap the detector actually sees — and therefore the coordinate
     *  space of every box in dets.jsonl / candidate_dets.jsonl. Two halvings from the RAW
     *  size: [decode] block-skips 2× (forcing even dims), then `Formation.demosaic2x2`
     *  halves again. Recorded in manifest.json so offline tools never have to guess the
     *  scale factor between capture_resolution and the logged boxes. */
    val bitmapWidth: Int get() = ((captureWidth / 2) and 1.inv()) / 2
    val bitmapHeight: Int get() = ((captureHeight / 2) and 1.inv()) / 2
    /** SENSOR_INFO_TIMESTAMP_SOURCE — UNKNOWN(0) or REALTIME(1). Tells offline analysis whether
     *  camera SENSOR_TIMESTAMP shares a clock with the IMU sample timestamps. */
    var timestampSource: Int = 0; private set

    private val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private lateinit var cameraId: String
    private lateinit var chars: CameraCharacteristics
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var width = 0
    private var height = 0
    private var blackAvg = 64

    private val thread = HandlerThread("RawCapturer").apply { start() }
    private val handler = Handler(thread.looper)

    /** Open the back camera that advertises RAW; build a RAW16 reader + session. Blocks. */
    @Suppress("MissingPermission")
    fun open() {
        cameraId = cm.cameraIdList.firstOrNull { id ->
            val c = cm.getCameraCharacteristics(id)
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
            c.get(CameraCharacteristics.LENS_FACING) == CameraMetadata.LENS_FACING_BACK &&
                caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW)
        } ?: throw IllegalStateException("No back camera with RAW capability")

        chars = cm.getCameraCharacteristics(cameraId)
        maxDn = (chars.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023).toDouble()
        syncMaxLatency = chars.get(CameraCharacteristics.SYNC_MAX_LATENCY)
            ?: CameraMetadata.SYNC_MAX_LATENCY_UNKNOWN
        cfaPattern = when (chars.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)) {
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> "RGGB"
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> "GRBG"
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> "GBRG"
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> "BGGR"
            else -> "RGGB"
        }
        chars.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.let { p ->
            val v = IntArray(4); p.copyTo(v, 0); blackAvg = v.average().toInt()
        }
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val size = map.getOutputSizes(ImageFormat.RAW_SENSOR).maxByOrNull { it.width * it.height }
            ?: throw IllegalStateException("No RAW_SENSOR output size")
        width = size.width; height = size.height
        captureWidth = width; captureHeight = height
        timestampSource = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ?: 0
        reader = ImageReader.newInstance(width, height, ImageFormat.RAW_SENSOR, /*maxImages*/ 16)

        val opened = CountDownLatch(1)
        cm.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(c: CameraDevice) { device = c; opened.countDown() }
            override fun onDisconnected(c: CameraDevice) { c.close(); device = null }
            override fun onError(c: CameraDevice, e: Int) { c.close(); device = null; opened.countDown() }
        }, handler)
        opened.await(3, TimeUnit.SECONDS)
        val dev = device ?: throw IllegalStateException("camera open failed")

        val configured = CountDownLatch(1)
        @Suppress("DEPRECATION")
        dev.createCaptureSession(listOf(reader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) { session = s; configured.countDown() }
            override fun onConfigureFailed(s: CameraCaptureSession) { configured.countDown() }
        }, handler)
        configured.await(3, TimeUnit.SECONDS)
        if (session == null) throw IllegalStateException("capture session failed")
        // Manual-control ranges and capabilities, for sizing the Grid and diagnosing silent
        // overrides. MANUAL_SENSOR is required for SENSOR_SENSITIVITY to be honored at all —
        // without it the request is accepted and ignored.
        val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
        val hasManualSensor = caps.contains(
            CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
        val hasManualPostProc = caps.contains(
            CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING)
        Log.d(TAG, "RAW open: ${width}x$height cfa=$cfaPattern white=$maxDn syncLat=$syncMaxLatency " +
                "iso=$isoRange exp_ns=$expRange manualSensor=$hasManualSensor manualPostProc=$hasManualPostProc " +
                "caps=${caps.toList()}")
    }

    override fun capture(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        // Deliberately minimal. Adding CONTROL_MODE=OFF / NOISE_REDUCTION=OFF /
        // POST_RAW_SENSITIVITY_BOOST changed nothing measurable, and CONTROL_MODE=OFF risks
        // suppressing the 3A pipeline the sensor gain rides on.
        val req = device!!.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureUs.toLong() * 1000L)
            set(CaptureRequest.SENSOR_SENSITIVITY, iso)
            addTarget(reader!!.surface)
        }.build()
        return runBurst(req, maxOf(1, nBurst), requestedIso = iso, requestedExpUs = exposureUs.toLong())
    }

    /** AE-on capture: the HAL picks ISO + exposure, read back into [lastMeta].
     *
     *  Caveat for the AE baseline: this is a one-shot capture, not a repeating request, so
     *  AE gets one observation per call with the whole detect/log pipeline in between. It has
     *  far less opportunity to converge than AE in a normal 30fps preview stream. */
    fun captureAe(nBurst: Int): List<RawFrame> {
        val req = device!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
            set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            addTarget(reader!!.surface)
        }.build()
        // -1 sentinels: there is no requested value, so frames.csv records only the applied one.
        return runBurst(req, maxOf(1, nBurst), requestedIso = -1, requestedExpUs = -1L)
    }

    /** Shared burst execution + image/result pairing for both manual and AE capture. */
    private fun runBurst(req: CaptureRequest, n: Int, requestedIso: Int, requestedExpUs: Long): List<RawFrame> {
        val dev = device ?: error("not opened"); val sess = session ?: error("no session")
        val images = LinkedBlockingQueue<Image>()
        reader!!.setOnImageAvailableListener({ r -> r.acquireNextImage()?.let { images.add(it) } }, handler)

        val results = ArrayList<TotalCaptureResult>()
        val done = CountDownLatch(n)
        sess.captureBurst(List(n) { req }, object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(s: CameraCaptureSession, rq: CaptureRequest, res: TotalCaptureResult) {
                synchronized(results) { results.add(res) }; done.countDown()
            }
            override fun onCaptureFailed(s: CameraCaptureSession, rq: CaptureRequest, f: android.hardware.camera2.CaptureFailure) {
                done.countDown()
            }
        }, handler)
        done.await(5, TimeUnit.SECONDS)

        val frames = ArrayList<RawFrame>()
        val metas = ArrayList<RawMeta>()
        try {
            for (k in 0 until n) {
                val img = images.poll(2, TimeUnit.SECONDS) ?: break
                try {
                    val res = synchronized(results) { results.getOrNull(k) }
                    // Per-frame DYNAMIC_BLACK_LEVEL, not the static SENSOR_BLACK_LEVEL_PATTERN:
                    // on S25 the static value is 64 while the measured dynamic black is ~256, so
                    // the static one leaves ~192 ADU of noise floor being treated as signal.
                    val dynArr = res?.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
                    val perFrameBlack = dynArr?.average()?.toInt() ?: blackAvg
                    val decoded = decode(img, perFrameBlack)
                    frames.add(decoded)
                    metas.add(RawMeta(
                        requestedIso = requestedIso,
                        appliedIso = res?.get(CaptureResult.SENSOR_SENSITIVITY) ?: -1,
                        requestedExpUs = requestedExpUs,
                        appliedExpUs = (res?.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: -1000L) / 1000L,
                        frameNumber = res?.frameNumber ?: -1L,
                        timestamp = res?.get(CaptureResult.SENSOR_TIMESTAMP) ?: -1L,
                        black = perFrameBlack, white = maxDn.toInt()))
                    val mean = decoded.bayer.average()
                    Log.d(TAG, "frame k=$k iso=${res?.get(CaptureResult.SENSOR_SENSITIVITY)} " +
                            "exp=${res?.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.div(1000L)}us " +
                            "rawMean=%.1f black=$perFrameBlack".format(mean))
                } finally { img.close() }
            }
        } finally {
            drainReader(images)
        }
        lastMeta = metas
        return frames
    }

    /**
     * Release every Image this burst did not consume. Without this, a burst that ends
     * early (capture failure, `done.await` timeout, decode throw) leaves acquired Images
     * in the queue, and images arriving after runBurst returns land in a dead queue that
     * nobody closes. The reader is built with maxImages=16, so a handful of leaked bursts
     * exhausts it and `acquireNextImage()` starts returning null / throwing — capture then
     * dies silently mid-run, which is the worst possible failure mode for a 300-frame pass.
     *
     * Ordering matters: detach the listener FIRST so no new Images are acquired into
     * [images], then do the drain ON THE CAPTURE HANDLER so it serializes behind any
     * listener callback already posted to that thread (the callback and this method run on
     * different threads otherwise). Finally pull any buffers the reader still holds that
     * the listener never got to acquire.
     */
    private fun drainReader(images: LinkedBlockingQueue<Image>) {
        val r = reader ?: return
        runCatching { r.setOnImageAvailableListener(null, null) }
        val drained = CountDownLatch(1)
        val posted = handler.post {
            try {
                while (true) (images.poll() ?: break).close()
                while (true) {
                    val extra = runCatching { r.acquireNextImage() }.getOrNull() ?: break
                    extra.close()
                }
            } finally { drained.countDown() }
        }
        // If the handler thread is gone (close() already ran), fall back to draining here.
        if (!posted) { while (true) (images.poll() ?: break).close(); return }
        drained.await(1, TimeUnit.SECONDS)
    }

    /** RAW16 Image -> black-subtracted linear Bayer IntArray (row-major).
     *
     *  Downsampled by taking every other CFA 2×2 block, which preserves the Bayer pattern at
     *  half dimensions for 1/4 the memory — full-res would be ~50 MB per frame and a burst
     *  would blow past largeHeap. Detector input is letterboxed to 640 anyway. Output dims
     *  are forced even so demosaic2x2 stays aligned. */
    private fun decode(img: Image, blackToSubtract: Int = blackAvg): RawFrame {
        val plane = img.planes[0]
        val buf = plane.buffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val rowStride = plane.rowStride / 2
        val ow = (width / 2) and 1.inv()
        val oh = (height / 2) and 1.inv()
        val out = IntArray(ow * oh)
        for (oy in 0 until oh) {
            val iy = (oy and 1.inv()) * 2 + (oy and 1)   // (oy/2)*4 + (oy%2)
            val rowIn = iy * rowStride
            val rowOut = oy * ow
            for (ox in 0 until ow) {
                val ix = (ox and 1.inv()) * 2 + (ox and 1)
                val v = (buf.get(rowIn + ix).toInt() and 0xFFFF) - blackToSubtract
                out[rowOut + ox] = if (v < 0) 0 else v
            }
        }
        return RawFrame(out, ow, oh, cfaPattern, maxDn)
    }

    fun close() {
        runCatching { session?.close() }; runCatching { device?.close() }
        runCatching { reader?.close() }; runCatching { thread.quitSafely() }
        session = null; device = null; reader = null
    }
}
