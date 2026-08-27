package com.example.activeperception

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.util.Size
import com.example.activeperception.acquire.RawCapturer
import com.example.activeperception.acquire.RawFrame
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.LongAdder
import kotlin.math.max

/**
 * RayNeo X3 Pro camera-0 RAW capturer.
 *
 * Camera callbacks only hand Images to a bounded worker pool. Workers pair each Image with
 * its CaptureResult by SENSOR_TIMESTAMP, copy/decode Bayer data, and close the Image. A RAW
 * becomes visible to the SoS pipeline only when the result's exposure/ISO actually matches
 * the requested state. This is intentionally independent of SYNC_MAX_LATENCY because the
 * device declares 0 while physical measurements show 9-10 frames of delay.
 */
class RawSensorCapturer(private val context: Context) : RawCapturer {

    companion object {
        private const val TAG = "RayNeoRawCapturer"
        private const val CAMERA_ID = "0"
        private const val BASE_FRAME_DURATION_NS = 33_329_000L
        private const val SETTING_GUARD_FRAMES = 12
        private const val MAX_IMAGES = 32
        private const val CONTINUOUS_RING_CAPACITY = 6
        // Only used if the HAL omits per-frame AWB metadata. Measured from RayNeo camera 0;
        // normal operation uses CaptureResult.COLOR_CORRECTION_GAINS instead.
        private val RAYNEO_WB_FALLBACK = floatArrayOf(1.8896484f, 1f, 1.7802734f)
        private val IDENTITY_CCM = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    }

    data class RawMeta(
        val requestedIso: Int,
        val appliedIso: Int,
        val requestedExpUs: Long,
        val appliedExpUs: Long,
        val frameNumber: Long,
        val timestamp: Long,
        val black: Int,
        val blackLevels: IntArray,
        val white: Int,
        val frameDurationNs: Long,
        val applyDelayFrames: Long?,
        val rowStrideBytes: Int,
        val pixelStrideBytes: Int,
        val whiteBalance: FloatArray,
        val cameraToSrgb: DoubleArray
    )

    data class CaptureProfile(
        val strategy: String,
        val requestedFrames: Int,
        val firstResultMs: Double,
        val firstImageMs: Double,
        val firstMatchedMs: Double,
        val decodeWallMs: Double,
        val decodeCpuSumMs: Double,
        val cleanupMs: Double,
        val totalMs: Double
    )

    data class ContinuousProfile(
        val requestedFrames: Int,
        val firstResultMs: Double,
        val firstMatchedMs: Double,
        val readyMs: Double,
        val decodeCpuSumMs: Double,
        val staleFramesDropped: Int,
        val decodedFrames: Int
    )

    data class RawStreamConfig(
        val format: Int,
        val formatName: String,
        val width: Int,
        val height: Int,
        val minFrameDurationNs: Long,
        val stallDurationNs: Long,
        val estimatedBufferBytes: Long
    )

    class FastSequenceHandle internal constructor(
        val commandId: Long,
        val requestedExpUs: Int,
        val requestedIso: Int,
        val count: Int,
        val startedNs: Long,
        val startedSensorNs: Long
    )

    data class CaptureSnapshot(
        val metas: List<RawMeta>,
        val profile: CaptureProfile?
    )

    private data class Decoded(val frame: RawFrame, val meta: RawMeta)

    /** One finite exact-N request routed through the persistent low-latency listener. */
    private class FastTarget(
        val expected: SensorState,
        val commandId: Long,
        val wanted: Int,
        val startedNs: Long
    ) {
        val lock = java.lang.Object()
        val decoded = ArrayList<Decoded>()
        val ready = LinkedBlockingQueue<Decoded>()
        var firstResultNs = 0L
        var firstImageNs = 0L
        var firstMatchedNs = 0L
        var firstDecodeStartNs = 0L
        var lastDecodeEndNs = 0L
        var decodeCpuNs = 0L
        var failures = 0
        /** Frames whose metadata missed the request (settling) — skipped, not fatal. */
        var settlingSkips = 0
        /** Extra requests submitted to replace settling frames, bounded by the guard. */
        var resubmits = 0
        var error: Throwable? = null
    }

    private class ContinuousTarget(
        val expected: SensorState,
        val commandId: Long,
        val wanted: Int,
        val startedNs: Long
    ) {
        val lock = java.lang.Object()
        val ring = ArrayList<Decoded>(CONTINUOUS_RING_CAPACITY)
        var firstResultNs = 0L
        var firstMatchedNs = 0L
        var readyNs = 0L
        var decodeCpuNs = 0L
        var staleFrames = 0
        var decodedFrames = 0
        var error: Throwable? = null
    }

    @Volatile var lastMeta: List<RawMeta> = emptyList(); private set
    @Volatile var lastSettlingMeta: List<RawMeta> = emptyList(); private set
    @Volatile var lastApplyDelayFrames: Long? = null; private set
    @Volatile var lastCaptureProfile: CaptureProfile? = null; private set
    @Volatile var lastContinuousProfile: ContinuousProfile? = null; private set
    @Volatile var lastNativeDecodeMismatchCount: Int = -1; private set
    val pendingCommandCount: Int get() = scheduler.pendingSnapshot().size

    var cfaPattern: String = "RGGB"; private set
    var maxDn: Double = 1023.0; private set
    var syncMaxLatency: Int = CameraMetadata.SYNC_MAX_LATENCY_UNKNOWN; private set
    var captureWidth: Int = 0; private set
    var captureHeight: Int = 0; private set
    private val sensorBitmapWidth: Int get() = ((captureWidth / 2) and 1.inv()) / 2
    private val sensorBitmapHeight: Int get() = ((captureHeight / 2) and 1.inv()) / 2
    val bitmapWidth: Int get() = if (sensorOrientation.mod(180) == 0) sensorBitmapWidth else sensorBitmapHeight
    val bitmapHeight: Int get() = if (sensorOrientation.mod(180) == 0) sensorBitmapHeight else sensorBitmapWidth
    var timestampSource: Int = 0; private set
    var sensorOrientation: Int = 0; private set
    var streamFormat: Int = ImageFormat.RAW_SENSOR; private set
    val streamFormatName: String get() = formatName(streamFormat)
    /** Min frame duration of the ACTIVE RAW stream, from the stream configuration map;
     *  falls back to the RayNeo-measured 33.329ms when the HAL reports none. Querying the
     *  advertised value instead of pinning the constant lets the same code drive the S25's
     *  60fps RAW16 stream and the RayNeo's 30fps RAW10 stream without device branches. */
    @Volatile var streamMinFrameDurationNs: Long = BASE_FRAME_DURATION_NS; private set

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private lateinit var chars: CameraCharacteristics
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var width = 0
    private var height = 0
    private var staticBlack = intArrayOf(0, 0, 0, 0)
    @Volatile private var sessionClosedLatch: CountDownLatch? = null

    // Camera metadata and ImageReader delivery must not wait behind formation workers.
    // Android's DISPLAY priority is a scheduling hint only; it does not busy-spin or reserve
    // a core, but lets these very short callbacks pre-empt ordinary CPU work promptly.
    private val cameraThread = HandlerThread(
        "RayNeo-CameraResult", Process.THREAD_PRIORITY_DISPLAY).apply { start() }
    private val cameraHandler = Handler(cameraThread.looper)
    private val imageThread = HandlerThread(
        "RayNeo-RawImage", Process.THREAD_PRIORITY_DISPLAY).apply { start() }
    private val imageHandler = Handler(imageThread.looper)
    @Volatile private var decodePool = Executors.newFixedThreadPool(4)
    @Volatile var decodeThreadCount: Int = 4; private set
    // isoTolerance: the S25 HAL quantizes applied sensitivity (req 100 -> applied 99,
    // 400 -> 398), so exact ISO matching never approves a frame there. 8 covers the
    // observed <=0.5% rounding up to ISO 1600 while staying far below the 2x grid spacing.
    private val scheduler = SensorStateScheduler(isoTolerance = 8)
    private val nativeValidationRemaining = AtomicInteger(0)
    private val continuousResults = ConcurrentHashMap<Long, TotalCaptureResult>()
    private val continuousResultLock = java.lang.Object()
    @Volatile private var activeContinuousTarget: ContinuousTarget? = null
    @Volatile private var continuousStarted = false
    private val fastResults = ConcurrentHashMap<Long, TotalCaptureResult>()
    private val fastResultLock = java.lang.Object()
    private val fastTargets = ConcurrentHashMap<Long, FastTarget>()
    private val captureSnapshot = ThreadLocal<CaptureSnapshot>()
    @Volatile private var fastCaptureStarted = false
    @Volatile private var currentState: SensorState? = null
    @Volatile private var lastWhiteBalance = RAYNEO_WB_FALLBACK.copyOf()
    @Volatile private var lastCameraToSrgb = IDENTITY_CCM.copyOf()

    @Suppress("MissingPermission")
    fun open() {
        require(manager.cameraIdList.contains(CAMERA_ID)) { "RayNeo camera 0 is unavailable" }
        chars = manager.getCameraCharacteristics(CAMERA_ID)
        val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
        require(caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW)) {
            "RayNeo camera 0 does not expose RAW"
        }
        require(caps.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)) {
            "RayNeo camera 0 does not expose MANUAL_SENSOR"
        }

        maxDn = (chars.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023).toDouble()
        syncMaxLatency = chars.get(CameraCharacteristics.SYNC_MAX_LATENCY)
            ?: CameraMetadata.SYNC_MAX_LATENCY_UNKNOWN
        timestampSource = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ?: 0
        sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        cfaPattern = when (chars.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)) {
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> "RGGB"
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> "GRBG"
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> "GBRG"
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> "BGGR"
            else -> error("Unsupported CFA pattern")
        }
        chars.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.copyTo(staticBlack, 0)

        val size = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputSizes(ImageFormat.RAW_SENSOR)
            ?.maxByOrNull { it.width * it.height }
            ?: error("RayNeo camera 0 has no RAW_SENSOR size")
        width = size.width
        height = size.height
        captureWidth = width
        captureHeight = height
        reader = ImageReader.newInstance(width, height, ImageFormat.RAW_SENSOR, MAX_IMAGES)
        streamFormat = ImageFormat.RAW_SENSOR
        streamMinFrameDurationNs = (chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputMinFrameDuration(ImageFormat.RAW_SENSOR, size) ?: 0L)
            .takeIf { it > 0L } ?: BASE_FRAME_DURATION_NS

        val opened = CountDownLatch(1)
        manager.openCamera(CAMERA_ID, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) { device = camera; opened.countDown() }
            override fun onDisconnected(camera: CameraDevice) {
                camera.close(); device = null; opened.countDown()
            }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close(); device = null; opened.countDown()
            }
        }, cameraHandler)
        check(opened.await(4, TimeUnit.SECONDS) && device != null) { "camera 0 open failed" }

        val configured = CountDownLatch(1)
        @Suppress("DEPRECATION")
        device!!.createCaptureSession(listOf(reader!!.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(value: CameraCaptureSession) {
                    session = value; configured.countDown()
                }
                override fun onConfigureFailed(value: CameraCaptureSession) {
                    configured.countDown()
                }
                override fun onClosed(value: CameraCaptureSession) {
                    sessionClosedLatch?.countDown()
                }
            }, cameraHandler)
        check(configured.await(4, TimeUnit.SECONDS) && session != null) {
            "camera 0 RAW session failed"
        }
        Log.i(TAG, "camera0=${width}x$height cfa=$cfaPattern white=$maxDn " +
            "syncDeclared=$syncMaxLatency timestamp=$timestampSource orientation=$sensorOrientation")
    }

    fun availableRawStreamConfigs(): List<RawStreamConfig> {
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return emptyList()
        return intArrayOf(ImageFormat.RAW_SENSOR, ImageFormat.RAW10, ImageFormat.RAW12)
            .flatMap { format ->
                (map.getOutputSizes(format) ?: emptyArray<Size>()).map { size ->
                    val bits = when (format) {
                        ImageFormat.RAW10 -> 10L
                        ImageFormat.RAW12 -> 12L
                        else -> 16L
                    }
                    RawStreamConfig(format, formatName(format), size.width, size.height,
                        map.getOutputMinFrameDuration(format, size),
                        map.getOutputStallDuration(format, size),
                        size.width.toLong() * size.height * bits / 8L)
                }
            }.sortedWith(compareBy<RawStreamConfig> { it.format }
                .thenByDescending { it.width.toLong() * it.height })
    }

    /** EXP5.3 only: reconfigure the existing camera device to another advertised RAW format. */
    @Synchronized
    fun reconfigureRawStream(format: Int, requestedWidth: Int, requestedHeight: Int) {
        require(format == ImageFormat.RAW_SENSOR || format == ImageFormat.RAW10 ||
            format == ImageFormat.RAW12)
        val supported = availableRawStreamConfigs().any {
            it.format == format && it.width == requestedWidth && it.height == requestedHeight
        }
        require(supported) { "Unsupported ${formatName(format)} ${requestedWidth}x$requestedHeight" }
        stopContinuousCapture()
        stopFastCapture()
        val closed = CountDownLatch(1)
        sessionClosedLatch = closed
        runCatching { session?.close() }
        closed.await(2, TimeUnit.SECONDS)
        sessionClosedLatch = null
        reader?.setOnImageAvailableListener(null, null)
        reader?.let { drainReader(it) }
        reader?.close()
        session = null

        width = requestedWidth; height = requestedHeight
        captureWidth = width; captureHeight = height; streamFormat = format
        streamMinFrameDurationNs = (chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputMinFrameDuration(format, android.util.Size(width, height)) ?: 0L)
            .takeIf { it > 0L } ?: BASE_FRAME_DURATION_NS
        reader = ImageReader.newInstance(width, height, format, MAX_IMAGES)
        val configured = CountDownLatch(1)
        @Suppress("DEPRECATION")
        device!!.createCaptureSession(listOf(reader!!.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(value: CameraCaptureSession) {
                    session = value; configured.countDown()
                }
                override fun onConfigureFailed(value: CameraCaptureSession) {
                    configured.countDown()
                }
                override fun onClosed(value: CameraCaptureSession) {
                    sessionClosedLatch?.countDown()
                }
            }, cameraHandler)
        check(configured.await(4, TimeUnit.SECONDS) && session != null) {
            "RAW session reconfigure failed for ${formatName(format)}"
        }
        currentState = null
        drainReader(reader!!)
        Log.i(TAG, "EXP5.3 stream=${formatName(format)} ${width}x$height")
    }

    private fun formatName(format: Int): String = when (format) {
        ImageFormat.RAW_SENSOR -> "RAW_SENSOR"
        ImageFormat.RAW10 -> "RAW10"
        ImageFormat.RAW12 -> "RAW12"
        else -> "FORMAT_$format"
    }

    override fun capture(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        return captureInternal(exposureUs, iso, nBurst, metadataFirst = false)
    }

    /** Experiment path: submit at most the measured 12-frame actuation window but stop as
     * soon as metadata exposes [nBurst] frames at the requested state. This keeps the safety
     * ceiling while removing the fixed wait from the accepted image path. */
    fun captureMetadataFirst(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        return captureInternal(exposureUs, iso, nBurst, metadataFirst = true)
    }

    /** EXP5.1 B: submit exactly the frames needed, with no fixed settling guard. A mismatch
     * is reported to the caller instead of silently accepting stale sensor state. */
    fun captureNoGuard(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val request = manualRequest(expected, command.id)
        return runCapture(request, max(1, nBurst), 0, expected, command.id)
    }

    /** EXP5.1.1 optimized B: exact-N request plus native NEON RAW decode. */
    fun captureNoGuardNative(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val request = manualRequest(expected, command.id)
        return runCapture(request, max(1, nBurst), 0, expected, command.id,
            nativeDecode = true)
    }

    /** EXP5.1.1 E: same exact-N/native path with the low-latency preview request template. */
    fun captureNoGuardNativePreview(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val request = manualRequest(expected, command.id, CameraDevice.TEMPLATE_PREVIEW)
        return runCapture(request, max(1, nBurst), 0, expected, command.id,
            nativeDecode = true)
    }

    /** Best safe EXP5.1.1 combination: preview template for latency-sensitive single RAW,
     * still template for four-frame bursts where preview increased tail latency. */
    fun captureExp511Best(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> =
        if (nBurst <= 1) captureNoGuardNativePreview(exposureUs, iso, 1)
        else captureNoGuardNative(exposureUs, iso, nBurst)

    /**
     * EXP5.6: install ImageReader routing once, keep Camera result and RAW image delivery on
     * separate high-priority HandlerThreads, and decode every image directly from its native
     * plane buffer as soon as its timestamp-matched result is available. Unlike EXP5.2 this
     * does not run a repeating stream: each call still submits exactly N finite requests.
     */
    @Synchronized
    fun startFastCapture() {
        if (fastCaptureStarted) return
        check(!continuousStarted) { "continuous capture is active" }
        val imageReader = reader ?: error("RAW reader is closed")
        fastResults.clear(); fastTargets.clear(); drainReader(imageReader)
        imageReader.setOnImageAvailableListener({ source ->
            val image = runCatching { source.acquireNextImage() }.getOrNull()
                ?: return@setOnImageAvailableListener
            val imageArrivalNs = System.nanoTime()
            // Ownership of Image moves to exactly one decode worker. No 12 MP plane copy is
            // made: native code reads plane.buffer directly and Image closes after decode.
            decodePool.execute {
                try {
                    val result = awaitResult(image.timestamp, fastResults,
                        fastResultLock, 2_000) ?: return@execute
                    val commandId = result.request.tag as? Long ?: return@execute
                    val target = fastTargets[commandId] ?: return@execute
                    val actual = result.sensorState()
                    if (!scheduler.matches(target.expected, actual)) {
                        // Settling frame: a HAL may deliver a tagged request's frame with
                        // pre-settling metadata (seen on the S25 from a cold stream). Skip
                        // it — the result callback resubmits the slot — rather than failing
                        // the whole capture.
                        synchronized(target.lock) { target.settlingSkips++ }
                        return@execute
                    }
                    val decodeStart = System.nanoTime()
                    val decoded = decode(image, result, target.expected, commandId,
                        nativeDecode = true)
                    val decodeEnd = System.nanoTime()
                    synchronized(target.lock) {
                        if (target.firstImageNs == 0L) target.firstImageNs = imageArrivalNs
                        if (target.firstMatchedNs == 0L) target.firstMatchedNs = decodeStart
                        if (target.firstDecodeStartNs == 0L ||
                            decodeStart < target.firstDecodeStartNs) {
                            target.firstDecodeStartNs = decodeStart
                        }
                        if (decodeEnd > target.lastDecodeEndNs) target.lastDecodeEndNs = decodeEnd
                        target.decodeCpuNs += decodeEnd - decodeStart
                        target.decoded.add(decoded)
                        target.ready.offer(decoded)
                        if (target.decoded.size >= target.wanted) target.lock.notifyAll()
                    }
                } catch (t: Throwable) {
                    // Route an error only when its request can still be identified. The caller
                    // otherwise reaches the bounded timeout and reports the missing frame.
                    fastTargets.values.forEach { target ->
                        synchronized(target.lock) {
                            if (target.error == null) target.error = t
                            target.lock.notifyAll()
                        }
                    }
                } finally {
                    image.close()
                }
            }
        }, imageHandler)
        fastCaptureStarted = true
        Log.i(TAG, "EXP5.6 persistent finite listener armed")
    }

    /** Exact-N on-demand capture using [startFastCapture]'s persistent routing. */
    fun captureFast(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        check(fastCaptureStarted) { "startFastCapture() must be called first" }
        val wanted = max(1, nBurst)
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val target = FastTarget(expected, command.id, wanted, System.nanoTime())
        check(fastTargets.putIfAbsent(command.id, target) == null)
        val template = if (wanted == 1) CameraDevice.TEMPLATE_PREVIEW
            else CameraDevice.TEMPLATE_STILL_CAPTURE
        val request = manualRequest(expected, command.id, template)
        val callback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                val nowNs = System.nanoTime()
                synchronized(target.lock) {
                    if (target.firstResultNs == 0L) target.firstResultNs = nowNs
                }
                val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: -1L
                scheduler.markRequestFrame(command.id, result.frameNumber)
                val actual = result.sensorState()
                scheduler.observe(result.frameNumber, ts, actual)?.let {
                    lastApplyDelayFrames = it.applyDelayFrames; currentState = actual
                }
                fastResults[ts] = result
                synchronized(fastResultLock) { fastResultLock.notifyAll() }
                // Replace a settling frame's slot so exact-N still completes; bounded so a
                // persistently wrong HAL state cannot loop forever.
                if (!scheduler.matches(target.expected, actual)) {
                    val resubmit = synchronized(target.lock) {
                        if (target.resubmits < SETTING_GUARD_FRAMES) { target.resubmits++; true }
                        else false
                    }
                    if (resubmit) runCatching { session.capture(request, this, cameraHandler) }
                }
            }

            override fun onCaptureFailed(
                session: CameraCaptureSession,
                request: CaptureRequest,
                failure: CaptureFailure
            ) {
                synchronized(target.lock) {
                    target.failures++
                    target.error = IllegalStateException(
                        "Fast capture failed frame=${failure.frameNumber} reason=${failure.reason}")
                    target.lock.notifyAll()
                }
            }
        }
        val sess = session ?: error("camera session is closed")
        sess.captureBurst(List(wanted) { request }, callback, cameraHandler)
        val timeoutNs = TimeUnit.MILLISECONDS.toNanos(max(6_000L, wanted * 400L))
        val deadline = System.nanoTime() + timeoutNs
        synchronized(target.lock) {
            while (target.decoded.size < wanted && target.error == null) {
                val left = deadline - System.nanoTime()
                if (left <= 0L) break
                target.lock.wait(minOf(100L,
                    TimeUnit.NANOSECONDS.toMillis(left).coerceAtLeast(1L)))
            }
        }
        fastTargets.remove(command.id)
        target.error?.let { throw it }
        val accepted = synchronized(target.lock) { target.decoded.sortedBy { it.meta.frameNumber } }
        check(accepted.size == wanted) {
            "Fast capture obtained ${accepted.size}/$wanted RAW frames; " +
                "requested=${exposureUs}us ISO$iso failures=${target.failures}"
        }
        val endNs = System.nanoTime()
        fun sinceStart(value: Long): Double = if (value == 0L) Double.NaN
            else (value - target.startedNs) / 1e6
        lastCaptureProfile = CaptureProfile(
            strategy = "persistent_finite_direct_native",
            requestedFrames = wanted,
            firstResultMs = sinceStart(target.firstResultNs),
            firstImageMs = sinceStart(target.firstImageNs),
            firstMatchedMs = sinceStart(target.firstMatchedNs),
            decodeWallMs = if (target.firstDecodeStartNs == 0L || target.lastDecodeEndNs == 0L)
                Double.NaN else (target.lastDecodeEndNs - target.firstDecodeStartNs) / 1e6,
            decodeCpuSumMs = target.decodeCpuNs / 1e6,
            cleanupMs = 0.0,
            totalMs = (endNs - target.startedNs) / 1e6)
        lastSettlingMeta = accepted.map { it.meta }
        lastMeta = lastSettlingMeta
        accepted.lastOrNull()?.let {
            currentState = SensorState(it.meta.appliedExpUs, it.meta.appliedIso)
        }
        return accepted.map { it.frame }
    }

    /**
     * Submit a finite run of same-exposure RAW requests without waiting for all of them.
     * [takeFastSequenceFrame] releases each decoded frame to formation immediately, allowing
     * later sensor frames to arrive while the current frame is on CPU/GPU.
     */
    fun beginFastSequence(exposureUs: Int, iso: Int, count: Int): FastSequenceHandle {
        check(fastCaptureStarted) { "startFastCapture() must be called first" }
        require(count in 1..MAX_IMAGES)
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val target = FastTarget(expected, command.id, count, System.nanoTime())
        val startedSensorNs = SystemClock.elapsedRealtimeNanos()
        check(fastTargets.putIfAbsent(command.id, target) == null)
        val request = manualRequest(expected, command.id, CameraDevice.TEMPLATE_PREVIEW)
        val callback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                val nowNs = System.nanoTime()
                synchronized(target.lock) {
                    if (target.firstResultNs == 0L) target.firstResultNs = nowNs
                }
                val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: -1L
                scheduler.markRequestFrame(command.id, result.frameNumber)
                val actual = result.sensorState()
                scheduler.observe(result.frameNumber, ts, actual)?.let {
                    lastApplyDelayFrames = it.applyDelayFrames; currentState = actual
                }
                fastResults[ts] = result
                synchronized(fastResultLock) { fastResultLock.notifyAll() }
                // Same settling-slot replacement as captureFast.
                if (!scheduler.matches(target.expected, actual)) {
                    val resubmit = synchronized(target.lock) {
                        if (target.resubmits < SETTING_GUARD_FRAMES) { target.resubmits++; true }
                        else false
                    }
                    if (resubmit) runCatching { session.capture(request, this, cameraHandler) }
                }
            }

            override fun onCaptureFailed(
                session: CameraCaptureSession,
                request: CaptureRequest,
                failure: CaptureFailure
            ) {
                synchronized(target.lock) {
                    target.failures++
                    target.error = IllegalStateException(
                        "Fast sequence failed frame=${failure.frameNumber} reason=${failure.reason}")
                    target.lock.notifyAll()
                }
            }
        }
        (session ?: error("camera session is closed"))
            .captureBurst(List(count) { request }, callback, cameraHandler)
        return FastSequenceHandle(command.id, exposureUs, iso, count, target.startedNs,
            startedSensorNs)
    }

    fun takeFastSequenceFrame(
        handle: FastSequenceHandle,
        timeoutMs: Long = 6_000L
    ): RawFrame {
        val target = fastTargets[handle.commandId]
            ?: error("Fast sequence ${handle.commandId} is no longer active")
        val decoded = target.ready.poll(timeoutMs, TimeUnit.MILLISECONDS)
            ?: run {
                target.error?.let { throw it }
                error("Timed out waiting for fast sequence ${handle.commandId}")
            }
        val nowNs = System.nanoTime()
        synchronized(target.lock) {
            lastCaptureProfile = CaptureProfile(
                strategy = "persistent_finite_streaming",
                requestedFrames = handle.count,
                firstResultMs = if (target.firstResultNs == 0L) Double.NaN
                    else (target.firstResultNs - target.startedNs) / 1e6,
                firstImageMs = if (target.firstImageNs == 0L) Double.NaN
                    else (target.firstImageNs - target.startedNs) / 1e6,
                firstMatchedMs = if (target.firstMatchedNs == 0L) Double.NaN
                    else (target.firstMatchedNs - target.startedNs) / 1e6,
                decodeWallMs = if (target.firstDecodeStartNs == 0L || target.lastDecodeEndNs == 0L)
                    Double.NaN else (target.lastDecodeEndNs - target.firstDecodeStartNs) / 1e6,
                decodeCpuSumMs = target.decodeCpuNs / 1e6,
                cleanupMs = 0.0,
                totalMs = (nowNs - target.startedNs) / 1e6)
        }
        lastMeta = listOf(decoded.meta); lastSettlingMeta = lastMeta
        currentState = SensorState(decoded.meta.appliedExpUs, decoded.meta.appliedIso)
        if (target.ready.isEmpty() && synchronized(target.lock) {
                target.decoded.size >= target.wanted
            }) {
            fastTargets.remove(handle.commandId)
        }
        return decoded.frame
    }

    @Synchronized
    fun stopFastCapture() {
        if (!fastCaptureStarted) return
        reader?.setOnImageAvailableListener(null, null)
        reader?.let { drainReader(it) }
        fastTargets.clear(); fastResults.clear()
        fastCaptureStarted = false
    }

    /** EXP5.2: keep one listener and one repeating RAW stream alive for a bounded experiment.
     * Images whose request tag or applied sensor state is stale are closed before decode.
     * Only metadata-confirmed frames enter the six-frame decoded ring. */
    @Synchronized
    fun startContinuousCapture() {
        if (continuousStarted) return
        val imageReader = reader ?: error("RAW reader is closed")
        val sess = session ?: error("camera session is closed")
        activeContinuousTarget = null
        continuousResults.clear()
        drainReader(imageReader)
        imageReader.setOnImageAvailableListener({ source ->
            val image = runCatching { source.acquireNextImage() }.getOrNull()
                ?: return@setOnImageAvailableListener
            decodePool.execute {
                try {
                    val result = awaitResult(image.timestamp, continuousResults,
                        continuousResultLock, 2_000) ?: return@execute
                    val target = activeContinuousTarget
                    val tag = result.request.tag as? Long
                    val actual = result.sensorState()
                    if (target == null || tag != target.commandId ||
                        !scheduler.matches(target.expected, actual)) {
                        target?.let { synchronized(it.lock) { it.staleFrames++ } }
                        return@execute
                    }
                    val decodeStart = System.nanoTime()
                    val decoded = decode(image, result, target.expected, target.commandId,
                        nativeDecode = true)
                    val decodeEnd = System.nanoTime()
                    synchronized(target.lock) {
                        if (activeContinuousTarget !== target) return@synchronized
                        if (target.firstMatchedNs == 0L) target.firstMatchedNs = decodeStart
                        target.decodeCpuNs += decodeEnd - decodeStart
                        target.decodedFrames++
                        target.ring.add(decoded)
                        target.ring.sortBy { it.meta.frameNumber }
                        while (target.ring.size > CONTINUOUS_RING_CAPACITY) {
                            target.ring.removeAt(0)
                        }
                        if (target.ring.size >= target.wanted && target.readyNs == 0L) {
                            target.readyNs = decodeEnd
                        }
                        target.lock.notifyAll()
                    }
                } catch (t: Throwable) {
                    activeContinuousTarget?.let {
                        synchronized(it.lock) { it.error = t; it.lock.notifyAll() }
                    }
                } finally {
                    image.close()
                }
            }
        }, cameraHandler)
        // The first concrete target is installed by captureContinuous().
        continuousStarted = true
        Log.i(TAG, "EXP5.2 continuous listener armed session=$sess")
    }

    @Synchronized
    fun captureContinuous(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        check(continuousStarted) { "startContinuousCapture() must be called first" }
        val wanted = max(1, nBurst)
        require(wanted <= CONTINUOUS_RING_CAPACITY)
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val target = ContinuousTarget(expected, command.id, wanted, System.nanoTime())
        activeContinuousTarget = target
        val template = if (wanted == 1) CameraDevice.TEMPLATE_PREVIEW
            else CameraDevice.TEMPLATE_STILL_CAPTURE
        val request = manualRequest(expected, command.id, template)
        val callback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: -1L
                val tag = result.request.tag as? Long
                if (tag == target.commandId) {
                    synchronized(target.lock) {
                        if (target.firstResultNs == 0L) target.firstResultNs = System.nanoTime()
                    }
                    scheduler.markRequestFrame(command.id, result.frameNumber)
                    val actual = result.sensorState()
                    scheduler.observe(result.frameNumber, ts, actual)?.let {
                        lastApplyDelayFrames = it.applyDelayFrames
                        currentState = actual
                    }
                }
                continuousResults[ts] = result
                synchronized(continuousResultLock) { continuousResultLock.notifyAll() }
            }

            override fun onCaptureFailed(
                session: CameraCaptureSession,
                request: CaptureRequest,
                failure: CaptureFailure
            ) {
                synchronized(target.lock) {
                    target.error = IllegalStateException("continuous capture failed: ${failure.reason}")
                    target.lock.notifyAll()
                }
            }
        }
        (session ?: error("camera session is closed"))
            .setRepeatingRequest(request, callback, cameraHandler)
        val timeoutNs = TimeUnit.MILLISECONDS.toNanos(max(6_000L, wanted * 1_500L))
        synchronized(target.lock) {
            while (target.ring.size < wanted && target.error == null) {
                val left = timeoutNs - (System.nanoTime() - target.startedNs)
                if (left <= 0L) break
                target.lock.wait(minOf(100L,
                    TimeUnit.NANOSECONDS.toMillis(left).coerceAtLeast(1L)))
            }
        }
        target.error?.let { throw it }
        check(target.ring.size >= wanted) {
            "Continuous capture only obtained ${target.ring.size}/$wanted frames for " +
                "${exposureUs}us ISO$iso"
        }
        val accepted = synchronized(target.lock) { target.ring.takeLast(wanted) }
        fun elapsed(ns: Long): Double = if (ns == 0L) Double.NaN
            else (ns - target.startedNs) / 1e6
        lastContinuousProfile = ContinuousProfile(
            requestedFrames = wanted,
            firstResultMs = elapsed(target.firstResultNs),
            firstMatchedMs = elapsed(target.firstMatchedNs),
            readyMs = elapsed(target.readyNs),
            decodeCpuSumMs = target.decodeCpuNs / 1e6,
            staleFramesDropped = target.staleFrames,
            decodedFrames = target.decodedFrames)
        lastMeta = accepted.map { it.meta }
        captureSnapshot.set(CaptureSnapshot(lastMeta.map { it.copy(
            blackLevels = it.blackLevels.copyOf(),
            whiteBalance = it.whiteBalance.copyOf(),
            cameraToSrgb = it.cameraToSrgb.copyOf()) }, lastCaptureProfile))
        lastSettlingMeta = lastMeta
        accepted.lastOrNull()?.let {
            currentState = SensorState(it.meta.appliedExpUs, it.meta.appliedIso)
        }
        return accepted.map { it.frame }
    }

    /** Snapshot belonging to the calling capture worker, immune to concurrent streaming frames. */
    fun consumeCaptureSnapshot(): CaptureSnapshot? {
        val value = captureSnapshot.get()
        captureSnapshot.remove()
        return value
    }

    @Synchronized
    fun stopContinuousCapture() {
        if (!continuousStarted) return
        activeContinuousTarget = null
        runCatching { session?.stopRepeating() }
        reader?.setOnImageAvailableListener(null, null)
        reader?.let { drainReader(it) }
        continuousResults.clear()
        continuousStarted = false
    }

    fun armNativeDecodeValidation() {
        lastNativeDecodeMismatchCount = -1
        nativeValidationRemaining.set(1)
    }

    /** EXP5.4: change RAW decode parallelism only between completed captures. */
    @Synchronized
    fun configureDecodeThreads(threads: Int) {
        require(threads in 1..4)
        if (threads == decodeThreadCount) return
        val old = decodePool
        old.shutdown()
        check(old.awaitTermination(3, TimeUnit.SECONDS)) {
            "RAW decode workers did not become idle"
        }
        decodePool = Executors.newFixedThreadPool(threads)
        decodeThreadCount = threads
        Log.i(TAG, "EXP5.4 decodeThreads=$threads")
    }

    /** EXP5.1 C: do not queue 12 surplus requests. Submit one frame until metadata matches,
     * then request only the remaining burst frames. This is intentionally camera-only and
     * conservative; the production path remains unchanged until the experiment validates it. */
    @Synchronized
    fun captureAdaptive(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> {
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val request = manualRequest(expected, command.id)
        val wanted = max(1, nBurst)
        val acceptedFrames = ArrayList<RawFrame>(wanted)
        val acceptedMeta = ArrayList<RawMeta>(wanted)
        var submissions = 0
        val maxSubmissions = SETTING_GUARD_FRAMES + wanted
        while (acceptedFrames.size < wanted && submissions < maxSubmissions) {
            val remaining = wanted - acceptedFrames.size
            // Before the first matching result, advance one request at a time. Afterwards,
            // collect only the exact number of additional burst frames needed.
            val count = if (acceptedFrames.isEmpty()) 1 else remaining
            val frames = runCapture(request, count, 0, null, command.id)
            val metas = lastMeta
            submissions += count
            for (i in frames.indices) {
                val meta = metas.getOrNull(i) ?: continue
                if (scheduler.matches(expected,
                        SensorState(meta.appliedExpUs, meta.appliedIso))) {
                    acceptedFrames.add(frames[i])
                    acceptedMeta.add(meta.copy(
                        requestedExpUs = exposureUs.toLong(), requestedIso = iso))
                    if (acceptedFrames.size == wanted) break
                }
            }
        }
        lastSettlingMeta = acceptedMeta.toList()
        lastMeta = acceptedMeta.toList()
        check(acceptedFrames.size == wanted) {
            "Adaptive capture only obtained ${acceptedFrames.size}/$wanted matching frames; " +
                "requested=${exposureUs}us ISO$iso, submissions=$submissions"
        }
        return acceptedFrames
    }

    private fun captureInternal(
        exposureUs: Int, iso: Int, nBurst: Int, metadataFirst: Boolean
    ): List<RawFrame> {
        val expected = SensorState(exposureUs.toLong(), iso)
        val command = scheduler.enqueue(expected)
        val alreadyApplied = currentState?.let { scheduler.matches(expected, it) } == true
        val guard = if (alreadyApplied) 0 else SETTING_GUARD_FRAMES
        val request = manualRequest(expected, command.id)
        return runCapture(request, max(1, nBurst), guard, expected, command.id, metadataFirst)
    }

    fun captureAe(nBurst: Int): List<RawFrame> {
        val request = device!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
            addTarget(reader!!.surface)
        }.build()
        return runCapture(request, max(1, nBurst), 0, null, null)
    }

    private fun manualRequest(
        expected: SensorState, commandId: Long,
        template: Int = CameraDevice.TEMPLATE_STILL_CAPTURE
    ): CaptureRequest =
        device!!.createCaptureRequest(template).apply {
            // Keep AE/manual exposure under SoS control while allowing the HAL AWB estimator
            // to publish per-frame gains and its camera-RGB -> linear-sRGB transform.
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
            set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
            set(CaptureRequest.COLOR_CORRECTION_MODE, CameraMetadata.COLOR_CORRECTION_MODE_FAST)
            set(CaptureRequest.SENSOR_EXPOSURE_TIME, expected.exposureUs * 1_000L)
            set(CaptureRequest.SENSOR_SENSITIVITY, expected.iso)
            set(CaptureRequest.SENSOR_FRAME_DURATION,
                max(streamMinFrameDurationNs, expected.exposureUs * 1_000L + 1_000_000L))
            setTag(commandId)
            addTarget(reader!!.surface)
        }.build()

    private fun runCapture(
        request: CaptureRequest,
        wanted: Int,
        guard: Int,
        expected: SensorState?,
        commandId: Long?,
        metadataFirst: Boolean = false,
        nativeDecode: Boolean = false
    ): List<RawFrame> {
        val captureStartNs = System.nanoTime()
        val sess = session ?: error("camera session is closed")
        val imageReader = reader ?: error("RAW reader is closed")
        val count = guard + wanted
        require(count <= MAX_IMAGES) { "capture count $count exceeds ImageReader capacity" }

        val resultByTimestamp = ConcurrentHashMap<Long, TotalCaptureResult>()
        val resultLock = java.lang.Object()
        val futures = LinkedBlockingQueue<Future<Decoded?>>()
        val done = CountDownLatch(count)
        val decodeSlots = AtomicInteger(0)
        val firstResultNs = AtomicLong(0L)
        val firstImageNs = AtomicLong(0L)
        val firstMatchedNs = AtomicLong(0L)
        val firstDecodeStartNs = AtomicLong(0L)
        val lastDecodeEndNs = AtomicLong(0L)
        val decodeCpuNs = LongAdder()
        val allResultMeta = java.util.Collections.synchronizedList(ArrayList<RawMeta>())

        imageReader.setOnImageAvailableListener({ source ->
            val image = runCatching { source.acquireNextImage() }.getOrNull() ?: return@setOnImageAvailableListener
            firstImageNs.compareAndSet(0L, System.nanoTime())
            futures.offer(decodePool.submit<Decoded?> {
                try {
                    val result = awaitResult(image.timestamp, resultByTimestamp, resultLock, 2_000)
                        ?: return@submit null
                    if (metadataFirst && expected != null) {
                        val actual = result.sensorState()
                        if (!scheduler.matches(expected, actual)) return@submit null
                        // Settling frames and surplus matching frames do not enter the costly
                        // 12 MP RAW decode pool. Reserve exactly the frames the caller needs.
                        if (decodeSlots.getAndIncrement() >= wanted) return@submit null
                    }
                    firstMatchedNs.compareAndSet(0L, System.nanoTime())
                    val decodeStart = System.nanoTime()
                    firstDecodeStartNs.compareAndSet(0L, decodeStart)
                    decode(image, result, expected, commandId, nativeDecode).also {
                        val end = System.nanoTime()
                        decodeCpuNs.add(end - decodeStart)
                        while (true) {
                            val old = lastDecodeEndNs.get()
                            if (end <= old || lastDecodeEndNs.compareAndSet(old, end)) break
                        }
                    }
                } finally {
                    image.close()
                }
            })
        }, cameraHandler)

        val callback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                firstResultNs.compareAndSet(0L, System.nanoTime())
                val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: -1L
                if (commandId != null) {
                    scheduler.markRequestFrame(commandId, result.frameNumber)
                    val actual = result.sensorState()
                    scheduler.observe(result.frameNumber, ts, actual)?.let {
                        lastApplyDelayFrames = it.applyDelayFrames
                        currentState = actual
                    }
                }
                resultByTimestamp[ts] = result
                synchronized(resultLock) { resultLock.notifyAll() }
                done.countDown()
            }

            override fun onCaptureFailed(
                session: CameraCaptureSession,
                request: CaptureRequest,
                failure: CaptureFailure
            ) { done.countDown() }
        }

        sess.captureBurst(List(count) { request }, callback, cameraHandler)
        val timeoutMs = max(6_000L, count * 400L)
        val decoded = ArrayList<Decoded>()
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs)
        var ready = false
        while (System.nanoTime() < deadline) {
            val future = futures.poll(100, TimeUnit.MILLISECONDS) ?: continue
            runCatching { future.get(2, TimeUnit.SECONDS) }.getOrNull()?.let { decoded.add(it) }
            val eligibleCount = if (expected == null) decoded.size else decoded.count {
                scheduler.matches(expected, SensorState(it.meta.appliedExpUs, it.meta.appliedIso))
            }
            ready = eligibleCount >= wanted
            if (metadataFirst && ready) break
            if (!metadataFirst && decoded.size >= count) break
        }
        val cleanupStartNs = System.nanoTime()
        if (metadataFirst && ready && done.count > 0) {
            // Remaining requests are settling frames that are no longer needed. Abort only
            // this finite burst; the next explicit capture starts from the observed state.
            runCatching { sess.abortCaptures() }
            done.await(400, TimeUnit.MILLISECONDS)
        } else {
            done.await(400, TimeUnit.MILLISECONDS)
        }
        imageReader.setOnImageAvailableListener(null, null)
        drainReader(imageReader)
        while (true) {
            val surplus = futures.poll() ?: break
            surplus.cancel(true)
        }
        val cleanupEndNs = System.nanoTime()

        decoded.sortBy { it.meta.frameNumber }
        allResultMeta.addAll(decoded.map { it.meta })
        lastSettlingMeta = allResultMeta.toList()
        val eligible = if (expected == null) decoded else decoded.filter {
            scheduler.matches(expected, SensorState(it.meta.appliedExpUs, it.meta.appliedIso))
        }
        val accepted = eligible.takeLast(wanted)
        fun sinceStart(value: Long): Double =
            if (value == 0L) Double.NaN else (value - captureStartNs) / 1e6
        val decodeStartNs = firstDecodeStartNs.get()
        val decodeEndNs = lastDecodeEndNs.get()
        lastCaptureProfile = CaptureProfile(
            strategy = if (nativeDecode) "native_neon" else
                if (metadataFirst) "metadata_first_abort" else "kotlin",
            requestedFrames = wanted,
            firstResultMs = sinceStart(firstResultNs.get()),
            firstImageMs = sinceStart(firstImageNs.get()),
            firstMatchedMs = sinceStart(firstMatchedNs.get()),
            decodeWallMs = if (decodeStartNs == 0L || decodeEndNs == 0L) Double.NaN
                else (decodeEndNs - decodeStartNs) / 1e6,
            decodeCpuSumMs = decodeCpuNs.sum() / 1e6,
            cleanupMs = (cleanupEndNs - cleanupStartNs) / 1e6,
            totalMs = (cleanupEndNs - captureStartNs) / 1e6)
        check(accepted.size == wanted) {
            "Only ${accepted.size}/$wanted metadata-confirmed RAW frames; " +
                "requested=${expected?.exposureUs}us ISO${expected?.iso}, guard=$guard"
        }
        lastMeta = accepted.map { it.meta }
        accepted.lastOrNull()?.let {
            currentState = SensorState(it.meta.appliedExpUs, it.meta.appliedIso)
        }
        return accepted.map { it.frame }
    }

    private fun awaitResult(
        timestamp: Long,
        results: ConcurrentHashMap<Long, TotalCaptureResult>,
        lock: java.lang.Object,
        timeoutMs: Long
    ): TotalCaptureResult? {
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs)
        synchronized(lock) {
            while (results[timestamp] == null) {
                val left = deadline - System.nanoTime()
                if (left <= 0) return null
                lock.wait(minOf(50L, TimeUnit.NANOSECONDS.toMillis(left).coerceAtLeast(1L)))
            }
        }
        return results.remove(timestamp)
    }

    private fun TotalCaptureResult.sensorState(): SensorState = SensorState(
        exposureUs = (get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: -1L) / 1_000L,
        iso = get(CaptureResult.SENSOR_SENSITIVITY) ?: -1
    )

    private fun decode(
        image: Image,
        result: TotalCaptureResult,
        expected: SensorState?,
        commandId: Long?,
        nativeDecode: Boolean = false
    ): Decoded {
        val plane = image.planes[0]
        val bytes = plane.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val dynamic = result.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
        val black = if (dynamic != null && dynamic.size >= 4) {
            IntArray(4) { dynamic[it].toInt() }
        } else staticBlack.copyOf()

        val outWidth = (width / 2) and 1.inv()
        val outHeight = (height / 2) and 1.inv()
        val out = IntArray(outWidth * outHeight)
        val nativeOk = nativeDecode && if (streamFormat == ImageFormat.RAW10) {
            NativeTensorPreprocessor.decodeSubsampledRaw10(
                bytes, bytes.position(), width, height, rowStride, black, out)
        } else {
            NativeTensorPreprocessor.decodeSubsampledBayer(
                bytes, bytes.position(), width, height, rowStride, pixelStride, black, out)
        }
        if (nativeOk && nativeValidationRemaining.compareAndSet(1, 0)) {
            val reference = IntArray(out.size)
            if (streamFormat == ImageFormat.RAW10) {
                decodeRaw10Reference(bytes, rowStride, black, outWidth, outHeight, reference)
            } else {
                decodeReference(bytes, rowStride, pixelStride, black, outWidth, outHeight, reference)
            }
            var mismatches = 0
            for (i in out.indices) if (out[i] != reference[i]) mismatches++
            lastNativeDecodeMismatchCount = mismatches
        }
        if (!nativeOk) {
            if (streamFormat == ImageFormat.RAW10) {
                decodeRaw10Reference(bytes, rowStride, black, outWidth, outHeight, out)
            } else {
                decodeReference(bytes, rowStride, pixelStride, black, outWidth, outHeight, out)
            }
        }

        val actual = result.sensorState()
        val calibration = colorCalibration(result)
        val applied = commandId?.let { id ->
            scheduler.historySnapshot().lastOrNull { it.id == id }?.applyDelayFrames
        }
        val meta = RawMeta(
            requestedIso = expected?.iso ?: -1,
            appliedIso = actual.iso,
            requestedExpUs = expected?.exposureUs ?: -1,
            appliedExpUs = actual.exposureUs,
            frameNumber = result.frameNumber,
            timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: image.timestamp,
            black = black.average().toInt(),
            blackLevels = black,
            white = maxDn.toInt(),
            frameDurationNs = result.get(CaptureResult.SENSOR_FRAME_DURATION) ?: -1L,
            applyDelayFrames = applied,
            rowStrideBytes = rowStride,
            pixelStrideBytes = pixelStride,
            whiteBalance = calibration.first.copyOf(),
            cameraToSrgb = calibration.second.copyOf()
        )
        return Decoded(RawFrame(out, outWidth, outHeight, cfaPattern, maxDn,
            sensorOrientation, calibration.first.copyOf(), calibration.second.copyOf()), meta)
    }

    private fun decodeReference(
        bytes: java.nio.ByteBuffer, rowStride: Int, pixelStride: Int,
        black: IntArray, outWidth: Int, outHeight: Int, out: IntArray
    ) {
        for (oy in 0 until outHeight) {
            val iy = (oy and 1.inv()) * 2 + (oy and 1)
            val rowOut = oy * outWidth
            for (ox in 0 until outWidth) {
                val ix = (ox and 1.inv()) * 2 + (ox and 1)
                val offset = iy * rowStride + ix * pixelStride
                val raw = bytes.getShort(offset).toInt() and 0xFFFF
                val channelBlack = black[(iy and 1) * 2 + (ix and 1)]
                out[rowOut + ox] = (raw - channelBlack).coerceAtLeast(0)
            }
        }
    }

    private fun decodeRaw10Reference(
        bytes: java.nio.ByteBuffer, rowStride: Int, black: IntArray,
        outWidth: Int, outHeight: Int, out: IntArray
    ) {
        for (oy in 0 until outHeight) {
            val iy = (oy and 1.inv()) * 2 + (oy and 1)
            val rowOut = oy * outWidth
            val blackRow = (iy and 1) * 2
            for (group in 0 until outWidth / 2) {
                val offset = iy * rowStride + group * 5
                val tails = bytes.get(offset + 4).toInt() and 0xff
                val p0 = ((bytes.get(offset).toInt() and 0xff) shl 2) or (tails and 3)
                val p1 = ((bytes.get(offset + 1).toInt() and 0xff) shl 2) or
                    ((tails ushr 2) and 3)
                out[rowOut + group * 2] = (p0 - black[blackRow]).coerceAtLeast(0)
                out[rowOut + group * 2 + 1] = (p1 - black[blackRow + 1]).coerceAtLeast(0)
            }
        }
    }

    /** Camera2 AWB operates independently of manual AE. Apply its gains in linear camera RGB,
     *  followed by the HAL's row-major camera-RGB -> linear-sRGB transform. */
    private fun colorCalibration(result: TotalCaptureResult): Pair<FloatArray, DoubleArray> {
        val vector = result.get(CaptureResult.COLOR_CORRECTION_GAINS)
        val measuredWb = vector?.let {
            floatArrayOf(it.red, (it.greenEven + it.greenOdd) * 0.5f, it.blue)
        }?.takeIf { values -> values.all { it.isFinite() && it > 0f } }
        val neutralWb = result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)?.let { neutral ->
            if (neutral.size >= 3) floatArrayOf(
                1f / neutral[0].toFloat(), 1f / neutral[1].toFloat(), 1f / neutral[2].toFloat()
            ) else null
        }?.takeIf { values -> values.all { it.isFinite() && it > 0f } }
        val wb = measuredWb ?: neutralWb ?: lastWhiteBalance
        if (measuredWb != null || neutralWb != null) lastWhiteBalance = wb.copyOf()

        val transform = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)
        val measuredCcm = transform?.let { value ->
            DoubleArray(9) { i -> value.getElement(i % 3, i / 3).toDouble() }
        }?.takeIf { values -> values.all { it.isFinite() } && values.any { kotlin.math.abs(it) > 1e-9 } }
        val ccm = measuredCcm ?: lastCameraToSrgb
        if (measuredCcm != null) lastCameraToSrgb = ccm.copyOf()
        return wb to ccm
    }

    private fun drainReader(imageReader: ImageReader) {
        while (true) {
            val image = runCatching { imageReader.acquireNextImage() }.getOrNull() ?: break
            image.close()
        }
    }

    fun close() {
        runCatching { stopContinuousCapture() }
        runCatching { stopFastCapture() }
        runCatching { reader?.setOnImageAvailableListener(null, null) }
        runCatching { session?.close() }
        runCatching { device?.close() }
        runCatching { reader?.close() }
        decodePool.shutdown()
        runCatching { decodePool.awaitTermination(2, TimeUnit.SECONDS) }
        cameraThread.quitSafely()
        imageThread.quitSafely()
        session = null; device = null; reader = null
    }
}
