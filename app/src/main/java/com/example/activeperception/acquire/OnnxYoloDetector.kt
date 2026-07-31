package com.example.activeperception.acquire

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import java.nio.FloatBuffer

/**
 * Detector<Bitmap> via ONNX Runtime (yolov8n_640.onnx in assets). NNAPI EP for the
 * S25 NPU/GPU. Letterbox preprocess -> run -> YoloDecode (decode + NMS, tested).
 * BLOCKING — call off the main thread (the acquire loop runs on a worker).
 *
 * Gradle: implementation("com.microsoft.onnxruntime:onnxruntime-android:1.+")
 * For the Qualcomm NPU (QNN EP) use onnxruntime-qnn and opts.addQnn(...) instead.
 * detectBatch loops per image (true-batch is a Bench variable; export a batched
 * model + stack inputs to measure it).
 */
class OnnxYoloDetector(
    context: Context,
    modelAsset: String = "yolov8n_640_fp16.onnx",  // FP16 weights, FP32 I/O. Original FP32 is at "yolov8n_640.onnx".
    private val imgsz: Int = 640,
    private val confThresh: Float = 0.01f,         // canonical DETECTOR floor (regen_lowconf CONF=0.01);
                                                   // keeps the 0.01-0.25 tail for the offload signal.
                                                   // SELECTION filters at 0.25 in the controller.
    private val iouThresh: Double = 0.45,
    private val maxDet: Int = 100,                 // matches regen_lowconf max_det
    private val allowed: Set<Int>? = setOf(2, 3, 5, 7),   // COCO vehicle classes
    private val numClasses: Int = 80,
    accelerator: Accelerator = Accelerator.QNN_HTP   // try real NPU first; falls back to CPU on failure
) : Detector<Bitmap> {

    /** Inference backend. QNN_HTP targets the Snapdragon Hexagon NPU (S25). XNNPACK
     *  uses ARM NEON CPU. NNAPI is the deprecated path that falls back to
     *  nnapi-reference (CPU reference impl) on Android 15+. CPU is plain ORT CPU. */
    enum class Accelerator { QNN_HTP, XNNPACK, NNAPI, CPU }

    companion object { private const val TAG = "OnnxYoloDetector" }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        when (accelerator) {
            Accelerator.QNN_HTP -> {
                // Minimal QNN config: only backend_path, no perf/optim tuning. S25 has
                // Hexagon V79 (seen in vendor logs as libbitml_nsp_79na_skel.so), but ORT
                // 1.22 only bundles V68/69/73/75 HtpSkel. Forcing burst/optimization=3
                // triggered QNN_DEVICE_ERROR_INVALID_CONFIG; defaults let the runtime
                // pick whatever the device actually supports.
                val libDir = context.applicationInfo.nativeLibraryDir
                // Last-ditch: explicitly negotiate V75 arch. ORT 1.22 QNN supports
                // up to v75; S25's V79 might accept a V75-compatible device config.
                val qnnOpts = mapOf(
                    "backend_path" to "$libDir/libQnnHtp.so",
                    "htp_arch" to "75",
                    "soc_model" to "0"      // 0 = auto-detect
                )
                Log.d(TAG, "trying QNN HTP backend_path=${qnnOpts["backend_path"]} htp_arch=75")
                runCatching { opts.addQnn(qnnOpts) }
                    .onSuccess { Log.d(TAG, "QNN HTP EP added (model=$modelAsset)") }
                    .onFailure { Log.w(TAG, "QNN EP add failed; falling back to XNNPACK", it)
                        runCatching { opts.addXnnpack(emptyMap()) } }
            }
            Accelerator.XNNPACK -> runCatching { opts.addXnnpack(emptyMap()) }
                .onSuccess { Log.d(TAG, "XNNPACK EP added") }
            Accelerator.NNAPI -> runCatching { opts.addNnapi() }
                .onSuccess { Log.d(TAG, "NNAPI EP added (may fall back to nnapi-reference on Android 15+)") }
            Accelerator.CPU -> Log.d(TAG, "CPU EP only")
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.iterator().next()
    }

    override fun detectBatch(images: List<Bitmap>): List<List<Detection>> =
        images.map { detectOne(it) }

    private fun detectOne(bmp: Bitmap): List<Detection> {
        val pp = preprocess(bmp, imgsz)
        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(pp.chw), longArrayOf(1, 3, imgsz.toLong(), imgsz.toLong()))
        tensor.use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                val outT = res[0] as OnnxTensor
                val fb = outT.floatBuffer
                val arr = FloatArray(fb.remaining()); fb.get(arr)   // [(.4+numClasses)*numAnchors]
                val numAnchors = arr.size / (4 + numClasses)
                val dets = YoloDecode.decode(arr, numClasses, numAnchors, confThresh, allowed, iouThresh, maxDet)
                return YoloDecode.unletterbox(dets, pp.scale, pp.padX, pp.padY)
            }
        }
    }

    private class Preproc(val chw: FloatArray, val scale: Double, val padX: Double, val padY: Double)

    /** Aspect-preserving letterbox to size×size, gray pad, RGB CHW float [0,1]. */
    private fun preprocess(bmp: Bitmap, size: Int): Preproc {
        val scale = minOf(size.toFloat() / bmp.width, size.toFloat() / bmp.height)
        val nw = Math.round(bmp.width * scale); val nh = Math.round(bmp.height * scale)
        val padX = (size - nw) / 2f; val padY = (size - nh) / 2f
        val canvas = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(canvas).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(Bitmap.createScaledBitmap(bmp, nw, nh, true), padX, padY, null)
        }
        val px = IntArray(size * size); canvas.getPixels(px, 0, size, 0, 0, size, size)
        val area = size * size
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = px[i]
            chw[i] = ((p ushr 16) and 0xFF) / 255f          // R
            chw[area + i] = ((p ushr 8) and 0xFF) / 255f     // G
            chw[2 * area + i] = (p and 0xFF) / 255f           // B
        }
        return Preproc(chw, scale.toDouble(), padX.toDouble(), padY.toDouble())
    }

    fun close() {
        session.close()
    }
}
