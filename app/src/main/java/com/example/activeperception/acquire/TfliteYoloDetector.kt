package com.example.activeperception.acquire

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * `Detector<Bitmap>` via TensorFlow Lite with the Adreno GPU (OpenCL) delegate, falling
 * back to CPU/XNNPACK if the delegate can't be built.
 *
 * True batching needs a fixed batch dim, so one Interpreter is loaded per batch size and
 * [detectBatch] picks the smallest one that fits. For K ∈ {3, 9} — the gain-column and
 * full-grid steps — the GPU runs all candidates in one launch, several times faster on
 * Adreno than looping. The batched assets are exported from `tools/yolov8n_640_dyn.onnx`,
 * an Ultralytics export patched to use Reshape `0` so the batch dim propagates.
 *
 * Every slot's output is [4 + numClasses, 8400] per item, so the decode path is shared.
 */
class TfliteYoloDetector(
    context: Context,
    /** (asset, batchSize) in ascending batch order; inputs larger than the last entry chunk. */
    batchedAssets: List<Pair<String, Int>> = listOf(
        "yolov8n_640_fp16.tflite" to 1,
        "yolov8n_640_b3_fp16.tflite" to 3,
        "yolov8n_640_b9_fp16.tflite" to 9
    ),
    private val imgsz: Int = 640,
    private val confThresh: Float = 0.01f,
    private val iouThresh: Double = 0.45,
    private val maxDet: Int = 100,
    private val allowed: Set<Int>? = setOf(2, 3, 5, 7),
    private val numClasses: Int = 80,
    accelerator: Accelerator = Accelerator.GPU
) : Detector<Bitmap> {

    enum class Accelerator { GPU, NNAPI, CPU }
    companion object { private const val TAG = "TfliteYoloDetector" }

    /** One Interpreter pinned at a batch size, with its dedicated I/O buffers.
     *  The buffers are reused across calls, so [detectBatch] is NOT thread-safe. */
    private class BatchSlot(
        val batch: Int,
        val interp: Interpreter,
        val gpuDelegate: GpuDelegate?,
        val input: ByteBuffer,
        val output: Array<Array<FloatArray>>   // [batch][4+numClasses][8400]
    )

    private val slots: List<BatchSlot>
    private val preprocPool = Executors.newFixedThreadPool(4)

    /** Last detectBatch breakdown (preprocess / GPU run / decode ms). */
    @Volatile var lastPreprocessMs: Double = 0.0; private set
    @Volatile var lastRunMs: Double = 0.0; private set
    @Volatile var lastDecodeMs: Double = 0.0; private set
    @Volatile var lastSlotBatch: Int = 0; private set

    init {
        slots = batchedAssets.sortedBy { it.second }.mapNotNull { (asset, batch) ->
            runCatching { loadSlot(context, asset, batch, accelerator) }
                .onFailure { Log.w(TAG, "skip $asset (B=$batch): ${it.message}") }
                .getOrNull()
        }
        require(slots.isNotEmpty()) { "no TFLite interpreter could be loaded" }
        Log.d(TAG, "loaded batches=${slots.map { it.batch }}")
    }

    private fun loadSlot(context: Context, asset: String, batch: Int, accel: Accelerator): BatchSlot {
        val bytes = context.assets.open(asset).use { it.readBytes() }
        val buf = ByteBuffer.allocateDirect(bytes.size).apply {
            order(ByteOrder.nativeOrder()); put(bytes); rewind()
        }
        val opts = Interpreter.Options()
        val gpu: GpuDelegate? = when (accel) {
            Accelerator.GPU -> {
                val gpuOpts = GpuDelegateFactory.Options().apply {
                    setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
                    setQuantizedModelsAllowed(true)
                    setInferencePreference(
                        GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    setSerializationParams(context.cacheDir.absolutePath, "$asset.v1")
                }
                runCatching { GpuDelegate(gpuOpts) }
                    .onSuccess { opts.addDelegate(it); Log.d(TAG, "$asset (B=$batch): GPU OpenCL") }
                    .onFailure { Log.w(TAG, "$asset (B=$batch): GPU failed -> CPU/XNNPACK", it); opts.setNumThreads(4) }
                    .getOrNull()
            }
            Accelerator.NNAPI -> { @Suppress("DEPRECATION") opts.setUseNNAPI(true); null }
            Accelerator.CPU -> { opts.setNumThreads(4); null }
        }
        val interp = Interpreter(buf, opts)
        val input = ByteBuffer.allocateDirect(batch * imgsz * imgsz * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        val output = Array(batch) { Array(4 + numClasses) { FloatArray(8400) } }
        return BatchSlot(batch, interp, gpu, input, output)
    }

    override fun detectBatch(images: List<Bitmap>): List<List<Detection>> {
        if (images.isEmpty()) return emptyList()
        // Smallest slot that fits, else the largest and chunk. A size that falls between two
        // slots pays the larger slot's full cost (the unused lanes are zero-padded).
        val slot = slots.firstOrNull { it.batch >= images.size } ?: slots.last()
        val out = ArrayList<List<Detection>>(images.size)
        var i = 0
        while (i < images.size) {
            val end = minOf(i + slot.batch, images.size)
            out.addAll(detectChunk(images.subList(i, end), slot))
            i = end
        }
        return out
    }

    private class Preproc(val scale: Double, val padX: Double, val padY: Double)

    private fun detectChunk(chunk: List<Bitmap>, slot: BatchSlot): List<List<Detection>> {
        val tPre = System.nanoTime()
        val frameBytes = imgsz * imgsz * 3 * 4
        // Workers write to disjoint slices of the shared DirectByteBuffer via absolute-index
        // puts, so there is no shared position state to contend on.
        val futures = chunk.mapIndexed { i, bmp ->
            preprocPool.submit<Preproc> { preprocessAt(slot.input, bmp, i * frameBytes) }
        }
        val pps = futures.map { it.get() }
        // Zero the unused lanes so the GPU sees deterministic data.
        for (i in chunk.size until slot.batch) {
            val base = i * frameBytes
            for (k in 0 until imgsz * imgsz * 3) slot.input.putFloat(base + k * 4, 0f)
        }
        slot.input.rewind()

        val tRun = System.nanoTime()
        slot.interp.run(slot.input, slot.output)

        val tDec = System.nanoTime()
        val out = ArrayList<List<Detection>>(chunk.size)
        for (i in chunk.indices) {
            val pp = pps[i]
            val flat = FloatArray((4 + numClasses) * 8400)
            for (c in 0 until 4 + numClasses) {
                System.arraycopy(slot.output[i][c], 0, flat, c * 8400, 8400)
            }
            val dets = YoloDecode.decode(flat, numClasses, 8400, confThresh, allowed, iouThresh, maxDet)
            out.add(YoloDecode.unletterbox(dets, pp.scale, pp.padX, pp.padY))
        }
        val tEnd = System.nanoTime()

        lastPreprocessMs = (tRun - tPre) / 1e6
        lastRunMs = (tDec - tRun) / 1e6
        lastDecodeMs = (tEnd - tDec) / 1e6
        lastSlotBatch = slot.batch
        Log.d(TAG, "B=${slot.batch} chunk=${chunk.size}  pre=${"%.1f".format(lastPreprocessMs)}  run=${"%.1f".format(lastRunMs)}  dec=${"%.1f".format(lastDecodeMs)} ms")

        return out
    }

    /** Aspect-preserving letterbox to imgsz², written as RGB HWC float [0,1] into [dst]
     *  at absolute byte offset [baseOffset]. Returns the mapping needed to unletterbox. */
    private fun preprocessAt(dst: ByteBuffer, bmp: Bitmap, baseOffset: Int): Preproc {
        val scale = minOf(imgsz.toFloat() / bmp.width, imgsz.toFloat() / bmp.height)
        val nw = Math.round(bmp.width * scale); val nh = Math.round(bmp.height * scale)
        val padX = (imgsz - nw) / 2f; val padY = (imgsz - nh) / 2f
        val canvas = Bitmap.createBitmap(imgsz, imgsz, Bitmap.Config.ARGB_8888)
        Canvas(canvas).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(Bitmap.createScaledBitmap(bmp, nw, nh, true), padX, padY, null)
        }
        val px = IntArray(imgsz * imgsz)
        canvas.getPixels(px, 0, imgsz, 0, 0, imgsz, imgsz)
        var pos = baseOffset
        for (p in px) {
            dst.putFloat(pos, ((p ushr 16) and 0xFF) / 255f); pos += 4
            dst.putFloat(pos, ((p ushr 8) and 0xFF) / 255f); pos += 4
            dst.putFloat(pos, (p and 0xFF) / 255f); pos += 4
        }
        return Preproc(scale.toDouble(), padX.toDouble(), padY.toDouble())
    }

    fun close() {
        slots.forEach {
            runCatching { it.interp.close() }
            runCatching { it.gpuDelegate?.close() }
        }
        runCatching { preprocPool.shutdown(); preprocPool.awaitTermination(2, TimeUnit.SECONDS) }
    }
}
