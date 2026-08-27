package com.example.activeperception

import android.graphics.Bitmap
import com.example.activeperception.acquire.CandidateSource
import com.example.activeperception.acquire.Formation
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.RawCapturer
import com.example.activeperception.acquire.RawFrame
import com.example.activeperception.acquire.DirectTensorBatch
import com.example.activeperception.acquire.TensorLetterbox
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.floor

enum class ColorPipeline(val tag: String) {
    RAYNEO_AWB_CCM("A_awb_ccm"),
    ORIGINAL_GAIN_SRGB("B_original_gain_srgb")
}

enum class BurstWindow { CENTERED, FIRST_N }

data class FormationProfile(
    val pipeline: String,
    val k: Int,
    val distinctRows: Int,
    val rowWallMs: Double,
    val sumCpuMs: Double,
    val demosaicCpuMs: Double,
    val packWallMs: Double,
    val packCpuMs: Double,
    val bitmapMs: Double,
    val totalMs: Double
)

data class Exp21PreparedRgb(
    val cells: IntArray,
    val rgbByRow: Map<Int, Array<IntArray>>,
    val width: Int,
    val height: Int,
    val orientation: Int,
    val maxDn: Double,
    val prepareMs: Double
)

data class Exp21PathProfile(
    val path: String,
    val k: Int,
    val transformWallMs: Double,
    val transformCpuMs: Double,
    val bitmapMs: Double,
    val totalMs: Double
)

data class Exp21BitmapResult(val images: List<Bitmap>, val profile: Exp21PathProfile)
data class Exp21TensorResult(val batch: DirectTensorBatch, val profile: Exp21PathProfile)

/**
 * Drop-in for [com.example.activeperception.acquire.RawCandidateSource], bit-equivalent but
 * much faster. Two changes: an sRGB lookup table replacing the per-pixel `pow(x, 1/2.4)`,
 * and a worker pool splitting the independent per-row and per-cell work.
 *
 * Four workers keep camera delivery independent from formation. The pure math in
 * `acquire/Formation.kt` is left untouched.
 */
class ParallelRawCandidateSource(
    private val grid: Grid,
    private val capturer: RawCapturer,
    nThreads: Int = 4,
    private val colorPipeline: ColorPipeline = ColorPipeline.RAYNEO_AWB_CCM,
    private val burstWindow: BurstWindow = BurstWindow.FIRST_N
) : CandidateSource<Bitmap> {

    private val pool = Executors.newFixedThreadPool(nThreads)
    private val exp21TensorBuffers = ConcurrentHashMap<Int, ByteBuffer>()
    @Volatile private var srgbLut: IntArray = IntArray(0)
    @Volatile private var lutMaxDn: Double = -1.0
    @Volatile var lastProfile: FormationProfile? = null
        private set

    private data class RowResult(
        val shutterIdx: Int,
        val rgb: Array<IntArray>,
        val sumMs: Double,
        val demosaicMs: Double
    )

    private data class PackResult(val argb: Formation.OrientedArgb, val cpuMs: Double)

    /** sRGB LUT for the given white level, rebuilt only when it changes. Values fed to it are
     *  already clipped to [0, maxDn] by the caller, so the table needs no headroom. */
    private fun lutFor(maxDn: Double): IntArray {
        if (lutMaxDn != maxDn) {
            val n = maxDn.toInt() + 1
            srgbLut = IntArray(n) { Formation.srgbU8(it.toDouble(), maxDn) }
            lutMaxDn = maxDn
        }
        return srgbLut
    }

    override fun render(cells: IntArray): List<Bitmap> {
        val rows = cells.map { grid.indices(it).second }.toSortedSet()
        val frames: List<RawFrame> = if (rows.size == 1) {
            capturer.capture(grid.exposuresUs[rows.first()], grid.baseGain, 1)
        } else {
            capturer.capture(grid.fastestExposureUs, grid.baseGain, grid.maxBurst)
        }
        require(frames.isNotEmpty()) { "RawCapturer returned no frames" }
        return formAllCells(frames, cells)
    }

    /** Form bitmaps from already-captured frames, so callers that want to time formation
     *  alone (Bench, Verify) can keep capture out of the measurement. Always returns exactly
     *  `cells.size` bitmaps, in order — callers index detections against `cells`. */
    fun formAllCells(frames: List<RawFrame>, cells: IntArray): List<Bitmap> {
        val totalStart = System.nanoTime()
        val avail = frames.size
        val w = frames[0].width; val h = frames[0].height
        val maxDn = frames[0].maxDn; val cfa = frames[0].cfaPattern
        // One calibration for the whole bracket prevents AWB drift from changing candidate
        // ranking. The temporal center corresponds best to all centered shutter windows.
        val calibration = frames[avail / 2]
        val lut = lutFor(maxDn)
        val lutMax = lut.size - 1

        // Per shutter row: burst-sum + demosaic. This is the expensive stage and it scales
        // with the number of DISTINCT rows in `cells`, not with cells.size.
        val rows = cells.map { grid.indices(it).second }.toSortedSet()
        val rowStart = System.nanoTime()
        val rowFutures = rows.map { sj ->
            pool.submit<RowResult> {
                val n = minOf(grid.burstN(sj), avail)
                val sumStart = System.nanoTime()
                val summed = IntArray(frames[0].bayer.size)
                // The deployed original-SoS path sums the first N planes. Keep the centered
                // RayNeo window only in the legacy A arm so the historical A/B is reproducible.
                val start = if (burstWindow == BurstWindow.FIRST_N) 0 else (avail - n) / 2
                for (k in start until start + n) {
                    val bk = frames[k].bayer
                    for (p in summed.indices) summed[p] += bk[p]
                }
                val demosaicStart = System.nanoTime()
                val rgb = Formation.demosaic2x2(summed, w, h, cfa)
                val end = System.nanoTime()
                RowResult(sj, rgb,
                    (demosaicStart - sumStart) / 1e6,
                    (end - demosaicStart) / 1e6)
            }
        }
        val rgbByRow = HashMap<Int, Array<IntArray>>()
        var sumCpuMs = 0.0
        var demosaicCpuMs = 0.0
        for (f in rowFutures) {
            val result = f.get()
            rgbByRow[result.shutterIdx] = result.rgb
            sumCpuMs += result.sumMs
            demosaicCpuMs += result.demosaicMs
        }
        val rowEnd = System.nanoTime()

        // Per cell: digital re-gain + sRGB encode + ARGB pack.
        val ow = w / 2; val oh = h / 2
        val orientation = calibration.sensorOrientation
        val packStart = System.nanoTime()
        val cellFutures = cells.map { c ->
            pool.submit<PackResult> {
                val start = System.nanoTime()
                val (gi, sj) = grid.indices(c)
                val rgb = rgbByRow[sj]!!
                val argb = when (colorPipeline) {
                    ColorPipeline.RAYNEO_AWB_CCM -> packOrientedArgbWithLut(
                        rgb[0], rgb[1], rgb[2], ow, oh, orientation,
                        grid.gainRatio(gi), calibration.whiteBalance,
                        calibration.cameraToSrgb, lut, lutMax)
                    ColorPipeline.ORIGINAL_GAIN_SRGB -> packOriginalOrientedArgbWithLut(
                        rgb[0], rgb[1], rgb[2], ow, oh, orientation,
                        grid.gainRatio(gi), lut, lutMax)
                }
                PackResult(argb, (System.nanoTime() - start) / 1e6)
            }
        }
        val packed = cellFutures.map { it.get() }
        val packEnd = System.nanoTime()
        val bitmapStart = System.nanoTime()
        val bitmaps = packed.map {
            val oriented = it.argb
            Bitmap.createBitmap(oriented.pixels, oriented.width, oriented.height, Bitmap.Config.ARGB_8888)
        }
        val end = System.nanoTime()
        lastProfile = FormationProfile(
            pipeline = colorPipeline.tag,
            k = cells.size,
            distinctRows = rows.size,
            rowWallMs = (rowEnd - rowStart) / 1e6,
            sumCpuMs = sumCpuMs,
            demosaicCpuMs = demosaicCpuMs,
            packWallMs = (packEnd - packStart) / 1e6,
            packCpuMs = packed.sumOf { it.cpuMs },
            bitmapMs = (end - bitmapStart) / 1e6,
            totalMs = (end - totalStart) / 1e6
        )
        return bitmaps
    }

    /** EXP2.1 setup shared by both measured arms. RAW summation and demosaic are deliberately
     *  done once so the benchmark starts at the named boundary: RGB -> YOLO tensor. */
    fun prepareExp21Rgb(frames: List<RawFrame>, cells: IntArray): Exp21PreparedRgb {
        require(frames.isNotEmpty())
        val started = System.nanoTime()
        val avail = frames.size
        val w = frames[0].width; val h = frames[0].height
        val cfa = frames[0].cfaPattern
        val rows = cells.map { grid.indices(it).second }.toSortedSet()
        val futures = rows.map { sj ->
            pool.submit<Pair<Int, Array<IntArray>>> {
                val n = minOf(grid.burstN(sj), avail)
                val summed = IntArray(frames[0].bayer.size)
                val start = if (burstWindow == BurstWindow.FIRST_N) 0 else (avail - n) / 2
                for (k in start until start + n) {
                    val plane = frames[k].bayer
                    for (p in summed.indices) summed[p] += plane[p]
                }
                sj to Formation.demosaic2x2(summed, w, h, cfa)
            }
        }
        val rgb = futures.associate { it.get() }
        return Exp21PreparedRgb(cells.clone(), rgb, w / 2, h / 2,
            frames[avail / 2].sensorOrientation, frames[0].maxDn,
            (System.nanoTime() - started) / 1e6)
    }

    /** Current production representation path, but starting from the shared RGB planes. */
    fun formExp21Bitmaps(prepared: Exp21PreparedRgb): Exp21BitmapResult {
        val started = System.nanoTime()
        val lut = lutFor(prepared.maxDn); val lutMax = lut.lastIndex
        val transformStart = System.nanoTime()
        val futures = prepared.cells.map { cell ->
            pool.submit<PackResult> {
                val cpuStart = System.nanoTime()
                val (gi, sj) = grid.indices(cell)
                val rgb = requireNotNull(prepared.rgbByRow[sj])
                val argb = packOriginalOrientedArgbWithLut(
                    rgb[0], rgb[1], rgb[2], prepared.width, prepared.height,
                    prepared.orientation, grid.gainRatio(gi), lut, lutMax)
                PackResult(argb, (System.nanoTime() - cpuStart) / 1e6)
            }
        }
        val packed = futures.map { it.get() }
        val transformEnd = System.nanoTime()
        val bitmapStart = System.nanoTime()
        val images = packed.map {
            Bitmap.createBitmap(it.argb.pixels, it.argb.width, it.argb.height, Bitmap.Config.ARGB_8888)
        }
        val end = System.nanoTime()
        return Exp21BitmapResult(images, Exp21PathProfile(
            "C_bitmap", prepared.cells.size,
            (transformEnd - transformStart) / 1e6, packed.sumOf { it.cpuMs },
            (end - bitmapStart) / 1e6, (end - started) / 1e6))
    }

    /** EXP2.1 direct path. It preserves gain, sRGB LUT, orientation, filtered letterbox,
     *  RGB channel order and [0,1] normalization while skipping ARGB and Bitmap objects. */
    fun formExp21Tensor(prepared: Exp21PreparedRgb, imgsz: Int = 640): Exp21TensorResult {
        val started = System.nanoTime()
        val count = prepared.cells.size
        val floatsPerFrame = imgsz * imgsz * 3
        val input = exp21TensorBuffers.computeIfAbsent(count) {
            ByteBuffer.allocateDirect(count * floatsPerFrame * 4).order(ByteOrder.nativeOrder())
        }
        val lut = lutFor(prepared.maxDn); val lutMax = lut.lastIndex
        val degrees = (prepared.orientation % 360 + 360) % 360
        require(degrees == 0 || degrees == 90 || degrees == 180 || degrees == 270)
        val orientedW = if (degrees == 90 || degrees == 270) prepared.height else prepared.width
        val orientedH = if (degrees == 90 || degrees == 270) prepared.width else prepared.height
        val scale = minOf(imgsz.toFloat() / orientedW, imgsz.toFloat() / orientedH)
        val nw = Math.round(orientedW * scale); val nh = Math.round(orientedH * scale)
        val padX = (imgsz - nw) / 2f; val padY = (imgsz - nh) / 2f
        val mapping = TensorLetterbox(scale.toDouble(), padX.toDouble(), padY.toDouble())
        val futures = prepared.cells.mapIndexed { lane, cell ->
            pool.submit<Double> {
                val cpuStart = System.nanoTime()
                val (gi, sj) = grid.indices(cell)
                val rgb = requireNotNull(prepared.rgbByRow[sj])
                val gain = grid.gainRatio(gi)
                val baseFloat = lane * floatsPerFrame
                val pad = 114f / 255f
                for (i in 0 until floatsPerFrame) input.putFloat((baseFloat + i) * 4, pad)

                fun sourceIndex(ox: Int, oy: Int): Int {
                    val sx: Int; val sy: Int
                    when (degrees) {
                        0 -> { sx = ox; sy = oy }
                        90 -> { sx = oy; sy = prepared.height - 1 - ox }
                        180 -> { sx = prepared.width - 1 - ox; sy = prepared.height - 1 - oy }
                        else -> { sx = prepared.width - 1 - oy; sy = ox }
                    }
                    return sy * prepared.width + sx
                }

                val xPad = padX.toInt(); val yPad = padY.toInt()
                for (dy in 0 until nh) {
                    val sy = ((dy + 0.5) * orientedH / nh - 0.5)
                        .coerceIn(0.0, (orientedH - 1).toDouble())
                    val y0 = floor(sy).toInt(); val y1 = minOf(y0 + 1, orientedH - 1)
                    val fy = sy - y0
                    for (dx in 0 until nw) {
                        val sx = ((dx + 0.5) * orientedW / nw - 0.5)
                            .coerceIn(0.0, (orientedW - 1).toDouble())
                        val x0 = floor(sx).toInt(); val x1 = minOf(x0 + 1, orientedW - 1)
                        val fx = sx - x0
                        val p00 = sourceIndex(x0, y0); val p10 = sourceIndex(x1, y0)
                        val p01 = sourceIndex(x0, y1); val p11 = sourceIndex(x1, y1)
                        val outPixel = ((dy + yPad) * imgsz + dx + xPad) * 3
                        for (channel in 0..2) {
                            val plane = rgb[channel]
                            fun u8(p: Int): Int = lut[(plane[p] * gain).toInt().coerceIn(0, lutMax)]
                            val top = u8(p00) * (1.0 - fx) + u8(p10) * fx
                            val bottom = u8(p01) * (1.0 - fx) + u8(p11) * fx
                            val value = ((top * (1.0 - fy) + bottom * fy) / 255.0).toFloat()
                            input.putFloat((baseFloat + outPixel + channel) * 4, value)
                        }
                    }
                }
                (System.nanoTime() - cpuStart) / 1e6
            }
        }
        val cpu = futures.map { it.get() }
        input.rewind()
        val end = System.nanoTime()
        return Exp21TensorResult(DirectTensorBatch(input, List(count) { mapping }),
            Exp21PathProfile("D_direct_tensor", count, (end - started) / 1e6,
                cpu.sum(), 0.0, (end - started) / 1e6))
    }

    /** Native C++/ARM NEON variant of the same direct path. Four workers are kept busy even
     *  for K=1 by splitting the output into horizontal stripes. */
    fun formExp21NativeTensor(prepared: Exp21PreparedRgb, imgsz: Int = 640): Exp21TensorResult {
        val started = System.nanoTime()
        val count = prepared.cells.size
        val floatsPerFrame = imgsz * imgsz * 3
        val input = exp21TensorBuffers.computeIfAbsent(count) {
            ByteBuffer.allocateDirect(count * floatsPerFrame * 4).order(ByteOrder.nativeOrder())
        }
        val lut = lutFor(prepared.maxDn)
        val degrees = (prepared.orientation % 360 + 360) % 360
        val orientedW = if (degrees == 90 || degrees == 270) prepared.height else prepared.width
        val orientedH = if (degrees == 90 || degrees == 270) prepared.width else prepared.height
        val scale = minOf(imgsz.toFloat() / orientedW, imgsz.toFloat() / orientedH)
        val nw = Math.round(orientedW * scale); val nh = Math.round(orientedH * scale)
        val padX = (imgsz - nw) / 2f; val padY = (imgsz - nh) / 2f
        NativeTensorPreprocessor.clearTensor(
            input, 0, count * floatsPerFrame, 114f / 255f)

        val stripes = when (count) { 1 -> 4; 3 -> 2; else -> 1 }
        val futures = ArrayList<java.util.concurrent.Future<Double>>(count * stripes)
        for ((lane, cell) in prepared.cells.withIndex()) {
            val (gi, sj) = grid.indices(cell)
            val rgb = requireNotNull(prepared.rgbByRow[sj])
            for (stripe in 0 until stripes) {
                val y0 = stripe * nh / stripes
                val y1 = (stripe + 1) * nh / stripes
                futures += pool.submit<Double> {
                    val cpuStart = System.nanoTime()
                    check(NativeTensorPreprocessor.fillOriginalSrgbStripe(
                        rgb[0], rgb[1], rgb[2], prepared.width, prepared.height,
                        prepared.orientation, grid.gainRatio(gi), lut, input,
                        lane * floatsPerFrame, imgsz, y0, y1))
                    (System.nanoTime() - cpuStart) / 1e6
                }
            }
        }
        val cpu = futures.map { it.get() }
        input.rewind()
        val end = System.nanoTime()
        return Exp21TensorResult(DirectTensorBatch(input,
            List(count) { TensorLetterbox(scale.toDouble(), padX.toDouble(), padY.toDouble()) }),
            Exp21PathProfile("E_native_neon", count, (end - started) / 1e6,
                cpu.sum(), 0.0, (end - started) / 1e6))
    }

    /**
     * EXP6 integrated path: RAW burst sum, 2x2 demosaic, gain, sRGB and letterbox are fused
     * in native code. It avoids per-row summed Bayer and three full RGB IntArrays while
     * retaining the exact FIRST_N and integer-green-average semantics of the reference.
     */
    fun formFusedNativeTensor(
        frames: List<RawFrame>, cells: IntArray, imgsz: Int = 640
    ): Exp21TensorResult {
        require(frames.isNotEmpty())
        val started = System.nanoTime()
        val count = cells.size
        val floatsPerFrame = imgsz * imgsz * 3
        val input = exp21TensorBuffers.computeIfAbsent(count) {
            ByteBuffer.allocateDirect(count * floatsPerFrame * 4).order(ByteOrder.nativeOrder())
        }
        val lut = lutFor(frames[0].maxDn)
        NativeTensorPreprocessor.clearTensor(
            input, 0, count * floatsPerFrame, 114f / 255f)
        val frameArrays = frames.map { it.bayer }.toTypedArray()
        val indexedByRow = cells.indices.groupBy { grid.indices(cells[it]).second }
        val futures = ArrayList<java.util.concurrent.Future<Double>>()
        for ((sj, lanes) in indexedByRow) {
            val gains = DoubleArray(lanes.size) { i ->
                grid.gainRatio(grid.indices(cells[lanes[i]]).first)
            }
            val bases = IntArray(lanes.size) { i -> lanes[i] * floatsPerFrame }
            val degrees = (frames[0].sensorOrientation % 360 + 360) % 360
            val rgbW = frames[0].width / 2; val rgbH = frames[0].height / 2
            val orientedH = if (degrees == 90 || degrees == 270) rgbW else rgbH
            val orientedW = if (degrees == 90 || degrees == 270) rgbH else rgbW
            val scale = minOf(imgsz.toFloat() / orientedW, imgsz.toFloat() / orientedH)
            val nh = Math.round(orientedH * scale)
            val stripes = if (indexedByRow.size == 1) 4 else 1
            repeat(stripes) { stripe ->
                val y0 = stripe * nh / stripes; val y1 = (stripe + 1) * nh / stripes
                futures += pool.submit<Double> {
                    val cpuStart = System.nanoTime()
                    check(NativeTensorPreprocessor.fillFusedBayerSrgbRowStripe(
                        frameArrays, minOf(grid.burstN(sj), frames.size),
                        frames[0].width, frames[0].height, frames[0].cfaPattern,
                        frames[0].sensorOrientation, gains, bases, lut, input,
                        imgsz, y0, y1))
                    (System.nanoTime() - cpuStart) / 1e6
                }
            }
        }
        val cpuMs = futures.sumOf { it.get() }
        input.rewind()
        val degrees = (frames[0].sensorOrientation % 360 + 360) % 360
        val rgbW = frames[0].width / 2; val rgbH = frames[0].height / 2
        val orientedW = if (degrees == 90 || degrees == 270) rgbH else rgbW
        val orientedH = if (degrees == 90 || degrees == 270) rgbW else rgbH
        val scale = minOf(imgsz.toFloat() / orientedW, imgsz.toFloat() / orientedH)
        val nw = Math.round(orientedW * scale); val nh = Math.round(orientedH * scale)
        val mapping = TensorLetterbox(scale.toDouble(),
            ((imgsz - nw) / 2f).toDouble(), ((imgsz - nh) / 2f).toDouble())
        val end = System.nanoTime()
        return Exp21TensorResult(DirectTensorBatch(input, List(count) { mapping }),
            Exp21PathProfile("EXP6_fused_bayer_native", count,
                (end - started) / 1e6, cpuMs, 0.0, (end - started) / 1e6))
    }

    /** Materializes only the selected 640 input after detection. All non-selected lanes stay
     * tensor-only, so display does not restore the old K-way ARGB/Bitmap bottleneck. */
    fun selectedTensorBitmap(result: Exp21TensorResult, lane: Int, imgsz: Int = 640): Bitmap {
        require(lane in result.batch.transforms.indices)
        val pixels = IntArray(imgsz * imgsz)
        val base = lane * imgsz * imgsz * 3
        for (p in pixels.indices) {
            val i = base + p * 3
            val r = (result.batch.input.getFloat(i * 4) * 255f).toInt().coerceIn(0, 255)
            val g = (result.batch.input.getFloat((i + 1) * 4) * 255f).toInt().coerceIn(0, 255)
            val b = (result.batch.input.getFloat((i + 2) * 4) * 255f).toInt().coerceIn(0, 255)
            pixels[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, imgsz, imgsz, Bitmap.Config.ARGB_8888)
    }

    /** Detector boxes are in the pre-letterbox image space; the final preview is the exact
     * model tensor, so map boxes into that 640-square coordinate system. */
    fun detectionsForTensorPreview(
        detections: List<com.example.activeperception.acquire.Detection>,
        result: Exp21TensorResult, lane: Int
    ): List<com.example.activeperception.acquire.Detection> {
        val t = result.batch.transforms[lane]
        return detections.map { d ->
            val p = d.xyxy
            com.example.activeperception.acquire.Detection(floatArrayOf(
                (p[0] * t.scale + t.padX).toFloat(),
                (p[1] * t.scale + t.padY).toFloat(),
                (p[2] * t.scale + t.padX).toFloat(),
                (p[3] * t.scale + t.padY).toFloat()
            ), d.confidence, d.classId)
        }
    }

    /** Original SoS minimal ISP: digital gain, clipping, sRGB OETF, orientation and packing.
     *  RayNeo-only timestamp pairing and per-CFA black correction happen before this method. */
    private fun packOriginalOrientedArgbWithLut(
        r: IntArray, g: IntArray, b: IntArray, width: Int, height: Int,
        sensorOrientation: Int, gainRatio: Double, lut: IntArray, lutMax: Int
    ): Formation.OrientedArgb {
        val out = IntArray(r.size)
        val degrees = (sensorOrientation % 360 + 360) % 360
        require(degrees == 0 || degrees == 90 || degrees == 180 || degrees == 270)
        for (p in r.indices) {
            val ri = (r[p] * gainRatio).toInt().coerceIn(0, lutMax)
            val gi = (g[p] * gainRatio).toInt().coerceIn(0, lutMax)
            val bi = (b[p] * gainRatio).toInt().coerceIn(0, lutMax)
            val x = p % width
            val y = p / width
            val dst = when (degrees) {
                0 -> p
                90 -> x * height + (height - 1 - y)
                180 -> out.lastIndex - p
                else -> (width - 1 - x) * height + y
            }
            out[dst] = (0xFF shl 24) or (lut[ri] shl 16) or (lut[gi] shl 8) or lut[bi]
        }
        return if (degrees == 90 || degrees == 270) {
            Formation.OrientedArgb(out, height, width)
        } else Formation.OrientedArgb(out, width, height)
    }

    /** Equivalent to `Formation.packCandidateArgb`, minus the per-pixel `Math.pow`. */
    private fun packOrientedArgbWithLut(
        r: IntArray, g: IntArray, b: IntArray, width: Int, height: Int,
        sensorOrientation: Int, gainRatio: Double,
        whiteBalance: FloatArray, cameraToSrgb: DoubleArray,
        lut: IntArray, lutMax: Int
    ): Formation.OrientedArgb {
        val out = IntArray(r.size)
        val degrees = (sensorOrientation % 360 + 360) % 360
        require(degrees == 0 || degrees == 90 || degrees == 180 || degrees == 270)
        for (p in r.indices) {
            val wr = r[p] * gainRatio * whiteBalance[0]
            val wg = g[p] * gainRatio * whiteBalance[1]
            val wb = b[p] * gainRatio * whiteBalance[2]
            val ri = (cameraToSrgb[0] * wr + cameraToSrgb[1] * wg + cameraToSrgb[2] * wb)
                .toInt().coerceIn(0, lutMax)
            val gi = (cameraToSrgb[3] * wr + cameraToSrgb[4] * wg + cameraToSrgb[5] * wb)
                .toInt().coerceIn(0, lutMax)
            val bi = (cameraToSrgb[6] * wr + cameraToSrgb[7] * wg + cameraToSrgb[8] * wb)
                .toInt().coerceIn(0, lutMax)
            val x = p % width
            val y = p / width
            val dst = when (degrees) {
                0 -> p
                90 -> x * height + (height - 1 - y)
                180 -> out.lastIndex - p
                else -> (width - 1 - x) * height + y // 270
            }
            out[dst] = (0xFF shl 24) or (lut[ri] shl 16) or (lut[gi] shl 8) or lut[bi]
        }
        return if (degrees == 90 || degrees == 270) {
            Formation.OrientedArgb(out, height, width)
        } else Formation.OrientedArgb(out, width, height)
    }

    fun shutdown() {
        pool.shutdown()
        pool.awaitTermination(2, TimeUnit.SECONDS)
        exp21TensorBuffers.clear()
    }
}
