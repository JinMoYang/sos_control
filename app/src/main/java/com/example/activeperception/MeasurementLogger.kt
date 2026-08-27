package com.example.activeperception

import android.content.Context
import android.graphics.Bitmap
import android.os.Process
import com.example.activeperception.acquire.Csv
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * One directory per run:
 *
 *   sos/run_<tag>_<ts>/
 *     manifest.json         method, method_params, grid, detector config, device caps
 *     summary.json          per-run detection + frame totals
 *     frames.csv            per-frame log
 *     dets.jsonl            per-frame detections, incl. the 0.01-0.25 tail
 *     candidates.csv        per-candidate Σconf (Proposed)
 *     candidate_dets.jsonl  per-candidate boxes (Proposed)
 *     imu.csv               raw IMU samples, one row per sensor sample
 *     bench.csv, lag.csv    Bench only
 *     img/                  formed bitmaps, .jpg
 *     raw/                  uint16 LE Bayer, .raw16, Verify only
 *
 * Camera, inference and sensor threads only enqueue small write operations. A dedicated
 * writer owns all BufferedWriters and flushes in batches, preventing storage I/O from
 * stalling RAW delivery. Queue overflow is counted in logger_stats.json.
 */
class MeasurementLogger(context: Context, runName: String) {

    /** Public Documents/sos when the user has granted All files access, so runs show up in
     *  a file manager without adb (the sos_control phone workflow); otherwise the
     *  app-private external dir, which records identically and stays adb-accessible.
     *  RayNeo never grants the special access, so its behavior is unchanged. */
    val dir: File = run {
        val canWritePublic = android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.R ||
            android.os.Environment.isExternalStorageManager()
        val target = if (canWritePublic) {
            @Suppress("DEPRECATION")
            File(android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_DOCUMENTS), "sos/$runName")
        } else {
            File(context.getExternalFilesDir(null), "sos/$runName")
        }
        target.apply { mkdirs() }
    }
    val imgDir: File = File(dir, "img").apply { mkdirs() }
    val rawDir: File = File(dir, "raw").apply { mkdirs() }
    private val writers = HashMap<String, BufferedWriter>()
    private val csvNames = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
    private val queue = ArrayBlockingQueue<Runnable>(4096)
    private val dropped = AtomicLong(0)
    private val artifactDropped = AtomicLong(0)
    @Volatile private var accepting = true
    @Volatile private var acceptingArtifacts = true
    private val writerThread = Thread({ writerLoop() }, "SoS-LogWriter").apply { start() }
    // Large bitmap/RAW writes have their own tiny bounded queue. They cannot delay CSV logs,
    // camera callbacks, or detection; overload drops an artifact instead of retaining many
    // 12 MP-derived buffers and triggering GC.
    private val artifactQueue = ArrayBlockingQueue<Runnable>(4)
    private val artifactThread = Thread({ artifactLoop() }, "SoS-ArtifactWriter").apply { start() }

    private fun enqueue(op: () -> Unit) {
        if (!accepting || !queue.offer(Runnable(op))) dropped.incrementAndGet()
    }

    private fun writerLoop() {
        var sinceFlush = 0
        while (accepting || queue.isNotEmpty()) {
            val op = queue.poll(200, TimeUnit.MILLISECONDS)
            if (op != null) {
                runCatching { op.run() }
                sinceFlush++
            }
            if (sinceFlush >= 64 || (op == null && sinceFlush > 0)) {
                writers.values.forEach { runCatching { it.flush() } }
                sinceFlush = 0
            }
        }
        writers.values.forEach { runCatching { it.flush(); it.close() } }
        writers.clear()
    }

    private fun artifactLoop() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
        while (acceptingArtifacts || artifactQueue.isNotEmpty()) {
            artifactQueue.poll(200, TimeUnit.MILLISECONDS)?.let { runCatching { it.run() } }
        }
    }

    private fun enqueueArtifact(op: () -> Unit): Boolean {
        val accepted = acceptingArtifacts && artifactQueue.offer(Runnable(op))
        if (!accepted) artifactDropped.incrementAndGet()
        return accepted
    }

    fun manifest(json: JSONObject) {
        File(dir, "manifest.json").writeText(json.toString(2))
    }

    /** Open (or reuse) a CSV with the given header. Names are shared with [jsonl], so a
     *  `.csv` and a `.jsonl` in the same run must not use the same name. */
    fun csv(name: String, header: List<String>) {
        if (!csvNames.add(name)) return
        enqueue {
            val w = File(dir, "$name.csv").bufferedWriter()
            w.write(Csv.row(header)); w.newLine(); writers[name] = w
        }
    }

    /** No-op if [csv] was never called for this name. */
    fun row(name: String, values: List<Any?>) {
        val encoded = Csv.row(values)
        enqueue { writers[name]?.apply { write(encoded); newLine() } }
    }

    fun jsonl(name: String, obj: JSONObject) {
        val encoded = obj.toString()
        enqueue {
            val w = writers.getOrPut(name) { File(dir, "$name.jsonl").bufferedWriter() }
            w.write(encoded); w.newLine()
        }
    }

    /** Returns the path relative to the run dir, which is what goes in frames.csv. */
    fun saveJpeg(name: String, bmp: Bitmap, quality: Int = 95): String {
        val f = File(imgDir, "$name.jpg")
        FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, quality, it) }
        return "img/${f.name}"
    }

    /** Non-blocking live path. Bitmap ownership is retained until the background write ends. */
    fun saveJpegAsync(name: String, bmp: Bitmap, quality: Int = 95): String {
        val f = File(imgDir, "$name.jpg")
        return if (enqueueArtifact {
                FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, quality, it) }
            }) "img/${f.name}" else ""
    }

    /** Lossless candidate/overlay output for paired visual comparisons. */
    fun savePng(name: String, bmp: Bitmap): String {
        val f = File(imgDir, "$name.png")
        FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return "img/${f.name}"
    }

    /** Linear Bayer plane as uint16 LE binary. Returns the path relative to the run dir. */
    fun saveRaw(name: String, bayer: IntArray): String {
        val f = File(rawDir, "$name.raw16")
        val bb = ByteBuffer.allocate(bayer.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in bayer) bb.putShort((v and 0xFFFF).toShort())
        FileOutputStream(f).use { it.write(bb.array()) }
        return "raw/${f.name}"
    }

    /** Non-blocking RAW persistence. The immutable RawFrame array is transferred by reference. */
    fun saveRawAsync(name: String, bayer: IntArray): String {
        val f = File(rawDir, "$name.raw16")
        return if (enqueueArtifact {
                val bb = ByteBuffer.allocate(bayer.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                for (v in bayer) bb.putShort((v and 0xFFFF).toShort())
                FileOutputStream(f).use { it.write(bb.array()) }
            }) "raw/${f.name}" else ""
    }

    fun flush() {
        if (!accepting) return
        val done = CountDownLatch(1)
        queue.put(Runnable {
            writers.values.forEach { runCatching { it.flush() } }
            done.countDown()
        })
        done.await(5, TimeUnit.SECONDS)
    }

    fun close() {
        flush()
        acceptingArtifacts = false
        artifactThread.join(5_000)
        accepting = false
        writerThread.join(5_000)
        File(dir, "logger_stats.json").writeText(JSONObject().apply {
            put("queue_capacity", 4096)
            put("dropped_operations", dropped.get())
            put("artifact_queue_capacity", 4)
            put("dropped_artifacts", artifactDropped.get())
            put("artifact_writer_closed_cleanly", !artifactThread.isAlive)
            put("closed_cleanly", !writerThread.isAlive)
        }.toString(2))
    }
}
