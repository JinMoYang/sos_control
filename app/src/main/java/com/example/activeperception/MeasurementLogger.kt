package com.example.activeperception

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import com.example.activeperception.acquire.Csv
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

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
 * [row] and [jsonl] flush on every write so a run killed mid-pass keeps its data.
 * Writers are hit from both the capture thread and the sensor thread, hence the
 * concurrent map.
 */
class MeasurementLogger(context: Context, runName: String) {

    /** Public Documents/sos when MANAGE_EXTERNAL_STORAGE is granted, so runs show up in
     *  "My Files" without adb; otherwise the app-private external dir, which records just as
     *  well but is invisible to the file manager. */
    val dir: File = run {
        val canWritePublic = Build.VERSION.SDK_INT < Build.VERSION_CODES.R ||
            Environment.isExternalStorageManager()
        val target = if (canWritePublic) {
            @Suppress("DEPRECATION")
            File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOCUMENTS), "sos/$runName")
        } else {
            File(context.getExternalFilesDir(null), "sos/$runName")
        }
        target.apply { mkdirs() }
    }
    val imgDir: File = File(dir, "img").apply { mkdirs() }
    val rawDir: File = File(dir, "raw").apply { mkdirs() }
    private val writers = java.util.concurrent.ConcurrentHashMap<String, BufferedWriter>()

    fun manifest(json: JSONObject) {
        File(dir, "manifest.json").writeText(json.toString(2))
    }

    /** Open (or reuse) a CSV with the given header. Names are shared with [jsonl], so a
     *  `.csv` and a `.jsonl` in the same run must not use the same name. */
    fun csv(name: String, header: List<String>) {
        if (writers.containsKey(name)) return
        val w = File(dir, "$name.csv").bufferedWriter()
        w.write(Csv.row(header)); w.newLine(); w.flush(); writers[name] = w
    }

    /** No-op if [csv] was never called for this name. */
    fun row(name: String, values: List<Any?>) {
        writers[name]?.apply { write(Csv.row(values)); newLine(); flush() }
    }

    fun jsonl(name: String, obj: JSONObject) {
        val w = writers.getOrPut(name) { File(dir, "$name.jsonl").bufferedWriter() }
        w.write(obj.toString()); w.newLine(); w.flush()
    }

    /** Returns the path relative to the run dir, which is what goes in frames.csv. */
    fun saveJpeg(name: String, bmp: Bitmap, quality: Int = 95): String {
        val f = File(imgDir, "$name.jpg")
        FileOutputStream(f).use { bmp.compress(Bitmap.CompressFormat.JPEG, quality, it) }
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

    fun flush() { writers.values.forEach { it.flush() } }

    fun close() {
        writers.values.forEach { runCatching { it.flush(); it.close() } }
        writers.clear()
    }
}
