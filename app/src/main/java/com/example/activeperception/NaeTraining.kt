package com.example.activeperception

import android.graphics.BitmapFactory
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.NaeFeatures
import com.example.activeperception.acquire.NaeSnap
import com.example.activeperception.acquire.NaeTrainer
import com.example.activeperception.acquire.NeuralAeNet
import com.example.activeperception.acquire.ShinMetric
import java.io.File

/**
 * Neural-AE training samples from run directories recorded by THIS app, mirroring
 * `sense/nae_train_app.py` build_samples: feature = 59x256 multi-scale histogram of the
 * step's chosen JPEG; target = clamp(ln(e_best_next / e_current), +-ln10), where best_next
 * is the argmax of the NEXT step's candidates.csv surface (ties broken by smallest
 * |dlog e| from the current cell, the sim's best_next_cell rule). Steps missing a JPEG or
 * a next-step surface are skipped. Proposed runs are the intended diet — their probe steps
 * carry full-grid surfaces; single-cell surfaces (PhysSweep/ShinNM rows) degenerate to a
 * trivial target and are better excluded by the caller.
 */
/**
 * Append-only store for Neural-AE training samples collected on the phone, so several
 * collection passes (different scenes, different light) accumulate into one pool before
 * training. Fixed-size records make append and count trivial: no index, no rewrite.
 *
 *   header  "naeds-v1" + int32 histLen        (12 bytes, little-endian)
 *   record  histLen float32 hist + float64 target   (histLen*4 + 8 bytes)
 *
 * One record is ~59 KB, so the pool is capped by record count rather than left to grow.
 */
object NaeDataset {
    private const val MAGIC = "naeds-v1"
    private val HIST_LEN = NeuralAeNet.N_HIST * NeuralAeNet.N_BINS
    val RECORD_BYTES = HIST_LEN * 4 + 8
    private const val HEADER_BYTES = 12
    /** ~140 MB. Beyond this the file is rewritten keeping the newest records. */
    const val MAX_RECORDS = 2400

    fun count(f: File): Int =
        if (!f.isFile || f.length() < HEADER_BYTES) 0
        else ((f.length() - HEADER_BYTES) / RECORD_BYTES).toInt()

    /** Appends [samples]; creates (or replaces a foreign/truncated) header as needed. */
    fun append(f: File, samples: List<NaeTrainer.Sample>) {
        if (samples.isEmpty()) return
        val fresh = count(f) == 0
        if (fresh) {
            f.parentFile?.mkdirs()
            java.io.DataOutputStream(java.io.BufferedOutputStream(f.outputStream())).use { o ->
                o.write(MAGIC.toByteArray(Charsets.US_ASCII))
                o.write(le32(HIST_LEN))
            }
        }
        java.io.BufferedOutputStream(java.io.FileOutputStream(f, true)).use { o ->
            val bb = java.nio.ByteBuffer.allocate(RECORD_BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            for (s in samples) {
                bb.clear()
                for (x in s.hist) bb.putFloat(x)
                bb.putDouble(s.targetLogU)
                o.write(bb.array())
            }
        }
        if (count(f) > MAX_RECORDS) trimToNewest(f, MAX_RECORDS)
    }

    fun load(f: File): List<NaeTrainer.Sample> {
        val n = count(f)
        if (n == 0) return emptyList()
        val out = ArrayList<NaeTrainer.Sample>(n)
        java.io.DataInputStream(java.io.BufferedInputStream(f.inputStream())).use { i ->
            i.skipNBytes(HEADER_BYTES.toLong())
            val buf = ByteArray(RECORD_BYTES)
            repeat(n) {
                i.readFully(buf)
                val bb = java.nio.ByteBuffer.wrap(buf).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val hist = FloatArray(HIST_LEN) { bb.float }
                out.add(NaeTrainer.Sample(hist, bb.double))
            }
        }
        return out
    }

    private fun trimToNewest(f: File, keep: Int) {
        val n = count(f)
        if (n <= keep) return
        val skip = (n - keep).toLong() * RECORD_BYTES
        val tmp = File(f.parentFile, f.name + ".trim")
        java.io.DataInputStream(java.io.BufferedInputStream(f.inputStream())).use { i ->
            java.io.BufferedOutputStream(tmp.outputStream()).use { o ->
                val head = ByteArray(HEADER_BYTES); i.readFully(head); o.write(head)
                i.skipNBytes(skip)
                i.copyTo(o)
            }
        }
        if (tmp.isFile && tmp.length() > 0) { f.delete(); tmp.renameTo(f) } else tmp.delete()
    }

    private fun le32(v: Int) = byteArrayOf(
        (v and 0xFF).toByte(), ((v ushr 8) and 0xFF).toByte(),
        ((v ushr 16) and 0xFF).toByte(), ((v ushr 24) and 0xFF).toByte())
}

object NaeTraining {

    private const val LN_M = 2.302585092994046   // ln(10), the Eq. 4 output bound

    class StepRow(val frame: Int) {
        val cells = ArrayList<Int>()
        val scores = ArrayList<Double>()
        var chosen = -1
    }

    /** candidates.csv rows grouped per frame. Writer layout (all modes):
     *  frame, cell, gain(effective), exposure_us, score, chosen, tie_break. */
    fun parseCandidates(file: File): List<StepRow> {
        if (!file.isFile) return emptyList()
        val byFrame = LinkedHashMap<Int, StepRow>()
        file.useLines { lines ->
            lines.drop(1).forEach { line ->
                val t = line.split(',')
                if (t.size < 6) return@forEach
                val fr = t[0].toIntOrNull() ?: return@forEach
                val cell = t[1].toIntOrNull() ?: return@forEach
                val score = t[4].toDoubleOrNull() ?: return@forEach
                val row = byFrame.getOrPut(fr) { StepRow(fr) }
                row.cells.add(cell); row.scores.add(score)
                if (t[5].trim() == "1") row.chosen = cell
            }
        }
        return byFrame.values.toList()
    }

    /** Argmax of the surface with the sim's tie-break (smallest |dlog e| from [curCell]). */
    fun bestNextCell(row: StepRow, curCell: Int, grid: Grid): Int {
        var top = Double.NEGATIVE_INFINITY
        for (s in row.scores) if (s > top) top = s
        val eCur = NaeSnap.cellE(curCell, grid.gains, grid.exposuresUs)
        var pick = -1; var bestD = Double.MAX_VALUE
        for (k in row.cells.indices) {
            if (row.scores[k] < top - 1e-9) continue
            val d = Math.abs(Math.log(
                NaeSnap.cellE(row.cells[k], grid.gains, grid.exposuresUs) / eCur))
            if (d < bestD) { bestD = d; pick = row.cells[k] }
        }
        return pick
    }

    /** All samples from one run directory (candidates.csv + img/frame_XXXX_*.jpg). */
    fun samplesFromRun(runDir: File, grid: Grid): List<NaeTrainer.Sample> {
        val steps = parseCandidates(File(runDir, "candidates.csv"))
        if (steps.size < 2) return emptyList()
        val jpegByFrame = HashMap<Int, File>()
        File(runDir, "img").listFiles()?.forEach { f ->
            Regex("frame_(\\d+)_").find(f.name)?.let {
                jpegByFrame.putIfAbsent(it.groupValues[1].toInt(), f)
            }
        }
        val out = ArrayList<NaeTrainer.Sample>()
        for (i in 0 until steps.size - 1) {
            val cur = steps[i]; val next = steps[i + 1]
            if (cur.chosen < 0) continue
            val jpg = jpegByFrame[cur.frame] ?: continue
            val bmp = BitmapFactory.decodeFile(jpg.absolutePath) ?: continue
            val px = IntArray(bmp.width * bmp.height)
            bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            val lum = ShinMetric.lumFromArgb(px, bmp.width, bmp.height)
            val hist = NaeFeatures.multiScaleHist(lum, bmp.width, bmp.height)
            bmp.recycle()
            val target = bestNextCell(next, cur.chosen, grid)
            if (target < 0) continue
            val eCur = NaeSnap.cellE(cur.chosen, grid.gains, grid.exposuresUs)
            val eTgt = NaeSnap.cellE(target, grid.gains, grid.exposuresUs)
            out.add(NaeTrainer.Sample(hist,
                Math.log(eTgt / eCur).coerceIn(-LN_M, LN_M)))
        }
        return out
    }
}
