package com.example.activeperception

import android.graphics.BitmapFactory
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.NaeFeatures
import com.example.activeperception.acquire.NaeSnap
import com.example.activeperception.acquire.NaeTrainer
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
