package com.example.activeperception.acquire

/** One detection: box in image pixel space, confidence, COCO class id. */
class Detection(val xyxy: FloatArray, val confidence: Float, val classId: Int)

/** Renders the given grid cells of the current frame. Generic over the image type so the
 *  controller is testable without Android (T = Bitmap on device). */
interface CandidateSource<T> {
    fun render(cells: IntArray): List<T>
}

/** Detects on a batch of candidates. May true-batch or loop internally — transparent to
 *  the controller, so batch-vs-loop is measurable without changing this code. */
interface Detector<T> {
    fun detectBatch(images: List<T>): List<List<Detection>>
}

class StepResult(
    val cell: Int,
    val detections: List<Detection>,
    val scores: DoubleArray,
    val cells: IntArray,
    val chosen: Int
)

/**
 * Candidate schedule for step [t].
 * Probe step (`t % period == 0`, includes t=0): the full grid.
 * Otherwise: the gain column at the anchor's shutter row.
 */
fun plan(t: Int, anchor: Int, grid: Grid, period: Int): IntArray {
    val sj = grid.indices(anchor).second
    return if (t % period == 0) {
        IntArray(grid.nGain * grid.nShutter) { it }
    } else {
        IntArray(grid.nGain) { gi -> grid.cell(gi, sj) }
    }
}

/** Summed confidence over detections at/above [selectConf]. The detector floors much lower
 *  (0.01) to keep the offload tail, so selection has to re-threshold here. */
fun sumConf(dets: List<Detection>, selectConf: Float): Double =
    dets.sumOf { if (it.confidence >= selectConf) it.confidence.toDouble() else 0.0 }

/**
 * Acquire-and-select loop. State is only `anchor` + `t`.
 * Per step: plan candidates -> render -> detect -> keep the highest Σconf and anchor on it.
 * With no detection anywhere the anchor holds.
 *
 * NOTE: the deployed path is MeasurementController.EntropyFallbackController, which adds a
 * tie-break for the all-zero case and therefore moves the anchor where this class holds it.
 */
class AcquireSelectController<T>(
    private val source: CandidateSource<T>,
    private val detector: Detector<T>,
    val grid: Grid,
    private val period: Int = 5,
    initAnchor: Int = 0,
    private val selectConf: Float = 0.25f
) {
    var anchor: Int = initAnchor
        private set
    var t: Int = 0
        private set

    fun step(): StepResult {
        val cells = plan(t, anchor, grid, period)
        val images = source.render(cells)
        val dets = detector.detectBatch(images)
        val scores = DoubleArray(dets.size) { i -> sumConf(dets[i], selectConf) }
        var chosen = 0
        for (i in scores.indices) if (scores[i] > scores[chosen]) chosen = i   // first-max on ties
        t += 1
        // The chosen candidate's FULL detections are emitted (tail kept for offload/logging);
        // the anchor only moves on a real (>= selectConf) detection.
        return if (scores[chosen] > 0.0) {
            anchor = cells[chosen]
            StepResult(cells[chosen], dets[chosen], scores, cells, chosen)
        } else {
            StepResult(anchor, dets[chosen], scores, cells, chosen)
        }
    }
}
