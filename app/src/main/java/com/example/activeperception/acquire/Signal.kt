package com.example.activeperception.acquire

/**
 * Selection-signal conventions shared with the offline simulation (`sense/proxy.py`,
 * V3 canon of 2026-08-27): vehicle classes INCLUDING bicycle, operating threshold on
 * confidence, and class-agnostic greedy dedup at IoU 0.3 BEFORE summing. The evaluation
 * counts post-NMS objects, so the raw sum rewarded an exposure three times for drawing
 * three boxes on one car; dedup at the evaluation's matching IoU restores agreement
 * (sim audit S1t: F1 +0.32* / recall +0.65*).
 *
 * Pure logic, no Android deps — keep in lockstep with `sense.proxy.sumconf`.
 */
object Signal {
    /** COCO bicycle, car, motorcycle, bus, truck — `sense.proxy.VEH_CLASSES`. */
    val VEH_CLASSES = intArrayOf(1, 2, 3, 5, 7)
    const val CONF_THR = 0.25f
    const val IOU_MATCH = 0.3

    fun iou(a: FloatArray, b: FloatArray): Double {
        val x1 = maxOf(a[0], b[0]).toDouble(); val y1 = maxOf(a[1], b[1]).toDouble()
        val x2 = minOf(a[2], b[2]).toDouble(); val y2 = minOf(a[3], b[3]).toDouble()
        if (x2 <= x1 || y2 <= y1) return 0.0
        val inter = (x2 - x1) * (y2 - y1)
        val union = (a[2] - a[0]).toDouble() * (a[3] - a[1]) +
            (b[2] - b[0]).toDouble() * (b[3] - b[1]) - inter
        return if (union > 0) inter / union else 0.0
    }

    /** V3 sumconf — the controller's selection signal. Mirrors `sense.proxy.sumconf`:
     *  vehicle classes at/above [thr], greedy highest-first dedup at [IOU_MATCH]. */
    fun sumConfV3(dets: List<Detection>, thr: Float = CONF_THR): Double {
        val cand = dets.filter { VEH_CLASSES.contains(it.classId) && it.confidence >= thr }
            .sortedByDescending { it.confidence }
        val kept = ArrayList<Detection>(cand.size)
        var sum = 0.0
        for (d in cand) {
            if (kept.none { iou(d.xyxy, it.xyxy) >= IOU_MATCH }) {
                kept.add(d); sum += d.confidence
            }
        }
        return sum
    }
}
