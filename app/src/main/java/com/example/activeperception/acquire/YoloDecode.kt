package com.example.activeperception.acquire

/**
 * YOLOv8 output decode + NMS (pure).
 *
 * Output layout is `[4 + numClasses, numAnchors]` flattened channel-major:
 * `out[c * numAnchors + a]`. Channels 0..3 are cx,cy,w,h in the letterboxed model-input
 * pixel space; 4.. are per-class scores with sigmoid already applied by the export.
 * Boxes come back in input space — call [unletterbox] to map them to the original image.
 */
object YoloDecode {

    fun iou(a: FloatArray, b: FloatArray): Double {
        val x1 = maxOf(a[0], b[0]); val y1 = maxOf(a[1], b[1])
        val x2 = minOf(a[2], b[2]); val y2 = minOf(a[3], b[3])
        val inter = maxOf(0f, x2 - x1).toDouble() * maxOf(0f, y2 - y1)
        val areaA = (a[2] - a[0]).toDouble() * (a[3] - a[1])
        val areaB = (b[2] - b[0]).toDouble() * (b[3] - b[1])
        val union = areaA + areaB - inter
        return if (union <= 0.0) 0.0 else inter / union
    }

    /** Greedy per-class NMS. */
    fun nms(dets: List<Detection>, iouThresh: Double): List<Detection> {
        val kept = ArrayList<Detection>()
        for ((_, group) in dets.groupBy { it.classId }) {
            val taken = ArrayList<Detection>()
            for (d in group.sortedByDescending { it.confidence }) {
                if (taken.none { iou(it.xyxy, d.xyxy) > iouThresh }) taken.add(d)
            }
            kept.addAll(taken)
        }
        return kept
    }

    fun decode(output: FloatArray, numClasses: Int, numAnchors: Int, confThresh: Float,
               allowed: Set<Int>?, iouThresh: Double, maxDet: Int = 100): List<Detection> {
        val cand = ArrayList<Detection>()
        for (a in 0 until numAnchors) {
            var best = 0f; var bestCls = -1
            for (c in 0 until numClasses) {
                val s = output[(4 + c) * numAnchors + a]
                if (s > best) { best = s; bestCls = c }
            }
            if (best < confThresh) continue
            if (allowed != null && bestCls !in allowed) continue
            val cx = output[a]; val cy = output[numAnchors + a]
            val w = output[2 * numAnchors + a]; val h = output[3 * numAnchors + a]
            cand.add(Detection(floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                               best, bestCls))
        }
        val kept = nms(cand, iouThresh)
        return if (kept.size <= maxDet) kept
               else kept.sortedByDescending { it.confidence }.take(maxDet)
    }

    /** Map boxes from letterboxed input space back to original image coords. */
    fun unletterbox(dets: List<Detection>, scale: Double, padX: Double, padY: Double): List<Detection> =
        dets.map { d ->
            val b = d.xyxy
            Detection(floatArrayOf(
                ((b[0] - padX) / scale).toFloat(), ((b[1] - padY) / scale).toFloat(),
                ((b[2] - padX) / scale).toFloat(), ((b[3] - padY) / scale).toFloat()),
                d.confidence, d.classId)
        }
}
