package com.example.activeperception.acquire

import kotlin.math.max
import kotlin.math.min

data class RoutingDecision(
    val shouldOffload: Boolean,
    val score: Double,
    val modelLimitedClusters: Int,
    val recoveredLocallyClusters: Int
) {
    companion object { val KEEP_LOCAL = RoutingDecision(false, 0.0, 0, 0) }
}

/**
 * Training-free SoS router from the detector outputs the sensing bracket already produced.
 * Same-class boxes are associated across exposure candidates by IoU. A recurring weak
 * cluster that never crosses the local operating threshold is the model-limited remainder;
 * a cluster resolved by any exposure is kept local.
 */
class CrossExposureRouter(
    private val operatingThreshold: Float = 0.25f,
    private val associationIou: Double = 0.5,
    private val minExposureSupport: Int = 2
) {
    private data class Observation(val candidate: Int, val detection: Detection)
    private class Cluster(val classId: Int, val observations: MutableList<Observation>)

    fun decide(bracket: List<List<Detection>>): RoutingDecision {
        if (bracket.size < minExposureSupport) return RoutingDecision.KEEP_LOCAL
        val clusters = ArrayList<Cluster>()
        bracket.forEachIndexed { candidate, detections ->
            detections.forEach { detection ->
                val cluster = clusters
                    .filter { it.classId == detection.classId }
                    .maxByOrNull { candidateCluster ->
                        candidateCluster.observations.maxOf {
                            iou(it.detection.xyxy, detection.xyxy)
                        }
                    }
                val overlap = cluster?.observations?.maxOf {
                    iou(it.detection.xyxy, detection.xyxy)
                } ?: 0.0
                if (cluster != null && overlap >= associationIou) {
                    cluster.observations.add(Observation(candidate, detection))
                } else {
                    clusters.add(Cluster(detection.classId,
                        mutableListOf(Observation(candidate, detection))))
                }
            }
        }

        var modelLimited = 0
        var recovered = 0
        var score = 0.0
        clusters.forEach { cluster ->
            val support = cluster.observations.map { it.candidate }.toSet().size
            if (support < minExposureSupport) return@forEach
            val resolved = cluster.observations.any {
                it.detection.confidence >= operatingThreshold
            }
            if (resolved) {
                recovered++
            } else {
                modelLimited++
                val meanTail = cluster.observations.map { it.detection.confidence }.average()
                score += support * meanTail
            }
        }
        return RoutingDecision(modelLimited > 0, score, modelLimited, recovered)
    }

    private fun iou(a: FloatArray, b: FloatArray): Double {
        val left = max(a[0], b[0]).toDouble()
        val top = max(a[1], b[1]).toDouble()
        val right = min(a[2], b[2]).toDouble()
        val bottom = min(a[3], b[3]).toDouble()
        val intersection = max(0.0, right - left) * max(0.0, bottom - top)
        val areaA = max(0.0, (a[2] - a[0]).toDouble()) * max(0.0, (a[3] - a[1]).toDouble())
        val areaB = max(0.0, (b[2] - b[0]).toDouble()) * max(0.0, (b[3] - b[1]).toDouble())
        val union = areaA + areaB - intersection
        return if (union > 0.0) intersection / union else 0.0
    }
}
