package com.example.activeperception.acquire

import java.util.Arrays

/** Metrics exposed by the allocation-free decoder for EXP2.2. */
data class OptimizedDecodeResult(
    val detections: List<Detection>,
    val preNmsCandidates: Int,
    val topKCandidates: Int
)

/**
 * YOLOv8 decoder that reads TFLite's [channel][anchor] output in-place.
 *
 * The hot path owns one reusable scratch object per decode worker: no 84x8400 flatten copy,
 * no Detection/FloatArray allocation before NMS, bounded pre-NMS work, and primitive-array
 * greedy NMS. Only the final (at most maxDet) detections become Kotlin objects.
 */
class OptimizedYoloDecode(
    private val numClasses: Int,
    private val numAnchors: Int,
    private val confThresh: Float,
    private val allowed: Set<Int>?,
    private val iouThresh: Double,
    private val maxDet: Int,
    private val preNmsTopK: Int = 1000,
    /** Model-output class channels to scan; null scans every channel. */
    classIndices: IntArray? = null,
    /** Optional local-output -> COCO class mapping for a head-pruned model. */
    private val classIdMap: IntArray? = null
) {
    private val scanClasses = classIndices?.clone() ?: IntArray(numClasses) { it }

    init {
        require(scanClasses.all { it in 0 until numClasses })
        require(classIdMap == null || classIdMap.size == numClasses)
    }
    private class Scratch(numAnchors: Int, topK: Int, maxDet: Int) {
        val x1 = FloatArray(numAnchors)
        val y1 = FloatArray(numAnchors)
        val x2 = FloatArray(numAnchors)
        val y2 = FloatArray(numAnchors)
        val score = FloatArray(numAnchors)
        val clazz = IntArray(numAnchors)
        val bestScore = FloatArray(numAnchors)
        val bestClass = IntArray(numAnchors)
        val heap = IntArray(topK)
        val order = IntArray(topK)
        val kept = IntArray(maxDet)
    }

    private val scratch = ThreadLocal.withInitial {
        Scratch(numAnchors, minOf(preNmsTopK, numAnchors), maxDet)
    }

    fun decode(
        output: Array<FloatArray>,
        transform: TensorLetterbox
    ): OptimizedDecodeResult {
        require(output.size >= 4 + numClasses)
        val s = requireNotNull(scratch.get())
        val topK = minOf(preNmsTopK, numAnchors)
        var candidateCount = 0
        var heapSize = 0

        // TFLite output is channel-major. Walking each class channel sequentially avoids
        // 80 pointer/cache jumps per anchor while preserving argmax over all COCO classes.
        Arrays.fill(s.bestScore, 0, numAnchors, 0f)
        Arrays.fill(s.bestClass, 0, numAnchors, -1)
        for (c in scanClasses) {
            val channel = output[4 + c]
            for (anchor in 0 until numAnchors) {
                val confidence = channel[anchor]
                if (confidence > s.bestScore[anchor]) {
                    s.bestScore[anchor] = confidence
                    s.bestClass[anchor] = c
                }
            }
        }
        for (anchor in 0 until numAnchors) {
            val best = s.bestScore[anchor]
            val bestClass = s.bestClass[anchor]
            val mappedClass = if (bestClass >= 0 && classIdMap != null) classIdMap[bestClass]
                else bestClass
            if (best < confThresh || (allowed != null && mappedClass !in allowed)) continue

            val index = candidateCount++
            val cx = output[0][anchor]
            val cy = output[1][anchor]
            val halfW = output[2][anchor] * 0.5f
            val halfH = output[3][anchor] * 0.5f
            s.x1[index] = cx - halfW
            s.y1[index] = cy - halfH
            s.x2[index] = cx + halfW
            s.y2[index] = cy + halfH
            s.score[index] = best
            s.clazz[index] = mappedClass

            if (heapSize < topK) {
                s.heap[heapSize] = index
                siftUp(s, heapSize)
                heapSize++
            } else if (best > s.score[s.heap[0]]) {
                s.heap[0] = index
                siftDown(s, 0, heapSize)
            }
        }

        // Pop the min-heap from the back: order[0] is the highest confidence candidate.
        val selected = heapSize
        var write = selected - 1
        while (heapSize > 0) {
            s.order[write--] = s.heap[0]
            heapSize--
            if (heapSize > 0) {
                s.heap[0] = s.heap[heapSize]
                siftDown(s, 0, heapSize)
            }
        }

        var keptCount = 0
        for (p in 0 until selected) {
            val candidate = s.order[p]
            var suppressed = false
            for (k in 0 until keptCount) {
                val prior = s.kept[k]
                if (s.clazz[prior] == s.clazz[candidate] && iou(s, prior, candidate) > iouThresh) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) {
                s.kept[keptCount++] = candidate
                if (keptCount == maxDet) break
            }
        }

        val scale = transform.scale
        val padX = transform.padX
        val padY = transform.padY
        val detections = ArrayList<Detection>(keptCount)
        for (i in 0 until keptCount) {
            val index = s.kept[i]
            detections.add(Detection(floatArrayOf(
                ((s.x1[index] - padX) / scale).toFloat(),
                ((s.y1[index] - padY) / scale).toFloat(),
                ((s.x2[index] - padX) / scale).toFloat(),
                ((s.y2[index] - padY) / scale).toFloat()
            ), s.score[index], s.clazz[index]))
        }
        return OptimizedDecodeResult(detections, candidateCount, selected)
    }

    private fun siftUp(s: Scratch, position: Int) {
        var child = position
        val value = s.heap[child]
        while (child > 0) {
            val parent = (child - 1) ushr 1
            if (s.score[s.heap[parent]] <= s.score[value]) break
            s.heap[child] = s.heap[parent]
            child = parent
        }
        s.heap[child] = value
    }

    private fun siftDown(s: Scratch, position: Int, size: Int) {
        var parent = position
        val value = s.heap[parent]
        while (true) {
            val left = parent * 2 + 1
            if (left >= size) break
            val right = left + 1
            var smaller = left
            if (right < size && s.score[s.heap[right]] < s.score[s.heap[left]]) smaller = right
            if (s.score[value] <= s.score[s.heap[smaller]]) break
            s.heap[parent] = s.heap[smaller]
            parent = smaller
        }
        s.heap[parent] = value
    }

    private fun iou(s: Scratch, a: Int, b: Int): Double {
        val x1 = maxOf(s.x1[a], s.x1[b])
        val y1 = maxOf(s.y1[a], s.y1[b])
        val x2 = minOf(s.x2[a], s.x2[b])
        val y2 = minOf(s.y2[a], s.y2[b])
        val intersection = maxOf(0f, x2 - x1).toDouble() * maxOf(0f, y2 - y1)
        val areaA = maxOf(0f, s.x2[a] - s.x1[a]).toDouble() * maxOf(0f, s.y2[a] - s.y1[a])
        val areaB = maxOf(0f, s.x2[b] - s.x1[b]).toDouble() * maxOf(0f, s.y2[b] - s.y1[b])
        val union = areaA + areaB - intersection
        return if (union > 0.0) intersection / union else 0.0
    }
}
