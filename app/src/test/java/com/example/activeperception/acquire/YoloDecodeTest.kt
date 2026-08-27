package com.example.activeperception.acquire

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Pure tests for YOLOv8 output decode + NMS (run on the JVM). */
class YoloDecodeTest {

    private fun synthetic(): Triple<FloatArray, Int, Int> {
        val a = 3; val c = 3
        val out = FloatArray((4 + c) * a)
        fun set(ch: Int, an: Int, v: Float) { out[ch * a + an] = v }
        set(0,0,50f); set(1,0,50f); set(2,0,20f); set(3,0,20f); set(4+1,0,0.9f); set(4+0,0,0.1f)
        set(0,1,51f); set(1,1,51f); set(2,1,20f); set(3,1,20f); set(4+1,1,0.8f)   // overlaps a0, cls1
        set(0,2,200f);set(1,2,200f);set(2,2,10f); set(3,2,10f); set(4+2,2,0.05f)  // below thresh
        return Triple(out, c, a)
    }

    @Test fun decodeThreshAndNms() {
        val (out, c, a) = synthetic()
        val d = YoloDecode.decode(out, c, a, 0.1f, null, 0.45)
        assertEquals(1, d.size)
        assertEquals(1, d[0].classId)
    }

    @Test fun allowedFilter() {
        val (out, c, a) = synthetic()
        assertTrue(YoloDecode.decode(out, c, a, 0.1f, setOf(2), 0.45).isEmpty())
    }

    @Test fun iouEndpoints() {
        assertEquals(1.0, YoloDecode.iou(floatArrayOf(0f,0f,10f,10f), floatArrayOf(0f,0f,10f,10f)), 1e-9)
        assertEquals(0.0, YoloDecode.iou(floatArrayOf(0f,0f,10f,10f), floatArrayOf(20f,20f,30f,30f)), 1e-9)
    }

    @Test fun unletterbox() {
        val ub = YoloDecode.unletterbox(
            listOf(Detection(floatArrayOf(100f,100f,200f,200f), 0.9f, 2)), 0.5, 20.0, 10.0)
        assertEquals(160f, ub[0].xyxy[0], 1e-4f)
        assertEquals(180f, ub[0].xyxy[1], 1e-4f)
    }

    @Test fun optimizedDecoderMatchesReferenceWithoutFlattenCopy() {
        val (flat, classes, anchors) = synthetic()
        val nested = Array(4 + classes) { channel ->
            FloatArray(anchors) { anchor -> flat[channel * anchors + anchor] }
        }
        val reference = YoloDecode.unletterbox(
            YoloDecode.decode(flat, classes, anchors, 0.1f, null, 0.45, 100),
            0.5, 20.0, 10.0)
        val optimized = OptimizedYoloDecode(
            classes, anchors, 0.1f, null, 0.45, 100, preNmsTopK = 100)
            .decode(nested, TensorLetterbox(0.5, 20.0, 10.0))
        assertEquals(reference.size, optimized.detections.size)
        assertEquals(reference[0].classId, optimized.detections[0].classId)
        assertEquals(reference[0].confidence, optimized.detections[0].confidence, 0f)
        for (i in 0..3) assertEquals(reference[0].xyxy[i], optimized.detections[0].xyxy[i], 0f)
    }

    @Test fun optimizedDecoderBoundsPreNmsCandidates() {
        val anchors = 5; val classes = 1
        val out = Array(4 + classes) { FloatArray(anchors) }
        for (a in 0 until anchors) {
            out[0][a] = a * 50f; out[1][a] = 20f
            out[2][a] = 10f; out[3][a] = 10f; out[4][a] = 0.1f * (a + 1)
        }
        val result = OptimizedYoloDecode(
            classes, anchors, 0.01f, null, 0.45, 100, preNmsTopK = 2)
            .decode(out, TensorLetterbox(1.0, 0.0, 0.0))
        assertEquals(5, result.preNmsCandidates)
        assertEquals(2, result.topKCandidates)
        assertEquals(listOf(0.5f, 0.4f), result.detections.map { it.confidence })
    }

    @Test fun optimizedDecoderClassRestrictionMatchesPrunedHeadMapping() {
        val anchors = 2
        val all = Array(4 + 80) { FloatArray(anchors) }
        for (a in 0 until anchors) {
            all[0][a] = 20f + a * 40f; all[1][a] = 20f
            all[2][a] = 10f; all[3][a] = 10f
        }
        all[4 + 41][0] = 0.8f // cup
        all[4 + 60][1] = 0.7f // dining table
        all[4 + 2][0] = 0.99f // ignored non-target car must not hide the cup
        val ids = intArrayOf(41, 40, 46, 5, 60)
        val restricted = OptimizedYoloDecode(
            80, anchors, 0.01f, null, 0.45, 100, classIndices = ids)
            .decode(all, TensorLetterbox(1.0, 0.0, 0.0)).detections

        val head = Array(4 + 5) { FloatArray(anchors) }
        for (channel in 0 until 4) all[channel].copyInto(head[channel])
        for (local in ids.indices) all[4 + ids[local]].copyInto(head[4 + local])
        val pruned = OptimizedYoloDecode(
            5, anchors, 0.01f, null, 0.45, 100, classIdMap = ids)
            .decode(head, TensorLetterbox(1.0, 0.0, 0.0)).detections
        assertEquals(listOf(41, 60), restricted.map { it.classId })
        assertEquals(restricted.map { it.classId }, pruned.map { it.classId })
        assertEquals(restricted.map { it.confidence }, pruned.map { it.confidence })
    }
}
