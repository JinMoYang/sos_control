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
}
