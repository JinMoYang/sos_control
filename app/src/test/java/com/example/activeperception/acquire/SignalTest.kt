package com.example.activeperception.acquire

import org.junit.Assert.assertEquals
import org.junit.Test

class SignalTest {

    private fun det(x1: Float, y1: Float, x2: Float, y2: Float, conf: Float, cls: Int) =
        Detection(floatArrayOf(x1, y1, x2, y2), conf, cls)

    @Test fun `duplicate boxes on one object are suppressed before summing`() {
        // Two near-identical boxes on the same car: raw sum = 1.3, V3 keeps the peak only.
        val dets = listOf(
            det(10f, 10f, 110f, 110f, 0.7f, 2),
            det(12f, 12f, 112f, 112f, 0.6f, 2)
        )
        assertEquals(0.7, Signal.sumConfV3(dets), 1e-6)
    }

    @Test fun `distinct objects both count`() {
        val dets = listOf(
            det(0f, 0f, 100f, 100f, 0.7f, 2),
            det(200f, 200f, 300f, 300f, 0.6f, 7)
        )
        assertEquals(1.3, Signal.sumConfV3(dets), 1e-4)
    }

    @Test fun `non-vehicle classes and sub-threshold tail are excluded`() {
        val dets = listOf(
            det(0f, 0f, 100f, 100f, 0.9f, 0),     // person
            det(200f, 0f, 300f, 100f, 0.9f, 9),   // traffic light
            det(0f, 200f, 100f, 300f, 0.10f, 2),  // vehicle below thr
            det(200f, 200f, 300f, 300f, 0.5f, 1)  // bicycle IS a vehicle class (V3)
        )
        assertEquals(0.5, Signal.sumConfV3(dets), 1e-6)
    }

    @Test fun `greedy dedup keeps the higher-confidence box`() {
        // Chain: A(0.9) overlaps B(0.8) at IoU 1/3; C(0.7) clears both. Greedy keeps A,
        // drops B (vs A), keeps C.
        val dets = listOf(
            det(0f, 0f, 100f, 100f, 0.9f, 2),
            det(50f, 0f, 150f, 100f, 0.8f, 2),
            det(150f, 0f, 250f, 100f, 0.7f, 2)
        )
        assertEquals(1.6, Signal.sumConfV3(dets), 1e-4)
    }

    @Test fun `iou matches hand computation`() {
        val a = floatArrayOf(0f, 0f, 10f, 10f)
        val b = floatArrayOf(5f, 0f, 15f, 10f)
        assertEquals(50.0 / 150.0, Signal.iou(a, b), 1e-9)
    }
}
