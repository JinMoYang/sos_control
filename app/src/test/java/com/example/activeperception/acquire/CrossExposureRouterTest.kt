package com.example.activeperception.acquire

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CrossExposureRouterTest {
    private fun det(conf: Float, x: Float = 0f) =
        Detection(floatArrayOf(x, 0f, x + 10f, 10f), conf, 2)

    @Test fun recurringWeakClusterIsOffloaded() {
        val decision = CrossExposureRouter().decide(listOf(
            listOf(det(0.10f)), listOf(det(0.18f)), emptyList()))
        assertTrue(decision.shouldOffload)
        assertEquals(1, decision.modelLimitedClusters)
    }

    @Test fun bracketResolvedClusterStaysLocal() {
        val decision = CrossExposureRouter().decide(listOf(
            listOf(det(0.10f)), listOf(det(0.40f))))
        assertFalse(decision.shouldOffload)
        assertEquals(1, decision.recoveredLocallyClusters)
    }

    @Test fun singleFalsePositiveIsNotSent() {
        val decision = CrossExposureRouter().decide(listOf(
            listOf(det(0.20f)), emptyList(), emptyList()))
        assertFalse(decision.shouldOffload)
    }
}
