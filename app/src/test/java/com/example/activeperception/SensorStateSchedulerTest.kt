package com.example.activeperception

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SensorStateSchedulerTest {
    @Test fun waitsForCaptureResultInsteadOfDeclaredLatency() {
        val scheduler = SensorStateScheduler()
        val command = scheduler.enqueue(SensorState(64_000, 100), nowNs = 1)
        scheduler.markRequestFrame(command.id, 50)

        repeat(9) { offset ->
            assertNull(scheduler.observe(50L + offset, 100L + offset,
                SensorState(32_000, 100)))
        }
        val applied = scheduler.observe(59, 109, SensorState(64_000, 100))

        assertEquals(command.id, applied?.id)
        assertEquals(9L, applied?.applyDelayFrames)
        assertTrue(scheduler.pendingSnapshot().isEmpty())
    }

    @Test fun exposureToleranceAcceptsMetadataRoundingButNotWrongIso() {
        val scheduler = SensorStateScheduler(exposureToleranceUs = 750)
        val expected = SensorState(64_000, 200)
        assertTrue(scheduler.matches(expected, SensorState(63_994, 200)))
        assertTrue(!scheduler.matches(expected, SensorState(63_994, 201)))
    }
}
