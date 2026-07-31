package com.example.activeperception.acquire

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Pure tests for RegimeClassifier + CSV logging (run on the JVM). */
class SensingLogTest {

    @Test fun regimeHysteresisNoFlicker() {
        val rc = RegimeClassifier(turnEnter = 0.30, turnExit = 0.15)
        assertEquals(Regime.STRAIGHT, rc.update(0.05))
        assertEquals(Regime.STRAIGHT, rc.update(0.20))   // in-band from straight
        assertEquals(Regime.TURN, rc.update(0.35))       // above enter
        assertEquals(Regime.TURN, rc.update(0.20))       // in-band from turn (hysteresis)
        assertEquals(Regime.STRAIGHT, rc.update(0.10))   // below exit
        assertEquals(Regime.TURN, rc.update(-0.40))      // sign-agnostic
        var toggles = 0; var prev = rc.regime
        for (y in listOf(0.16, 0.29, 0.16, 0.29, 0.20)) {
            val cur = rc.update(y); if (cur != prev) toggles++; prev = cur
        }
        assertEquals(0, toggles)
    }

    @Test fun csvRoundTrip() {
        val vals = listOf("a", "b,c", "d\"e", "", null, 3, 1.5)
        assertEquals(listOf("a", "b,c", "d\"e", "", "", "3", "1.5"),
            Csv.parseRow(Csv.row(vals)))
        assertEquals("\"x,y\"", Csv.escape("x,y"))
        assertEquals("plain", Csv.escape("plain"))
    }

    @Test fun schemaStability() {
        assertTrue(LogSchema.L2_SWEEP.containsAll(LogSchema.L1_VERIFY))
        assertEquals(LogSchema.L1_VERIFY.size + 8, LogSchema.L2_SWEEP.size)
        assertTrue(LogSchema.L3_LIVE.contains("formation_ms"))
        assertTrue(LogSchema.L3_LIVE.contains("infer_ms"))
    }
}
