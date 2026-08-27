package com.example.activeperception.acquire

/** Motion regime from yaw rate, for segmenting straight vs turning driving. */
enum class Regime { STRAIGHT, TURN }

/**
 * Classifies straight vs turn from the platform yaw rate (rad/s) — the gravity-axis
 * gyro component (`verticalGyroSpeedRad` in SensorDataManager), NOT the gyro
 * magnitude (which conflates vibration). Hysteresis (enter > exit) prevents
 * flicker in the band; stateful.
 */
class RegimeClassifier(
    private val turnEnter: Double = 0.30,   // |yaw| >= this while STRAIGHT -> TURN
    private val turnExit: Double = 0.15     // |yaw| <= this while TURN -> STRAIGHT
) {
    init {
        require(turnExit <= turnEnter) { "turnExit must be <= turnEnter" }
    }

    var regime: Regime = Regime.STRAIGHT
        private set

    fun update(yawRate: Double): Regime {
        val y = Math.abs(yawRate)
        regime = when (regime) {
            Regime.STRAIGHT -> if (y >= turnEnter) Regime.TURN else Regime.STRAIGHT
            Regime.TURN -> if (y <= turnExit) Regime.STRAIGHT else Regime.TURN
        }
        return regime
    }
}
