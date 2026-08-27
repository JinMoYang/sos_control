package com.example.activeperception

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.util.Log
import kotlin.math.*

/**
 * Accelerometer, gyroscope, and light sensor fusion.
 *
 * The gyro's gravity-axis component is the yaw rate that drives [currentHeadingAngle] and the
 * straight-vs-turn classification; the gyro MAGNITUDE is kept separately because it conflates
 * rotation with vibration. Gravity is locked from a detected stable pose and then drift-
 * corrected, so heading integration doesn't wander with device tilt.
 *
 * Heading crossing 360° counts a lap, which segments a repeatable-track experiment.
 */
class SensorDataManager(
    private val context: Context,
    private val mainHandler: Handler = Handler(Looper.getMainLooper())
) : SensorEventListener {
    companion object {
        private const val TAG = "SensorDataManager"
        private const val NS2S = 1.0f / 1000000000.0f
        private const val GYRO_DEADZONE = 0.05f   // below this, treat yaw as noise
        private const val LAP_THRESHOLD = 360.0   // degrees
    }

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var lightSensor: Sensor? = null

    private val gravity = floatArrayOf(0f, 0f, 0f)

    var smoothedAccel: Double = 0.0
        private set
    var smoothedGyro: Double = 0.0
        private set
    var currentLux: Float = 0.0f
        private set

    var currentAccel: Double = 0.0
        private set
    var currentGyro: Double = 0.0
        private set

    /** Gravity-axis rotation in rad/s — the straight-vs-turn signal, distinct from
     *  [currentGyro], which is a magnitude and so conflates turning with vibration. */
    var currentYawRate: Double = 0.0
        private set

    /** Integrated heading in degrees, reset each lap. Recorded in frames.csv as the
     *  path-position proxy used to align passes offline. */
    var currentHeadingAngle: Double = 0.0
        private set

    /** Per-sample hooks, each fired from the sensor thread with that sensor's OWN timestamp.
     *  Whether those timestamps share a clock with the camera's SENSOR_TIMESTAMP is recorded
     *  in the manifest as timestamp_source. */
    var onAccelSample: ((tsNs: Long, ax: Float, ay: Float, az: Float) -> Unit)? = null
    var onGyroSample:  ((tsNs: Long, gx: Float, gy: Float, gz: Float) -> Unit)? = null
    var onLuxSample:   ((tsNs: Long, lux: Float) -> Unit)? = null

    private var lapCount = 1
    private var lastGyroTimestamp = 0L

    private var lapAccelSum = 0.0
    private var lapGyroSum = 0.0
    private var lapLuxSum = 0.0f
    private var lapAccelCount = 0
    private var lapGyroCount = 0
    private var lapLuxCount = 0

    private var recentFramesWindow = 15

    private val dataLock = Any()

    private val referenceGravity = floatArrayOf(0f, 0f, 0f)
    private var isReferenceGravitySet = false
    private val recentAccel = ArrayDeque<Double>()
    private val recentGyro = ArrayDeque<Double>()
    private val recentLux = ArrayDeque<Float>()

    private var lastMotionTime = 0L
    private var lastLapTime = 0L
    private val MIN_LAP_INTERVAL_MS = 1000L

    var onLapCompleted: ((Int) -> Unit)? = null
    var onSensorDataUpdated: ((Double, Double, Float) -> Unit)? = null

    private var toneGenerator: ToneGenerator? = null

    fun release() {
        try {
            toneGenerator?.release()
            toneGenerator = null
            Log.d(TAG, "ToneGenerator released")
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing toneGenerator", e)
        }
    }

    var ACCEL_MIN = 10.5
    var ACCEL_MAX = 15.0
    var GYRO_MIN = 0.5
    var GYRO_MAX = 2.5
    var DECAY_FACTOR = 0.8

    init {
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
    }

    fun registerListeners() {
        lastGyroTimestamp = 0L

        if (toneGenerator == null) {
            toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)
        }

        // GAME (~50 Hz) is ample against ~1-2 Hz camera frames; FASTEST would be 200-400 Hz
        // here and would bloat imu.csv for no gain.
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        lightSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        Log.d(TAG, "Sensors registered")
    }

    fun unregisterListeners() {
        sensorManager.unregisterListener(this)
        lastGyroTimestamp = 0L
        Log.d(TAG, "All sensors unregistered")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            var isUpdated = false

            when (it.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    processAccelerometer(it.values, it.timestamp)
                    isUpdated = true
                }
                Sensor.TYPE_GYROSCOPE -> {
                    processGyroscope(it.values, it.timestamp)
                    isUpdated = true
                }
                Sensor.TYPE_LIGHT -> {
                    processLightSensor(it.values[0], it.timestamp)
                }
            }

            if (isUpdated) {
                if (smoothedAccel > ACCEL_MIN || smoothedGyro > GYRO_MIN) {
                    lastMotionTime = System.currentTimeMillis()
                }
                onSensorDataUpdated?.invoke(smoothedAccel, smoothedGyro, currentLux)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun processAccelerometer(values: FloatArray, timestamp: Long) {
        val x = values[0]
        val y = values[1]
        val z = values[2]
        onAccelSample?.invoke(timestamp, x, y, z)
        val rawAccel = sqrt((x * x + y * y + z * z).toDouble())

        // Stable = gravity only (no centrifugal component) and no rotation.
        val isStable = rawAccel in 9.5..10.2 && smoothedGyro < 0.1 && lastGyroTimestamp != 0L

        if (isStable) {
            if (!isReferenceGravitySet) {
                referenceGravity[0] = x
                referenceGravity[1] = y
                referenceGravity[2] = z
                isReferenceGravitySet = true
                Log.d(TAG, "Reference Gravity Locked!")
            } else {
                // Slow correction for sensor drift.
                referenceGravity[0] = 0.99f * referenceGravity[0] + 0.01f * x
                referenceGravity[1] = 0.99f * referenceGravity[1] + 0.01f * y
                referenceGravity[2] = 0.99f * referenceGravity[2] + 0.01f * z
            }
        } else if (!isReferenceGravitySet) {
            // Provisional value until a stable pose appears.
            referenceGravity[0] = x
            referenceGravity[1] = y
            referenceGravity[2] = z
        }

        if (rawAccel > smoothedAccel) {
            smoothedAccel = rawAccel
        } else {
            smoothedAccel = (smoothedAccel * DECAY_FACTOR) + (rawAccel * (1 - DECAY_FACTOR))
        }
        currentAccel = smoothedAccel

        synchronized(dataLock) {
            if (recentAccel.size >= recentFramesWindow) recentAccel.removeFirst()
            recentAccel.addLast(currentAccel)

            lapAccelSum += currentAccel
            lapAccelCount++
        }
    }

    private fun processGyroscope(values: FloatArray, timestamp: Long) {
        val x = values[0]
        val y = values[1]
        val z = values[2]

        if (lastGyroTimestamp != 0L && isReferenceGravitySet) {
            val dt = (timestamp - lastGyroTimestamp) * NS2S

            val gx = referenceGravity[0].toDouble()
            val gy = referenceGravity[1].toDouble()
            val gz = referenceGravity[2].toDouble()

            // Rotation projected onto gravity = yaw, independent of how the device is held.
            val gNorm = sqrt(gx * gx + gy * gy + gz * gz)
            val verticalGyroSpeedRad = if (gNorm > 0) {
                (x * gx + y * gy + z * gz) / gNorm
            } else {
                z.toDouble()
            }
            currentYawRate = verticalGyroSpeedRad

            if (Math.abs(verticalGyroSpeedRad) < GYRO_DEADZONE) {
                lastGyroTimestamp = timestamp
                updateGyroMagnitude(x, y, z)
                return
            }

            val deltaAngle = Math.toDegrees(verticalGyroSpeedRad) * dt
            currentHeadingAngle += deltaAngle

            if (Math.abs(currentHeadingAngle) >= LAP_THRESHOLD) {
                val currentTime = System.currentTimeMillis()

                if (currentTime - lastLapTime < MIN_LAP_INTERVAL_MS) {
                    Log.w(TAG, "Lap too fast (${currentTime - lastLapTime}ms), ignoring")
                    currentHeadingAngle = 0.0
                    lastGyroTimestamp = timestamp
                    updateGyroMagnitude(x, y, z)
                    return
                }

                lastLapTime = currentTime
                val completedLap = lapCount
                lapCount++

                currentHeadingAngle -= LAP_THRESHOLD * Math.signum(currentHeadingAngle)

                if (Math.abs(currentHeadingAngle) > 45.0) {
                    currentHeadingAngle = 0.0
                }

                if (lapCount % 10 == 0) {
                    playLapSound()
                }

                onLapCompleted?.invoke(completedLap)
                resetLapAverages()

                Log.d(TAG, "Lap $completedLap completed! Now at Lap $lapCount")
            }
        }

        lastGyroTimestamp = timestamp
        updateGyroMagnitude(x, y, z)
        onGyroSample?.invoke(timestamp, x, y, z)
    }

    private fun updateGyroMagnitude(x: Float, y: Float, z: Float) {
        val rawGyro = sqrt((x * x + y * y + z * z).toDouble())
        if (rawGyro > smoothedGyro) {
            smoothedGyro = rawGyro
        } else {
            smoothedGyro = (smoothedGyro * DECAY_FACTOR) + (rawGyro * (1 - DECAY_FACTOR))
        }
        currentGyro = smoothedGyro

        synchronized(dataLock) {
            if (recentGyro.size >= recentFramesWindow) recentGyro.removeFirst()
            recentGyro.addLast(currentGyro)

            lapGyroSum += currentGyro
            lapGyroCount++
        }
    }

    private fun processLightSensor(lux: Float, timestamp: Long) {
        currentLux = (currentLux * 0.7f) + (lux * 0.3f)
        onLuxSample?.invoke(timestamp, lux)

        synchronized(dataLock) {
            if (recentLux.size >= recentFramesWindow) recentLux.removeFirst()
            recentLux.addLast(currentLux)

            lapLuxSum += currentLux
            lapLuxCount++
        }
    }

    fun getLapAverages(): Triple<Double, Double, Float> {
        synchronized(dataLock) {
            val avgAccel = if (lapAccelCount > 0) lapAccelSum / lapAccelCount else currentAccel
            val avgGyro = if (lapGyroCount > 0) lapGyroSum / lapGyroCount else currentGyro
            val avgLux = if (lapLuxCount > 0) lapLuxSum / lapLuxCount else currentLux
            return Triple(avgAccel, avgGyro, avgLux)
        }
    }

    fun getRecentAverages(): Triple<Double, Double, Float> {
        synchronized(dataLock) {
            val avgAccel = if (recentAccel.isNotEmpty()) recentAccel.average() else currentAccel
            val avgGyro = if (recentGyro.isNotEmpty()) recentGyro.average() else currentGyro
            val avgLux = if (recentLux.isNotEmpty()) recentLux.average().toFloat() else currentLux
            return Triple(avgAccel, avgGyro, avgLux)
        }
    }

    fun resetLapAverages() {
        synchronized(dataLock) {
            lapAccelSum = 0.0
            lapGyroSum = 0.0
            lapLuxSum = 0.0f
            lapAccelCount = 0
            lapGyroCount = 0
            lapLuxCount = 0
        }
    }

    fun getCurrentLap(): Int = lapCount

    fun resetLapCount() {
        lapCount = 1
        currentHeadingAngle = 0.0
        lastGyroTimestamp = 0L
    }

    private fun playLapSound() {
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_CDMA_PIP, 150)
            Log.d(TAG, "Lap sound played")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to play lap sound", e)
        }
    }

    fun mapValue(value: Double, inMin: Double, inMax: Double, outMin: Long, outMax: Long): Long {
        val result = (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin
        return if (outMin < outMax) result.coerceIn(outMin.toDouble(), outMax.toDouble()).toLong()
        else result.coerceIn(outMax.toDouble(), outMin.toDouble()).toLong()
    }


    fun mapLogLog(
        value: Float,
        inInit: Float,
        inEnd: Float,
        outInit: Float,
        outEnd: Float
    ): Float {
        val sInInit = inInit.coerceAtLeast(1e-6f)
        val sInEnd = inEnd.coerceAtLeast(1e-6f)
        val sOutInit = outInit.coerceAtLeast(1e-6f)
        val sOutEnd = outEnd.coerceAtLeast(1e-6f)
        val clampedValue = value.coerceIn(min(sInInit, sInEnd), max(sInInit, sInEnd))
        val logInInit = ln(sInInit)
        val logInEnd = ln(sInEnd)
        val logOutInit = ln(sOutInit)
        val logOutEnd = ln(sOutEnd)
        val logValue = ln(clampedValue)
        val ratio = (logValue - logInInit) / (logInEnd - logInInit)
        val logResult = logOutInit + ratio * (logOutEnd - logOutInit)
        return exp(logResult)
    }

    fun setRecentFramesWindow(size: Int) {
        synchronized(dataLock) {
            recentFramesWindow = size
            recentAccel.clear()
            recentGyro.clear()
            recentLux.clear()
        }
    }

    fun getRecentFramesWindow(): Int = recentFramesWindow
}