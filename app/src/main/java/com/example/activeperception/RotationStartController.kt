package com.example.activeperception

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.io.FileWriter
import java.util.Locale
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * One-shot turntable gate copied from the validated diagnostic motion experiment.
 *
 * It learns both oscillation endpoints from relative Game Rotation Vector quaternions,
 * computes their shortest-path midpoint, then fires once while crossing that midpoint in
 * the positive gyro-Y direction. Euler yaw is deliberately not used because the glasses sit
 * near its pitch singularity. The learned range works unchanged for 45/90/180 degree rigs.
 */
class RotationStartController(
    context: Context,
    private val runDir: File,
    private val status: (String) -> Unit
) : SensorEventListener, Closeable {
    data class Quaternion(val x: Double, val y: Double, val z: Double, val w: Double)
    data class PoseSample(val timestampNs: Long, val q: Quaternion, val gyroY: Double)
    data class Profile(
        val endpointPositive: Quaternion,
        val endpointNegative: Quaternion,
        val center: Quaternion,
        val rangeDeg: Double,
        val halfPeriodMs: Double
    )
    data class TriggerEvent(
        val sensorTimestampNs: Long,
        val gyroY: Double,
        val centerDistanceDeg: Double,
        val predictedTimeToCenterNs: Long,
        val leadNs: Long
    )
    data class PoseAt(val sampleTimestampNs: Long, val centerErrorDeg: Double, val gyroY: Double)

    private data class Endpoint(
        val timestampNs: Long, val fromSign: Int, val toSign: Int, val q: Quaternion
    )
    private data class Armed(val leadNs: Long, val callback: (TriggerEvent) -> Unit)

    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gameRotation = manager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR, false)
        ?: error("Game Rotation Vector unavailable")
    private val gyro = manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE, false)
        ?: error("Gyroscope unavailable")
    private val thread = HandlerThread("SoS-RotationStart").apply { start() }
    private val handler = Handler(thread.looper)
    private val writer: BufferedWriter = BufferedWriter(
        FileWriter(File(runDir, "rotation_trigger.csv")), 256 * 1024
    ).apply {
        appendLine("wall_time_ms,sensor_timestamp_ns,callback_elapsed_ns,sensor_type,x,y,z,w,gyro_y_rad_s,event,detail")
    }

    private val history = ArrayDeque<PoseSample>()
    private val endpoints = mutableListOf<Endpoint>()
    private val positiveCenterOffsetsNs = ArrayDeque<Long>()
    private var latestGyroY = 0.0
    private var smoothedGyroY = 0.0
    private var directionSign = 0
    private var lastPositiveEndpointNs = 0L
    private var learning = false
    private var learningCallback: ((Profile) -> Unit)? = null
    private var armed: Armed? = null
    private var running = false
    private var profile: Profile? = null
    @Volatile private var closed = false

    @Synchronized
    fun startLearning(callback: (Profile) -> Unit) {
        check(!closed) { "rotation controller is closed" }
        if (!running) {
            manager.registerListener(this, gameRotation, SensorManager.SENSOR_DELAY_FASTEST, 0, handler)
            manager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST, 0, handler)
            running = true
        }
        endpoints.clear()
        positiveCenterOffsetsNs.clear()
        directionSign = 0
        learning = true
        learningCallback = callback
        logEvent(SystemClock.elapsedRealtimeNanos(), "learning_started",
            "game=${gameRotation.name} gyro=${gyro.name}")
        status("ROTATION · 양 끝점 학습 중 · 턴테이블을 계속 회전하세요")
    }

    @Synchronized
    fun arm(leadNs: Long, callback: (TriggerEvent) -> Unit) {
        check(profile != null) { "rotation profile not learned" }
        armed = Armed(leadNs.coerceAtLeast(0L), callback)
        logEvent(SystemClock.elapsedRealtimeNanos(), "armed", "lead_ns=$leadNs direction=gyro_y_positive")
        status("ROTATION · 같은 방향의 중앙 시작점 대기 중")
    }

    @Synchronized
    fun poseAt(timestampNs: Long): PoseAt? {
        val center = profile?.center ?: return null
        val sample = history.minByOrNull { abs(it.timestampNs - timestampNs) } ?: return null
        return PoseAt(sample.timestampNs, angleDeg(sample.q, center), sample.gyroY)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (closed) return
        val callbackNs = SystemClock.elapsedRealtimeNanos()
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> onGyro(event)
            Sensor.TYPE_GAME_ROTATION_VECTOR -> onRotation(event)
        }
        writeSensor(event, callbackNs)
    }

    @Synchronized
    private fun onGyro(event: SensorEvent) {
        latestGyroY = event.values[1].toDouble()
        smoothedGyroY += 0.08 * (latestGyroY - smoothedGyroY)
        val newSign = when {
            smoothedGyroY > DIRECTION_THRESHOLD -> 1
            smoothedGyroY < -DIRECTION_THRESHOLD -> -1
            else -> 0
        }
        if (newSign != 0 && directionSign != 0 && newSign != directionSign && (learning || profile != null)) {
            val cutoff = event.timestamp - ENDPOINT_SEARCH_NS
            val endpoint = history.asSequence().filter { it.timestampNs >= cutoff }
                .minByOrNull { abs(it.gyroY) }
            if (endpoint != null && (endpoints.lastOrNull()?.timestampNs ?: 0L) < endpoint.timestampNs - 500_000_000L) {
                endpoints += Endpoint(endpoint.timestampNs, directionSign, newSign, endpoint.q)
                if (directionSign == -1 && newSign == 1) lastPositiveEndpointNs = endpoint.timestampNs
                logEvent(endpoint.timestampNs, "endpoint", "from=$directionSign to=$newSign count=${endpoints.size}")
                if (learning) maybeFinishLearning() else if (directionSign == 1 && newSign == -1) {
                    updateCenterOffsetFromLatestPositiveHalf()
                }
            }
        }
        if (newSign != 0) directionSign = newSign
    }

    @Synchronized
    private fun onRotation(event: SensorEvent) {
        val xyzNorm = event.values.take(3).sumOf { it.toDouble() * it }
        val q = normalize(Quaternion(
            event.values[0].toDouble(), event.values[1].toDouble(), event.values[2].toDouble(),
            event.values.getOrNull(3)?.toDouble() ?: sqrt((1.0 - xyzNorm).coerceAtLeast(0.0))
        ))
        val sample = PoseSample(event.timestamp, q, latestGyroY)
        history.addLast(sample)
        while (history.size > MAX_HISTORY) history.removeFirst()
        maybeTrigger(sample)
    }

    @Synchronized
    private fun maybeFinishLearning() {
        val positive = endpoints.filter { it.fromSign == -1 && it.toSign == 1 }
        val negative = endpoints.filter { it.fromSign == 1 && it.toSign == -1 }
        if (positive.size < 2 || negative.size < 2) return
        val a = average(positive.takeLast(2).map { it.q })
        val b = average(negative.takeLast(2).map { it.q })
        val learned = Profile(
            a, b, midpoint(a, b), angleDeg(a, b),
            endpoints.takeLast(4).map { it.timestampNs }.zipWithNext()
                .map { (x, y) -> (y - x) / 1e6 }.average()
        )
        profile = learned
        learning = false
        recomputeCenterOffsets()
        logEvent(SystemClock.elapsedRealtimeNanos(), "profile",
            "range_deg=${learned.rangeDeg} half_period_ms=${learned.halfPeriodMs}")
        File(runDir, "rotation_profile.json").writeText(JSONObject().apply {
            put("range_deg", learned.rangeDeg)
            put("half_period_ms", learned.halfPeriodMs)
            put("trigger_direction", "gyro_y_positive")
            put("endpoint_count", endpoints.size)
        }.toString(2))
        val callback = learningCallback
        learningCallback = null
        callback?.invoke(learned)
    }

    @Synchronized
    private fun maybeTrigger(sample: PoseSample) {
        val target = armed ?: return
        val learned = profile ?: return
        if (smoothedGyroY <= DIRECTION_THRESHOLD || lastPositiveEndpointNs == 0L) return
        val centerOffsetNs = median(positiveCenterOffsetsNs.toList())
            ?: (learned.halfPeriodMs * 500_000.0).toLong()
        val phaseTimeToCenterNs = lastPositiveEndpointNs + centerOffsetNs - sample.timestampNs
        if (phaseTimeToCenterNs < 0L || phaseTimeToCenterNs > target.leadNs) return
        if (phaseTimeToCenterNs < (target.leadNs - TRIGGER_WINDOW_NS).coerceAtLeast(0L)) return
        armed = null
        val trigger = TriggerEvent(
            sample.timestampNs, smoothedGyroY, angleDeg(sample.q, learned.center),
            phaseTimeToCenterNs, target.leadNs
        )
        logEvent(sample.timestampNs, "trigger",
            "center_distance_deg=${trigger.centerDistanceDeg} predicted_time_to_center_ns=$phaseTimeToCenterNs lead_ns=${target.leadNs}")
        target.callback(trigger)
    }

    @Synchronized
    private fun recomputeCenterOffsets() {
        positiveCenterOffsetsNs.clear()
        endpoints.zipWithNext().forEach { (start, end) ->
            if (start.fromSign == -1 && start.toSign == 1 && end.fromSign == 1 && end.toSign == -1) {
                centerOffset(start.timestampNs, end.timestampNs)?.let(::addCenterOffset)
            }
        }
    }

    @Synchronized
    private fun updateCenterOffsetFromLatestPositiveHalf() {
        val end = endpoints.lastOrNull() ?: return
        val start = endpoints.dropLast(1).lastOrNull { it.fromSign == -1 && it.toSign == 1 } ?: return
        centerOffset(start.timestampNs, end.timestampNs)?.let(::addCenterOffset)
    }

    @Synchronized
    private fun centerOffset(startNs: Long, endNs: Long): Long? {
        val center = profile?.center ?: return null
        val closest = history.asSequence().filter { it.timestampNs in startNs..endNs }
            .minByOrNull { angleRad(it.q, center) } ?: return null
        return (closest.timestampNs - startNs).takeIf { it > 0L }
    }

    private fun addCenterOffset(value: Long) {
        positiveCenterOffsetsNs.addLast(value)
        while (positiveCenterOffsetsNs.size > 5) positiveCenterOffsetsNs.removeFirst()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        if (running) manager.unregisterListener(this)
        running = false
        armed = null
        learning = false
        writer.flush()
        writer.close()
        thread.quitSafely()
    }

    @Synchronized
    private fun writeSensor(event: SensorEvent, callbackNs: Long) {
        val v = event.values
        fun f(index: Int) = v.getOrNull(index)?.let { String.format(Locale.US, "%.9f", it) } ?: ""
        writer.append(System.currentTimeMillis().toString()).append(',').append(event.timestamp.toString()).append(',')
            .append(callbackNs.toString()).append(',').append(event.sensor.type.toString()).append(',')
            .append(f(0)).append(',').append(f(1)).append(',').append(f(2)).append(',').append(f(3)).append(',')
            .append(String.format(Locale.US, "%.9f", latestGyroY)).append(",,\n")
    }

    @Synchronized
    private fun logEvent(timestampNs: Long, event: String, detail: String) {
        writer.append(System.currentTimeMillis().toString()).append(',').append(timestampNs.toString()).append(',')
            .append(SystemClock.elapsedRealtimeNanos().toString()).append(",,,,,,,")
            .append(event).append(',').append('"').append(detail.replace("\"", "\"\"")).append('"').append('\n')
        writer.flush()
    }

    private fun average(values: List<Quaternion>): Quaternion {
        val reference = values.first()
        var x = 0.0; var y = 0.0; var z = 0.0; var w = 0.0
        values.forEach { raw ->
            val q = if (dot(raw, reference) < 0) Quaternion(-raw.x, -raw.y, -raw.z, -raw.w) else raw
            x += q.x; y += q.y; z += q.z; w += q.w
        }
        return normalize(Quaternion(x, y, z, w))
    }

    private fun midpoint(a: Quaternion, rawB: Quaternion): Quaternion {
        val b = if (dot(a, rawB) < 0) Quaternion(-rawB.x, -rawB.y, -rawB.z, -rawB.w) else rawB
        return normalize(Quaternion(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w))
    }

    private fun median(values: List<Long>): Long? {
        if (values.isEmpty()) return null
        val sorted = values.sorted(); val middle = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[middle]
        else sorted[middle - 1] / 2L + sorted[middle] / 2L
    }

    private fun angleDeg(a: Quaternion, b: Quaternion) = Math.toDegrees(angleRad(a, b))
    private fun angleRad(a: Quaternion, b: Quaternion) = 2.0 * acos(abs(dot(a, b)).coerceIn(0.0, 1.0))
    private fun dot(a: Quaternion, b: Quaternion) = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
    private fun normalize(q: Quaternion): Quaternion {
        val n = sqrt(dot(q, q)).coerceAtLeast(1e-12)
        return Quaternion(q.x/n, q.y/n, q.z/n, q.w/n)
    }

    companion object {
        private const val DIRECTION_THRESHOLD = 0.08
        private const val ENDPOINT_SEARCH_NS = 600_000_000L
        private const val TRIGGER_WINDOW_NS = 20_000_000L
        private const val MAX_HISTORY = 30_000
    }
}
