package com.example.activeperception

import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

/** Requested and metadata-observed Camera2 sensor state. */
data class SensorState(val exposureUs: Long, val iso: Int)

/**
 * Tracks commands independently from Camera2's declared SYNC_MAX_LATENCY.
 *
 * RayNeo X3 Pro reports latency 0 but measurements show 9-10 frames. A captured RAW is
 * therefore eligible only after its CaptureResult matches the command. The class is pure
 * Kotlin so the delayed-state behavior can be unit tested without an Android camera.
 */
class SensorStateScheduler(
    private val exposureToleranceUs: Long = 750L,
    private val isoTolerance: Int = 0,
    private val historyLimit: Int = 64
) {
    data class Command(
        val id: Long,
        val requested: SensorState,
        val requestedAtNs: Long,
        var firstRequestFrame: Long? = null,
        var appliedFrame: Long? = null,
        var appliedTimestampNs: Long? = null
    ) {
        val applyDelayFrames: Long?
            get() = firstRequestFrame?.let { start -> appliedFrame?.minus(start) }
    }

    private val nextId = AtomicLong(1)
    private val pending = ArrayDeque<Command>()
    private val history = ArrayDeque<Command>()

    @Synchronized
    fun enqueue(state: SensorState, nowNs: Long = System.nanoTime()): Command {
        val cmd = Command(nextId.getAndIncrement(), state, nowNs)
        pending.addLast(cmd)
        return cmd
    }

    @Synchronized
    fun markRequestFrame(commandId: Long, frameNumber: Long) {
        pending.firstOrNull { it.id == commandId && it.firstRequestFrame == null }
            ?.firstRequestFrame = frameNumber
    }

    /** Returns the command that became physically active on this result, if any. */
    @Synchronized
    fun observe(frameNumber: Long, timestampNs: Long, actual: SensorState): Command? {
        val match = pending.firstOrNull { matches(it.requested, actual) } ?: return null
        match.appliedFrame = frameNumber
        match.appliedTimestampNs = timestampNs
        while (pending.isNotEmpty()) {
            val completed = pending.removeFirst()
            history.addLast(completed)
            if (completed.id == match.id) break
        }
        while (history.size > historyLimit) history.removeFirst()
        return match
    }

    fun matches(expected: SensorState, actual: SensorState): Boolean =
        abs(expected.exposureUs - actual.exposureUs) <= exposureToleranceUs &&
            abs(expected.iso - actual.iso) <= isoToleranceFor(expected.iso)

    /** HAL sensitivity rounding is proportional (S25 family: 100->99, 400->398, ~0.5-1%),
     *  so a fixed slack that covers ISO 400 starves ISO 1600 and the state never approves.
     *  Scale as 1% of the request, floored at the configured tolerance; exact matching
     *  (tolerance 0) stays exact. */
    private fun isoToleranceFor(requestedIso: Int): Int =
        if (isoTolerance == 0) 0 else maxOf(isoTolerance, requestedIso / 100)

    @Synchronized fun pendingSnapshot(): List<Command> = pending.toList()
    @Synchronized fun historySnapshot(): List<Command> = history.toList()
}
