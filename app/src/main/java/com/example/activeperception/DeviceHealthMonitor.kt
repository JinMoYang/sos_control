package com.example.activeperception

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import android.os.Process
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

data class DeviceHealth(
    val batteryTemperatureC: Double,
    val thermalStatus: Int,
    val totalPssKb: Int
)

/** Low-rate cached health sample so per-frame logging does not add per-frame binder work. */
class DeviceHealthMonitor(private val context: Context) {
    @Volatile private var lastAtMs = 0L
    @Volatile private var cached = DeviceHealth(Double.NaN, -1, -1)
    private val sampler = Executors.newSingleThreadScheduledExecutor { task ->
        Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            task.run()
        }, "SoS-HealthSampler")
    }

    init {
        sampler.scheduleAtFixedRate({ runCatching { refresh() } },
            0L, 2L, TimeUnit.SECONDS)
    }

    /** Critical-path reads never perform binder/PSS work. */
    fun sample(): DeviceHealth = cached

    @Synchronized
    private fun refresh() {
        val now = SystemClock.elapsedRealtime()
        val battery = context.registerReceiver(null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenthsC = battery?.getIntExtra("temperature", Int.MIN_VALUE) ?: Int.MIN_VALUE
        val thermal = (context.getSystemService(Context.POWER_SERVICE) as PowerManager)
            .currentThermalStatus
        val memory = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
        cached = DeviceHealth(
            if (tenthsC == Int.MIN_VALUE) Double.NaN else tenthsC / 10.0,
            thermal,
            memory.totalPss
        )
        lastAtMs = now
    }

    fun close() { sampler.shutdownNow() }
}
