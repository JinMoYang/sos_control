package com.example.activeperception.acquire

/** RFC-4180-ish CSV serialization: quote fields containing comma/quote/newline, double
 *  inner quotes. Pure + tested — the escaping is the bug-prone part. */
object Csv {
    fun escape(v: Any?): String {
        val s = v?.toString() ?: ""
        return if (s.any { it == ',' || it == '"' || it == '\n' || it == '\r' })
            "\"" + s.replace("\"", "\"\"") + "\""
        else s
    }

    fun row(values: List<Any?>): String = values.joinToString(",") { escape(it) }

    fun parseRow(line: String): List<String> {
        val out = ArrayList<String>()
        val sb = StringBuilder()
        var i = 0
        var inQuotes = false
        while (i < line.length) {
            val c = line[i]
            when {
                inQuotes -> when {
                    c == '"' && i + 1 < line.length && line[i + 1] == '"' -> { sb.append('"'); i++ }
                    c == '"' -> inQuotes = false
                    else -> sb.append(c)
                }
                c == '"' -> inQuotes = true
                c == ',' -> { out.add(sb.toString()); sb.setLength(0) }
                else -> sb.append(c)
            }
            i++
        }
        out.add(sb.toString())
        return out
    }
}

/** Column order per record type, as defined in MEASUREMENT_SPEC.md. */
object LogSchema {
    val L0_LINEARITY = listOf("iso", "channel", "exposure_us", "mean_raw", "black", "white", "saturated")
    val L1_VERIFY = listOf("ts", "scene", "pass", "cell", "gain", "iso_applied", "exp_applied", "black", "white", "raw_path")
    val L2_SWEEP = L1_VERIFY + listOf("frame", "lap", "regime", "yaw_rate", "lux", "accel", "n_det", "sum_conf")
    val L3_LIVE = listOf("ts", "frame", "regime", "lux", "yaw_rate", "chosen_cell", "k", "batch_mode",
                         "formation_ms", "infer_ms", "total_ms", "iso_req", "iso_applied", "exp_req", "exp_applied")
    val L4_LATENCY = listOf("k", "imgsz", "quant", "batch_mode", "formation_ms", "infer_ms_p50", "infer_ms_p95")
}
