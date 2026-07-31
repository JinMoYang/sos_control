package com.example.activeperception

import android.util.Log
import okhttp3.Call
import okhttp3.Callback
import okhttp3.ConnectionPool
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Async cloud offload: JPEG-encode a frame, POST it to the lab GPU detect server, and log
 * the round-trip plus staleness when the reply lands. Never blocks the capture loop.
 *
 * Writes three files into the run dir: `offload_log.csv` (one row per attempt),
 * `cloud_dets.jsonl` (raw boxes/confs per OK response, for offline accuracy + trigger-policy
 * ablation), and `offload_meta.json` (the config the CSV rows omit).
 *
 * OkHttp callbacks run on its dispatcher pool, so both file appends are lock-guarded.
 */
class OffloadClient(
    private val serverUrl: String,        // e.g. "http://192.168.50.10:8000"
    logDir: File,                         // same run dir as frames.csv
    private val regime: String,           // network regime label; server applies the delay
    private val currentFrame: () -> Int   // live frame id, for staleness
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .connectionPool(ConnectionPool(4, 5, TimeUnit.MINUTES))
        .build()
    private val octet = "application/octet-stream".toMediaType()

    private val inFlight = AtomicInteger(0)
    var maxInFlight = 2                    // backpressure cap on outstanding requests

    private val logFile = File(logDir, "offload_log.csv")
    private val logLock = Any()
    private val detsFile = File(logDir, "cloud_dets.jsonl")
    private val detsLock = Any()
    private val metaFile = File(logDir, "offload_meta.json")

    init {
        // injected_ms = server-applied artificial delay for the regime; regime = the label
        // echoed back by the server, which may differ from the one requested.
        if (!logFile.exists()) logFile.writeText(
            "sent_epoch_ms,recv_epoch_ms,frame_id,current_frame,staleness," +
            "bytes,roundtrip_ms,server_ms,infer_ms,injected_ms,regime,n_cloud_det,ok\n")
        if (!detsFile.exists()) detsFile.createNewFile()
        writeMeta()
    }

    private fun writeMeta() {
        val meta = JSONObject()
            .put("server_url", serverUrl)
            .put("regime", regime)
            .put("max_in_flight", maxInFlight)
            .put("jpeg_quality", JPEG_QUALITY)
            .put("connect_timeout_s", 2)
            .put("read_timeout_s", 10)
            .put("connection_pool_max", 4)
        metaFile.writeText(meta.toString(2))
    }

    data class CloudResult(
        val frameId: Int, val boxes: List<FloatArray>, val confs: List<Float>,
        val serverMs: Double, val roundtripMs: Double, val staleness: Int
    )

    /** Prime the keep-alive connection (call once, off the UI thread). */
    fun warmConnection() {
        try {
            client.newCall(Request.Builder().url("$serverUrl/health").build())
                .execute().use { /* ignore */ }
        } catch (_: IOException) { Log.w("Offload", "server not reachable at $serverUrl") }
    }

    /**
     * Fire-and-forget offload. Returns false if dropped by backpressure.
     * [onResult] runs on a background thread ~k frames later; the CSV row is logged either way.
     */
    fun offload(frameId: Int, jpeg: ByteArray, onResult: ((CloudResult) -> Unit)? = null): Boolean {
        if (inFlight.get() >= maxInFlight) {
            logRow(System.currentTimeMillis(), 0, frameId, currentFrame(), -1,
                   jpeg.size, 0.0, 0.0, 0.0, 0.0, regime, 0, "dropped")
            return false
        }
        inFlight.incrementAndGet()
        val sentMs = System.currentTimeMillis()
        val tSend = System.nanoTime()
        val req = Request.Builder()
            .url("$serverUrl/detect")
            .header("X-Frame-Id", frameId.toString())
            .header("X-Regime", regime)
            .post(jpeg.toRequestBody(octet))
            .build()
        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                inFlight.decrementAndGet()
                logRow(sentMs, System.currentTimeMillis(), frameId, currentFrame(), -1,
                       jpeg.size, 0.0, 0.0, 0.0, 0.0, regime, 0, "fail")
            }
            override fun onResponse(call: Call, response: Response) {
                // The decrement MUST be in a finally. Anything thrown while parsing the body
                // (non-JSON error page, malformed box array) would otherwise leak a permit;
                // after maxInFlight such responses the backpressure check refuses every later
                // frame and offload stops for the rest of the run, silently.
                try {
                    response.use { resp ->
                        val tRecv = System.nanoTime()
                        val rtMs = (tRecv - tSend) / 1e6
                        val recvMs = System.currentTimeMillis()
                        val j = JSONObject(resp.body?.string() ?: "{}")
                        val ba = j.optJSONArray("boxes"); val ca = j.optJSONArray("confs")
                        val cla = j.optJSONArray("classes") ?: j.optJSONArray("class_ids")
                        val boxes = ArrayList<FloatArray>(); val confs = ArrayList<Float>()
                        if (ba != null) for (i in 0 until ba.length()) {
                            val b = ba.getJSONArray(i)
                            boxes.add(floatArrayOf(b.getDouble(0).toFloat(), b.getDouble(1).toFloat(),
                                                   b.getDouble(2).toFloat(), b.getDouble(3).toFloat()))
                            confs.add(ca?.getDouble(i)?.toFloat() ?: 0f)
                        }
                        val cur = currentFrame()
                        val staleness = cur - frameId
                        val srv = j.optDouble("server_ms", 0.0)
                        val injected = j.optDouble("injected_ms", 0.0)
                        // Logged separately from the requested regime so the two can be
                        // cross-checked offline.
                        val regimeEcho = j.optString("regime", "")
                        logRow(sentMs, recvMs, frameId, cur, staleness,
                               jpeg.size, rtMs, srv, j.optDouble("infer_ms", 0.0),
                               injected, regimeEcho.ifBlank { regime }, boxes.size, "ok")
                        // Raw JSONArrays pass straight through to preserve the server's
                        // coordinate space and class-id encoding.
                        val rec = JSONObject()
                            .put("frame_id", frameId)
                            .put("sent_ms", sentMs)
                            .put("recv_ms", recvMs)
                            .put("staleness", staleness)
                            .put("server_ms", srv)
                            .put("roundtrip_ms", rtMs)
                            .put("injected_ms", injected)
                            .put("regime", regimeEcho.ifBlank { regime })
                            .put("boxes", ba ?: JSONArray())
                            .put("confs", ca ?: JSONArray())
                            .put("class_ids", cla ?: JSONArray())
                        appendDetsLine(rec.toString())
                        onResult?.invoke(CloudResult(frameId, boxes, confs, srv, rtMs, staleness))
                    }
                } catch (e: Exception) {
                    // Reached the server but the reply was unusable. Distinct from "fail"
                    // (transport error) so the two are separable in offload_log.csv.
                    Log.w("Offload", "bad response for frame $frameId: ${e.message}")
                    logRow(sentMs, System.currentTimeMillis(), frameId, currentFrame(), -1,
                           jpeg.size, 0.0, 0.0, 0.0, 0.0, regime, 0, "bad_response")
                } finally {
                    inFlight.decrementAndGet()
                }
            }
        })
        return true
    }

    private fun logRow(sent: Long, recv: Long, fid: Int, cur: Int, stale: Int,
                       bytes: Int, rt: Double, srv: Double, inf: Double,
                       injected: Double, regimeStr: String, ndet: Int, ok: String) {
        synchronized(logLock) {
            logFile.appendText(
                "$sent,$recv,$fid,$cur,$stale,$bytes,${"%.2f".format(rt)}," +
                "${"%.2f".format(srv)},${"%.2f".format(inf)},${"%.2f".format(injected)}," +
                "$regimeStr,$ndet,$ok\n")
        }
    }

    private fun appendDetsLine(jsonLine: String) {
        synchronized(detsLock) { detsFile.appendText("$jsonLine\n") }
    }

    companion object {
        /** Applied by the caller when encoding; kept here so it is also recorded in
         *  offload_meta.json from a single source. */
        const val JPEG_QUALITY = 85
    }
}

/** Offload policy stub. Currently unused — the app offloads every frame and trigger
 *  policies are evaluated offline as a frame subset over the collected logs. */
object OffloadPolicy {
    /** Offload when the local detector is NOT confident (model-limited -> cloud). */
    fun shouldOffload(localSumConf: Double, threshold: Double = 0.25): Boolean =
        localSumConf < threshold
}
