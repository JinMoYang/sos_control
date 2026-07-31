package com.example.activeperception

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.GridLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import java.io.ByteArrayOutputStream
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.activeperception.acquire.Detection
import com.example.activeperception.acquire.Grid
import com.example.activeperception.acquire.REAL_3x3_INDOOR
import com.example.activeperception.acquire.TfliteYoloDetector

/**
 * UI shell: pick a mode and its parameters, then run it on a worker thread. All measurement
 * behaviour lives in [MeasurementController] and the output layout in [MeasurementLogger].
 *
 * The camera and detector are opened once per Activity session, but a fresh logger and
 * controller are built per run so each lands in its own directory.
 */
class MeasurementActivity : AppCompatActivity() {

    private val grid: Grid = REAL_3x3_INDOOR
    private lateinit var raw: RawSensorCapturer
    private lateinit var detector: TfliteYoloDetector
    private lateinit var sensors: SensorDataManager
    private lateinit var status: TextView
    private lateinit var preview: ImageView
    private lateinit var overlay: OverlayView
    private lateinit var cellText: TextView
    private lateinit var methodGroup: RadioGroup
    private lateinit var cellGridWrap: LinearLayout
    private lateinit var cellGrid: GridLayout
    private lateinit var periodGroup: RadioGroup
    private lateinit var fallbackGroup: RadioGroup
    private lateinit var boostGroup: RadioGroup
    private lateinit var proposedSettings: LinearLayout
    private lateinit var aeSettings: LinearLayout
    private lateinit var aeStrategyGroup: RadioGroup
    private lateinit var confThreshSpinner: Spinner
    private lateinit var offloadCheck: CheckBox
    private lateinit var offloadUrl: EditText
    private lateinit var offloadRegime: Spinner
    private val cellButtons = ArrayList<Button>(16)
    private var selectedCell: Int = 0

    private var mc: MeasurementController? = null
    private var worker: Thread? = null
    private var opened = false

    /** Non-null when the Offload checkbox is set, or when the Activity was launched with an
     *  `--es server_url http://...` extra. */
    private var offloader: OffloadClient? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_measurement)
        status = findViewById(R.id.statusText)
        preview = findViewById(R.id.previewImage)
        overlay = findViewById(R.id.overlayView)
        cellText = findViewById(R.id.cellText)
        methodGroup = findViewById(R.id.methodGroup)
        cellGridWrap = findViewById(R.id.cellGridWrap)
        cellGrid = findViewById(R.id.cellGrid)
        periodGroup = findViewById(R.id.periodGroup)
        fallbackGroup = findViewById(R.id.fallbackGroup)
        boostGroup = findViewById(R.id.boostGroup)
        proposedSettings = findViewById(R.id.proposedSettings)
        aeSettings = findViewById(R.id.aeSettings)
        aeStrategyGroup = findViewById(R.id.aeStrategyGroup)
        confThreshSpinner = findViewById(R.id.confThreshSpinner)
        val confChoices = (1..10).map { "%.2f".format(it * 0.05) }   // 0.05 .. 0.50
        confThreshSpinner.adapter = ArrayAdapter(this,
            android.R.layout.simple_spinner_item, confChoices
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
        confThreshSpinner.setSelection(confChoices.indexOf("0.25").coerceAtLeast(0))
        offloadCheck = findViewById(R.id.offloadCheck)
        offloadUrl = findViewById(R.id.offloadUrl)
        offloadRegime = findViewById(R.id.offloadRegime)
        // Server-side delay-injection profiles; "clear" means no added delay. New labels must
        // be added to the server's regime table too.
        offloadRegime.adapter = ArrayAdapter(this,
            android.R.layout.simple_spinner_item,
            listOf("clear", "wifi", "5g", "lte", "congested")
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
        sensors = SensorDataManager(this)

        // Edge-to-edge on Android 15+. All four sides, because the nav bar moves to the right
        // edge in landscape on some devices. Base padding is captured once so repeated
        // callbacks don't compound it.
        val controlsInner = findViewById<LinearLayout>(R.id.controlsInner)
        val basePad = intArrayOf(
            controlsInner.paddingLeft, controlsInner.paddingTop,
            controlsInner.paddingRight, controlsInner.paddingBottom)
        ViewCompat.setOnApplyWindowInsetsListener(controlsInner) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(
                basePad[0] + bars.left,
                basePad[1] + bars.top,
                basePad[2] + bars.right,
                basePad[3] + bars.bottom)
            insets
        }

        applyOrientation(resources.configuration.orientation)
        // Must precede buildCellGrid so the cell labels render with the right effective ISO.
        grid.digitalBoost = boostFromCheckedId(boostGroup.checkedRadioButtonId)
        buildCellGrid()
        methodGroup.setOnCheckedChangeListener { _, id ->
            cellGridWrap.visibility = if (id == R.id.methodFixed) View.VISIBLE else View.GONE
            proposedSettings.visibility = if (id == R.id.methodProposed) View.VISIBLE else View.GONE
            aeSettings.visibility =
                if (id == R.id.methodAe || id == R.id.methodAeQuant) View.VISIBLE else View.GONE
        }
        // Initial state mirrors the default-checked methodProposed radio.
        cellGridWrap.visibility = View.GONE
        proposedSettings.visibility = View.VISIBLE
        aeSettings.visibility = View.GONE

        // Grid is mutated in place; the rebuild refreshes the effective-ISO labels.
        boostGroup.setOnCheckedChangeListener { _, id ->
            grid.digitalBoost = boostFromCheckedId(id)
            buildCellGrid()
        }

        findViewById<Button>(R.id.btnStart).setOnClickListener { startSelectedMethod() }
        findViewById<Button>(R.id.btnVerify).setOnClickListener {
            start("verify") { mc!!.runVerify(false, ::post, ::showFrame) }
        }
        findViewById<Button>(R.id.btnVerifyProbe).setOnClickListener {
            start("verifyprobe") { mc!!.runVerifyProbe(false, ::post, ::showFrame) }
        }
        findViewById<Button>(R.id.btnBench).setOnClickListener { start("bench") { mc!!.runBench(::post) } }
        findViewById<Button>(R.id.btnIsoDiag).setOnClickListener {
            start("iso_diag") { mc!!.runIsoDiag(onStatus = ::post) }
        }
        findViewById<Button>(R.id.btnStop).setOnClickListener {
            mc?.stop(); post("stopped"); runOnUiThread { overlay.clear(); cellText.text = "—" }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }

        // Without "All files access" the runs land in the app-private dir, which the file
        // manager won't show. Prompt for it so output ends up in Documents/sos/ instead.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
            && !Environment.isExternalStorageManager()) {
            post("Tap Settings → Permissions → Allow All files access for output to appear in My Files")
            runCatching {
                startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.fromParts("package", packageName, null)))
            }.onFailure {
                runCatching { startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
            }
        }
    }

    /** Reflows the root between portrait (preview above controls) and landscape (controls
     *  left, preview right) by swapping LayoutParams rather than reinflating — reinflating
     *  would tear down the camera and detector. View objects keep their state across the
     *  detach and reattach. */
    private fun applyOrientation(orientation: Int) {
        val root = findViewById<LinearLayout>(R.id.rootLayout)
        // Grab the children before removeAllViews; afterwards findViewById returns null.
        val previewWrap = findViewById<android.widget.FrameLayout>(R.id.previewWrap)
        val controls = findViewById<android.widget.ScrollView>(R.id.controls)
        root.removeAllViews()
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            root.orientation = LinearLayout.HORIZONTAL
            root.addView(controls, LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.MATCH_PARENT, 2f))
            root.addView(previewWrap, LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.MATCH_PARENT, 3f))
        } else {
            root.orientation = LinearLayout.VERTICAL
            root.addView(previewWrap, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
            root.addView(controls, LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        applyOrientation(newConfig.orientation)
        // System bars move on rotation, so re-fire the inset listener to follow them.
        ViewCompat.requestApplyInsets(findViewById(R.id.controlsInner))
    }

    private fun boostFromCheckedId(id: Int): Double = when (id) {
        R.id.boost1x -> 1.0
        R.id.boost4x -> 4.0
        else -> 2.0
    }

    private fun aeStrategyFromCheckedId(id: Int): AeStrategy = when (id) {
        R.id.aeCustom -> AeStrategy.CUSTOM_BRIGHTNESS
        else -> AeStrategy.PHONE
    }

    private fun effectiveIso(gainIdx: Int): Int =
        (grid.gains[gainIdx] * grid.digitalBoost).toInt()

    /** Fixed-mode cell selector: rows are gain, columns are shutter. Labels show EFFECTIVE
     *  ISO so the number matches the brightness the formed cell will actually have. Call
     *  again after digitalBoost changes to refresh them. */
    private fun buildCellGrid() {
        cellGrid.removeAllViews()
        cellButtons.clear()
        // Overrides the XML's 3×3 hint so wider grids get enough rows and columns.
        cellGrid.rowCount = grid.nGain
        cellGrid.columnCount = grid.nShutter
        for (gi in 0 until grid.nGain) {
            for (sj in 0 until grid.nShutter) {
                val cell = grid.cell(gi, sj)
                val effIso = effectiveIso(gi); val expUs = grid.exposuresUs[sj]
                // Two lines, no unit words — anything longer stops fitting on a phone.
                val expMs = expUs / 1000.0
                val expLabel = if (expMs >= 1.0) "%.0fms".format(expMs) else "%.1fms".format(expMs)
                val b = Button(this).apply {
                    text = "$effIso\n$expLabel"
                    textSize = 10f
                    minWidth = 0; minimumWidth = 0
                    setPadding(0, 0, 0, 0)
                    setOnClickListener { setSelectedCell(cell) }
                }
                val lp = GridLayout.LayoutParams().apply {
                    width = 0; height = GridLayout.LayoutParams.WRAP_CONTENT
                    columnSpec = GridLayout.spec(sj, 1f); rowSpec = GridLayout.spec(gi, 1f)
                    setMargins(2, 2, 2, 2)
                }
                cellGrid.addView(b, lp)
                cellButtons.add(b)
            }
        }
        setSelectedCell(0)
    }

    private fun setSelectedCell(cell: Int) {
        selectedCell = cell
        val idle = ContextCompat.getColor(this, R.color.cell_unselected)
        val active = ContextCompat.getColor(this, R.color.cell_selected)
        for (b in cellButtons) b.setBackgroundColor(idle)
        cellButtons.getOrNull(cell)?.setBackgroundColor(active)
    }

    private fun startSelectedMethod() {
        // GT-reference checkbox removed from UI; hardcoded false. Re-add a flag plumb
        // if a GT pass mode is needed (or a long-press gesture on Start to toggle).
        val gtRef = false
        when (methodGroup.checkedRadioButtonId) {
            R.id.methodFixed -> {
                val (gi, sj) = grid.indices(selectedCell)
                start("fixed_g${grid.gains[gi]}_e${grid.exposuresUs[sj]}") {
                    mc!!.runFixed(gi, sj, 300, gtRef, ::post, ::onFrameWithOffload)
                }
            }
            R.id.methodAe -> {
                val ae = aeStrategyFromCheckedId(aeStrategyGroup.checkedRadioButtonId)
                start("ae_${ae.tag()}") { mc!!.runAe(300, gtRef, ae, ::post, ::onFrameWithOffload) }
            }
            R.id.methodAeQuant -> {
                val ae = aeStrategyFromCheckedId(aeStrategyGroup.checkedRadioButtonId)
                start("ae_paired_${ae.tag()}") { mc!!.runAeQuant(300, gtRef, ae, ::post, ::onFrameWithOffload) }
            }
            R.id.methodProposed -> {
                val period = when (periodGroup.checkedRadioButtonId) {
                    R.id.period10 -> 10
                    R.id.period15 -> 15
                    R.id.period20 -> 20
                    R.id.period25 -> 25
                    R.id.period30 -> 30
                    else -> 5
                }
                val fallback = when (fallbackGroup.checkedRadioButtonId) {
                    R.id.fbLaplacian -> FallbackMetric.LAPLACIAN_VAR
                    R.id.fbTenengrad -> FallbackMetric.TENENGRAD_NORM
                    R.id.fbCreteRoffet -> FallbackMetric.CRETE_ROFFET
                    R.id.fbSafeCell -> FallbackMetric.SAFE_CELL
                    else -> FallbackMetric.ENTROPY
                }
                start("proposed_p${period}_${fallback.tag()}") {
                    mc!!.runProposed(period, 300, gtRef, fallback,
                        onStatus = ::post, onFrame = ::onFrameWithOffload)
                }
            }
            else -> post("pick a method")
        }
    }

    private fun post(s: String) = runOnUiThread { status.text = s }

    /** Preview plus, if offload is on, an async send of EVERY frame. Sending everything is
     *  deliberate: trigger policies are then evaluated offline as a subset of the collected
     *  logs, instead of re-running the experiment once per policy. */
    private fun onFrameWithOffload(bmp: Bitmap, dets: List<Detection>, iso: Int, expUs: Int) {
        showFrame(bmp, dets, iso, expUs)
        offloader?.let { o ->
            val jpeg = ByteArrayOutputStream()
                .also { bmp.compress(Bitmap.CompressFormat.JPEG, OffloadClient.JPEG_QUALITY, it) }
                .toByteArray()
            o.offload(mc?.lastFrameIdx ?: 0, jpeg)
        }
    }

    /** Detection.xyxy is already in `bmp` pixel space, so the bitmap's own dims are what the
     *  overlay needs to scale against. */
    private fun showFrame(bmp: Bitmap, dets: List<Detection>, iso: Int, expUs: Int) {
        val drawn = dets.map {
            val xy = it.xyxy
            OverlayView.DrawInfo(
                Rect(xy[0].toInt(), xy[1].toInt(), xy[2].toInt(), xy[3].toInt()),
                "${it.classId} %.2f".format(it.confidence))
        }
        val cellLabel = "ISO %d\nexp %.1f ms".format(iso, expUs / 1000.0)
        runOnUiThread {
            preview.setImageBitmap(bmp)
            overlay.setResults(drawn, bmp.width, bmp.height)
            cellText.text = cellLabel
        }
    }

    /** Runs [block] on a worker thread with a fresh logger and controller, so each press gets
     *  its own `run_<modeTag>_<ts>/` directory. Camera and detector are opened once and reused. */
    private fun start(modeTag: String, block: () -> Unit) {
        if (worker?.isAlive == true) { post("busy — stop first"); return }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) { post("no camera permission"); return }
        // All View reads happen here, on the UI thread, before the worker starts.
        val offUrl = (if (offloadCheck.isChecked) offloadUrl.text.toString().trim().ifBlank { null }
                      else null) ?: intent.getStringExtra("server_url")
        val offRegime = (offloadRegime.selectedItem?.toString() ?: "clear").ifBlank { "clear" }
        val selectConf = (confThreshSpinner.selectedItem?.toString()?.toFloatOrNull()) ?: 0.25f
        worker = Thread {
            try {
                if (!opened) {
                    post("opening camera…")
                    raw = RawSensorCapturer(this); raw.open()
                    detector = TfliteYoloDetector(this)
                    opened = true
                }
                val runName = "run_${modeTag}_${System.currentTimeMillis()}"
                val logger = MeasurementLogger(this, runName)
                val mcLocal = MeasurementController(raw, detector, grid, sensors, logger, selectConf)
                mc = mcLocal
                // The currentFrame supplier reads lastFrameIdx, so staleness is measured
                // against the same frame index that frames.csv records.
                offloader = offUrl?.let {
                    post("offload -> $it ($offRegime)")
                    OffloadClient(it, logger.dir, offRegime) { mcLocal.lastFrameIdx }
                        .apply { warmConnection() }
                }
                block()
                // Vehicle classes only — the detector already filters.
                val above = mcLocal.detectionTotalAboveThresh
                val floor = mcLocal.detectionTotalAtFloor
                val frames = mcLocal.totalFramesLogged
                post("done — vehicles: $above above thresh (${floor} at floor), $frames frames")
                android.os.Handler(android.os.Looper.getMainLooper())
                    .postDelayed({ status.text = "" }, 5000)
            } catch (e: Exception) {
                post("error: ${e.message}")
            }
        }.also { it.start() }
    }

    override fun onResume() { super.onResume(); sensors.registerListeners() }
    override fun onPause() {
        super.onPause()
        sensors.unregisterListeners()
        // Backgrounding revokes camera access and nulls our CameraDevice, so the whole stack
        // is torn down here and reopened on the next start().
        mc?.stop()
        if (opened) {
            runCatching { raw.close() }
            runCatching { detector.close() }
            opened = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mc?.stop()
        runCatching { if (opened) { raw.close(); detector.close() } }
        sensors.release()
    }
}
