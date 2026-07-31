# acquire-and-select — integration guide (ATI app)

Successor to the `Grid2DRLAgent` (CMAB predict-from-state). Instead of learning a
state→setting policy, each frame it **acquires candidate exposures from one RAW
capture and selects the one the detector reads best** (MEASURE, not PREDICT).

## What is verified vs not

- **Pure core** (`Grid`, `Controller`/`plan`, `Formation`) — compiled and tested
  with `kotlinc` on the JVM (26 checks pass); a faithful port of the tested Python
  `port/sensor_control_v2`. JUnit mirror in `src/test/.../acquire/AcquireCoreTest.kt`
  (`./gradlew testDebugUnitTest`).
- **Android glue** (`RawCandidateSource`, `MlKitObjectDetector`) — code only, NOT
  run here (Bitmap/ML Kit/Camera2 are device-side). The per-pixel math they call
  (`Formation`) is tested.
- **`RawCapturer`** — the one piece you implement on Camera2 RAW; verify on device.

## Pieces

| file | role | status |
|---|---|---|
| `Grid.kt` | gain/shutter grid: ratios, burst counts, cell↔idx | verified |
| `Controller.kt` | `plan()` schedule + `AcquireSelectController` (Σconf argmax + anchor) | verified |
| `Formation.kt` | burst-sum, digital re-gain, demosaic, sRGB, ARGB pack | verified |
| `RawCandidateSource.kt` | `CandidateSource<Bitmap>`: RAW → candidates (digital re-gain) | code-only |
| `MlKitObjectDetector.kt` | `Detector<Bitmap>` via ML Kit | code-only |
| **`RawCapturer` (TODO)** | Camera2 RAW16 capture → `RawFrame` | **you implement** |

## The one thing to implement: `RawCapturer` on Camera2

```kotlin
interface RawCapturer { fun capture(exposureUs: Int, iso: Int, nBurst: Int): List<RawFrame> }
```

In `CameraController.kt`:

1. **Check RAW support** (once, at open):
   ```kotlin
   val caps = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)!!
   val hasRaw = caps.contains(
       CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
   // also confirm RAW_SENSOR is in the stream config map.
   ```
   Galaxy S9 reports RAW (Camera2 level FULL). If `!hasRaw`, fall back to the
   physical-burst path (below).

2. **RAW ImageReader** (replace/augment the YUV reader at CameraController:510):
   ```kotlin
   val sz = map.getOutputSizes(ImageFormat.RAW_SENSOR).maxByOrNull { it.width * it.height }!!
   rawReader = ImageReader.newInstance(sz.width, sz.height, ImageFormat.RAW_SENSOR, /*maxImages*/ grid.maxBurst + 1)
   ```

3. **Capture `nBurst` frames** at manual ISO/exposure (AE/AWB off — you already do
   AE_OFF + SENSOR_SENSITIVITY/EXPOSURE_TIME at CameraController:630). Use
   `captureSession.captureBurst(requests, cb, handler)` with `nBurst` identical
   requests, collect the RAW images.

4. **RAW16 → linear Bayer IntArray** (the bug-prone bit — verify on device):
   ```kotlin
   val plane = image.planes[0]
   val buf = plane.buffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
   val rowStride = plane.rowStride / 2            // shorts per row (>= width)
   val black = characteristics.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
   val whiteLevel = characteristics.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL)!!
   val blackAvg = /* average of the 4 CFA black levels */ 64
   val bayer = IntArray(w * h)
   for (y in 0 until h) for (x in 0 until w) {
       val v = (buf.get(y * rowStride + x).toInt() and 0xFFFF) - blackAvg
       bayer[y * w + x] = if (v < 0) 0 else v
   }
   val maxDn = (whiteLevel - blackAvg).toDouble()
   ```
   CFA pattern: `characteristics.get(SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)` →
   map {0:"RGGB",1:"GRBG",2:"GBRG",3:"BGGR"} into `RawFrame.cfaPattern`.

5. Return `List<RawFrame>(bayer, w, h, cfaPattern, maxDn)`.

## Wire the loop (MainActivity, new "AcquireSelect" mode)

Run on the existing background handler (the detector blocks):
```kotlin
val grid = REAL_4x4
val ctrl = AcquireSelectController(
    RawCandidateSource(grid, cameraController.rawCapturer),
    MlKitObjectDetector(), grid, period = 5, initAnchor = grid.cell(0, 0))
backgroundHandler.post(object : Runnable {
    override fun run() {
        val r = ctrl.step()
        // r.detections = this frame's output; r.cell = chosen (gain, shutter)
        overlay.show(r.detections); log(r)
        backgroundHandler.post(this)
    }
})
```
Keep the existing modes (passive/RL/grid) intact — add this as a new switch so the
MobiSys artifact still works.

## Decisions

- **ISO candidates: digital re-gain (this code) vs physical burst.** Digital
  (capture once at base ISO, scale in `Formation`) is canonical and lagless but
  needs RAW. If your device lacks RAW, implement `RawCapturer` as a physical burst
  with per-request `SENSOR_SENSITIVITY` returning YUV→bitmap candidates instead —
  the loop is unchanged, but it is the slower, optimistic "true-cell" path.
- **Grid ranges (S9).** Set `REAL_4x4` from `SENSOR_INFO_SENSITIVITY_RANGE` (S9 ≈
  50–800) and `SENSOR_INFO_EXPOSURE_TIME_RANGE`. Exposure ratios should be 1:2:4:8
  so burst counts are 1,2,4,8.
- **Detector.** ML Kit gives coarse categories; for the vehicle task swap a custom
  TFLite detector behind the same `Detector<Bitmap>` interface. Σconf selection is
  detector-agnostic.
