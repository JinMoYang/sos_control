# SoS — acquire-and-select sensor control

Android measurement app for a camera controller that picks **(gain, exposure) per frame**
instead of deferring to auto-exposure.

Visiting every candidate setting physically is far too slow, so the app captures once at the
base ISO and forms the other cells **virtually** — burst-summing for longer exposure, digitally
re-gaining for higher ISO — then scores all of them in one batched YOLOv8n pass and anchors on
the winner.

It is a **recorder**: it writes RAW, JPEGs, CSV and JSONL. Accuracy analysis happens offline.

---

## Method

```
              exposure  16ms  32ms  64ms        burst frames: 1, 2, 4
      ISO 100    cell 0     1     2               gain ratio: ×1
      ISO 200         3     4     5                           ×2
      ISO 400         6     7     8                           ×4
```

Exposure (columns) costs one burst-sum + demosaic **per row**; gain (rows) is one multiply
**per cell**. So the schedule alternates:

- **probe step** (`t % period == 0`) — all 9 cells from one burst
- **otherwise** — the 3-cell gain column at the anchor's exposure, from one capture

Selection is `Σ conf` above the operating threshold. When nothing is detected anywhere, a
configurable tie-break picks the cell instead of stalling.

> In Proposed mode the sensor stays at base gain — only exposure is actuated. "Chose ISO 400"
> means it chose a digitally re-gained candidate. Verify + VerifyProbe measure whether that is
> a faithful stand-in.

## Modes

| Mode | What it does |
|---|---|
| **Fixed** | Every frame at one selected cell, AE off |
| **AE** | Phone AE, or a deterministic custom AE |
| **AE_quant** | AE's choice snapped to the nearest grid cell, then captured there |
| **Proposed** | The acquire-and-select loop above |
| **Verify** | Physically visits every cell twice, static scene |
| **VerifyProbe** | Forms every cell virtually from one burst, same scene |
| **Bench** | Formation / inference / tie-break latency per K, plus ISO-change lag |
| **IsoDiag** | ISO sweep at fixed exposure — finds where analog gain stops working |

The first four run 300 frames or until **Stop**. Grid and defaults are in `acquire/Grid.kt`
(`REAL_3x3_INDOOR`: ISO 100/200/400 × 16/32/64 ms, digital boost 2×).

---

## Build

```bash
./gradlew assembleDebug            # needs Android SDK; minSdk 29, compileSdk 36
./gradlew testDebugUnitTest        # pure logic under acquire/, no device
```

Needs a back camera with `RAW` + `MANUAL_SENSOR`. Grant **Camera** and **All files access** —
without the latter, runs land in the app-private directory instead of `Documents/sos/`.

Measured on a Galaxy S25 (Snapdragon 8 Elite, Adreno via OpenCL).

## Output

One directory per run: `Documents/sos/run_<mode>_<timestamp>/`

```
manifest.json    grid, method params, detector config, resolutions
summary.json     frame and detection totals
frames.csv       one row per frame (columns = MeasurementController.headers())
dets.jsonl       per-frame detections, including the sub-threshold tail
candidates.csv   per-candidate Σconf                      (Proposed)
imu.csv          one row per sensor sample
img/  raw/       formed bitmaps, uint16 Bayer planes
```

Plus `bench.csv`, `lag.csv`, `iso_diag.csv`, `verify*`, `probe*`, `cloud_dets.jsonl` per mode.
Everything joins on the frame index.

**Four things that will bite you when reading the logs:**

- `gain` is *effective* ISO (physical × `digital_boost`); `iso_req`/`iso_applied` stay physical.
- `iso_req`/`exp_req` describe the **capture**, not the chosen cell — in Proposed these differ.
- Box coordinates live in `manifest.capture_resolution.detector_input_*`, a quarter of the RAW
  dimensions per axis.
- `formation_ms` includes capture in `frames.csv` but not in `bench.csv`.

---

## Detector

YOLOv8n 640 fp16 on the TFLite GPU delegate, falling back to CPU. Three interpreters at batch
1 / 3 / 9 so each K in the schedule runs in one launch. Confidence floor 0.01, NMS 0.45, COCO
vehicle classes `[2, 3, 5, 7]`.

Model provenance — only the `.tflite` files are loaded at runtime:

```
assets/yolov8n_640.onnx              FP32 Ultralytics export, the root checkpoint
  ├─ tools/convert_fp16.py  → assets/yolov8n_640_fp16.onnx      (ORT path, unused)
  └─ Reshape `0` patch      → tools/yolov8n_640_dyn.onnx
                                └─ onnx2tf → assets/*_fp16.tflite  ×3   ← loaded
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
