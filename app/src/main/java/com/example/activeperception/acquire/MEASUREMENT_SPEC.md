# Measurement / Eval app — spec (target: Galaxy S25)

App = a **mode-selectable recorder** + on-device detector. Accuracy analysis is
**offline in Python** (reuses the sim pipeline, CARLA radiance → real RAW). No live
server (8x offline; instantaneous-cloud; bandwidth = send-fraction). **GT is built
post-hoc by the user** from clean frames in the AE / Sweep passes.

Experiments (motion rig, phase, lighting, repeatable track) are **field protocols**;
the app provides the capture primitives + the real-time-only modes.

## Three goals

| Goal | Measures |
|---|---|
| **#1 Probing realism** | digital re-gain & burst ≈ real HW (physical visit ↔ simulated probe) |
| **#2 Sensor controller** | acquire-select beats AE/fixed + on-device feasible (latency, batch) |
| **#3 Router co-design** | MEASURE offload signal vs ORIC, bandwidth @ matched recall |

## Modes (pick at startup)

| Mode | Sensor policy | Captures / does | Serves |
|---|---|---|---|
| **Fixed** | AE off, **UI-selected cell** (ISO×exp) | record at one constant cell | static baseline (real) / physical-visit primitive |
| **AE** | phone real AE on | RAW + AE's chosen ISO/exp per frame | **AE-b** (faithful AE baseline) |
| **Sweep** | all cells/frame (or base RAW + burst) | save all + meta | **substrate** → offline derive **AE-a + any fixed + controller-replay** (same frames) |
| **Probe** | live acquire-select (real-time 8n, closed loop) | emit + log per-candidate latency/Σconf | deployed controller + systems (#2) |
| **Verify** | per cell: physical-visit ×2 (null) + sim-base, **settled** | paired capture | #1 probing realism (physical ↔ simulated) |
| **Bench** | — | K × imgsz × INT8 × batch/loop latency | on-device B-budget (#2) |

`AE-a` is **derived from Sweep** (metering → which cell AE would pick), not a separate
capture. To get both AE paths: run **Sweep** (→AE-a) and **AE** (→AE-b).

## #2 comparison map

```
controller (Probe live + Sweep replay)  vs
 ├─ AE-a   (from Sweep, same frames, approx)
 ├─ AE-b   (AE mode, real phone AE, faithful)
 └─ fixed  (from Sweep: per-scene-best / global-best)
```
On a repeatable track, Sweep / AE / Probe passes align by position. Fixed pass is an
optional real-static sanity check.

## Capture primitive + features

- **`captureRaw(exposureUs, iso, nBurst)`** + save RAW (DNG or raw16+JSON) + metadata.
  Every mode is this primitive sequenced differently. Per frame log:
  applied-vs-requested ISO/exp (`CaptureResult`), **dynamic black level**, white level,
  CFA pattern, `SENSOR_TIMESTAMP`.
- **Settling filter** — accept only when `applied == requested`; else log + drop.
- **Linearity pre-exp** — flat target, exposure ramp (per ISO, per CFA channel) →
  mean(RAW−black) vs exposure → linear range. (G6 prerequisite.)
- **YOLOv8n online** — NCNN(Vulkan) or ONNX-RT(QNN/NNAPI); `detectBatch` supports
  true-batch AND K-loop (batching is a Bench measurement, not an assumption);
  imgsz/INT8/delegate/K configurable.
- **Formation** (RAW→demosaic→digital re-gain→bitmap) — ✅ pure math (kotlinc).
- **IMU regime** — `RegimeClassifier` on yaw-rate (`verticalGyroSpeedRad`), hysteresis.
  Verify camera/IMU clock (`SENSOR_INFO_TIMESTAMP_SOURCE`).

## Logging (record types — `LogSchema`)

- **L0 linearity**: iso, channel, exposure_us, mean_raw, black, white, saturated
- **L1 verify**: ts, scene, pass(phys1|phys2|sim_base|burst), cell, gain, iso/exp_applied, black, white, raw_path
- **L2 sweep/fixed/AE**: L1 + frame, lap, regime, yaw_rate, lux, accel (AE/Fixed log the policy's applied cell)
- **L3 live (Probe)**: ts, frame, regime, lux, yaw_rate, chosen_cell, k, batch_mode, formation_ms, infer_ms, total_ms, iso/exp req+applied
- **L4 bench**: k, imgsz, quant, batch_mode, formation_ms, infer_ms_p50, infer_ms_p95

CSV via `Csv` (RFC-4180 escaping). Common clock for all `ts`.

## Tests

**Pure (kotlinc, verified here — 20 checks):** Grid, plan, `AcquireSelectController`
(via `detectBatch`, argmax/anchor/empty), Formation (re-gain/burst/sRGB/demosaic/pack),
`RegimeClassifier` (hysteresis/no-flicker/abs), `Csv`/`LogSchema` (round-trip/escaping/stability).

**Device (JUnit instrumented / manual):** RAW16 read (stride/dynamic-black/white/CFA),
settling filter, detector inference + `detectBatch` batch==loop equivalence, Bench
harness, file save.

## Offline (Python — sim infra, not app)

RAW→bitmap → 8n + 8x → recall / C-tax / router(send-fraction) / GT(pseudo-oracle
8x+ByteTrack, exposure-invariant transfer). #1 = rank-corr(reward_sim, reward_physical)
vs physical-vs-physical null, on decision-relevant (non-flat) frames.
**⚠️ offline 8n MUST be the deployed model** (export the exact INT8 NCNN/ONNX and run
THAT offline) or characterize the full-precision↔INT8 gap — else #2 numbers aren't the
phone's. (audit gap C.)

## Capture-strategy / storage

RAW16 × cells × frames = GBs. Default Sweep: base RAW + burst per frame (form gain
cells offline digitally — valid iff #1 passes); cap frames/run. Full physical sweep
only for Verify and as the #1-fail fallback.

## Status

✅ pure core (Grid, Controller/detectBatch, Formation, RegimeClassifier, LogRecord)
— implemented + kotlinc-verified. ⬜ device: captureRaw + mode selector (Fixed/AE/
Sweep/Probe/Verify/Bench), RAW16/settling/black-white, YOLOv8n (NCNN/ONNX) +
detectBatch impl, Bench harness, linearity pre-exp. Replace `MlKitObjectDetector`.
