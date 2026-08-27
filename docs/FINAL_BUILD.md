# Final build record

- Built: 2026-08-27 (Asia/Seoul)
- Project: `/Users/silver/Desktop/sos_glass_final`
- APK: `dist/rayneo-sos-final-debug.apk`
- Application ID: `com.sos.rayneox3.final`
- Version: `1.0-final` (`versionCode=1`)
- ABI: `arm64-v8a`
- SHA-256: `6275a58eff191eefb8e42dd6d0bbabdb18a61cc891200596645b4bb2e3f56d7f`

## Completed checks

- `clean` build succeeded.
- Debug unit tests succeeded.
- APK manifest reports the final application ID, label `RayNeo SoS`, minSdk 29 and targetSdk 36.
- APK contains only the selected 640 FP16 COCO5 B=1/B=3/B=9 model assets.
- Experimental GPU probe activity is not exported or packaged from source.
- Existing `/Users/silver/Desktop/sos_glass` experiment project was not modified by finalization.

## Pending because the glass was disconnected

- APK install and launcher discovery
- Camera permission and RAW10 stream open
- OpenCL delegate B=1/B=3/B=9 cold initialization
- Proposed P=5 live K=9→4×K=3 cycle
- Preview orientation and bounding-box alignment
- Metadata-match/log completeness and sustained thermal test
