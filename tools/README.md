# tools

One-off desktop scripts. Not part of the Android build.

## convert_fp16.py

Casts `yolov8n_640.onnx` (FP32, ~12.85 MB) to `yolov8n_640_fp16.onnx`
(~6.4 MB) for use with the ONNX Runtime QNN EP on Snapdragon NPU. I/O tensors
stay FP32 so `OnnxYoloDetector.kt` does not need to change input encoding.

### Run

```sh
pip install onnx onnxconverter-common onnxruntime
python tools/convert_fp16.py
```

The FP16 model is written next to the FP32 model in
`app/src/main/assets/yolov8n_640_fp16.onnx`. After that, in
`OnnxYoloDetector.kt`:

```kotlin
class OnnxYoloDetector(
    context: Context,
    modelAsset: String = "yolov8n_640_fp16.onnx",   // <- swap
    ...
)
```

…and add the QNN EP in the session options block (see step 2 of the latency
plan).
