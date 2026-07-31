#!/usr/bin/env python3
"""
Cast yolov8n_640.onnx (FP32) -> yolov8n_640_fp16.onnx, keeping I/O as FP32 so
the on-device detector code does not need to change. Run once on desktop:

    pip install onnx onnxconverter-common onnxruntime
    python tools/convert_fp16.py

Output lands next to the input in app/src/main/assets/. Then update
OnnxYoloDetector's modelAsset arg to "yolov8n_640_fp16.onnx" and add the QNN EP.
"""
import argparse
import os
import sys

import onnx
from onnx import shape_inference
from onnxconverter_common import float16


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.normpath(os.path.join(here, "..", "app", "src", "main", "assets"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=os.path.join(assets, "yolov8n_640.onnx"))
    ap.add_argument("--out", dest="dst", default=os.path.join(assets, "yolov8n_640_fp16.onnx"))
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"missing input: {args.src}", file=sys.stderr)
        return 1

    print(f"loading {args.src} ({os.path.getsize(args.src)/1e6:.2f} MB)")
    model = onnx.load(args.src)

    # Shape inference helps the FP16 cast handle dynamic axes cleanly.
    model = shape_inference.infer_shapes(model)

    # keep_io_types=True -> external inputs/outputs stay FP32 (the Kotlin code
    # builds FloatBuffer FP32 tensors; internal weights/activations become FP16).
    print("casting weights/activations to FP16, keeping I/O as FP32...")
    fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )

    # The converter sometimes leaves DUPLICATE value_info entries for cast outputs
    # (one FP16 from the new cast, one FP32 left over from the original Resize). ORT
    # picks the wrong one and refuses to load ("tensor(float) does not match
    # expected tensor(float16)"). Dedupe: per name, prefer the entry whose elem_type
    # matches what the producing node actually outputs.
    node_out_to_dtype = {}
    for n in fp16.graph.node:
        if n.op_type == "Cast":
            to = next((a.i for a in n.attribute if a.name == "to"), None)
            if to is not None and len(n.output) > 0:
                node_out_to_dtype[n.output[0]] = to
    seen, kept = set(), []
    for vi in fp16.graph.value_info:
        if vi.name in seen:
            continue
        expected = node_out_to_dtype.get(vi.name)
        actual = vi.type.tensor_type.elem_type
        if expected is not None and actual != expected:
            continue  # stale entry; the dedupe pass below will pick the correct one
        kept.append(vi)
        seen.add(vi.name)
    # Second pass: for cast outputs we skipped above, re-add a single corrected entry.
    for vi in fp16.graph.value_info:
        if vi.name in seen:
            continue
        expected = node_out_to_dtype.get(vi.name)
        if expected is not None:
            vi.type.tensor_type.elem_type = expected
        kept.append(vi)
        seen.add(vi.name)
    del fp16.graph.value_info[:]
    fp16.graph.value_info.extend(kept)
    print(f"deduped value_info: {len(kept)} entries kept")

    print("repairing remaining shapes via shape_inference.infer_shapes(strict_mode=False)...")
    fp16 = shape_inference.infer_shapes(fp16, strict_mode=False, check_type=False)

    onnx.save(fp16, args.dst)
    out_mb = os.path.getsize(args.dst) / 1e6
    print(f"wrote {args.dst} ({out_mb:.2f} MB)")

    # Smoke-test: load + check expected I/O signature ([1,3,640,640] -> [1,84,8400]).
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"input  : {inp.name} {inp.shape} {inp.type}")
        print(f"output : {out.name} {out.shape} {out.type}")
        if list(inp.shape) != [1, 3, 640, 640]:
            print(f"WARN: input shape != [1,3,640,640]")
        if list(out.shape) != [1, 84, 8400]:
            print(f"WARN: output shape != [1,84,8400]")
    except ImportError:
        print("(skipping smoke test: install onnxruntime to verify)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
