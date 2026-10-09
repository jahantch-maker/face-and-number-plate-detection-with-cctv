"""Make an exported plate-reader model loadable by onnxruntime.

Newer TensorFlow writes the GELU activation with an "Erfc" step that ONNX does
not have. This rewrites every Erfc(x) as 1 - Erf(x) (the same maths) and then
checks the model loads and reads like the Keras original.

    python training/fix_onnx.py <model.onnx> [--keras best.keras --plate-config plate_config.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def replace_erfc(model: onnx.ModelProto) -> int:
    g = model.graph
    nodes, n = [], 0
    for node in g.node:
        if node.op_type != "Erfc":
            nodes.append(node)
            continue
        n += 1
        erf_out = node.output[0] + "_erf"
        one = node.output[0] + "_one"
        g.initializer.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), one))
        nodes.append(helper.make_node("Erf", [node.input[0]], [erf_out], name=node.name + "_erf"))
        nodes.append(helper.make_node("Sub", [one, erf_out], [node.output[0]], name=node.name + "_sub"))
    del g.node[:]
    g.node.extend(nodes)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--keras", type=Path, help="compare against this Keras model")
    ap.add_argument("--plate-config", type=Path, help="the plate settings file (needed with --keras)")
    args = ap.parse_args()
    model = onnx.load(str(args.model))
    n = replace_erfc(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.model))
    print(f"{args.model}: replaced {n} Erfc steps")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [1 if not isinstance(d, int) else d for d in inp.shape]
    x = np.random.default_rng(0).integers(0, 256, size=shape).astype(np.uint8)
    out = sess.run(None, {inp.name: x})[0]
    print("loads in onnxruntime: yes, output", out.shape)
    if args.keras:
        from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
        from fast_plate_ocr.train.utilities.utils import load_keras_model
        km = load_keras_model(args.keras, load_plate_config_from_yaml(args.plate_config))
        ref = km.predict(x.astype(np.float32), verbose=0)
        ref = ref[0] if isinstance(ref, (list, tuple)) else (next(iter(ref.values())) if isinstance(ref, dict) else ref)
        diff = float(np.abs(np.asarray(ref).reshape(out.shape) - out).max())
        print(f"largest difference from the Keras model: {diff:.6f}", "(OK)" if diff < 1e-3 else "(TOO BIG)")
        if diff >= 1e-3:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
