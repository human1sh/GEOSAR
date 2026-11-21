"""Real FP32 vs INT8 benchmark with ONNX Runtime static quantization.

    python -m src.edge_benchmark --data-dir ./data --scenes 02_01,03_01 --calib-patches 32 --out-dir outputs/results

Numbers are measured, not assumed. They come from this machine's CPU (CPUExecutionProvider) and say
nothing about Jetson or FPGA hardware. INT8 can be slower than FP32 on CPUs without INT8 acceleration;
the result is reported either way. Calibration uses real SN9 patches (unregistered, georeferenced).
The network is untrained, so agreement/drift measure quantization error only, not task accuracy; the
fraction of pixels predicted as class 1 is reported so a degenerate (constant) output is visible.

Requires: torch, onnx, onnxruntime. Written against the documented onnxruntime.quantization API.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def collect_patches(data_dir, split, scenes, n, gsd=1.2, size=512, seed=0):
    """Real fusion-network inputs (1,4,size,size): optical luminance x3 + SAR, scaled to [0,1]."""
    from .dataset import ScenePair, discover_scenes
    from .geo import grid_transform, load_tiepoints, tiepoints_to_world
    from .registration import prep_optical, prep_sar

    scene_list = [s for s in discover_scenes(data_dir, split) if not scenes or s.scene_id in scenes]
    per_scene = int(np.ceil(n / max(len(scene_list), 1)))
    rng = np.random.default_rng(seed)
    out = []
    for sc in scene_list:
        tp = load_tiepoints(sc.tiepoints)
        got = 0
        with ScenePair(sc) as pair:
            opt_xy, _ = tiepoints_to_world(tp, pair.opt_transform, pair.sar_transform, pair.opt_to_work, pair.sar_to_work)
            for i in rng.permutation(len(tp)):
                if got >= per_scene:
                    break
                t6 = grid_transform(opt_xy[i], size, gsd)
                sar, opt = pair.read_sar(t6, size), pair.read_optical(t6, size)
                if np.isfinite(sar).mean() < 0.6 or np.isfinite(opt[0]).mean() < 0.6:
                    continue
                o = prep_optical(opt)
                s = prep_sar(sar, log_scale=pair.sar.dtypes[0] != "uint8")
                out.append((np.stack([o, o, o, s]).astype(np.float32)[None] / 255.0))
                got += 1
    if not out:
        raise RuntimeError("No calibration patches could be read")
    return out[:n]


def patches_from_examples(npz_path, size=512):
    z = np.load(npz_path)
    arrays = []
    for k in sorted({n.split("__")[0] for n in z.files}):
        opt, sar = z[f"{k}__opt"], z[f"{k}__sar_warped"]
        h, w = opt.shape
        r, c = (h - size) // 2, (w - size) // 2
        o = opt[r:r + size, c:c + size].astype(np.float32) / 255.0
        s = sar[r:r + size, c:c + size].astype(np.float32) / 255.0
        arrays.append(np.stack([o, o, o, s])[None])
    if not arrays:
        raise RuntimeError("examples.npz has no patches")
    return arrays


def export_fp32(path, size=512):
    import torch
    from .fusion_unet import FusionUNet
    model = FusionUNet().eval()
    torch.onnx.export(model, torch.zeros(1, 4, size, size), path, input_names=["fused_input"],
                      output_names=["logits"], opset_version=18)
    return path


def model_size_mb(path):
    """File size INCLUDING external weight files (the torch exporter may store weights in a .data file)."""
    import onnx
    m = onnx.load(path, load_external_data=False)
    total, seen = os.path.getsize(path), set()
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == "location" and e.value not in seen:
                seen.add(e.value)
                total += os.path.getsize(os.path.join(os.path.dirname(path), e.value))
    return round(total / 1e6, 2)


def quantize_int8(fp32_path, int8_path, calib_arrays):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self, arrays):
            self._it = iter([{"fused_input": a} for a in arrays])

        def get_next(self):
            return next(self._it, None)

    quantize_static(fp32_path, int8_path, Reader(calib_arrays), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    return int8_path


def latency_ms(path, x, warmup=5, runs=30):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    for _ in range(warmup):
        s.run(None, {name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(t)), float(np.percentile(t, 90))


def run(out_dir, out_json, arrays, size=512):
    import onnxruntime as ort
    os.makedirs(out_dir, exist_ok=True)
    fp32 = export_fp32(os.path.join(out_dir, "fusion_fp32.onnx"), size)
    int8 = quantize_int8(fp32, os.path.join(out_dir, "fusion_int8.onnx"), arrays)
    x = arrays[0]
    f_med, f_p90 = latency_ms(fp32, x)
    i_med, i_p90 = latency_ms(int8, x)
    sf = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    sq = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    agree, drift, rel, frac1 = [], [], [], []
    for a in arrays:
        lf, lq = sf.run(None, {"fused_input": a})[0], sq.run(None, {"fused_input": a})[0]
        agree.append(float((lf.argmax(1) == lq.argmax(1)).mean()))
        drift.append(float(np.abs(lf - lq).mean()))
        rel.append(float(np.abs(lf - lq).mean() / (np.abs(lf).mean() + 1e-12)))
        frac1.append(float((lf.argmax(1) == 1).mean()))
    out = {
        "device": "CPU (ONNX Runtime CPUExecutionProvider), this machine only",
        "input_shape": list(x.shape),
        "fp32_median_ms": round(f_med, 2), "fp32_p90_ms": round(f_p90, 2),
        "int8_median_ms": round(i_med, 2), "int8_p90_ms": round(i_p90, 2),
        "speedup_fp32_over_int8": round(f_med / i_med, 2),
        "fp32_size_mb": model_size_mb(fp32), "int8_size_mb": model_size_mb(int8),
        "argmax_agreement": round(float(np.mean(agree)), 4),
        "mean_abs_logit_drift": round(float(np.mean(drift)), 5),
        "relative_logit_drift": round(float(np.mean(rel)), 4),
        "fp32_class1_pixel_fraction": round(float(np.mean(frac1)), 4),
        "n_calibration_patches": len(arrays),
        "note": ("Untrained weights: agreement and drift measure quantization error only, not task accuracy. "
                 "If class-1 fraction is ~0 or ~1 the output is degenerate and agreement is uninformative."),
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", default="", help="comma-separated scene ids (default: all)")
    ap.add_argument("--calib-patches", type=int, default=32)
    ap.add_argument("--out-dir", default="outputs/results")
    ap.add_argument("--from-examples", action="store_true", help="use examples.npz from src.evaluate instead of reading data")
    a = ap.parse_args()
    if a.from_examples:
        arrs = patches_from_examples(os.path.join(a.out_dir, "examples.npz"))
    else:
        arrs = collect_patches(a.data_dir, a.split, [s for s in a.scenes.split(",") if s], a.calib_patches)
    run(a.out_dir, os.path.join(a.out_dir, "benchmark.json"), arrs)
