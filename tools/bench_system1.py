#!/usr/bin/env python3
"""System 1 time split on this device: preprocess (6 raw frames -> normalized
tensor), backbone, scorer head, postprocess. bf16 autocast (production).
Run after `da-models pull system1@<v>` and `da-models build system1@<v>`:

    python3 tools/bench_system1.py --model-dir /opt/driveragent/models/system1/1.0.1

Input: 6 seeded random uint8 frames (1600x900, the checkpoint raw size) and the
vehicle calibration from $DRIVERAGENT_ROOT/calibration (identity if missing).
Note: run `sudo jetson_clocks` first for fixed-clock figures.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--loops", type=int, default=40)
a = ap.parse_args()
V = Path(a.model_dir)
sys.path.insert(0, str(V / "runtime"))
sys.path.insert(0, str(V / "runtime" / "system1"))
from system1.run_system1 import load_model, preprocess_images  # noqa: E402

dev = "cuda"
model, pre = load_model(str(V / "system1_deploy.pth"), device=dev)
model.eval()
try:
    from runner.calibration import load_all
    proj, wh = load_all()
    proj, wh = torch.as_tensor(proj, dtype=torch.float32), torch.as_tensor(wh, dtype=torch.float32)
    calib = "vehicle"
except Exception:
    proj, wh, calib = torch.eye(4).expand(6, 4, 4).clone(), torch.tensor([[704.0, 256.0]]).expand(6, 2).clone(), "identity"
proj = proj.reshape(1, 6, 4, 4).to(dev)
wh = wh.reshape(1, 6, 2).to(dev)
metas = {"projection_mat": proj, "image_wh": wh}
ego = torch.zeros(1, 8, device=dev)
g = torch.Generator().manual_seed(0)
H, W = pre["raw_size"]
raw = torch.randint(0, 256, (6, 3, H, W), dtype=torch.uint8, generator=g)

names = ("preprocess", "backbone", "scorer_head", "postprocess", "total")
acc = {n: [] for n in names}
with torch.no_grad():
    for i in range(a.loops + 5):
        t0 = time.perf_counter()
        x = preprocess_images(raw, pre, device=dev); torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = model.backbone(x); torch.cuda.synchronize(); t2 = time.perf_counter()
            o, _ = model.scorer(f, ego, metas, None, scene_ctx=None); torch.cuda.synchronize()
        t3 = time.perf_counter()
        _ = o["trajectory"][0].float().cpu(); t4 = time.perf_counter()
        if i >= 5:
            for n, v in zip(names, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                acc[n].append(v * 1000)
res = {n: {"mean": round(float(np.mean(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)} for n, v in acc.items()}
res["calibration"] = calib
res["max_cuda_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
print(json.dumps(res, indent=1))
