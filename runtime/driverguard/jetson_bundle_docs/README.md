# Jetson AGX Orin deployment bundle — DTCP + YOLOPX

Everything you need to deploy the combined DTCP planner + YOLOPX perception
stack to a Jetson AGX Orin, packaged into one folder.

The PC reference pipeline this bundle was extracted from lives at
`/home/tonyho/development/combined_inference/` — its full-val output MP4
(`output/all_val_combined.mp4`, 7.7 min) is the visual ground truth your
Jetson port should match.

---

## Tree

```
jetson_bundle/
├── README.md                          ← you are here
│
├── docs/                              ← background reading (read once)
│   ├── DTCP_deployment.md             ← DTCP architecture, conventions, caveats
│   ├── YOLOPX_deployment.md           ← YOLOPX classes, preprocessing, TRT recipe
│   └── jetson_deployment_workplan.md  ← step-by-step plan from §0 to §13
│
├── onnx/                              ← pre-built ONNX (validated against PyTorch)
│   ├── dtcp_v1.onnx           (97 MB)
│   └── yolopx_v2.onnx         (132 MB)
│
├── weights/                           ← original .pth (backup; re-export source)
│   ├── dtcp_nusc_route_v1.pt  (103 MB)
│   └── yolopx_v2_epoch30.pth  (350 MB)
│
├── pc_export_scripts/                 ← if you need to re-export ONNX on a PC
│   ├── export_dtcp_onnx.py
│   └── export_yolopx_onnx.py
│
├── source/                            ← arch source (needed by export scripts only)
│   ├── dtcp/                          ← model.py, resnet.py, dtcp_infer.py
│   └── yolopx/lib/                    ← full YOLOPX lib tree
│
├── jetson_runtime/                    ← what actually runs ON the Jetson
│   ├── trt_runner.py                  ← TRT engine wrapper
│   ├── preprocess.py                  ← image preprocessing for both engines
│   ├── yolopx_postprocess.py          ← numpy NMS + scale_coords + mask un-pad
│   ├── beta_mode.py                   ← deterministic Beta-mode action
│   ├── viz_helpers.py                 ← waypoint projection, mask blend, box drawing
│   └── jetson_pipeline.py             ← main entry point
│
└── samples/                           ← parity-check fixtures
    ├── cam_front/                     ← 3 nuScenes JPGs
    ├── scene_manifest_subset.json     ← matching speed/cmd/target per frame
    └── reference_yolopx/              ← PC's YOLOPX outputs on these frames (.npz)
```

---

## Quick start on the Jetson

```bash
# 1) Copy the bundle to the Jetson
scp -r jetson_bundle/ jetson:~/

# 2) On the Jetson — build engines (one-time, takes a few minutes each)
mkdir -p ~/jetson_bundle/engines
/usr/src/tensorrt/bin/trtexec \
    --onnx=~/jetson_bundle/onnx/yolopx_v2.onnx \
    --saveEngine=~/jetson_bundle/engines/yolopx_v2_fp16.engine \
    --fp16 --workspace=4096 \
    --shapes=image:1x3x384x640

/usr/src/tensorrt/bin/trtexec \
    --onnx=~/jetson_bundle/onnx/dtcp_v1.onnx \
    --saveEngine=~/jetson_bundle/engines/dtcp_v1_fp16.engine \
    --fp16 --workspace=4096 \
    --shapes=image:1x3x256x928,state:1x9,target_point:1x2

# 3) Install Python deps (use NVIDIA's pip index for tensorrt/pycuda on JetPack)
pip install --user numpy opencv-python pillow matplotlib

# 4) Run the smoke test on the 3 bundled sample frames
cd ~/jetson_bundle/jetson_runtime
python jetson_pipeline.py \
    --yolopx-engine ~/jetson_bundle/engines/yolopx_v2_fp16.engine \
    --dtcp-engine   ~/jetson_bundle/engines/dtcp_v1_fp16.engine \
    --frames-dir    ~/jetson_bundle/samples/cam_front \
    --manifest      ~/jetson_bundle/samples/scene_manifest_subset.json \
    --out-dir       ~/jetson_out \
    --mp4
```

If everything works, you'll get 3 PNGs and a tiny MP4 in `~/jetson_out/`,
visually identical to the corresponding frames in the PC reference video.

---

## What lives where

### Want to run inference?
→ `jetson_runtime/jetson_pipeline.py`. Edit `--frames-dir` / `--manifest` to
point at your data. Everything else in `jetson_runtime/` is imported by it
and does not need to be invoked directly.

### ONNX is out of date or needs different input shape?
→ Re-export on a PC: install the bundle on the PC, then
`python pc_export_scripts/export_dtcp_onnx.py --img-h N --img-w M`
(adjust shape args). The export script imports from `source/dtcp/`. Do not
expect the export to work on the Jetson — it needs the full PyTorch +
training-time stack.

### Want to verify the engine matches the PC reference?
→ Run the pipeline on `samples/cam_front/` and diff against
`samples/reference_yolopx/<scene>_yolopx.npz` (boxes, da_mask, ll_mask). For
DTCP, run the PC `DTCPPlanner.infer()` on the same frames and compare
`pred_wp`, `mu`, `sigma`, `pred_speed`. Acceptance criteria are in
`docs/jetson_deployment_workplan.md` §6.

### Need to understand the DTCP frame convention / command mapping?
→ `docs/DTCP_deployment.md` §inputs-and-outputs-in-detail. **Read this
before integrating with a route planner** — the lateral axis is
right-positive and the commands are TCP-convention
(`0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANE_FOLLOW, 4/5=CHANGE_*`).

### Want to do INT8 / C++ runtime / live camera?
→ `docs/jetson_deployment_workplan.md` §10–11. Defer until baseline FP16
Python pipeline is shipping correctly.

---

## Per-file notes worth knowing

- **`onnx/dtcp_v1.onnx`** drops the second (control-future) GRU loop from
  `TCP.forward` — that loop is unused at inference (see `dtcp_infer.py`).
  This shrinks the graph by roughly half without changing any output the
  pipeline consumes.

- **`onnx/yolopx_v2.onnx`** exports the **inference** detection branch only
  (`inf_out`, not `train_out`). NMS is done in numpy at runtime via
  `yolopx_postprocess.nms_yolopx` — you can swap this for the
  `EfficientNMS_TRT` plugin later for ~1–2 ms saving (workplan §10.2).

- **`jetson_runtime/beta_mode.py`** replaces the stochastic Beta.rsample()
  in `TCP.get_action` with the deterministic Beta **mode**. Required for
  deployment determinism. The math is `(α-1)/(α+β-2)`; valid because the
  model was trained with `α, β > 1` by construction.

- **`source/`** is only needed if you re-export ONNX. The bundle's ONNX is
  already validated (parity max-abs-diff < 5e-3 on YOLOPX raw logits,
  < 1e-6 on seg masks, < 1e-6 on DTCP outputs). Leave `source/` alone
  unless you need to retrain or change export shape.

---

## Total bundle size
≈ 700 MB (most of it is `weights/yolopx_v2_epoch30.pth` at 350 MB — drop
that if you're confident you won't need to re-export).

---

## When to defer to which doc

| Question | Doc |
|---|---|
| "How does DTCP's state vector work?" | `docs/DTCP_deployment.md` |
| "What are the 10 YOLOPX classes?" | `docs/YOLOPX_deployment.md` §2 |
| "What's the FP16 vs INT8 trade-off here?" | `docs/jetson_deployment_workplan.md` §10.1 |
| "How do I add `EfficientNMS_TRT`?" | `docs/jetson_deployment_workplan.md` §10.2 |
| "How do I migrate to C++?" | `docs/jetson_deployment_workplan.md` §11 |
| "What's been done, what's left?" | `docs/jetson_deployment_workplan.md` §13 hand-off log |
