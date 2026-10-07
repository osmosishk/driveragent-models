# Jetson AGX Orin deployment workplan — DTCP + YOLOPX combined

**Status:** PC-side end-to-end pipeline is working (see `/home/tonyho/development/combined_inference/`). This plan covers the Jetson port.

**Hand-off note for the next agent:** read this file end-to-end before touching any model file. The "decisions for you to make" at the bottom flag what is not yet pinned down. When you check off a step, edit this file and move it to the "done" log at the end so the *next* hand-off after you knows where you stopped.

---

## 0. What you're inheriting

| Artifact | Path | Purpose |
|---|---|---|
| DTCP weights (PyTorch) | `/home/tonyho/development/DTCP/deploy/dtcp_nusc_route_v1.pt` | Route-trained nuScenes planner. Wrapped dict: `{'state_dict', 'meta'}`. |
| DTCP inference class | `/home/tonyho/development/DTCP/deploy/dtcp_infer.py` | `DTCPPlanner(weights, device).infer(image, speed_mps, target_point, command)`. |
| DTCP arch | `/home/tonyho/development/DTCP/DTCP/DTCP/model.py` (class `TCP`) | Imported by `dtcp_infer.py`. |
| YOLOPX weights | `/home/tonyho/development/YOLOPX/runs/BddDataset/_2026-05-10-08-32/epoch-30.pth` | Wrapped dict: `{'state_dict', 'best_state_dict', 'epoch', ...}`. |
| YOLOPX arch | `/home/tonyho/development/YOLOPX/lib/models/YOLOP.py` (class `MCnet`) | Built via `lib.models.get_net(cfg)`. |
| Combined pipeline (PC) | `/home/tonyho/development/combined_inference/` | Two-stage: `run_yolopx_stage.py` (env `yolox`) → `run_dtcp_and_viz.py` (env `DTCP`). 4,610 val frames in ~15 min wall time. |
| Full-val MP4 reference | `/home/tonyho/development/combined_inference/output/all_val_combined.mp4` | 7.7 min, 1800×600 @ 10 fps. Use as a visual ground truth — Jetson output should look identical at the same frames. |
| Per-frame cached YOLOPX outputs | `/home/tonyho/development/combined_inference/cache/<scene_token>_yolopx.npz` | 136 files. Stage-1 boxes/masks. Useful as fixtures for Jetson-side TRT correctness checks. |

**Two background docs you should read once before starting:**
- `/home/tonyho/development/DTCP/deployment.md` (DTCP architecture, conventions, known caveats)
- `/home/tonyho/development/YOLOPX/deployment.md` (YOLOPX classes, preprocessing, **and an already-drafted ONNX-export script in §5.1**)

---

## 1. Target

- **Hardware:** Jetson AGX Orin, JetPack 5.1+ (TensorRT 8.5+).
- **Runtime precision:** **FP16** as the baseline. INT8 is a follow-up optimisation, documented in §10.
- **Runtime language:** **Python** for the first port (faster to debug, lower risk). C++ migration is an optional second pass, documented in §11.
- **Combined-pipeline goal:** match PC's 3-panel viz (front-cam overlay | BEV trajectory | action readout). Same frame layout and rendering — only the inference engines change.
- **Performance target:** the two models running back-to-back at ≥10 FPS end-to-end on a 1600×900 nuScenes-equivalent frame. Per YOLOPX/deployment.md §5.3, YOLOPX alone is ~150–250 FPS FP16 on AGX. DTCP is smaller (25.79 M params vs YOLOPX's ~30 M but with autoregression unrolled, only 4 steps) — expect 100–200 FPS. End-to-end should comfortably exceed 10 FPS.

---

## 2. PC-side: export both models to ONNX

**Run on the PC (not the Jetson) — uses full PyTorch + more disk/RAM.**

### 2.1 YOLOPX → ONNX (already drafted, just create + run)

`/home/tonyho/development/YOLOPX/deployment.md` §5.1 has the **complete script content** for `tools/export_onnx.py`. The script is *not yet on disk*; create it from the deployment.md block, then run:

```bash
cd /home/tonyho/development/YOLOPX
/home/tonyho/anaconda3/envs/yolox/bin/python tools/export_onnx.py \
    --weights runs/BddDataset/_2026-05-10-08-32/epoch-30.pth \
    --out /home/tonyho/development/combined_inference/onnx/yolopx_v2.onnx
```

Then validate:
```bash
/home/tonyho/anaconda3/envs/yolox/bin/pip install onnx onnxruntime
/home/tonyho/anaconda3/envs/yolox/bin/python -c \
  "import onnx; m=onnx.load('/home/tonyho/development/combined_inference/onnx/yolopx_v2.onnx'); onnx.checker.check_model(m); print('OK', m.ir_version)"
```

**Gotchas:**
- Use opset 13 (matches deployment.md).
- The export uses fixed input size 1×3×640×640, but the model with `auto=True` letterbox actually wants 384×640 for 16:9 input — confirm with `tools/demo.py` what spatial size goes into the network on real frames. If it's not 640×640, change `--img-size` and re-export. (You can check by adding `print(img.shape)` just before `model(img)` in `tools/demo.py:111`.)
- Dynamic batch in the export is fine; we'll fix batch=1 in the TRT engine.

### 2.2 DTCP → ONNX (new — script to write)

DTCP's `TCP.forward()` (in `DTCP/DTCP/model.py:138-210`) computes a lot of training-only outputs we don't need: `future_feature`, `future_attn_weighted`, `future_mu`, `future_sigma`, `cnn_ctrl_weighted`, `pred_value_traj`, `pred_value_ctrl`, etc., plus a second autoregressive GRU loop (lines 179–203). **Inference only consumes** `pred_wp`, `pred_speed`, `mu_branches`, `sigma_branches` (verify by reading `DTCP/deploy/dtcp_infer.py:157-165`).

Build an inference wrapper that returns *only* those four tensors. The second GRU loop becomes dead code and ONNX will drop it.

Create `/home/tonyho/development/combined_inference/export_dtcp_onnx.py`:

```python
import os, sys, argparse, torch
sys.path.insert(0, '/home/tonyho/development/DTCP/DTCP')
sys.path.insert(0, '/home/tonyho/development/DTCP/deploy')
from dtcp_infer import _MinimalConfig          # reuse the same config dtcp_infer uses
from DTCP.model import TCP

class DTCPInfer(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.m = TCP(_MinimalConfig())
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        self.m.load_state_dict(sd, strict=True)
        self.m.eval()

    @torch.no_grad()
    def forward(self, img, state, target_point):
        # img: (1,3,256,928) float32 ImageNet-normalised
        # state: (1,9)
        # target_point: (1,2)
        feature_emb, cnn_feature = self.m.perception(img)
        pred_speed = self.m.speed_branch(feature_emb)

        measurement_feature = self.m.measurements(state)
        j_traj = self.m.join_traj(torch.cat([feature_emb, measurement_feature], 1))
        z = j_traj
        x = torch.zeros((img.shape[0], 2), dtype=img.dtype, device=img.device)
        wps = []
        for _ in range(self.m.config.pred_len):     # 4 unrolled steps
            x_in = torch.cat([x, target_point], dim=1)
            z = self.m.decoder_traj(x_in, z)
            x = self.m.output_traj(z) + x
            wps.append(x)
        pred_wp = torch.stack(wps, dim=1)            # (B, 4, 2)

        init_att = self.m.init_att(measurement_feature).view(-1, 1, 8, 29)
        feature_emb_ctrl = (cnn_feature * init_att).sum(dim=(2, 3))
        j_ctrl = self.m.join_ctrl(torch.cat([feature_emb_ctrl, measurement_feature], 1))
        policy = self.m.policy_head(j_ctrl)
        mu = self.m.dist_mu(policy)
        sigma = self.m.dist_sigma(policy)
        return pred_wp, mu, sigma, pred_speed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='/home/tonyho/development/DTCP/deploy/dtcp_nusc_route_v1.pt')
    ap.add_argument('--out', default='/home/tonyho/development/combined_inference/onnx/dtcp_v1.onnx')
    ap.add_argument('--opset', type=int, default=13)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    wrapper = DTCPInfer(args.weights).eval()
    img = torch.zeros(1, 3, 256, 928)
    state = torch.zeros(1, 9)
    target = torch.zeros(1, 2)
    torch.onnx.export(
        wrapper, (img, state, target), args.out,
        opset_version=args.opset,
        input_names=['image', 'state', 'target_point'],
        output_names=['pred_wp', 'mu', 'sigma', 'pred_speed'],
        dynamic_axes={'image': {0: 'batch'}, 'state': {0: 'batch'},
                      'target_point': {0: 'batch'},
                      'pred_wp': {0: 'batch'}, 'mu': {0: 'batch'},
                      'sigma': {0: 'batch'}, 'pred_speed': {0: 'batch'}},
    )
    print(f'wrote {args.out}')

if __name__ == '__main__':
    main()
```

Run with the DTCP env so the architecture imports resolve:
```bash
/home/tonyho/anaconda3/envs/DTCP/bin/python /home/tonyho/development/combined_inference/export_dtcp_onnx.py
```

**Parity check (do this immediately after export — do not skip):**
- Load the .onnx in `onnxruntime`, feed one real frame from the val set, compare outputs vs the PyTorch `DTCPPlanner.infer()` on the same frame.
- Tolerance: max abs diff < 1e-4 on `pred_wp`, < 1e-3 on mu/sigma, < 1e-3 on pred_speed.
- If anything is off, the most likely cause is a missing branch in the wrapper or a sneaky in-place op. Re-read `TCP.forward()` and confirm every used tensor is wired.

### 2.3 Copy ONNX to Jetson

```bash
scp /home/tonyho/development/combined_inference/onnx/*.onnx jetson:~/onnx/
```

---

## 3. Jetson-side: TensorRT engine build

**Engines are device- and JetPack-specific. Always build on the Jetson, never copy a `.engine` file across machines.**

### 3.1 YOLOPX engine

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=$HOME/onnx/yolopx_v2.onnx \
    --saveEngine=$HOME/engines/yolopx_v2_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --shapes=image:1x3x384x640 \
    --verbose 2>&1 | tee yolopx_build.log
```

Adjust `--shapes` to whatever spatial size matched `tools/demo.py` at export time.

### 3.2 DTCP engine

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=$HOME/onnx/dtcp_v1.onnx \
    --saveEngine=$HOME/engines/dtcp_v1_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --shapes=image:1x3x256x928,state:1x9,target_point:1x2 \
    --verbose 2>&1 | tee dtcp_build.log
```

**Both builds:** check that `trtexec` reports **0 unsupported nodes**. If a GRU or any op falls back to FP32, note which one in the log — you may need to either accept the perf hit or restructure the export. Modern TRT (≥8.5) supports GRU directly; older versions do not.

---

## 4. Jetson-side: Python TRT runtime — one engine at a time

**Goal:** prove each engine produces the same output as PyTorch on the same input. Do this *before* fusing into the combined pipeline.

### 4.1 Common harness (`trt_runner.py`)

A thin wrapper around `tensorrt` + `pycuda` that:
- loads an engine,
- allocates device buffers from the engine's binding shapes,
- exposes `infer(dict_of_named_inputs) -> dict_of_named_outputs`.

Don't reinvent — copy the skeleton from `YOLOPX/deployment.md` §5.3 and generalise for multi-input/multi-output by reading bindings via `engine.get_binding_name(i)` / `engine.get_binding_shape(i)`.

### 4.2 YOLOPX parity test on Jetson

- Pick one val frame whose `.npz` cache is on disk (copy a few `cache/*_yolopx.npz` files over).
- Preprocess the original `.jpg` (letterbox + ImageNet-normalize + FP16), run the engine.
- Postprocess: NMS (port `lib/core/general.non_max_suppression` to numpy or use the TensorRT `EfficientNMS_TRT` plugin — see §10.2), then `scale_coords`.
- Compare boxes, da_mask, ll_mask against the cached `.npz`. Tolerances: box IoU > 0.95, mask agreement > 99%.

### 4.3 DTCP parity test on Jetson

- Same idea: pick a frame, look up `(speed, target_point, command)` from `packed_nuscenes_route.npy` (or hand-craft the state vector), run the engine.
- Compare `pred_wp` against the PC `DTCPPlanner.infer()` output for the same frame. Tolerance: max abs diff < 1e-3.

If both engines pass parity, the rest is mechanical.

---

## 5. Jetson-side: combined pipeline

Port `/home/tonyho/development/combined_inference/run_yolopx_stage.py` + `run_dtcp_and_viz.py` to a single Python entry point on the Jetson:

```
jetson_pipeline.py
├── load both engines (YOLOPX + DTCP)
├── load nominal CAM_FRONT intrinsics (from viz_helpers.py — already vendored, copy it over)
├── for each input frame (folder, video, or live camera):
│     bgr = read_frame()
│     # Stage A — YOLOPX
│     boxes, da_mask, ll_mask = yolopx_engine.infer(preprocess_yolopx(bgr))
│     # Stage B — DTCP
│     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
│     state = build_state_vec(speed_mps, target_point, command)
│     wp, mu, sigma, speed_pred = dtcp_engine.infer(preprocess_dtcp(rgb), state, target_point)
│     throttle, steer, brake = beta_mode_to_action(mu, sigma)
│     # Stage C — render the same 3-panel viz as on PC
│     composite = render_three_panel(...)   # vendor from run_dtcp_and_viz.py
│     write_to_mp4(composite)
```

**Key files to vendor as-is** from the PC `combined_inference/`:
- `viz_helpers.py` (project_wp_to_image, blend_seg_masks, draw_box, color_for_class — no torch deps)
- `render_three_panel` from `run_dtcp_and_viz.py:62-118`

**Frame-source decision:**
- For initial bring-up: read JPGs from a folder (mirror the PC manifest flow).
- For real demo: read from `/dev/video*` via cv2.VideoCapture or from the Argus / NVMM camera stack.
- The state vector (`speed`, `target_point`, `command`) needs a real source — for the demo you can replay from the manifest; for live driving it needs to come from a GNSS/odometry/route-planner stack. **This is out of scope for the model port** but flag it to whoever owns the vehicle integration.

**Beta-mode action (deterministic, no sampling on Jetson):**
PyTorch `model.get_action()` samples from `Beta(mu, sigma)`. For deployment, prefer the **mode**: `acc = (α-1)/(α+β-2)` mapped to `[-1,1]`. Read `DTCP/DTCP/model.py:336` for the exact sampling logic, then write a numpy equivalent that returns the mode rather than a sample. This makes inference fully deterministic and lets you skip the PyTorch dist library entirely.

---

## 6. Validation harness

Before any deployment, run the Jetson pipeline on the same 4,610 val frames and verify:

1. **Per-frame parity vs PC:** for each frame, max abs diff between Jetson and PC outputs — `pred_wp` < 1e-2 m, action MAE < 1e-2, box IoU > 0.95, mask agreement > 99%. Some drift is expected because of FP16 vs FP32, but the magnitudes should be tiny.
2. **Aggregate parity:** ADE / FDE / wp_L1 on val should match deployment.md §validation-performance to within 1% (1.130 m → ≤ 1.142 m ADE, etc.).
3. **Visual parity:** render the same scene the PC did (`output/<scene>/combined.mp4`) and diff side-by-side — overlays should be visually identical. The reference is `output/e036014a715945aa965f4ec24e8639c9/combined.mp4`.

Build a small `compare_jetson_pc.py` that:
- reads the cached YOLOPX `.npz` and a parallel Jetson-produced `.npz`, computes the deltas above,
- prints a pass/fail table.

**If parity fails:** the first thing to check is preprocessing (BGR vs RGB swap, normalization stats, letterbox padding sign). The second is FP16 underflow in any branch — try re-building the engine with `--strict-types` or force-FP32 on the speed_branch and check.

---

## 7. Performance benchmarking

Once parity is established:

```bash
trtexec --loadEngine=yolopx_v2_fp16.engine --shapes=image:1x3x384x640 --warmUp=1000 --duration=10
trtexec --loadEngine=dtcp_v1_fp16.engine    --shapes=image:1x3x256x928,state:1x9,target_point:1x2 --warmUp=1000 --duration=10
```

Record per-model FPS, then measure end-to-end pipeline FPS (including camera read + preprocess + render). Camera I/O and rendering often dominate — confirm before optimising the models further.

---

## 8. What deliverable looks like at the end

A `~/jetson_deploy/` directory on the Jetson containing:
- `engines/yolopx_v2_fp16.engine`, `engines/dtcp_v1_fp16.engine`
- `jetson_pipeline.py` (the combined runtime)
- `viz_helpers.py` (vendored)
- `trt_runner.py` (the TRT inference wrapper)
- `beta_mode.py` (deterministic Beta-mode action)
- `validation/compare_jetson_pc.py` (parity harness)
- `README.md` describing how to run inference on a folder of JPGs or on a live camera

The PC-side `combined_inference/` repo stays untouched as the reference implementation.

---

## 9. Risk register — read before doing

- **DTCP's GRU autoregression** is 4 unrolled steps. If TRT's GRU support behaves oddly on Orin, you can drop to plain `nn.GRUCell` calls (already what `TCP.decoder_traj` uses, AFAICT — verify). The unrolled loop becomes 4 matmul + tanh + add chains which TRT trivially fuses.
- **Beta sampling vs mode** — the model was trained with `Beta(α = m·s+1, β = (1-m)·s+1)` where mode == ground-truth action (see deployment.md §how-we-got-here). So mode is the right deployment policy; sampling is only useful for exploration in training.
- **Image-size mismatch.** YOLOPX trained on letterboxed 640-multiple-of-32. DTCP trained on stretched 256×928 (16:9 → 3.6:1, not aspect-preserving). Do not "fix" the DTCP preprocess — re-use the same stretch as `dtcp_infer.py:140-142`, otherwise the model goes out of distribution.
- **Per-scene camera intrinsics** vary by ~5% in nuScenes. The PC viz uses nominal `fx=fy=1266, cx=816, cy=491`. On a real vehicle with one mounted camera, replace these with the *actual* calibrated intrinsics — the waypoint projection in the front-cam panel will otherwise drift.
- **Closed-loop / safety.** Per DTCP/deployment.md caveat #7: "No closed-loop simulator evaluation. Real driving behaviour with this model is unknown." Do not let this pipeline drive a real vehicle without a fallback safety layer (see YOLOPX/deployment.md §6.4 — depth-based proximity stop, parallel pedestrian detector).
- **Class names hardcoded** — keep `['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train']` in the Jetson code; do not import YOLOPX at runtime.
- **`weights_only` argument** — when loading the YOLOPX `.pth` on the Jetson with newer PyTorch versions, you may need `torch.load(..., weights_only=False)`. The export script in YOLOPX deployment.md §5.1 omits it; add it if the load errors.

---

## 10. Optional follow-ups (after baseline FP16 is shipping)

### 10.1 INT8 quantisation
- Collect ~500 representative frames from the actual deployment camera (or fall back to nuScenes val frames). Save preprocessed FP32 tensors.
- Write a `Int8EntropyCalibrator2` subclass that streams those tensors.
- Build with `--int8 --calib=<calib.cache>`.
- Re-run §6 parity. INT8 typically loses ~1–3% on detection mAP and may shift DTCP waypoints by 1–2 cm — acceptable for most use cases, but validate before shipping.

### 10.2 EfficientNMS_TRT plugin
The PC pipeline does NMS in PyTorch via `lib.core.general.non_max_suppression`. On Jetson it's faster to fuse NMS into the engine itself:
- Use `onnx-graphsurgeon` to insert an `EfficientNMS_TRT` node after the detection head.
- Re-build the engine — boxes come out post-NMS, no numpy NMS at runtime.
- Saves ~1–2 ms per frame on AGX.

### 10.3 Single fused engine
The two models share no weights, but could share the engine for scheduling. Marginal gain on AGX (which can run them concurrently anyway). Skip unless you've already optimised everything else.

---

## 11. Optional: C++ runtime

The Python TRT runtime is fine for ~30–60 FPS workloads. If you need more, port `jetson_pipeline.py` to C++ using TensorRT's C++ API plus NVMM / GStreamer for camera I/O. Expect ~2× FPS for the same engines. Heavy lift — only do this if Python actually bottlenecks the deploy target.

---

## 12. Decisions for you to make (no consensus yet)

| Decision | Default I'd take | Reason it's flagged |
|---|---|---|
| Precision: FP16 vs INT8 first | FP16 | Lower risk, no calibration data needed, gets you a running baseline fastest. Add INT8 after parity is proven. |
| NMS location: numpy vs `EfficientNMS_TRT` | numpy first | Easier debugging; swap to plugin only once everything else works. |
| Frame source: replay JPGs vs live camera | Replay JPGs from manifest | Decouples model bring-up from camera-integration work, which is its own can of worms (CSI driver, ISP tuning, sync). |
| Runtime lang: Python vs C++ | Python | See §11. |
| Combined-pipeline output: viz MP4 (like PC) vs minimal text-only telemetry | Same 3-panel MP4 | Keeps debug parity with PC; you can strip the viz layer later. |

If any of these defaults are wrong for your context, change them *before* starting §2 — they affect the export shape.

---

## 13. Hand-off log

> Append your progress here as you check off sections. Format: `[YYYY-MM-DD] agent-name §N — what you did, what's blocked, anything surprising`.

- `[2026-05-11] opus-4.7 §0–13` — wrote this plan based on the PC-side end-to-end pipeline. Nothing executed on Jetson yet.
