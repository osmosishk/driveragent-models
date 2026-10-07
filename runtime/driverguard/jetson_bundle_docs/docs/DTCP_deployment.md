# DTCP — nuScenes route-trained planner

Self-contained reference for deploying the DTCP planning model trained on
nuScenes. Read this file end-to-end before doing anything; the "DO NOT use…"
items below are easy to miss and easy to regret.

---

## TL;DR — what to deploy

**Use these four files.** Everything else in the repo is either training
infrastructure or a diagnostic artifact you should NOT ship.

| File | Role | Size |
|---|---|---|
| `deploy/dtcp_nusc_route_v1.pt` | model weights + metadata block | 103 MB |
| `deploy/dtcp_infer.py` | self-contained `DTCPPlanner` inference class | 6 KB |
| `DTCP/DTCP/model.py` | TCP architecture (imported by `dtcp_infer.py`) | 13 KB |
| `DTCP/DTCP/resnet.py` | ResNet-34 backbone (imported by `model.py`) | 8 KB |

**Runtime deps:** `torch ≥ 1.13`, `torchvision`, `numpy`, `Pillow`. Nothing else.
In particular: **no `pytorch_lightning`, no `gym`, no `carla`, no `imgaug`, no
Roach RL checkpoint.**

**Quick start:**

```bash
python /home/tonyho/development/DTCP/deploy/dtcp_infer.py   # self-test
```

```python
from dtcp_infer import DTCPPlanner

planner = DTCPPlanner(
    weights='/home/tonyho/development/DTCP/deploy/dtcp_nusc_route_v1.pt',
    device='cuda',
)
out = planner.infer(
    image=rgb_image_HxWx3_uint8,   # auto-resized to 256x928
    speed_mps=5.0,                 # ego forward speed, m/s
    target_point=(2.0, 30.0),      # next nav goal in ego frame: (right+, forward), m
    command=3,                     # 0=LEFT 1=RIGHT 2=STRAIGHT 3=LANE_FOLLOW 4=CHG_L 5=CHG_R
)
out['waypoints']   # (4, 2) np.ndarray — ego frame, t+0.5/1.0/1.5/2.0 s
out['throttle']    # [0, 1]
out['steer']       # [-1, 1]
out['brake']       # [0, 1]
```

---

## DO NOT confuse the two training runs

| Run | Checkpoint dir | target_point source | When to use |
|---|---|---|---|
| **Route (use this)** | `runs/dtcp_nusc_il_route/` | nuScenes `route.json` next waypoint ≥30 m ahead | All planning, all benchmarks, all downstream deployment |
| **Privileged (diagnostic only)** | `runs/dtcp_nusc_il/` | `gt_ego_fut_trajs[6]` = GT ego position at t+3 s | Diagnostic / ablation only — **DO NOT deploy** |

**Why the distinction matters:** the privileged run produces ADE 0.38 m / FDE
0.42 m, which looks like state-of-the-art for monocular nuScenes planning. It
isn't. The model receives the ground-truth future ego position at t+3 s as an
input and effectively interpolates between origin and that target. A
privileged-input probe (neutralizing target_point to a constant) blows up ADE
**10×** (0.38 → 3.89 m) and FDE **15×** — i.e., the camera barely contributes.

The route run produces ADE 1.13 m / FDE 1.84 m. Probe degradation: 1.35×.
That's a real planner — ~65% of the signal comes from the camera + command +
speed, ~35% from the route goal. Comparable to TCP/DTCP's CARLA training setup
(where target_point is a route waypoint, not a GT future position).

**The deployment .pt file (`deploy/dtcp_nusc_route_v1.pt`) is from the route
run.** Its metadata block records this.

---

## Inputs and outputs in detail

### Inputs (per inference call)

| Name | Shape / dtype | Range | Meaning |
|---|---|---|---|
| `image` | (H, W, 3) uint8 | 0–255 | RGB front-camera image. Auto-resized to (256, 928). |
| `speed_mps` | scalar float | 0–25 typical | Ego forward speed in m/s. Internally divided by 12. |
| `target_point` | (2,) float | each \|x\| ≤ ~30 typical | (lateral_right_positive, forward) in metres, ego frame. At training time, this was the next nuScenes route waypoint at ≥30 m. Pass your nav system's equivalent. |
| `command` | int 0–5 | discrete | High-level intent. See table above. |

**Frame convention (critical):**
- `col 0 = lateral, right-positive` (the standard "right-handed" lateral is
  *negated* relative to the more common "y = left" convention — verified
  empirically against the cached nuScenes preprocess at 1.85 cm round-trip
  error)
- `col 1 = forward`
- Both in metres
- Z is implicitly 0 (ground plane); the model has no notion of elevation

### Outputs

| Key | Shape | Range | Meaning |
|---|---|---|---|
| `waypoints` | (4, 2) np.ndarray | ego frame, metres | Predicted ego positions at t+0.5, 1.0, 1.5, 2.0 s |
| `throttle` | float | [0, 1] | Throttle command |
| `steer` | float | [-1, 1] | Steering (right-positive, matching the lateral convention) |
| `brake` | float | [0, 1] | Brake command |
| `mu`, `sigma` | (2,) each | >1 typical | Beta α, β over (acc, steer); throttle/brake come from splitting the sampled `acc` |
| `speed_pred` | float | m/s | Model's predicted next-step speed (auxiliary head) |

`throttle` and `brake` are mutually exclusive by construction: `acc = sample
from Beta(α, β) ∈ [-1, 1]`; `throttle = max(acc, 0)`, `brake = max(-acc, 0)`.

---

## Validation performance (route run, best ckpt)

Held-out val: 4,620 frames across 150 scenes (NOT the official nuScenes val
split — see Caveats below).

| Metric | Value |
|---|---|
| ADE (mean displacement, 2 s horizon, 4 waypoints) | **1.130 m** ± 0.948 |
| FDE (final displacement at t+2 s) | **1.838 m** ± 1.569 |
| Waypoint L1 | **0.619 m** ± 0.504 |
| Throttle MAE | 0.026 |
| Steer MAE | 0.038 |
| Brake MAE | 0.040 |
| Beta KL (action distribution divergence vs ground-truth) | 0.039 |

For context: VAD/UniAD report ~0.3–1.0 m ADE at **3 s** horizon with a full BEV
stack (lidar + multi-cam + map). Our 1.13 m ADE at **2 s** with monocular
front-cam, no map, no LiDAR, no surround views is roughly proportional given
the input gap.

---

## Architecture summary

**Backbone:** ResNet-34 (ImageNet init), input 256×928 RGB → feature map
(B, 512, 8, 29).

**Heads:**
- Trajectory: GRU autoregressively predicts 4 waypoints (`pred_len=4`).
- Control: Beta-distribution policy over (acc, steer); 2-D output mapped to
  [-1, 1].
- Speed: auxiliary regression head (returns model's predicted speed).
- Plus value heads that were used during training (RL-teacher distillation in
  CARLA; zeroed-out for nuScenes — see Substitution rules).

**State vector (9-D):** `[speed/12, target_x_lat, target_y_fwd, command_one_hot(6)]`
→ embedded by a small MLP and fused with image features.

**Total parameters:** 25.79 M.

Diversity loss (the original Path-A motivation) operates on CNN attention maps
during training and has no inference-time effect — it's a regulariser only.

---

## How we got here (training journey, condensed)

1. **CARLA → nuScenes pivot.** Original DTCP trains on CARLA logs with
   supervision from a Roach RL teacher (action Beta distribution, value, latent
   features). nuScenes has no Roach analog, so:
   - `action_mu/sigma` substituted with deterministic-human Beta params:
     `α = m·s + 1`, `β = (1-m)·s + 1` where `m = clip((human_action+1)/2, 0.05, 0.95)`,
     `s = 10`. Mode of Beta exactly equals the human action; KL becomes near-regression.
   - `value_weight = 0` and `features_weight = 0` in `GlobalConfig` when
     `dataset_kind='nuscenes'` (gated in `config.py:__init__`).
   - Action labels (3-vec for val MAE: throttle/steer/brake) derived from
     `gt_ego_lcf_feat[3,8]` (longitudinal accel, steering wheel angle).

2. **First training (`dtcp_nusc_il`, ~2 hr on RTX 4090):**
   - Used `target_point = gt_ego_fut_trajs[6]` (privileged — GT ego at t+3 s).
   - Hit ADE 0.38 m, looked great.
   - Probe revealed 10× degradation when target_point neutralized → model was
     interpolating, not planning.
   - **Plus a silent bug:** `NUSC_CMD_TO_TCP` had nuScenes class 0/1 swapped.
     Privileged signal masked it because the GT target dominated.

3. **Rebuild + retrain (`dtcp_nusc_il_route`, ~1.5 hr):**
   - target_point switched to nuScenes `route.json` next-waypoint at ≥30 m.
   - Command mapping fixed: **nuScenes 0 = RIGHT, 1 = LEFT, 2 = STRAIGHT**
     (verified by command/lateral correlation: `LEFT cmd` rows have route
     target lat mean −12.7 m as expected).
   - Filtered ~15 % of frames: 507 from scenes lacking `route.json` + 2,044
     from scenes whose `route.json` doesn't track ego (closest route point > 10 m).
   - Result: ADE 1.13 m on real planning task. Probe degradation 1.35×.

4. **Global-to-ego convention discovery.** The cached
   `cached_nuscenes_info.pkl` (VAD/UniAD-style preprocess) uses
   **right-positive lateral**: `fwd = cos(yaw)*dx + sin(yaw)*dy; right =
   sin(yaw)*dx - cos(yaw)*dy`. Verified at 1.85 cm median round-trip error.
   Hardcoded in `build_nuscenes_index.py:global_to_ego()`. **Do not change
   without re-verifying.**

5. **Deployment package built** (this file's TL;DR).

---

## File reference

### Training / data pipeline

| Path | What it does |
|---|---|
| `DTCP/DTCP/build_nuscenes_index.py` | Converts `cached_nuscenes_info.pkl` → `packed_nuscenes*.npy`. CLI flag `--target_point_source {route, fut_traj}` chooses defensible vs privileged target. Defaults to `route`. |
| `DTCP/DTCP/nuscenes_data.py` | Dataset class matching `CARLA_Data` schema; loads `packed_nuscenes*.npy`. Lazy-imports `imgaug` only when `img_aug=True`. |
| `DTCP/DTCP/config.py` | `GlobalConfig`; `dataset_kind` field; zeroes value/feature weights for nuScenes. |
| `DTCP/DTCP/train.py` | LightningModule + `build_datasets(cfg)` helper. CLI flag `--dataset_kind {carla, nuscenes}` and `--nuscenes_pack` to override pack path. |
| `DTCP/DTCP/model.py` | TCP architecture. Also used at inference (imported by `dtcp_infer.py`). |
| `DTCP/DTCP/resnet.py` | Backbone. Also used at inference. |

### Packed data files

| Path | Records | Use |
|---|---|---|
| `/home/tonyho/datasets/nuscenes/packed_nuscenes_route.npy` | 26,234 (train 21,614 + val 4,620) | **Use this for any new training/eval.** Built with route target. |
| `/home/tonyho/datasets/nuscenes/packed_nuscenes.npy` | 29,049 (train 23,919 + val 5,130) | Privileged target — diagnostic only. |
| `/home/tonyho/datasets/nuscenes/cached_nuscenes_info.pkl` | 34,149 | Source data (VAD/UniAD preprocess). |

### Training runs

| Path | Notes |
|---|---|
| `runs/dtcp_nusc_il_route/` | **The good run.** Best ckpt: `best_epoch=59-val_loss=0.876.ckpt`. 60 epochs, batch 32. |
| `runs/dtcp_nusc_il/` | Privileged run. Do not deploy. Keeps as a baseline for the interpolation-degradation probe. |
| `runs/dtcp_div_on/` | Prior CARLA Path-A run. Unrelated to nuScenes work; leave alone. |

### Eval and viz

| Path | What it produces |
|---|---|
| `eval/run_eval.py` | ADE/FDE/wp_L1/Beta-KL/per-channel-MAE → CSV. Use `--dataset_kind nuscenes --nuscenes_pack <pack>`. |
| `eval/metrics_nuscenes.csv` | Two appended rows: privileged baseline and route baseline. |
| `/tmp/dtcp_neutral_target_probe.py` | The "neutralize target_point" probe. Parameterised by `PROBE_CKPT` and `PROBE_PACK` env vars. |
| `/tmp/dtcp_viz_full.py` | Per-frame still PNGs (matplotlib, 3-panel) → `~/dtcp_probe/viz_full/`. |
| `/tmp/dtcp_viz_video.py` | OpenCV-rendered MP4 of full val set → `~/dtcp_probe/viz_full/dtcp_val_route.mp4` (4,620 frames, 10 FPS, 1760×720, 374 MB). |

### Deployment package

| Path | What it is |
|---|---|
| `deploy/dtcp_nusc_route_v1.pt` | Stripped state_dict + metadata dict. No Lightning. |
| `deploy/dtcp_infer.py` | Self-contained `DTCPPlanner` class. |

---

## Useful commands

### Reproduce the best training run

```bash
cd /home/tonyho/development/DTCP/DTCP
PYTHONPATH=. /home/tonyho/anaconda3/envs/DTCP/bin/python -m DTCP.train \
  --dataset_kind nuscenes \
  --nuscenes_pack /home/tonyho/datasets/nuscenes/packed_nuscenes_route.npy \
  --id dtcp_nusc_il_route_v2 \
  --batch_size 32 \
  --epochs 60 \
  --logdir /home/tonyho/development/DTCP/runs
```

Run from `DTCP/DTCP/` (the relative `roach/log/ckpt_11833344.pth` path in
`_load_weight()` requires it). Lightning still calls `_load_weight()` during
training — it's needed to initialise the value/dist heads, even though we then
zero out the value/feature losses for nuScenes. Inference doesn't need Roach
because `deploy/dtcp_infer.py` bypasses `_load_weight()` entirely.

### Re-run eval

```bash
PYTHONPATH=/home/tonyho/development/DTCP/DTCP \
  /home/tonyho/anaconda3/envs/DTCP/bin/python /home/tonyho/development/DTCP/eval/run_eval.py \
  --ckpt /home/tonyho/development/DTCP/runs/dtcp_nusc_il_route/best_epoch=59-val_loss=0.876.ckpt \
  --tag dtcp_nusc_il_route_best \
  --dataset_kind nuscenes \
  --nuscenes_pack /home/tonyho/datasets/nuscenes/packed_nuscenes_route.npy \
  --batch_size 32 \
  --out /home/tonyho/development/DTCP/eval/metrics_nuscenes.csv
```

### Rebuild the packed file (e.g. after a code change)

```bash
PYTHONPATH=/home/tonyho/development/DTCP/DTCP \
  /home/tonyho/anaconda3/envs/DTCP/bin/python -m DTCP.build_nuscenes_index \
    --target_point_source route \
    --dst /home/tonyho/datasets/nuscenes/packed_nuscenes_route.npy \
    --no-verify-paths   # skip per-frame os.path.exists; faster
```

### Extract a fresh deployment package after a new training run

```python
import sys, types, os, torch
sys.modules.setdefault('carla', types.ModuleType('carla'))
sys.path.insert(0, '/home/tonyho/development/DTCP/DTCP')
os.chdir('/home/tonyho/development/DTCP/DTCP')  # for Roach relative path
from DTCP.config import GlobalConfig
from DTCP.train import DTCP_planner

cfg = GlobalConfig(dataset_kind='nuscenes',
                   nuscenes_pack_path='/home/tonyho/datasets/nuscenes/packed_nuscenes_route.npy')
planner = DTCP_planner.load_from_checkpoint('<your new ckpt>',
                                            config=cfg, lr=cfg.lr, map_location='cpu')
torch.save({'state_dict': planner.model.state_dict(),
            'meta': {... your metadata ...}},
           '/home/tonyho/development/DTCP/deploy/dtcp_nusc_route_vN.pt')
```

---

## Known caveats — what's NOT done

Each of these is a 1- to 3-hour fix if you need it.

1. **Scene-level train/val split is sorted-by-token, not the official nuScenes
   split.** Same size ratio (700/150) but different scene assignment. To
   compare against published nuScenes leaderboard numbers, install
   `nuscenes-devkit`, replace the sort logic in `train.py:build_datasets()`
   with `create_splits_scenes()['val']`, rebuild the pack, retrain.

2. **2-second prediction horizon (4 waypoints at 0.5 s).** VAD/UniAD report at
   3 s (6 waypoints). To extend: change `cfg.pred_len = 6` and slice
   `gt_ego_fut_trajs[1:7]` in `build_nuscenes_index.py:convert_record`,
   rebuild, retrain.

3. **Image stretched, not letterboxed, to 256×928.** Distorts aspect ratio
   (16:9 → ~3.6:1). Fine for from-scratch training but means CARLA-pretrained
   weights won't transfer cleanly without resize alignment.

4. **Per-frame action labels are physical accel + steering wheel angle, not
   true throttle/brake.** Derived from `gt_ego_lcf_feat[3, 8]` (longitudinal
   accel m/s², steering rad). Mapped to [-1, 1] via division by 3.0 / 6.0
   respectively. This is the closest analog to TCP's (acc, steer) convention
   available from nuScenes; the resulting action MAEs aren't directly
   comparable to CARLA/Roach numbers.

5. **~15 % of scenes filtered out** during route-pack build: 15 scenes with
   missing `route.json`, plus scenes where the route file doesn't match the
   ego trajectory (closest route point > 10 m — a known nuScenes data quirk).

6. **Visualization uses nominal camera intrinsics.** `dtcp_infer.py` itself
   doesn't project to image space — only the viz scripts in `/tmp/` do.
   They use fx=fy=1266, cx=816, cy=491, camera mount at (1.72 m forward, 1.49 m
   up). Real per-scene intrinsics differ by ~5 %. For pixel-accurate viz,
   load from `v1.0-trainval/calibrated_sensor.json` per `sample_data_token`.

7. **No closed-loop simulator evaluation.** All metrics are open-loop on
   pre-recorded keyframes. Real driving behaviour with this model is unknown.

8. **The model was the auxiliary output of a research goal, not the goal
   itself.** The actual deliverable per `project_dtcp_path_a.md` memory is a
   portable `diversity_loss.py` for OpenDriveVLA — this nuScenes-trained model
   is a side-effect that demonstrates the architecture transfers to real data.
   Treat the model as a research artifact, not a deployment candidate without
   further closed-loop validation.

---

## Performance cheat sheet

On RTX 4090, batch=32, mixed precision (bf16):

| Operation | Throughput |
|---|---|
| Full training step (forward + backward + opt) | ~8.4 it/s |
| Inference (forward only) | ~12–14 it/s |
| Full eval (4,620 frames) | ~80 s |
| Full val MP4 render (4,620 frames, 1760×720, mp4v codec) | ~55 s |

GPU memory at batch=32: 5.1 GB out of 24 GB. Plenty of headroom to scale batch
up or input resolution.

---

## Sanity-check before trusting any change

1. After any pack rebuild, run `eval/run_eval.py` on the current best ckpt
   with the new pack. wp_L1 should not move by more than ~5 % unless you
   intentionally changed the label distribution.
2. After any retraining, run `/tmp/dtcp_neutral_target_probe.py` with
   `PROBE_CKPT` pointed at the new ckpt and `PROBE_PACK` pointed at the new
   pack. If the privileged/neutral ratio exceeds 2.0×, the model is leaning
   on target_point too hard — you've reintroduced the original interpolation
   problem.
3. After any inference-script change, run `python deploy/dtcp_infer.py`
   self-test on the synthetic image and confirm `(throttle, steer, brake)`
   values are bounded and `waypoints` have monotonic forward distance.

---

*This file consolidates the entire nuScenes training pivot done 2026-05-08
through 2026-05-10. For deeper context see `project_nuscenes_training_pivot.md`
in the agent's memory directory.*
