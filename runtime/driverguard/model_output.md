# DriverGuard Model Output Specification

For UI rendering of the `driverguard` model running on the on-vehicle Jetson.
Bundle version: **`v1-fp32-2026-05-25`** (TRT FP32 main + ORT-CPU control hybrid).

## Where the data comes from

- **ZMQ pub address:** `tcp://127.0.0.1:8014`
- **Schema type:** `DriverGuardResult` in `/home/tonyho/driveragent/message/message.capnp`
- **Publish rate:** 10 Hz (one message per camera frame)
- **Already wired** in `ui/newwidgets/data_bus.py:665` — `subscribe_driverguard()` populates `data.driverguard_*` attributes consumed by `main_camera.py`. This doc is the spec; the subscription already exists.

## Field reference

| Field | Type | Units / range | Meaning |
|---|---|---|---|
| `timestamp` | UInt64 | ms since epoch | wall-clock when the frame was captured |
| `frame` | UInt64 | monotonic | frame counter; useful for detecting gaps |
| `trajectory` | List(Point2D), len 4 | meters, ego frame | 4 waypoints at t+0.5/1.0/1.5/2.0 s; `x = lat_right_m`, `y = fwd_m` |
| `throttle` | Float32 | `[0, 1]` | acceleration command (0 = off, 1 = full) |
| `steer` | Float32 | `[-1, +1]` | **normalized wheel angle**, positive = right; multiply by 28° to get degrees |
| `brake` | Float32 | `[0, 1]` | brake command; mutually exclusive with `throttle` |
| `predSpeedMps` | Float32 | m/s | model's predicted next-step speed |
| `egoSpeedMps` | Float32 | m/s | observed current ego speed at inference time |
| `command` | UInt8 | 0..5 | nav command active at inference: 0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANE_FOLLOW, 4=CHANGE_LEFT, 5=CHANGE_RIGHT |
| `inferenceMs` | Float32 | ms | end-to-end YOLOPX+DTCP latency this frame |
| `finite` | Bool | — | true if every numeric output above is finite |
| `daMaskRle` | Data | RLE bytes | drivable-area binary mask; decode with `ui/newwidgets/mask_codec.py` |
| `llMaskRle` | Data | RLE bytes | lane-line binary mask; same codec |
| `maskWidth` / `maskHeight` | UInt16 | px | mask dimensions for both `daMaskRle` and `llMaskRle` |
| `detections` | List(DriverGuardDetection) | — | YOLOPX boxes: `x1, y1, x2, y2, conf, classId` (cls 0..9, see `yolopx_postprocess.py:CLASS_NAMES`) |
| `enginePrecision` | Text | — | provenance tag: `"FP32+ort_control"` for v1 |
| `modelVersion` | Text | — | bundle version string, e.g. `"v1-fp32-2026-05-25"` |

## Conventions to render correctly

### Ego frame (trajectory + target_point)
- Origin is the rear-axle center
- **x = lateral, right-positive (meters)** — opposite of most ROS conventions
- **y = forward (meters)**
- The 4 waypoints are at t+0.5, 1.0, 1.5, 2.0 s — render them as a 4-point polyline starting near the origin and extending forward

### Steering sign
- `steer > 0` → wheels turn right → vehicle yaws right
- The published `steer` is a **normalized** value. To display a degree readout: `wheel_angle_deg = steer * 28.0`
- This is computed in `runner.py:steer_from_pred_wp()` via a bicycle model from `trajectory[3]` (the t+2 s waypoint), not from the DTCP model's internal steer scalar. So `steer` and `trajectory[3]` will always agree

### Throttle / brake
- Always one or the other (mutually exclusive by construction in `beta_mode.py`)
- A natural visual: a single bipolar bar `[-1, 0, +1]` where `-1 = full brake`, `+1 = full throttle`, value = `throttle - brake`

### Masks
- RLE bytes — call `ui/newwidgets/mask_codec.py:decode_rle(rle, w, h)` → `np.uint8 [H, W]` (0/1)
- `da_mask` = drivable area, `ll_mask` = lane lines
- Already drawn in `main_camera.py` via the existing data_bus pipeline; nothing new needed unless you want a different overlay style

### Detections
- 10 classes: `['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic light', 'traffic sign', 'train']`
- Boxes are in **original camera resolution** (1280×720 typically), already scaled by `scale_coords` in the runner
- `conf` in `[0, 1]`

## Suggested UI panels

Two new readouts make the most sense given what's already on screen:

1. **Action panel** (near the existing trajectory overlay)
   - Throttle bar `[0, 1]` (green)
   - Brake bar `[0, 1]` (red)
   - Steer dial or horizontal bar `[-1, +1]` showing both normalized value and `deg = steer × 28°`
   - Command label (`LANE_FOLLOW` etc.) and remap badge if you want to show that 2/4/5 collapse to 3/0/1 (see "Caveats" below)

2. **Model health strip** (corner badge)
   - `modelVersion` (e.g. `v1-fp32-2026-05-25`)
   - `enginePrecision` (e.g. `FP32+ort_control`)
   - `inferenceMs` (current latency)
   - `finite` indicator — should always be true after this deploy; flash red if false
   - `predSpeedMps` vs `egoSpeedMps` diff — gives a quick "is the model in sync with reality" signal

The existing trajectory + mask + box overlays in `main_camera.py` already handle the spatial data — no change needed there.

## Live sample values (from 15 s static smoke test, parked vehicle)

```
throttle      = 0.000              # parked, no acceleration
steer         = -0.012             # ≈ -0.34° wheel angle, basically straight
brake         = 0.118              # gentle holding brake
trajectory[3] = (-0.0, +3.5)       # t+2s waypoint, ~3.5 m forward, no lateral offset
egoSpeedMps   = 0.0
predSpeedMps  = ~7.0               # model thinks ~7 m/s is appropriate
command       = 2 (STRAIGHT)       # remapped internally to 3 (LANE_FOLLOW)
inferenceMs   = ~80 ms             # well under 100 ms / 10 Hz budget
finite        = True               # 100% of 150 messages
enginePrecision = "FP32+ort_control"
modelVersion    = "v1-fp32-2026-05-25"
```

## Caveats worth surfacing in the UI

- **Command remap (runner_patches/0001).** The DTCP checkpoint was only trained on commands `{0, 1, 3}`. Indices 2 / 4 / 5 are remapped inside the runner to `3 / 0 / 1` before inference. The `command` field on the wire is the *pre-remap* value (what RouteGuidance asked for). If you want to show "what the model actually saw", apply: `2→3, 4→0, 5→1`. Otherwise just display the original.
- **Steer is geometry, not the model's scalar.** As noted above, `steer` comes from `trajectory[3]` via a bicycle model. This is intentional — the DTCP steer scalar is weak (spread ~0.076 across full sweep). Don't try to cross-check `steer` against any "raw model steer" — it isn't published.
- **`finite` should be 100%.** Pre-deploy baseline was 0%. If `finite=False` starts showing up, surface it loudly — it means the engine has regressed.
- **Hz vs frame age.** The 10 Hz cadence is target, not guarantee. If you want a "stale" indicator, compare `now_ms - timestamp` and warn at > 200 ms.
- **No shadow vs engaged distinction in this message.** Whether the model's actions actually drive the car is decided by `selfdrive/supervisor.py:AutopilotState` (on port 5607). DriverGuardResult is published regardless of engage state, unless `run.py` was started with `--gated`.
