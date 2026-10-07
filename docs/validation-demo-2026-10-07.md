# Validation on demo (AGX Orin 64 GB), 2026-10-07

Device: `agxorin64`, L4T 36.4.7, JetPack 6.2.1, TensorRT 10.3.0, CUDA 12.6,
torch 2.8.0, onnxruntime 1.23.2 (CPU), power mode MAXN. Clean store:
`/opt/driveragent/models`. Harness scripts are not in the repo; the method is
below.

## 1. DriverGuard 1.0.0: engine in use vs rebuilt engine

"In use" = `~/model/driverguard/engines/{yolopx_v2_fp16,dtcp_v1_fp32}.engine`.
"Rebuilt" = `da-models build driverguard` (flags from models.yaml).
Same 3 nuScenes front frames (`validation_samples.tar.gz`), same registry
preprocessing, same ORT control sub-graph.

| Output | Max abs diff | Mean abs diff | Note |
|---|---|---|---|
| DTCP `cnn_feature`, `measurement_feature` | 0 | 0 | bit-identical |
| DTCP `pred_wp` | 4.8e-7 m | 1e-7 m | float rounding |
| DTCP `pred_speed` | 1.2e-7 | | |
| throttle / steer / brake | 0 | 0 | identical on all 3 frames |
| YOLOPX `det` (raw) | 2.4 | 0.014 | mean abs value 38 (pixel coords); FP16 tactic difference |
| YOLOPX boxes after NMS | | | same count (8/12/14), 100 % matched at IoU 0.5, same class |
| YOLOPX `da_seg` / `ll_seg` argmax | | | 99.97 % / 99.99 % same pixels |

`validate_jetson.py` (registry runtime, rebuilt engines): **PASS**, throttle
MAE 0.0018, steer MAE 0.0022 (threshold 0.02), finite on all frames. Same as
the May 2026 install record.

TensorRT device warning ("Using an engine plan file across different models
of devices"): **only** the engine in use `dtcp_v1_fp32.engine`. The rebuilt
engines and `yolopx_v2_fp16.engine` give no warning.

Rate (trtexec `--loadEngine`, 15 s, GPU compute mean):

| Engine | In use | Rebuilt |
|---|---|---|
| yolopx (FP16) | 7.98 ms, 125 qps | 7.79 ms, 128 qps |
| dtcp_main (FP32) | 7.75 ms, 129 qps | 7.88 ms, 127 qps |

End-to-end Python loop (preprocess + YOLOPX + DTCP + ORT control, 200 loops):
in use 109.0 ms (9.2 Hz), rebuilt 108.0 ms (9.3 Hz).

Build time on demo: yolopx 835 s, dtcp_main 26 s. Cache download on a second
store: 4 s.

### Low-workspace build (for Orin NX 8 GB)

`build driverguard --rebuild --workspace-mb 1024` in a scratch store: 1078 s,
peak `trtexec` RSS 2079 MB. Against the 4096 MiB engines: masks 100 % equal,
same boxes, same controls, `det` max abs diff 0.5. GPU compute, back to back:
yolopx 8.71 ms vs 8.72-8.75 ms; dtcp_main 8.76 ms vs 8.00 ms (+9.5 %).
(The GPU was warmer than in the first rate table, so all times are higher.)

### DriverGuard time per stage (rebuilt engines, Python loop)

| Stage | 12 cores | 8 cores (taskset) | 6 cores (taskset) |
|---|---|---|---|
| YOLOPX preprocess | 18.5 ms | 16.9 ms | 16.6 ms |
| YOLOPX TRT (incl. copies) | 28.1 ms | 28.6 ms | 28.2 ms |
| NMS | 1.8 ms | 1.8 ms | 1.8 ms |
| DTCP preprocess | 14.8 ms | 14.6 ms | 14.5 ms |
| DTCP TRT (incl. copies) | 25.8 ms | 26.4 ms | 26.3 ms |
| ORT control (CPU) | 2.1 ms | 10.1 ms | 10.0 ms |
| Total | 91.1 ms (11.0 Hz) | 98.4 ms (10.2 Hz) | 97.4 ms (10.3 Hz) |

Process RSS: about 1.0 GB.

## 2. System 1 1.0.1: installation vs registry

"Installation" = `~/model/system1` with its prebuilt `ops/*.so` (with
`~/model/sparsedrive/models_convnext` added to `sys.path`, because
`~/model/models_convnext` does not exist). "Registry" = `system1@1.0.1` with
the ops built by `da-models build` (87 s). Same checkpoint (same sha256), same
input (seed 0 images, vehicle calibration).

| Precision | Max abs diff (trajectory) | Latency installation | Latency registry |
|---|---|---|---|
| bf16 | 0 (bit-identical) | 211.8 ms | 212.3 ms |
| fp32 | 0 (bit-identical) | 264.4 ms | 264.6 ms |

Peak memory: CUDA allocations 1065 MB, process RSS 3.3 GB.
Entry point `run.py --once --no-cameras`: loads, 1 frame finite, publishes on
8011 (first frame 822 ms with warm-up).

`system1@1.0.0` is archived: its runtime has no `models/backbone.py`, so the
model cannot load.

## 3. system1-backbone 0.1.0 (candidate)

Builds with TensorRT 10.3 FP16 in 601 s; 6x3x256x704 input; GPU compute
60.8 ms (16.4 qps). Not compared numerically (no engine in use). The engine is
not uploaded.
