# Validation on orin-nx (Orin NX 16 GB), 2026-10-07

Unit: `orin-nx` (p3767-0000 on a p3768-compatible carrier), L4T R36.3.0 =
JetPack 6.0, CUDA 12.2, cuDNN 8.9.4, **TensorRT 8.6.2.3**, Python 3.10.12,
torch 2.4.0a0 (nv24.7), onnxruntime 1.23.2 (CPU), pycuda 2026.1. Power mode
MAXN, DVFS on (GPU 306-918 MHz). The bench HMI of another team ran on the unit
during all steps (GPU 18 % alone).

Version: `driverguard@1.0.1` (candidate, TensorRT 8.6 variant of 1.0.0).
Tag: `orinnx16-jp6.0-trt8.6.2-<precision>`.

## 1. Build (da-models build, reader key, --no-upload)

| Engine | Build time | trtexec GPU compute (mean) |
|---|---|---|
| YOLOPX FP16 | 1421.5 s | 25.0 ms |
| DTCP main FP32 (`--noTF32`) | 52.9 s | 26.8 ms |

Peak system RAM during the build: 8891 MB of 15656 MB (HMI included).
`trtexec` RSS about 2.1 GB. Total GPU load (GR3D, HMI included) mean 52 %,
p95 96 %; only 6.5 % of the samples were at 90 % or more. trtexec timings ran
beside the HMI and warned that GPU compute time is unstable (CoV about 3.5 %).

TensorRT 8.6 warnings (no per-layer warnings at the default log level):
- Both: "ONNX model has been generated with INT64 weights ... cast down to INT32".
- YOLOPX: one value clamped: INT64_MAX (9223372036854775807) to INT32_MAX, the
  `ends` input of `Slice_641` / `Slice_724` ("slice to the end"; harmless).
- YOLOPX FP16: 128 weights with subnormal FP16 values, 42 weights below the
  smallest FP16 subnormal (set to the minimum). Offline analysis of the ONNX
  (147 / 47 tensors): all in `Conv` layers, mainly `Conv_380`, `Conv_359`,
  `Conv_121`, `Conv_349`, `Conv_338`.

## 2. Validation

`validate_jetson.py` (checks DTCP + control; YOLOPX only load-checked): **PASS**. Throttle MAE 0.0018, steer MAE 0.0022 (limit
0.02), finite on all 3 frames. Identical to AGX Orin / TensorRT 10.3.

Per component, Orin NX (TensorRT 8.6) vs AGX Orin (TensorRT 10.3), same 3 frames:

| Component | Result |
|---|---|
| YOLOPX | boxes 8/12/14 on both, 100 % matched (IoU 0.5); drivable-area mask IoU 0.9974-0.9999, lane mask IoU 0.9980-0.9982; raw `det` max abs diff 1.6-4.8 (mean 0.014, mean abs value 38) |
| YOLOPX vs PC reference (`reference_yolopx/*.npz` in `validation_samples.tar.gz`, frames f0000-f0002) | 100 % boxes matched; mask IoU 0.9975-0.9999 (AGX: 0.9968-0.9999) |
| DTCP main | max abs diff: `pred_wp` 3.5e-6, `pred_speed` 0, `cnn_feature` 7.6e-6, `measurement_feature` 1.2e-6 |
| Control (ORT) | `mu` / `sigma` max abs diff 4.8e-7; throttle/steer/brake identical |

Engine cache: both engines were imported with `da-models cache-import` (from
`demo`, publisher key). In a clean store on the unit, `build` took them from
the cache in 4 s (`TRTEXEC=/bin/false`), byte-identical, `verify --deep` OK,
`validate_jetson.py` PASS.

## 3. Time per frame (`tools/bench_driverguard.py --loops 300`)

One nuScenes front frame 1600x900 (JPEG), decoded again on each loop; a stream
sync after each phase; "full frame" is the sum of the stage means. HMI running,
no `jetson_clocks`.

| Stage | Mean | p95 |
|---|---|---|
| Decode JPEG | 16.73 ms | 25.02 ms |
| YOLOPX preprocess | 14.38 ms | 19.87 ms |
| YOLOPX H2D / compute / D2H | 0.85 / 29.39 / 1.50 ms | 0.98 / 35.94 / 1.76 ms |
| YOLOPX NMS / masks | 0.97 / 3.99 ms | 1.24 / 5.30 ms |
| DTCP preprocess | 10.36 ms | 15.54 ms |
| DTCP H2D / compute / D2H | 1.24 / 32.12 / 0.42 ms | 1.63 / 36.88 / 0.51 ms |
| Control: ORT / beta | 2.21 / 0.21 ms | 7.87 / 0.21 ms |
| DTCP postprocess | 0.02 ms | 0.02 ms |
| Sum of the stages of the "108 ms" definition | 92.7 ms | |
| **Full frame** | **114.4 ms (8.74 Hz)** | |

Process peak RSS 570 MB. During the benchmark: GR3D mean 59 %, p95 99 %;
system RAM 6784-6974 MB (6570 MB with the HMI alone; the Jetson nvmap page
pool makes this delta a lower bound); VDD_IN 16.2 W mean, 18.8 W max.

## 4. Against the Phase 4 estimates

| Item | Estimate (from demo) | Measured |
|---|---|---|
| DriverGuard frame | 125-165 ms (6-8 Hz) | 114.4 ms full frame (8.7 Hz); about 98 ms without JPEG decode |
| Fits in memory (16 GB) | yes | yes: 570 MB RSS; build peak 8.9 GB system |
| YOLOPX build time | 30-50 min | 23.7 min |

The estimate was pessimistic because the `demo` figures were measured with low
DVFS clocks. Pure GPU time on the Orin NX (trtexec) is about 2.9x that of AGX
(YOLOPX 25.0 vs 8.7 ms).
