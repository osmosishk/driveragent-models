# driveragent-models

Model registry for DriverAgent. All Jetson units (AGX Orin and Orin NX) pull
the same model versions from one place.

- **GitHub (this repo)** holds the manifest `models.yaml`, the tool
  `da-models`, the release specs and the documents. It holds no model
  binaries. Do not use Git LFS.
- **GCS bucket `gs://driveragent-model`** (project `ancient-binder-127813`)
  holds the binaries.

## 1. Rules

1. The source artifact is the ONNX file (or the PyTorch checkpoint for a
   PyTorch model) with its config and pre/post-processing parameters. Each
   device builds its TensorRT engine itself. An engine from an AGX Orin is not
   valid on an Orin NX, or on a different JetPack or TensorRT version.
2. A published version is **immutable**. To change a file, publish a new
   version. The publisher identity has no delete permission, so GCS itself
   refuses an overwrite.
3. Only the host `demo` (the AGX Orin publisher) has the write identity. All
   other devices use the read-only identity.
4. Do not commit keys or tokens. `.gitignore` blocks `*.json` outside `releases/`.

## 2. Bucket layout

```
gs://driveragent-model/
  models/<model>/<version>/<files...>          source artifacts + runtime.tar.gz
  models/<model>/<version>/model.json          written last: the version is complete
  engines/<model>/<version>/<tag>/<engine>     engine cache
  engines/<model>/<version>/<tag>/<engine>.json  build record (flags, versions, sha256)
```

`<tag>` is `<device>-jp<jetpack>-trt<tensorrt>-<precision>`, for example
`agxorin64-jp6.2.1-trt10.3.0-fp32`. A PyTorch extension build uses
`<device>-jp<jetpack>-torch<torch>-<precision>`. Run `da-models device` to
see the values for a unit.

| Device name | Module |
|---|---|
| `agxorin64` | P3701-0005, AGX Orin 64 GB |
| `agxorin32` | P3701-0000 / -0004, AGX Orin 32 GB |
| `orinnx16` | P3767-0000, Orin NX 16 GB |
| `orinnx8` | P3767-0001, Orin NX 8 GB |

## 3. Status

| Status | `pull <model>` | `pull <model>@<version>` | `list` |
|---|---|---|---|
| `production` | yes (newest production) | yes, and sets `current` | yes |
| `candidate` | no | yes, `current` changes only with `--set-current` | yes |
| `archived` | no | yes, `current` changes only with `--set-current` | yes |

You can change `status` in `models.yaml` by hand (then commit). You cannot
change the files of a version.

## 4. Local store

Default root: `/opt/driveragent/models` (set `DA_MODELS_ROOT` to change it).

```
/opt/driveragent/models/
  <model>/current -> <version>             symlink, switched atomically
  <model>/<version>/                       files from the bucket (sha256 checked)
  <model>/<version>/model.json             local record of the manifest entry
  <model>/<version>/runtime/               runtime code from runtime.tar.gz
  <model>/<version>/engines/<tag>/         engines built or downloaded for this device
```

`pull` downloads into `<root>/.tmp/`, checks the size and sha256 of each
file, and then renames the directory into place in one step. A failed pull
leaves nothing in `<model>/`.

## 5. Commands

```
da-models list                             models, versions, status, local state
da-models pull <model>[@version]           download, check sha256, move into place
da-models build <model>[@version]          engines for this device (default: current)
da-models verify [<model>[@version]]       check the local store
          [--deep]                         also load each engine/ONNX, check tensors
          [--remote]                       also check the bucket against models.yaml
da-models publish releases/<spec>.yaml     upload a new version (publisher only)
          [--dry-run]
da-models cache-put <model>@<version> <component> <engine>
                                           put an engine built here into the cache
da-models device                           device, JetPack, TensorRT, torch versions
```

### build

For each component of the version:

| Runtime | What `build` does |
|---|---|
| `tensorrt` | 1. Use the local engine if its record agrees. 2. Else download from the cache if the tag (device + JetPack + TensorRT + precision) agrees, check sha256 and load the engine. 3. Else run `trtexec` with the flags from `models.yaml`, then upload to the cache. |
| `pytorch` with `build:` | The same steps for the compiled extensions (`ops-build.tar.gz`). It loads each `.so` with torch to check the ABI. |
| `onnxruntime-cpu` | Nothing to build. It loads the ONNX in onnxruntime and checks the tensor names. |

If the credential cannot write (all devices except `demo`), the upload is
refused. `build` then prints a notice, keeps the engine on the device and
exits with code 0.

### publish

1. Write a release spec in `releases/<model>-<version>.yaml`. See the
   examples in that folder.
2. Commit and push the runtime code. `publish` stops when the git working
   tree is not clean or the commit is not on a remote branch. It makes
   `runtime.tar.gz` with `git archive` and records the commit SHA.
3. Run `./da-models publish releases/<spec>.yaml --dry-run`, then run it
   without `--dry-run`.
4. Commit `models.yaml` and push.

If an upload stops halfway, run the same command again. Objects with the same
sha256 are kept, and the rest are uploaded. `model.json` is uploaded last.

## 6. Credentials

| Host | File | Roles on the bucket only |
|---|---|---|
| `demo` | `~/.config/driveragent-models/publisher.json` | `roles/storage.objectCreator`, `roles/storage.objectViewer` |
| all other units | `~/.config/driveragent-models/reader.json` | `roles/storage.objectViewer` |

Make the folder with mode 700 and the key with mode 600. The tool also
reads `DA_MODELS_CREDENTIALS` or `--credentials`.

## 7. Files that are not published

| File on `demo` | Reason |
|---|---|
| `~/model/driverguard/engines/dtcp_v1_fp16.engine` | Old rollback engine. Its outputs are not finite (mu/sigma collapse, finite = 0 %). No file proves which ONNX made it, so no other device can rebuild it. The file stays on `demo`. |
| `~/model/jetson_bundle/onnx/dtcp_v1.onnx`, `dtcp_v1_fix.onnx`, `dtcp_v1.onnx.broken` | Full-graph DTCP exports. TensorRT 10.3 gives wrong control outputs for the full graph. DriverGuard 1.0.0 uses `dtcp_v1_main.onnx` + `dtcp_v1_control.onnx`. |
| `~/driveragent/calibration/*` | Camera calibration is data for one vehicle, not model data. |

## 8. Runtime code

The runtime code for each model is in `runtime/<model>/`. `publish` packs
that folder from a clean, pushed commit into `runtime.tar.gz`. `pull`
extracts it into `<root>/<model>/<version>/runtime/`.

| Folder | Content | Upstream |
|---|---|---|
| `runtime/driverguard/` | `run.py`, `runner/`, `jetson_runtime/`, `validate_jetson.py`, docs | Osmosis code (from `~/model/driverguard`, `~/model/jetson_bundle` on `demo`) |
| `runtime/system1/system1/` | System 1 package, custom CUDA ops source | Osmosis code (from `~/model/system1`) |
| `runtime/system1/models_convnext/`, `runtime/system1/models/backbone.py` | ConvNeXt V2 backbone (uses timm); FPN and GridMask that it loads by file path | From `~/model/sparsedrive` on `demo`. **No upstream counterpart** in https://github.com/swc-17/SparseDrive (upstream uses an mmdet3d ResNet backbone). Treat as Osmosis code. |
| `runtime/yolopx/` | `lib/`, `tools/demotext.py`, `LICENSE` | https://github.com/jiaoZ7688/YOLOPX, commit `35627f645ef84baee93eef33283b807c42da77d3` (MIT, see `runtime/yolopx/LICENSE`). `tools/demotext.py` is a local addition. |

### SparseDrive upstream (compared 2026-10-07)

Upstream: https://github.com/swc-17/SparseDrive, MIT License, Copyright (c)
2024 swc-17. A copy of the upstream LICENSE is in
`runtime/system1/LICENSE.SparseDrive` (unchanged since commit `ffebeb4`).

Only the System 1 custom CUDA ops come from upstream
`projects/mmdet3d_plugin/ops/`. Line similarity to the nearest upstream file:

| Local file (`runtime/system1/system1/ops/`) | Upstream file | Similarity |
|---|---|---|
| `src/deformable_aggregation_cuda.cu` | `src/deformable_aggregation_cuda.cu` | 0.72 |
| `setup.py` | `setup.py` | 0.72 |
| `src/deformable_aggregation.cpp` | `src/deformable_aggregation.cpp` | 0.64 |
| `src/deformable_aggregation_with_depth.cpp` | `src/deformable_aggregation.cpp` | 0.54 |
| `deformable_aggregation.py` | `__init__.py` | 0.37 (partly) |

Nearest commit: the upstream code of these files is the same in all commits
from `ffebeb4` (2024-06-24, "release") to `ec0225d` (2026-04-01). The local
files are dated 2026-05-07, so the nearest commit is **`ec0225d`** (exact base
commit unknown). Local changes: THC atomics replaced by ATen
(`ATen/cuda/Atomic.cuh`), changed kernel indexing (no `num_anchors`), a new
`deformable_aggregation_with_depth` extension, and `setup.py` builds both.
The upstream fix `4958e1e` (2026-09-15, large-batch indexing) is **not** in the
local code. Other System 1 files (`scorer/`, `models/`, `models_convnext/`) have
no upstream match (similarity 0.37 or less).

Commit `4ff9985` is a byte-identical copy of the code on `demo`. Later commits
change the code. The services on `demo` still use the old folders; wiring them
to `/opt/driveragent/models/<model>/current` is a separate task.

Paths in the runtime code:
- Model files: relative to the runtime folder (`<version>/runtime/..`).
- The DriverAgent repo (capnp schema, calibration): `$DRIVERAGENT_ROOT`,
  default `~/driveragent`.

## 9. Requirements on a device

- JetPack 6.1 or later, TensorRT 10.3 or later (`trtexec` at
  `/usr/src/tensorrt/bin/trtexec`, or set `TRTEXEC`)
- Python 3.10: `pip3 install google-cloud-storage pyyaml onnxruntime`
- `onnx` only on the publisher host
- PyTorch for the PyTorch models. System 1 also needs `timm==1.0.26`.
  Install it with `--no-deps` so that pip does not change torch (see
  `docs/install-device.md`).

## 10. Tests

```
python3 -m unittest discover -s tests -v
```

The tests use a local `file://` bucket and a very small ONNX model. They run
`trtexec`, so run them on a Jetson.
