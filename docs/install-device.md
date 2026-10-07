# Install a new device (AGX Orin or Orin NX)

This procedure installs `da-models` with the **read-only** credential, then
pulls and builds the models. Do it on each new unit. Do not put the publisher
key on these units.

## 1. Check the base system

1. Find the JetPack and TensorRT versions. They select the DriverGuard
   version:
   ```
   cat /etc/nv_tegra_release           # R36 (release), REVISION: x.y
   ls /usr/src/tensorrt/bin/trtexec    # must exist (not on PATH)
   python3 -c "import tensorrt; print(tensorrt.__version__)"
   ```

   | L4T | JetPack | TensorRT | DriverGuard version |
   |---|---|---|---|
   | R36.4.x | 6.1 / 6.2.x | 10.3 | `driverguard@1.0.0` (production) |
   | R36.3.0 | 6.0 | 8.6.2 | `driverguard@1.0.1` (candidate, TensorRT 8.6 variant) |

   The engine tag contains the TensorRT version, so engines for TensorRT 8.6
   and 10.3 never mix in the cache.
2. If `trtexec` or `tensorrt` is missing, install the JetPack packages:
   `sudo apt install nvidia-jetpack`. Do not do this on a unit that another
   team owns without their approval.

## 2. Install the tool

1. Get read access to the private repo. Add a **read-only deploy key** for
   this unit in GitHub (repo Settings > Deploy keys). Do not put a token in
   `~/.gitconfig`.
2. Clone the repo:
   ```
   git clone git@github.com:osmosishk/driveragent-models.git ~/driveragent-models
   ```
   **If the unit has no GitHub key yet** (as on `orin-nx`), push the repo from
   `demo` over SSH (only `~/driveragent-models` changes on the unit):
   ```
   # on the unit:
   git init -q -b main ~/driveragent-models
   git -C ~/driveragent-models config receive.denyCurrentBranch updateInstead
   # on demo:
   GIT_SSH_COMMAND="ssh -i ~/.ssh/<key>" git push ssh://<user>@<unit>/home/<user>/driveragent-models main
   # on the unit:
   git -C ~/driveragent-models config --unset receive.denyCurrentBranch
   git -C ~/driveragent-models remote add origin git@github.com:osmosishk/driveragent-models.git
   ```
   To update `models.yaml` by itself later (`git -C ~/driveragent-models pull`),
   the unit needs a read-only GitHub deploy key.
3. Install the Python packages that the manifest requires, **one at a time,
   without dependencies**, so that pip cannot change torch, numpy or other
   packages. Use PyPI only (the NVIDIA extra index can fail), and no cache:
   ```
   PIP="pip3 install --user --isolated --index-url https://pypi.org/simple --no-cache-dir --no-deps"
   ```
   Check first which packages are missing (`python3 -c "import importlib.metadata as m; print(m.version('<pkg>'))"`).
   The DriverGuard runtime and `da-models` need: `tensorrt` (JetPack), `pycuda`,
   `onnxruntime`, `cv2`, `numpy`, `zmq`, `capnp`, `yaml`, `google-cloud-storage`.
   Set used on `orin-nx` (JetPack 6.0), in this order:
   ```
   for p in humanfriendly==10.0 coloredlogs==15.0.1 flatbuffers==25.12.19 onnxruntime==1.23.2 \
            platformdirs==4.12.3 siphash24==1.9 pytools==2026.1.1; do
     $PIP --only-binary=:all: "$p"
   done
   # pycuda has no aarch64 wheel: it compiles (about 2 min on Orin NX). nvcc is not on PATH on JetPack 6.0.
   tmux new-session -d -s da-pycuda "export PATH=/usr/local/cuda/bin:\$PATH CUDA_ROOT=/usr/local/cuda; \
     nice -n 19 $PIP --no-build-isolation --no-binary pycuda pycuda==2026.1 > ~/handoff/logs/pycuda-build.log 2>&1"
   python3 -c "import pycuda.driver as d; d.init(); print(d.Device(0).name())"
   ```
4. For System 1 only: PyTorch 2.8 for JetPack 6 must already be installed.
   Install timm **without dependencies**, so that pip cannot change torch:
   ```
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # record
   for p in timm==1.0.26 safetensors==0.7.0 huggingface_hub==0.36.0 tqdm==4.67.3 hf-xet==1.5.0; do
     $PIP --no-deps "$p"
   done
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must be the same
   ```
5. Optional: put the tool on the PATH:
   `ln -s ~/driveragent-models/da-models ~/.local/bin/da-models`

## 3. Set the read-only credential

```
install -d -m 700 ~/.config/driveragent-models
cp <reader key file> ~/.config/driveragent-models/reader.json
chmod 600 ~/.config/driveragent-models/reader.json
```

## 4. Make the local store

```
sudo install -d -o $USER -g $USER /opt/driveragent/models
```

## 5. Check the installation

```
cd ~/driveragent-models
./da-models device            # device name, JetPack, TensorRT, tag
./da-models list
./da-models verify --remote   # proves that the read-only key can read the bucket
```

## 6. Pull and build

```
./da-models pull driverguard
./da-models build driverguard
./da-models verify --deep
python3 /opt/driveragent/models/driverguard/current/runtime/validate_jetson.py
```

- If the cache has an engine for this tag, `build` downloads it.
- If not, `build` runs `trtexec` on the unit. It then prints
  `upload not permitted with this credential (read-only)`. This is
  correct: the engine stays on this unit, and the exit code is 0.
- `validate_jetson.py` must print `PASS`.

A candidate or archived version needs the version number, for example
`./da-models pull system1@1.0.0`.

## 7. Orin NX settings

An engine cache entry is valid only for one tag. An Orin NX has a different
tag (`orinnx16-...` or `orinnx8-...`), so the **first** build on each Orin NX
runs `trtexec` on that unit. With the read-only credential the engine is not
uploaded, so **each** Orin NX unit builds its engines one time.

Before the build:
```
sudo nvpmodel -m 0        # MAXN (MAXN SUPER on JetPack 6.2 and later)
sudo jetson_clocks
free -h                   # make sure that swap (zram) is on
```

| Setting | Orin NX 16 GB | Orin NX 8 GB |
|---|---|---|
| TensorRT workspace | default (4096 MiB) | `./da-models build driverguard --workspace-mb 1024` |
| Peak build memory (measured on demo) | about 2.1 GB `trtexec` RSS + workspace + about 0.5 GB engine | about 3.5 GB in total with 1024 MiB |
| System 1 CUDA ops compile | `MAX_JOBS=4 ./da-models build system1@<v>` | `MAX_JOBS=2 ...` (each nvcc job can use 1-2 GB) |
| Other GPU programs during the build | stop them if possible | stop them |
| Expected build time (yolopx FP16) | about 30-50 min (demo: 14-17 min) | about 35-60 min |

The 1024 MiB workspace was tested on `demo`: same outputs (masks 100 % equal,
same controls), yolopx same rate, dtcp_main 9.5 % slower.

### Measured on orin-nx (Orin NX 16 GB, JetPack 6.0, TensorRT 8.6), 2026-10-07

| Item | Measured |
|---|---|
| DriverGuard 1.0.1 full frame (JPEG 1600x900) | 114.4 ms (8.7 Hz); about 98 ms without JPEG decode |
| Memory | 570 MB process RSS; build peak 8.9 GB system RAM (HMI included) |
| Build time (workspace 4096 MiB) | YOLOPX 23.7 min, DTCP main 53 s; cache download 4 s |
| Validation | PASS, throttle MAE 0.0018, steer MAE 0.0022 |

Details: `docs/validation-orinnx-2026-10-07.md`. Use `--rate 8` for DriverGuard
on an Orin NX 16 GB until a live run is measured.

### Fit and rate (estimates from demo measurements, before the real unit)

| Model | Memory | Orin NX 16 GB | Orin NX 8 GB |
|---|---|---|---|
| DriverGuard 1.0.0 | about 1.0 GB RSS + 0.3 GB GPU | fits | fits |
| DriverGuard rate (target 10 Hz) | demo: 91-98 ms per frame (10-11 Hz) | about 125-165 ms (6-8 Hz): **too slow for 10 Hz** | about 130-180 ms (5.5-7.5 Hz): **too slow for 10 Hz** |
| System 1 1.0.1 | 3.3 GB RSS + 1.1 GB CUDA | fits | **does not fit** with DriverGuard and the OS |
| System 1 rate (default 5 Hz) | demo: 212 ms (4.7 Hz), already below 5 Hz | about 470-720 ms (1.4-2.1 Hz): **too slow** | not applicable |

The DriverGuard estimate for the Orin NX 16 GB was pessimistic (see the measured table above). The Orin NX 8 GB and System 1 rows are still estimates.

### Measure on the unit

The tag comes from the unit: for example `orin-nx` (Orin NX 16 GB, JetPack 6.0,
TensorRT 8.6.2) gets `orinnx16-jp6.0-trt8.6.2-<precision>`, and an Orin NX with
JetPack 6.2 would get `orinnx16-jp6.2-trt10.3.0-<precision>`. Check it with
`./da-models device`.

```
sudo jetson_clocks                                   # fixed clocks: DVFS changes the figures
python3 tools/bench_driverguard.py                   # time split, ms
python3 tools/bench_system1.py --model-dir /opt/driveragent/models/system1/1.0.1
python3 tools/live_rate.py --seconds 600             # when the models run on live data
```

On `demo`, DVFS lowered the GPU to 306-408 MHz (max 1300 MHz) during the
DriverGuard loop. Report whether `jetson_clocks` was on with each result.
