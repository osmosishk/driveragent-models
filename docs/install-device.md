# Install a new device (AGX Orin or Orin NX)

This procedure installs `da-models` with the **read-only** credential, then
pulls and builds the models. Do it on each new unit. Do not put the publisher
key on these units.

## 1. Check the base system

1. Make sure that the unit has JetPack 6.1 or later. JetPack 6.2.1 is the
   tested version.
   ```
   cat /etc/nv_tegra_release           # R36 (release), REVISION: 4.x
   ls /usr/src/tensorrt/bin/trtexec    # must exist
   python3 -c "import tensorrt; print(tensorrt.__version__)"   # 10.3 or later
   ```
2. If `trtexec` or `tensorrt` is missing, install the JetPack packages:
   `sudo apt install nvidia-jetpack`.

## 2. Install the tool

1. Get read access to the private repo. Add a **read-only deploy key** for
   this unit in GitHub (repo Settings > Deploy keys). Do not put a token in
   `~/.gitconfig`.
2. Clone the repo:
   ```
   git clone git@github.com:osmosishk/driveragent-models.git ~/driveragent-models
   ```
3. Install the Python packages. Use PyPI only (the NVIDIA extra index in the
   default pip.conf can fail):
   ```
   PIP="pip3 install --user --isolated --index-url https://pypi.org/simple"
   $PIP google-cloud-storage pyyaml onnxruntime
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

### Fit and rate (estimates from demo measurements; confirm on a real unit)

| Model | Memory | Orin NX 16 GB | Orin NX 8 GB |
|---|---|---|---|
| DriverGuard 1.0.0 | about 1.0 GB RSS + 0.3 GB GPU | fits | fits |
| DriverGuard rate (target 10 Hz) | demo: 91-98 ms per frame (10-11 Hz) | about 125-165 ms (6-8 Hz): **too slow for 10 Hz** | about 130-180 ms (5.5-7.5 Hz): **too slow for 10 Hz** |
| System 1 1.0.1 | 3.3 GB RSS + 1.1 GB CUDA | fits | **does not fit** with DriverGuard and the OS |
| System 1 rate (default 5 Hz) | demo: 212 ms (4.7 Hz), already below 5 Hz | about 470-720 ms (1.4-2.1 Hz): **too slow** | not applicable |

Use `--rate 5` for DriverGuard on an Orin NX until a real unit is measured.
