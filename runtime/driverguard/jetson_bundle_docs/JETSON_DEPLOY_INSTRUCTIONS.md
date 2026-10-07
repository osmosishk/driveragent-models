# Jetson deploy runbook — bundle `v1-fp32-2026-05-24`

**Audience:** the agent running on the on-vehicle Jetson AGX Orin.
**Goal:** rebuild the DTCP engine in FP32, validate it against the PC parity reference, swap it in, restart the driveragent services, and confirm the new telemetry tags the engine correctly. Total wall time: ~20 min including trtexec build.

**Why this bundle exists:** field telemetry over 4,673 frames of Ami driving (`/home/tonyho/datasets/ami/output/PHASE_D_SUMMARY.md`) showed throttle ≡ 0 and `finite == False` on every frame because the FP16 DTCP engine collapses the Beta-mode acceleration head. Same checkpoint on PC PyTorch is alive (throttle mean 0.024, max 0.45). Rebuilding the engine in FP32 should restore the on-vehicle outputs to match.

If anything in this runbook fails, do **not** improvise — jump to §10 (Rollback) and report the failure.

---

## 0. Prerequisites (one-time)

- `/usr/src/tensorrt/bin/trtexec` exists and runs.
- `~/jetson_bundle/` is the deploy root (create it if absent).
- The driveragent repo is at `~/driveragent/` with `start.py` runnable as documented in `driveragent/CLAUDE.md`.
- `gcloud` SDK or `gsutil` is installed and authenticated for the `carvideo_osmosisai` bucket. (If not, fall back to a Python-side download using the credentials at `/home/tonyho/development/gcs_upload.json` and the pattern in `/home/tonyho/development/ami_pipeline/gcs_download.py`.)

## 1. Pull the bundle from GCS

```bash
VERSION=v1-fp32-2026-05-24
gsutil -m cp -r gs://carvideo_osmosisai/engines/${VERSION}/ ~/jetson_bundle.staged/
```

This puts everything under `~/jetson_bundle.staged/${VERSION}/`. Move into place only after §2 passes.

### 1.b — Wire `driveragent/sync.py` to the engines pointer (one-time)

`driveragent/sync.py` already does MD5 asset sync from GCS. Add a small subcommand so it polls `gs://carvideo_osmosisai/engines/current.json` and stages new bundles without auto-deploying. Append to `driveragent/sync.py`:

```python
def check_engines(local_root="~/jetson_bundle"):
    """Poll engines/current.json; if version differs from INSTALLED.json,
    pull the bundle to ~/jetson_bundle.staged/<version>/ and log a notice.
    NEVER auto-deploys — requires the human to re-run JETSON_DEPLOY_INSTRUCTIONS.md."""
    import json, pathlib
    from google.cloud import storage
    client = storage.Client.from_service_account_json("/path/to/gcs_upload.json")
    bucket = client.bucket("carvideo_osmosisai")
    cur = json.loads(bucket.blob("engines/current.json").download_as_text())
    installed_p = pathlib.Path(local_root).expanduser() / "INSTALLED.json"
    installed = json.loads(installed_p.read_text()) if installed_p.exists() else {}
    if installed.get("version") == cur["version"]:
        return
    print(f"[sync] new bundle available: {cur['version']} (installed: {installed.get('version','none')})")
    staged = pathlib.Path("~/jetson_bundle.staged").expanduser() / cur["version"]
    staged.mkdir(parents=True, exist_ok=True)
    for blob in bucket.list_blobs(prefix=f"engines/{cur['version']}/"):
        rel = blob.name.split(f"engines/{cur['version']}/", 1)[1]
        if not rel:
            continue
        dest = staged / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(dest)
    print(f"[sync] pulled to {staged} — DEPLOY MANUALLY via JETSON_DEPLOY_INSTRUCTIONS.md")
```

Wire it into your existing cron / scheduler at hourly cadence. **It must not auto-deploy.** The human (or you) re-runs this runbook against the new staged version.

## 2. Verify bundle integrity

```bash
cd ~/jetson_bundle.staged/${VERSION}/
python3 - <<'PY'
import json, hashlib, sys
m = json.load(open("MANIFEST.json"))
for fname, meta in m["onnx"].items():
    h = hashlib.sha256(open(f"onnx/{fname}", "rb").read()).hexdigest()
    ok = (h == meta["sha256"])
    print(f"{'OK ' if ok else 'BAD'} {fname}  expected {meta['sha256'][:16]}…  got {h[:16]}…")
    if not ok: sys.exit(1)
print("manifest integrity OK")
PY
```

If any line says `BAD`, re-pull from GCS — do **not** continue.

## 3. Promote staged bundle to live and build engines

```bash
rm -rf ~/jetson_bundle.prev
[ -d ~/jetson_bundle ] && mv ~/jetson_bundle ~/jetson_bundle.prev
mv ~/jetson_bundle.staged/${VERSION} ~/jetson_bundle
chmod +x ~/jetson_bundle/build_engines.sh

~/jetson_bundle/build_engines.sh --precision fp32
```

trtexec output is mirrored to `~/jetson_bundle/engines/build_fp32_*.log`. Expect ~3–5 min on AGX Orin. Engines land at:

- `~/jetson_bundle/engines/dtcp_v1_fp32.engine`  ← the new one
- `~/jetson_bundle/engines/yolopx_v2_fp16.engine`

## 4. Validate — **go/no-go gate**

```bash
PYTHONPATH=~/jetson_bundle/jetson_runtime python3 \
  ~/jetson_bundle/validate_jetson.py \
  --dtcp-engine   ~/jetson_bundle/engines/dtcp_v1_fp32.engine \
  --yolopx-engine ~/jetson_bundle/engines/yolopx_v2_fp16.engine
```

**PASS criteria** (also in `MANIFEST.json:sample_thresholds`):
- throttle MAE vs PC reference < 0.02 on all 3 sample frames
- steer MAE < 0.02
- `finite == True` on all 3 frames

If the script prints `FAIL`, stop here. The new engine is not safe to deploy — go to §10 (Rollback) and surface the report.

## 5. Apply runner patches

```bash
cd ~/driveragent
git apply ~/jetson_bundle/runner_patches/0001-runner-cmd-remap.diff
git apply ~/jetson_bundle/runner_patches/0002-runner-steer-from-pred-wp.diff
```

If `git apply` complains because the runner module lives at a different path, open each `.diff`, find the relevant hunk, and apply by hand. Both patches are short — the bodies are repeated inline in the patch files.

## 6. Apply schema patch + regenerate capnp bindings

```bash
cd ~/driveragent
git apply ~/jetson_bundle/runner_patches/0003-message-capnp-engine-tag.diff
```

The capnp schema picks up the new fields automatically the next time the proxy or any subscriber imports `message.capnp` via pycapnp (no codegen step required for pycapnp). Make sure to also update the runner emitter to populate the two new fields:

```python
result.enginePrecision = "FP32"            # or "FP16" / "FP16+beta_fp32"
result.modelVersion    = "v1-fp32-2026-05-24"   # MANIFEST.json:version
```

(Read the bundle's `MANIFEST.json:version` at startup so you don't hand-edit it on every release.)

## 7. Point the runner at the new engine + restart driveragent

```bash
# Update the engine path in your runner config (path varies — typically
# control_config.ini, runner.py top-level constant, or an env var):
#   DTCP_ENGINE=$HOME/jetson_bundle/engines/dtcp_v1_fp32.engine
# (Keep the FP16 engine on disk for rollback — do not delete it.)

# SIGHUP to start.py picks up the new engine for the relevant subprocess(es).
pkill -SIGHUP -f 'python.*start.py' || python ~/driveragent/start.py &
```

## 8. Watch for 60 s — confirm `finite == True` and engine tag is correct

```bash
# Tail the latest in-progress messages.bz2 (logger.uploader rotates ~5 min).
LATEST=$(ls -1t ~/<your-logger-output-dir>/messages*.bz2 | head -1)
PYTHONPATH=/home/tonyho/development/ami_pipeline python3 - <<PY
from parse_messages import load_segment_log
s = load_segment_log("$LATEST")
n = len(s["driver_guard"])
finite = sum(r.finite for r in s["driver_guard"])
thr = [r.throttle for r in s["driver_guard"] if r.throttle > 0]
print(f"driver_guard records: {n}")
print(f"  finite==True: {finite}/{n}  ({100.0*finite/max(n,1):.1f}%)")
print(f"  throttle nonzero: {len(thr)}/{n}  (max {max(thr or [0]):.4f})")
PY
```

**Expected:**
- `finite==True` on the great majority of records (≥99 %)
- throttle nonzero on a meaningful fraction (≥10 %) of records during driving

If those numbers don't move from the pre-deploy baseline (`PHASE_D_SUMMARY.md`: 0 % finite, 1.4 % nonzero throttle), the FP32 engine isn't actually loaded — check the runner config and the trtexec build log.

## 9. Mark the deploy + close the loop

Write `~/jetson_bundle/INSTALLED.json`:

```json
{
  "version": "v1-fp32-2026-05-24",
  "installed_at": "<ISO ts>",
  "engine_paths": {
    "dtcp":   "/home/<user>/jetson_bundle/engines/dtcp_v1_fp32.engine",
    "yolopx": "/home/<user>/jetson_bundle/engines/yolopx_v2_fp16.engine"
  },
  "validate_output": "<paste the PASS block from §4 here>"
}
```

The next `logger.uploader` cycle will push a `messages.bz2` segment tagged with the new engine; the server-side `ami_pipeline/ingest_loop.sh` will pick it up and surface it in the next `aggregate_metrics.csv` under `model_version == "v1-fp32-2026-05-24"`.

## 10. Rollback

If §4 fails, or §8 shows no improvement, restore the previous engine and bundle:

```bash
# 1) Point runner back at the FP16 engine (keep the FP32 .engine in place for inspection):
#    DTCP_ENGINE=$HOME/jetson_bundle.prev/engines/dtcp_v1_fp16.engine
# 2) Swap bundle dir back:
rm -rf ~/jetson_bundle.broken
mv ~/jetson_bundle ~/jetson_bundle.broken
mv ~/jetson_bundle.prev ~/jetson_bundle
# 3) Revert runner patches if you applied them in this session:
cd ~/driveragent && git checkout -- runner/runner.py message/message.capnp
# 4) SIGHUP start.py:
pkill -SIGHUP -f 'python.*start.py'
```

Report back with: the `validate_jetson.py` FAIL output (or §8 numbers), the build log path (`~/jetson_bundle.broken/engines/build_fp32_*.log`), and a copy of one fresh `messages.bz2` segment from the failing engine.

## 11. (Optional) Per-layer FP32 fallback if FP32 is too slow

If end-to-end latency at 10 Hz blows the budget (rough rule: DTCP infer > 60 ms on AGX Orin), build a third variant that keeps most layers FP16 but forces only the Beta-mode acceleration head layers to FP32:

```bash
# 1) Discover Beta-head layer names from the ONNX graph or trtexec --verbose:
/usr/src/tensorrt/bin/trtexec --onnx=~/jetson_bundle/onnx/dtcp_v1.onnx --verbose 2>&1 | \
  grep -i 'beta\|alpha\|action_mu\|action_sigma' | head -20
# 2) Build the hybrid engine (substitute the actual layer name(s) found above):
/usr/src/tensorrt/bin/trtexec \
    --onnx=~/jetson_bundle/onnx/dtcp_v1.onnx \
    --saveEngine=~/jetson_bundle/engines/dtcp_v1_fp16_beta_fp32.engine \
    --fp16 --workspace=4096 \
    --shapes=image:1x3x256x928,state:1x9,target_point:1x2 \
    --layerPrecisions=<beta_head_layer_name>:fp32 \
    --precisionConstraints=obey
# 3) Re-run validate_jetson.py with --dtcp-engine pointing at the hybrid.
#    Update enginePrecision tag in the runner: "FP16+beta_fp32"
```

This is **a follow-up** — not required for first deploy. Don't attempt unless the FP32 engine measurably fails latency budget.
