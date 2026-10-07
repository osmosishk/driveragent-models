# DriverGuard

Combined **YOLOPX perception + DTCP planner** running on the front camera
(`/tmp/cam0`). Engines come from `/home/tonyho/model/jetson_bundle/engines/`
(symlinked into `engines/`).

## Run standalone

```bash
python /home/tonyho/model/driverguard/run.py --pub-port 8014
```

Subscribe (any other shell):

```bash
python -c "
import sys; sys.path.insert(0, '/home/tonyho/driveragent')
from message.capnp_pubsub import Subscriber
s = Subscriber('/home/tonyho/driveragent/message/message.capnp',
               'DriverGuardResult', 'tcp://127.0.0.1:8014')
for _ in range(3):
    m = s.receive()
    print('wp =', [(p.x, p.y) for p in m.trajectory],
          'dets =', len(m.detections),
          'inf_ms =', m.inferenceMs,
          'da_rle =', len(m.daMaskRle), 'B')
"
```

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--pub-port` | 8014 | ZMQ publish port (DriverGuardResult) |
| `--cam` | `/tmp/cam0` | GStreamer shmsrc socket path |
| `--rate` | 10 (Hz) | Target inference rate |
| `--command` | 2 (STRAIGHT) | DTCP turn command (0=L 1=R 2=STRAIGHT 3=LANE_FOLLOW 4=CHANGE_L 5=CHANGE_R) |
| `--target` | `0 20` | DTCP target (lat_right_m, fwd_m) |
| `--gated` | off | Idle while `SelfDrivingStatus.enabled == false` |

## Wiring

| Where | Entry |
|---|---|
| Schema | `DriverGuardResult` + `DriverGuardDetection` in `driveragent/message/message.capnp` |
| Supervisor | `MODEL_REGISTRY["driverguard"]` in `driveragent/start.py` |
| UI databus | `_driverguard_thread` in `driveragent/ui/newwidgets/data_bus.py` |
| UI overlay | front-cam branch in `driveragent/ui/newwidgets/main_camera.py` |
