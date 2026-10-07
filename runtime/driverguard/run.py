"""DriverGuard entry point.

YOLOPX (drivable area + lane mask + 10-class detection) + DTCP (4-waypoint
trajectory + throttle/steer/brake) running off a single front camera at
/tmp/cam0.

Registry layout (da-models): this file is <version>/runtime/run.py. The model
files are in <version>/, and `da-models build driverguard` puts the engines for
this device in <version>/engines/<tag>/. The DriverAgent repo (capnp schema)
is $DRIVERAGENT_ROOT (default ~/driveragent).

    python <version>/runtime/run.py --pub-port 8014
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))     # <version>/runtime
MODEL_DIR = os.path.dirname(ROOT)                      # <version>
DEFAULT_DRIVERAGENT_ROOT = os.path.expanduser(os.environ.get("DRIVERAGENT_ROOT", "~/driveragent"))
DEFAULT_DTCP_CONTROL_ONNX = os.path.join(MODEL_DIR, "dtcp_v1_control.onnx")
DEFAULT_SCHEMA = os.path.join(DEFAULT_DRIVERAGENT_ROOT, "message", "message.capnp")


def _find_engine(name):
    """The engine that `da-models build` made for this device, or None."""
    hits = sorted(glob.glob(os.path.join(MODEL_DIR, "engines", "*", name)))
    return hits[0] if len(hits) == 1 else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolopx-engine", default=_find_engine("yolopx_v2.engine"),
                   help="default: <version>/engines/<tag>/yolopx_v2.engine")
    p.add_argument("--dtcp-engine", default=_find_engine("dtcp_v1_main.engine"),
                   help="default: <version>/engines/<tag>/dtcp_v1_main.engine")
    p.add_argument("--control-onnx", default=DEFAULT_DTCP_CONTROL_ONNX,
                   help="DTCP control sub-graph ONNX (mu/sigma via onnxruntime CPU)")
    p.add_argument("--rate", type=float, default=10.0,
                   help="Target inference rate in Hz (default 10)")
    p.add_argument("--pub-port", type=int, default=8014,
                   help="ZMQ port for DriverGuardResult publisher (default 8014)")
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--driveragent-root", default=DEFAULT_DRIVERAGENT_ROOT)
    p.add_argument("--cam", default="/tmp/cam0",
                   help="Path to GStreamer shmsrc socket (default /tmp/cam0)")
    p.add_argument("--command", type=int, default=2,
                   help="DTCP command index (0=L 1=R 2=STRAIGHT 3=LANE_FOLLOW "
                        "4=CHANGE_L 5=CHANGE_R). Default 2 (STRAIGHT)")
    p.add_argument("--target", type=float, nargs=2, default=[0.0, 20.0],
                   metavar=("LAT_RIGHT_M", "FWD_M"),
                   help="DTCP target point (lat_right, fwd) in meters, used as "
                        "the fallback when no route guidance. Default (0, 20)")
    p.add_argument("--route-guidance-addr", default="tcp://127.0.0.1:5605",
                   help="RouteGuidance ZMQ address. The model follows the live "
                        "dtcpCommand + target from the active map route, and "
                        "falls back to --command/--target when it is stale.")
    p.add_argument("--no-route-guidance", action="store_true",
                   help="Ignore RouteGuidance; use the static --command/--target")
    p.add_argument("--gated", action="store_true",
                   help="Only run inference while SelfDrivingStatus.enabled")
    p.add_argument("--sd-status-addr", default="tcp://127.0.0.1:5595")
    p.add_argument("--once", action="store_true",
                   help="Run a single inference and exit (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.yolopx_engine or not args.dtcp_engine:
        raise SystemExit("No engine for this device in " + os.path.join(MODEL_DIR, "engines") +
                         ". Run `da-models build driverguard`, or give --yolopx-engine/--dtcp-engine.")
    sys.path.insert(0, args.driveragent_root)
    sys.path.insert(0, os.path.join(ROOT, "jetson_runtime"))
    sys.path.insert(0, ROOT)
    from runner.runner import main as runner_main
    runner_main(args)


if __name__ == "__main__":
    main()
