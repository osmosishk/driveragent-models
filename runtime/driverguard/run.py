"""DriverGuard entry point.

YOLOPX (drivable area + lane mask + 10-class detection) + DTCP (4-waypoint
trajectory + throttle/steer/brake) running off a single front camera at
/tmp/cam0. Engines are the FP16 TRT engines already shipped in
/home/tonyho/model/jetson_bundle/engines/.

Spawned by /home/tonyho/driveragent/start.py's MODEL_REGISTRY when the UI
picker requests "driverguard".

    python /home/tonyho/model/driverguard/run.py --pub-port 8014
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DRIVERAGENT_ROOT = "/home/tonyho/driveragent"
DEFAULT_BUNDLE_ROOT = "/home/tonyho/model/jetson_bundle"
DEFAULT_YOLOPX_ENGINE = os.path.join(ROOT, "engines", "yolopx_v2_fp16.engine")
DEFAULT_DTCP_ENGINE = os.path.join(ROOT, "engines", "dtcp_v1_fp32.engine")
DEFAULT_DTCP_CONTROL_ONNX = os.path.join(ROOT, "engines", "dtcp_v1_control.onnx")
DEFAULT_SCHEMA = os.path.join(DEFAULT_DRIVERAGENT_ROOT, "message", "message.capnp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolopx-engine", default=DEFAULT_YOLOPX_ENGINE)
    p.add_argument("--dtcp-engine", default=DEFAULT_DTCP_ENGINE)
    p.add_argument("--control-onnx", default=DEFAULT_DTCP_CONTROL_ONNX,
                   help="DTCP control sub-graph ONNX (mu/sigma via onnxruntime CPU)")
    p.add_argument("--rate", type=float, default=10.0,
                   help="Target inference rate in Hz (default 10)")
    p.add_argument("--pub-port", type=int, default=8014,
                   help="ZMQ port for DriverGuardResult publisher (default 8014)")
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--driveragent-root", default=DEFAULT_DRIVERAGENT_ROOT)
    p.add_argument("--bundle-root", default=DEFAULT_BUNDLE_ROOT)
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
    sys.path.insert(0, args.driveragent_root)
    sys.path.insert(0, os.path.join(args.bundle_root, "jetson_runtime"))
    sys.path.insert(0, ROOT)
    from runner.runner import main as runner_main
    runner_main(args)


if __name__ == "__main__":
    main()
