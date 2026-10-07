"""System1 entry point.

Spawns the system1 inference daemon. Mirrors yolopx's tools/demotext.py shape
so the UI/supervisor can launch any model with the same calling convention:

    python <version>/runtime/system1/run.py \
        --checkpoint <version>/system1_deploy.pth \
        --rate 5 --pub-port 8011

The driveragent root is added to sys.path here so the runner can import
`message.capnp_pubsub`. The system1 root (this file's dir) is added so the
runner can import `runner.*` and `run_system1`.
"""

import argparse
import os
import sys

SYSTEM1_ROOT = os.path.dirname(os.path.abspath(__file__))       # <version>/runtime/system1
MODEL_DIR = os.path.dirname(os.path.dirname(SYSTEM1_ROOT))       # <version>
DEFAULT_DRIVERAGENT_ROOT = os.path.expanduser(os.environ.get("DRIVERAGENT_ROOT", "~/driveragent"))
DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, "system1_deploy.pth")
DEFAULT_SCHEMA = os.path.join(DEFAULT_DRIVERAGENT_ROOT, "message", "message.capnp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--rate", type=float, default=5.0,
                   help="Target inference rate in Hz (default 5)")
    p.add_argument("--pub-port", type=int, default=8011,
                   help="ZMQ port for System1Result publisher (default 8011)")
    p.add_argument("--schema", default=DEFAULT_SCHEMA,
                   help="Path to message.capnp schema")
    p.add_argument("--sd-status-addr", default="tcp://127.0.0.1:5595",
                   help="SelfDrivingStatus subscriber address (used with --gated)")
    p.add_argument("--driveragent-root", default=DEFAULT_DRIVERAGENT_ROOT,
                   help="Path to driveragent (for capnp pubsub imports)")
    p.add_argument("--once", action="store_true",
                   help="Run a single inference and exit (smoke test)")
    p.add_argument("--gated", action="store_true",
                   help="Only run inference while SelfDrivingStatus.enabled")
    p.add_argument("--precision", default="bf16",
                   choices=["fp32", "fp16", "bf16"],
                   help="Autocast precision (bf16 recommended)")
    p.add_argument("--no-cameras", action="store_true",
                   help="Use zero images instead of /tmp/cam* (offline test)")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.driveragent_root)
    sys.path.insert(0, SYSTEM1_ROOT)
    from runner.runner import main as runner_main
    runner_main(args)


if __name__ == "__main__":
    main()
