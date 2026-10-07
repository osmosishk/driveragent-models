#!/usr/bin/env python3
"""Measure the live rate and latency of DriverGuard and System 1.

Subscribes to DriverGuardResult and System1Result for --seconds and reports,
per model: messages per second, inferenceMs (mean, p50, p95, max), message
age at receive (now - timestamp), frame gaps and the finite rate. Read-only:
it only subscribes.

    python3 tools/live_rate.py --host 127.0.0.1 --seconds 600
    # schema: $DRIVERAGENT_ROOT/message/message.capnp (default ~/driveragent)
"""
import argparse
import json
import os
import time

import capnp
import numpy as np
import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--driverguard-port", type=int, default=8014)
ap.add_argument("--system1-port", type=int, default=8011)
ap.add_argument("--schema", default=os.path.join(
    os.path.expanduser(os.environ.get("DRIVERAGENT_ROOT", "~/driveragent")), "message", "message.capnp"))
ap.add_argument("--json", help="also write the result to this file")
a = ap.parse_args()

capnp.remove_import_hook()
schema = capnp.load(a.schema)
ctx = zmq.Context()
subs = {}
for name, port, typ in (("driverguard", a.driverguard_port, schema.DriverGuardResult),
                        ("system1", a.system1_port, schema.System1Result)):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVHWM, 1000)
    s.connect(f"tcp://{a.host}:{port}")
    subs[s] = (name, typ)
poller = zmq.Poller()
for s in subs:
    poller.register(s, zmq.POLLIN)

data = {n: {"recv": [], "inf": [], "age": [], "frame": [], "finite": []} for n, _ in subs.values()}
end = time.time() + a.seconds
while time.time() < end:
    for s, _ in poller.poll(500):
        name, typ = subs[s]
        payload = s.recv_multipart()[-1]
        now = time.time()
        with typ.from_bytes(payload) as m:
            d = data[name]
            d["recv"].append(now)
            d["inf"].append(float(m.inferenceMs))
            d["age"].append(now * 1000 - float(m.timestamp))
            d["frame"].append(int(m.frame))
            d["finite"].append(bool(m.finite))


def pct(v, q):
    return round(float(np.percentile(v, q)), 1) if v else None


report = {}
for name, d in data.items():
    n = len(d["recv"])
    span = d["recv"][-1] - d["recv"][0] if n > 1 else 0
    gaps = int(sum(max(0, b - a - 1) for a, b in zip(d["frame"], d["frame"][1:])))
    report[name] = {
        "messages": n, "rate_hz": round((n - 1) / span, 2) if span else 0.0,
        "inference_ms": {"mean": round(float(np.mean(d["inf"])), 1) if n else None,
                         "p50": pct(d["inf"], 50), "p95": pct(d["inf"], 95),
                         "max": round(max(d["inf"]), 1) if n else None},
        "interval_ms_p95": pct(list(np.diff(d["recv"]) * 1000), 95) if n > 1 else None,
        "age_ms": {"p50": pct(d["age"], 50), "p95": pct(d["age"], 95)},
        "frame_gaps": gaps,
        "finite_pct": round(100.0 * sum(d["finite"]) / n, 2) if n else None,
    }
print(json.dumps(report, indent=1))
if a.json:
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1)
