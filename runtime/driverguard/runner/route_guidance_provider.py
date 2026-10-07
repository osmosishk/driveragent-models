"""RouteGuidance subscriber -> live (command, target_point) for the DTCP planner.

DriverGuard's DTCP planner is conditional: every frame it needs a discrete
navigation command (one-hot) plus a target point. plan/route_guidance.py in the
driveragent repo already computes both from the active map route and publishes
them on RouteGuidance@5605 (the schema even labels dtcpCommand "DriverGuard
convention"). This provider feeds them into the model live, so it follows the
destination instead of the hardcoded startup --command/--target.

If RouteGuidance is unavailable / stale / inactive, latest_guidance() reports
fresh=False and the runner falls back to the CLI --command / --target defaults.
"""

import sys
import time
import threading

sys.path.insert(0, "/home/tonyho/driveragent")
from message.capnp_pubsub import Subscriber

CAPNP_SCHEMA = "/home/tonyho/driveragent/message/message.capnp"
ROUTE_GUIDANCE_ADDR = "tcp://127.0.0.1:5605"

FRESH_S = 1.0          # guidance older than this is ignored
TARGET_FWD_M = 20.0    # pick the route point ~this far ahead as the DTCP target
# Clamp the target so a far-off-route route point never feeds an
# out-of-distribution value into the DTCP planner (it was trained around
# the default (0, 20) target).
TARGET_MAX_LAT_M = 15.0
TARGET_MIN_FWD_M = 5.0
TARGET_MAX_FWD_M = 30.0


class RouteGuidanceProvider:
    """Background subscriber to RouteGuidance@5605; exposes the latest
    navigation command and target point for the DTCP planner."""

    def __init__(self, addr=ROUTE_GUIDANCE_ADDR, schema_file=CAPNP_SCHEMA):
        self._addr = addr
        self._schema_file = schema_file
        self._lock = threading.Lock()
        self._command = None       # int 0..5, or None when no active guidance
        self._target_xy = None     # (lat_right_m, fwd_m), or None
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="driverguard_route_guidance")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            sub = Subscriber(self._schema_file, "RouteGuidance", self._addr)
        except Exception as e:
            print(f"[driverguard] WARNING: cannot subscribe to RouteGuidance at "
                  f"{self._addr}: {e}. Falling back to CLI command/target.")
            return
        while not self._stop.is_set():
            try:
                msg = sub.receive()
                if not bool(getattr(msg, "active", False)):
                    # no active route — report "no guidance"
                    with self._lock:
                        self._command = None
                        self._target_xy = None
                        self._ts = time.time()
                    continue
                command = max(0, min(5, int(getattr(msg, "dtcpCommand", 3))))
                target = self._pick_target(msg.egoPoints)
                with self._lock:
                    self._command = command
                    self._target_xy = target
                    self._ts = time.time()
            except Exception:
                time.sleep(0.05)

    @staticmethod
    def _pick_target(ego_points):
        """Route point ~TARGET_FWD_M ahead, converted to the DTCP frame.

        RouteGuidance.egoPoints are the standard ego frame (x=forward,
        y=left); DTCP target_point is (lat_right, fwd). Returns None if there
        is no point ahead of the vehicle.
        """
        best = None
        best_err = float("inf")
        for p in ego_points:
            x_fwd = float(p.x)
            if x_fwd <= 0.0:
                continue
            err = abs(x_fwd - TARGET_FWD_M)
            if err < best_err:
                best_err = err
                best = (-float(p.y), x_fwd)   # (lat_right, fwd)
        if best is None:
            return None
        lat_right, fwd = best
        lat_right = max(-TARGET_MAX_LAT_M, min(TARGET_MAX_LAT_M, lat_right))
        fwd = max(TARGET_MIN_FWD_M, min(TARGET_MAX_FWD_M, fwd))
        return (lat_right, fwd)

    def latest_guidance(self):
        """Return (command, target_xy, fresh).

        command   : int 0..5 or None
        target_xy : (lat_right_m, fwd_m) or None
        fresh     : True only if an active route was seen within FRESH_S.
        When fresh is False the caller should use its own defaults.
        """
        with self._lock:
            fresh = (self._command is not None
                     and (time.time() - self._ts) <= FRESH_S)
            return self._command, self._target_xy, fresh
