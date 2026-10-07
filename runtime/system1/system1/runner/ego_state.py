"""CarState subscriber → 8-dim ego state tensor for System1.

System1 expects [vx, vy, ax, ay, yaw_rate, speed, cmd0, cmd1] where cmd is a
2-bit one-hot of turn intent ([1,0]=left, [0,0]=straight, [0,1]=right).

DriverAgent CarState only publishes speedKph (+ gear, brake). For now we
assume forward motion (vy=0, yaw_rate=0) and derive longitudinal accel from
finite differences of speed.
"""

import os
import sys
import time
import threading

import torch

# Allow running both as a package module and as a script
DRIVERAGENT_ROOT = os.path.expanduser(os.environ.get("DRIVERAGENT_ROOT", "~/driveragent"))
sys.path.insert(0, DRIVERAGENT_ROOT)
from message.capnp_pubsub import Subscriber

CAPNP_SCHEMA = os.path.join(DRIVERAGENT_ROOT, "message", "message.capnp")
CARSTATE_ADDR = "tcp://127.0.0.1:5592"

# Turn-intent one-hot encoding used by System1 ego_state[6:8]
CMD_STRAIGHT = (0.0, 0.0)
CMD_LEFT     = (1.0, 0.0)
CMD_RIGHT    = (0.0, 1.0)


class EgoStateProvider:
    """Background thread that tracks the latest CarState. Caller asks for an
    ego_state tensor on demand."""

    def __init__(self, addr=CARSTATE_ADDR, schema_file=CAPNP_SCHEMA,
                 timeout_s=2.0):
        self._addr = addr
        self._schema_file = schema_file
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._latest_speed_kph = 0.0
        self._latest_ts = 0.0
        self._prev_speed_mps = 0.0
        self._prev_ts = 0.0
        self._latest_ax = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ego_state_sub")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            sub = Subscriber(self._schema_file, "CarState", self._addr)
        except Exception as e:
            print(f"[ego_state] WARNING: cannot subscribe to CarState at "
                  f"{self._addr}: {e}. Will report zeros.")
            return
        while not self._stop.is_set():
            try:
                msg = sub.receive()
                speed_kph = float(getattr(msg, "speedKph", 0.0))
                now = time.time()
                speed_mps = speed_kph / 3.6
                with self._lock:
                    if self._prev_ts > 0 and now > self._prev_ts:
                        dt = now - self._prev_ts
                        if dt > 0.01:
                            self._latest_ax = (speed_mps - self._prev_speed_mps) / dt
                    self._prev_speed_mps = speed_mps
                    self._prev_ts = now
                    self._latest_speed_kph = speed_kph
                    self._latest_ts = now
            except Exception as e:
                # Don't spin-fail; brief backoff.
                time.sleep(0.05)

    def fresh(self):
        with self._lock:
            return self._latest_ts > 0 and (time.time() - self._latest_ts) < self._timeout_s

    def speed_kph(self):
        with self._lock:
            return self._latest_speed_kph

    def build_ego_state(self, cmd=CMD_STRAIGHT):
        """Returns torch.Tensor [1, 8] float32."""
        with self._lock:
            speed_mps = self._latest_speed_kph / 3.6
            ax = self._latest_ax
        cmd0, cmd1 = cmd
        t = torch.tensor([[
            speed_mps,  # vx (forward)
            0.0,        # vy (lateral) — no IMU
            ax,         # ax
            0.0,        # ay
            0.0,        # yaw_rate — no IMU
            speed_mps,  # speed magnitude
            cmd0,       # cmd one-hot bit 0
            cmd1,       # cmd one-hot bit 1
        ]], dtype=torch.float32)
        return t
