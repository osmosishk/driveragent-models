"""CarState subscriber → speed_mps for DTCP's state vector.

DriverAgent CarState publishes speedKph; we expose it as m/s. If the
subscriber can't connect (no CarState producer running) we report 0 m/s
so inference still runs — useful for bench testing.
"""

import os
import sys
import time
import threading

DRIVERAGENT_ROOT = os.path.expanduser(os.environ.get("DRIVERAGENT_ROOT", "~/driveragent"))
sys.path.insert(0, DRIVERAGENT_ROOT)
from message.capnp_pubsub import Subscriber

CAPNP_SCHEMA = os.path.join(DRIVERAGENT_ROOT, "message", "message.capnp")
CARSTATE_ADDR = "tcp://127.0.0.1:5592"


class SpeedProvider:
    def __init__(self, addr=CARSTATE_ADDR, schema_file=CAPNP_SCHEMA):
        self._addr = addr
        self._schema_file = schema_file
        self._lock = threading.Lock()
        self._latest_speed_mps = 0.0
        self._latest_ts = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="driverguard_speed")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            sub = Subscriber(self._schema_file, "CarState", self._addr)
        except Exception as e:
            print(f"[driverguard] WARNING: cannot subscribe to CarState at "
                  f"{self._addr}: {e}. Reporting 0 m/s.")
            return
        while not self._stop.is_set():
            try:
                msg = sub.receive()
                speed_kph = float(getattr(msg, "speedKph", 0.0))
                with self._lock:
                    self._latest_speed_mps = speed_kph / 3.6
                    self._latest_ts = time.time()
            except Exception:
                time.sleep(0.05)

    def latest_speed_mps(self):
        with self._lock:
            return self._latest_speed_mps
