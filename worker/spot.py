from __future__ import annotations
import logging
import threading
import urllib.error
import urllib.request

log = logging.getLogger("worker.spot")
IMDS = "http://169.254.169.254"

# Reads a value from the instance metadata
def imds_get(path: str, timeout: float = 2.0) -> str | None:
    try:
        tok_req = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                         headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
        token = urllib.request.urlopen(tok_req, timeout=timeout).read().decode()
        req = urllib.request.Request(f"{IMDS}/latest/meta-data/{path}",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()
    except (urllib.error.URLError, OSError):
        return None

class SpotWatcher(threading.Thread):
    # Keeps the event and the poll time
    def __init__(self, event: threading.Event, interval: float = 5.0, getter=imds_get):
        super().__init__(daemon=True, name="spot-watcher")
        self.event, self.interval, self._get = event, interval, getter
        self._halt = threading.Event()
    # Waits for a spot interruption notice
    def run(self):
        while not self._halt.wait(self.interval):
            if self._get("spot/instance-action"):
                log.warning("spot interruption notice received")
                self.event.set()
                return
    # Stops the watcher
    def stop(self):
        self._halt.set()
