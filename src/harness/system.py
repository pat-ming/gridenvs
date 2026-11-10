"""
Background GPU sampler (NVML). Run starts one automatically when NVML is
available; samples go to events.jsonl as sys/gpu<i>/<field> metrics (W&B's
system tab already charts the same thing, so they aren't re-sent), and
finish() adds per-GPU mean util, peak memory and energy to summary.json.
"""

import threading
import time


class SystemMonitor:
    FIELDS = ("util", "mem_util", "mem_gb", "power_w", "temp_c", "sm_mhz")

    def __init__(self, run, interval: float = 10.0):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        self.run = run
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="harness-gpu"
        )
        n = len(self._handles)
        self._util_sum = [0.0] * n
        self._samples = 0
        self._mem_peak = [0.0] * n
        self._energy_j = [0.0] * n
        self._last = time.monotonic()

    def start(self) -> "SystemMonitor":
        self._thread.start()
        return self

    def stop(self) -> dict:
        """Stop sampling; return the per-GPU summary."""
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)
        summary = {}
        for i in range(len(self._handles)):
            summary[f"gpu{i}_util_mean"] = round(
                self._util_sum[i] / max(self._samples, 1), 1
            )
            summary[f"gpu{i}_mem_peak_gb"] = round(self._mem_peak[i], 2)
            summary[f"gpu{i}_energy_kwh"] = round(self._energy_j[i] / 3.6e6, 4)
        return summary

    def sample(self) -> dict:
        nv = self._nvml
        now = time.monotonic()
        dt, self._last = now - self._last, now
        out = {}
        for i, h in enumerate(self._handles):
            util = nv.nvmlDeviceGetUtilizationRates(h)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            power = nv.nvmlDeviceGetPowerUsage(h) / 1000
            values = {
                "util": util.gpu,
                "mem_util": util.memory,
                "mem_gb": round(mem.used / 2**30, 2),
                "power_w": round(power, 1),
                "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            }
            self._util_sum[i] += util.gpu
            self._mem_peak[i] = max(self._mem_peak[i], values["mem_gb"])
            self._energy_j[i] += power * dt
            out.update({f"gpu{i}/{k}": v for k, v in values.items()})
        self._samples += 1
        return out

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.run.log(self.sample(), prefix="sys/", wandb=False)
            except Exception as e:  # noqa: BLE001 - a sampling hiccup shouldn't kill the thread
                self.run.event("system_monitor_error", error=repr(e))


def start_monitor(run, interval: float) -> SystemMonitor | None:
    """A started monitor, or None without NVML (no GPU, no driver, macOS)."""
    try:
        return SystemMonitor(run, interval).start()
    except Exception:  # noqa: BLE001 - ImportError or any NVMLError means no GPUs to watch
        return None
