"""
Metrics Store & Computation

Tracks all metrics across experiments and provides:
  - Cumulative metric logging (per-step, per-epoch)
  - VRAM tracking integrated with training loop
  - Timer context managers for wall-clock measurement
  - Training throughput (samples/sec) computation
  - Metric comparison utilities
"""

import time
from collections.abc import Generator
from contextlib import contextmanager


class MetricsStore:
    """
    Accumulates metrics during training and provides epoch-level summaries.

    Usage:
        ms = MetricsStore()

        for batch in dataloader:
            loss = train_step(batch)
            ms.update("train_loss", loss.item())
            ms.update("batch_size", batch.shape[0])

        epoch_summary = ms.summarize_and_reset()
        # {"train_loss_mean": ..., "train_loss_count": ..., "batch_size_sum": ..., ...}
    """

    def __init__(self):
        self._accumulators: dict[str, list[float]] = {}
        self._timers: dict[str, float] = {}
        self._counters: dict[str, int] = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Accumulate a metric value."""
        if name not in self._accumulators:
            self._accumulators[name] = []
        self._accumulators[name].append(value)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter (e.g., total samples processed)."""
        self._counters[name] = self._counters.get(name, 0) + amount

    def summarize(self) -> dict:
        """
        Compute summary statistics for all accumulated metrics.
        Returns dict with {name}_mean, {name}_min, {name}_max, {name}_count.
        """
        summary = {}
        for name, values in self._accumulators.items():
            if values:
                summary[f"{name}_mean"] = sum(values) / len(values)
                summary[f"{name}_min"] = min(values)
                summary[f"{name}_max"] = max(values)
                summary[f"{name}_last"] = values[-1]
                summary[f"{name}_count"] = len(values)

        for name, count in self._counters.items():
            summary[name] = count

        return summary

    def summarize_and_reset(self) -> dict:
        """Compute summary, then reset all accumulators."""
        summary = self.summarize()
        self.reset()
        return summary

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._accumulators.clear()
        self._counters.clear()

    def get_values(self, name: str) -> list[float]:
        """Get raw accumulated values for a metric."""
        return self._accumulators.get(name, [])

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        """
        Context manager that times a block and stores the elapsed time.

        Usage:
            with ms.timer("forward_pass"):
                output = model(batch)
            # ms now has "forward_pass_sec" accumulated
        """
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.update(f"{name}_sec", elapsed)


class EpochTimer:
    """
    Simple wall-clock timer for an entire epoch.

    Usage:
        timer = EpochTimer()
        timer.start()
        ... training loop ...
        elapsed = timer.stop()
    """

    def __init__(self):
        self._start = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started")
        elapsed = time.perf_counter() - self._start
        self._start = None
        return elapsed

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        self.start()
        yield
        # elapsed is accessible via self.elapsed after exiting

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start


class ThroughputTracker:
    """
    Tracks training throughput (samples/sec) over an epoch.

    Usage:
        tp = ThroughputTracker()
        tp.start()
        for batch in dataloader:
            train_step(batch)
            tp.add_samples(batch_size)
        result = tp.stop()
        # {"samples_per_sec": ..., "total_samples": ..., "elapsed_sec": ...}
    """

    def __init__(self):
        self._start_time = None
        self._total_samples = 0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._total_samples = 0

    def add_samples(self, n: int) -> None:
        self._total_samples += n

    def stop(self) -> dict:
        if self._start_time is None:
            raise RuntimeError("ThroughputTracker was never started")
        elapsed = time.perf_counter() - self._start_time
        sps = self._total_samples / elapsed if elapsed > 0 else 0.0
        return {
            "samples_per_sec": round(sps, 2),
            "total_samples": self._total_samples,
            "elapsed_sec": round(elapsed, 2),
        }

    @property
    def current_throughput(self) -> float:
        if self._start_time is None or self._total_samples == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._total_samples / elapsed if elapsed > 0 else 0.0
