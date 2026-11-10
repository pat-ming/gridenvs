"""
Latency & Throughput Benchmarking

Measurement protocol:
  - Warmup phase (discard first N batches)
  - Timed phase with torch.cuda.synchronize()
  - Outlier trimming (top/bottom percentile)
  - Separate timing for preprocess, forward, postprocess
  - Sustained throughput over fixed duration

Usage:
    bench = LatencyBenchmark(cfg)
    results = bench.run(model, dataloader)
    tracker.log_benchmark(results, label="baseline_fp32_B1")
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BenchmarkResult:
    """Structured benchmark output."""

    label: str = ""
    batch_size: int = 1
    # Latency (ms)
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    # Breakdown (ms)
    preprocess_mean_ms: float = 0.0
    forward_mean_ms: float = 0.0
    postprocess_mean_ms: float = 0.0
    # Throughput
    fps: float = 0.0
    # Raw data
    num_iterations: int = 0
    num_trimmed: int = 0
    raw_latencies_ms: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "raw_latencies_ms"}
        return d

    def summary_line(self) -> str:
        return (
            f"[{self.label}] B={self.batch_size} | "
            f"lat={self.latency_mean_ms:.2f}±{self.latency_std_ms:.2f}ms | "
            f"P95={self.latency_p95_ms:.2f}ms | "
            f"FPS={self.fps:.1f} | "
            f"fwd={self.forward_mean_ms:.2f}ms"
        )


class LatencyBenchmark:
    """
    Measures per-batch latency following the measurement protocol.

    Protocol:
      1. Warmup: run `warmup_batches` batches, discard timing
      2. Timed: run `timing_iterations` batches with cuda.synchronize()
      3. Trim: discard top/bottom `trim_percentile`%
      4. Report: mean, std, median, P95, P99
    """

    def __init__(self, cfg: dict):
        bench_cfg = cfg.get("benchmark", {})
        self.warmup_batches = bench_cfg.get("warmup_batches", 50)
        self.timing_iterations = bench_cfg.get("timing_iterations", 500)
        self.trim_percentile = bench_cfg.get("trim_percentile", 5)

    def run(
        self,
        forward_fn: Callable,
        data_iter,
        batch_size: int = 1,
        label: str = "",
        preprocess_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
    ) -> BenchmarkResult:
        """
        Run the latency benchmark.

        Args:
            forward_fn: Callable that takes a batch and returns raw model output.
                        Must already be on GPU.
            data_iter: Iterable that yields batches. Will be cycled if needed.
            batch_size: For reporting.
            label: Human-readable label for this benchmark.
            preprocess_fn: Optional preprocess step to time separately.
            postprocess_fn: Optional postprocess/NMS step to time separately.
        """
        import torch

        # Build a cycling iterator
        batches = list(data_iter)
        if not batches:
            raise ValueError("data_iter yielded no batches")

        def cycle():
            idx = 0
            while True:
                yield batches[idx % len(batches)]
                idx += 1

        it = cycle()

        # --- Warmup ---
        for _ in range(self.warmup_batches):
            batch = next(it)
            if preprocess_fn:
                batch = preprocess_fn(batch)
            _ = forward_fn(batch)
            if postprocess_fn:
                _ = postprocess_fn(_)
            torch.cuda.synchronize()

        # --- Timed iterations ---
        preprocess_times = []
        forward_times = []
        postprocess_times = []
        total_times = []

        for _ in range(self.timing_iterations):
            batch = next(it)
            torch.cuda.synchronize()

            t_total_start = time.perf_counter()

            # Preprocess
            if preprocess_fn:
                t0 = time.perf_counter()
                batch = preprocess_fn(batch)
                torch.cuda.synchronize()
                preprocess_times.append((time.perf_counter() - t0) * 1000)

            # Forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = forward_fn(batch)
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - t0) * 1000)

            # Postprocess
            if postprocess_fn:
                t0 = time.perf_counter()
                _ = postprocess_fn(output)
                torch.cuda.synchronize()
                postprocess_times.append((time.perf_counter() - t0) * 1000)

            torch.cuda.synchronize()
            total_times.append((time.perf_counter() - t_total_start) * 1000)

        # --- Trim outliers ---
        total_arr = np.array(total_times)
        lower = np.percentile(total_arr, self.trim_percentile)
        upper = np.percentile(total_arr, 100 - self.trim_percentile)
        mask = (total_arr >= lower) & (total_arr <= upper)
        trimmed = total_arr[mask]

        num_trimmed = len(total_arr) - len(trimmed)

        result = BenchmarkResult(
            label=label,
            batch_size=batch_size,
            latency_mean_ms=float(np.mean(trimmed)),
            latency_std_ms=float(np.std(trimmed)),
            latency_median_ms=float(np.median(trimmed)),
            latency_p95_ms=float(np.percentile(trimmed, 95)),
            latency_p99_ms=float(np.percentile(trimmed, 99)),
            preprocess_mean_ms=(
                float(np.mean(preprocess_times)) if preprocess_times else 0.0
            ),
            forward_mean_ms=float(np.mean(forward_times)),
            postprocess_mean_ms=(
                float(np.mean(postprocess_times)) if postprocess_times else 0.0
            ),
            fps=batch_size / (float(np.mean(trimmed)) / 1000),
            num_iterations=len(trimmed),
            num_trimmed=num_trimmed,
            raw_latencies_ms=total_times,
        )

        return result


class ThroughputBenchmark:
    """
    Measures sustained throughput over a fixed time window.

    Protocol: Run continuous inference for `duration_sec` seconds,
    count total frames processed, report FPS.
    """

    def __init__(self, cfg: dict):
        bench_cfg = cfg.get("benchmark", {})
        self.duration_sec = bench_cfg.get("sustained_duration_sec", 300)
        self.warmup_batches = bench_cfg.get("warmup_batches", 50)

    def run(
        self,
        forward_fn: Callable,
        data_iter,
        batch_size: int = 1,
        label: str = "",
    ) -> dict:
        """
        Run sustained throughput benchmark.

        Returns dict with: total_frames, duration_sec, fps, batches_processed.
        """
        import torch

        batches = list(data_iter)
        if not batches:
            raise ValueError("data_iter yielded no batches")

        def cycle():
            idx = 0
            while True:
                yield batches[idx % len(batches)]
                idx += 1

        it = cycle()

        # Warmup
        for _ in range(self.warmup_batches):
            _ = forward_fn(next(it))
            torch.cuda.synchronize()

        # Sustained run
        total_batches = 0
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        while True:
            _ = forward_fn(next(it))
            torch.cuda.synchronize()
            total_batches += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= self.duration_sec:
                break

        total_frames = total_batches * batch_size
        actual_duration = time.perf_counter() - t_start

        return {
            "label": label,
            "batch_size": batch_size,
            "total_frames": total_frames,
            "total_batches": total_batches,
            "duration_sec": round(actual_duration, 2),
            "fps": total_frames / actual_duration,
        }
