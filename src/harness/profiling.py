"""
torch.profiler and CUDA memory helpers. Outputs go to the run's profiles/.
Continuous GPU util/memory/power sampling lives in harness.system.

    with profile_steps(run, label="step_100") as prof:
        for batch in batches:
            train_step(batch)
            prof.step()

    run.log(vram_snapshot(), step=step, prefix="vram/")
"""

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def profile_steps(
    run,
    label: str = "profile",
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
) -> Generator:
    """Profile a few steps; writes a Chrome trace and the top-20 CUDA ops table."""
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    profiles_dir = run.dir / "profiles"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiles_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof

    trace_path = profiles_dir / f"{label}_trace.json"
    prof.export_chrome_trace(str(trace_path))
    top_ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    summary_path = profiles_dir / f"{label}_top_ops.txt"
    summary_path.write_text(top_ops)
    run.event("profile", label=label, trace=str(trace_path), top_ops=str(summary_path))


def vram_snapshot(device: int = 0) -> dict:
    """Allocator view of GPU memory in MB (what PyTorch holds, not what NVML sees)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize(device)
    mb = 2**20
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / mb, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / mb, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / mb, 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / mb, 1),
    }


def reset_vram_peak(device: int = 0) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
