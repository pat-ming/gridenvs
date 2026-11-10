"""
One experiment run: a self-contained directory, optionally mirrored to W&B.

    <output_dir>/<name>_<utc timestamp>/
        config.yaml            frozen config
        meta.json              provenance (git, slurm, host, GPUs), status, wandb id
        events.jsonl           every metric and event, append-only
        records/<name>.jsonl   named record streams (rollouts, eval predictions, ...)
        checkpoints/
        profiles/
        summary.json           written when the run ends

The JSONL files are the source of truth and W&B is a view of them. Logging
never raises into the training loop. Only rank 0 writes; on other ranks every
method is a no-op, so the same code runs under torchrun/accelerate.
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import yaml

from harness.system import start_monitor

_current: "Run | None" = None


class Run:
    """
    Usage:
        with Run(cfg) as run:      # or harness.init(cfg), then harness.log(...)
            run.log({"loss": 0.42, "lr": 1e-4}, step=100, prefix="train/")
            run.log({"success": 0.31}, step=100, prefix="eval/")
            run.record("rollouts", [{"prompt": ..., "reward": 1.0}], step=100)
            run.save_checkpoint({"model": model.state_dict()}, step=100)

    Status ends as "finished", "failed" (exception), or "interrupted"
    (Ctrl-C, or SIGTERM from a Slurm time limit/scancel).

    Config keys read (all optional):
        experiment: {name, tags, output_dir (default $HARNESS_RUNS_DIR or runs/)}
        tracking:   {wandb: {enabled, project, entity, group, mode},
                     best: {metric, mode}, keep_last_n_checkpoints,
                     system_interval (GPU sampling seconds, default 10, 0 = off)}
    """

    def __init__(
        self,
        cfg: dict,
        resume: str | Path | None = None,
        step_key: str = "step",
    ):
        global _current
        exp = cfg.get("experiment", {})
        track = cfg.get("tracking", {})
        self.cfg = cfg
        self.name = exp.get("name", "unnamed")
        self.tags = list(exp.get("tags", []))
        self.step_key = step_key
        self.summary: dict = {}
        self.active = int(os.environ.get("RANK", "0")) == 0
        self.wandb = None
        self._monitor = None
        self._lock = threading.Lock()
        self._closed = False
        _current = self
        if not self.active:
            return

        if resume:
            self.dir = Path(resume)
            self.meta = json.loads((self.dir / "meta.json").read_text())
        else:
            root = exp.get("output_dir") or os.environ.get("HARNESS_RUNS_DIR", "runs")
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            # Slurm array tasks start in the same second, so the job id keeps dirs apart.
            job = os.environ.get("SLURM_JOB_ID")
            suffix = f"{stamp}_{job}" if job else stamp
            self.dir = Path(os.path.expandvars(root)) / f"{self.name}_{suffix}"
            self.meta = provenance()
        for sub in ("records", "checkpoints", "profiles"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        self._records: dict = {}
        self._start = time.monotonic()

        best = track.get("best") or {}
        self.best_metric = best.get("metric")
        self._best_sign = -1 if best.get("mode", "max") == "min" else 1
        self.best_value = self.meta.get("best_value")
        self.best_step = self.meta.get("best_step")
        self.is_best = False
        self._keep_last_n = track.get("keep_last_n_checkpoints", 0)

        self.wandb = self._init_wandb(track.get("wandb", {}), self.meta.get("wandb_id"))
        self.meta.update(
            status="running",
            wandb_id=self.wandb.id if self.wandb else None,
            wandb_url=self.wandb.url if self.wandb else None,
        )
        self._write_meta()
        self.event("start", resumed=bool(resume))
        _raise_on_sigterm()
        interval = track.get("system_interval", 10)
        if interval:
            self._monitor = start_monitor(self, interval)

    def log(
        self,
        metrics: dict,
        step: int | None = None,
        *,
        prefix: str = "",
        wandb: bool = True,
    ) -> None:
        """Log scalars. wandb=False when something else (e.g. HF's WandbCallback) already sends them."""
        if not self.active:
            return
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        self.event("metrics", step=step, **metrics)
        self._update_best(metrics, step)
        if wandb and self.wandb:
            payload = {k: v for k, v in metrics.items() if _is_scalar(v)}
            if step is not None:
                payload[self.step_key] = step
            self._wandb_call(self.wandb.log, payload)

    def record(
        self,
        name: str,
        rows: list[dict],
        step: int | None = None,
        *,
        wandb_sample: int = 0,
    ) -> None:
        """Append rows to records/<name>.jsonl; mirror the first wandb_sample as a W&B table."""
        if not self.active:
            return
        if name not in self._records:
            path = self.dir / "records" / f"{name}.jsonl"
            self._records[name] = open(path, "a", encoding="utf-8")  # noqa: SIM115
        f = self._records[name]
        for row in rows:
            f.write(json.dumps({"step": step, **row}, default=str) + "\n")
        f.flush()
        if wandb_sample and self.wandb and rows:
            import wandb

            sample = rows[:wandb_sample]
            columns = sorted({k for row in sample for k in row})
            data = [[_stringify(row.get(c)) for c in columns] for row in sample]
            payload = {f"records/{name}": wandb.Table(columns=columns, data=data)}
            if step is not None:
                payload[self.step_key] = step
            self._wandb_call(self.wandb.log, payload)

    def event(self, kind: str, **data) -> None:
        """Write a non-metric event (checkpoint saved, eval started, ...) to events.jsonl."""
        if not self.active:
            return
        line = json.dumps({"event": kind, "time": _now(), **data}, default=str)
        with self._lock:  # the GPU sampler thread logs too
            self._events.write(line + "\n")
            self._events.flush()

    def save_checkpoint(self, state: dict, step: int) -> Path | None:
        """torch.save atomically; copy to best.pt if the last best-metric log improved."""
        if not self.active:
            return None
        import torch

        ckpt_dir = self.dir / "checkpoints"
        path = ckpt_dir / f"step_{step:07d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({**state, "step": step, "wandb_id": self.meta["wandb_id"]}, tmp)
        tmp.replace(path)
        if self.is_best:
            shutil.copy2(path, ckpt_dir / "best.tmp")
            (ckpt_dir / "best.tmp").replace(ckpt_dir / "best.pt")
        if self._keep_last_n > 0:
            for old in sorted(ckpt_dir.glob("step_*.pt"))[: -self._keep_last_n]:
                old.unlink()
        self.event("checkpoint", step=step, path=str(path), best=self.is_best)
        return path

    def finish(self, summary: dict | None = None, status: str = "finished") -> dict:
        global _current
        if _current is self:
            _current = None
        if self._closed or not self.active:
            return self.summary
        if self._monitor:
            self.summary.update(self._monitor.stop())
        self._closed = True
        elapsed = time.monotonic() - self._start
        self.meta["gpu_hours"] = (
            self.meta.get("gpu_hours", 0.0) + elapsed * _gpu_count() / 3600
        )
        self.meta.update(
            status=status, best_value=self.best_value, best_step=self.best_step
        )
        self._write_meta()

        self.summary.update(
            name=self.name,
            tags=self.tags,
            status=status,
            elapsed_sec=round(elapsed, 1),
            gpu_hours=round(self.meta["gpu_hours"], 4),
            best_metric=self.best_metric,
            best_value=self.best_value,
            best_step=self.best_step,
            **(summary or {}),
        )
        (self.dir / "summary.json").write_text(
            json.dumps(self.summary, indent=2, default=str)
        )
        self.event("end", status=status)
        self._events.close()
        for f in self._records.values():
            f.close()
        if self.wandb:
            clean = {k: v for k, v in self.summary.items() if _is_scalar(v)}
            self._wandb_call(self.wandb.summary.update, clean)
            exit_code = 0 if status == "finished" else 1
            self._wandb_call(self.wandb.finish, exit_code=exit_code)
        return self.summary

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            status = "finished"
        elif issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
        else:
            status = "failed"
            self.event("error", error=repr(exc))
        self.finish(status=status)
        return False

    def _update_best(self, metrics: dict, step: int | None) -> None:
        value = metrics.get(self.best_metric) if self.best_metric else None
        if not _is_number(value):
            return
        sign = self._best_sign
        self.is_best = self.best_value is None or sign * (value - self.best_value) > 0
        if self.is_best:
            self.best_value, self.best_step = value, step

    def _init_wandb(self, cfg: dict, resume_id: str | None):
        if not cfg.get("enabled", False):
            return None
        try:
            import wandb

            run = wandb.init(
                project=cfg.get("project", "harness"),
                entity=cfg.get("entity"),
                group=cfg.get("group"),
                mode=cfg.get("mode"),
                name=self.dir.name,
                tags=self.tags,
                config=self.cfg,
                dir=self.dir,
                id=resume_id,
                resume="allow" if resume_id else None,
            )
            run.define_metric(self.step_key)
            run.define_metric("*", step_metric=self.step_key)
            return run
        except Exception as e:  # noqa: BLE001 - tracking must never stop training
            print(f"[harness] wandb disabled: {e}", file=sys.stderr)
            return None

    def _wandb_call(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - tracking must never stop training
            print(f"[harness] wandb call failed: {e}", file=sys.stderr)

    def _write_meta(self) -> None:
        text = json.dumps(self.meta, indent=2, default=str)
        (self.dir / "meta.json").write_text(text)


def init(cfg: dict, **kwargs) -> Run:
    """Start a run and make it the one harness.log/record/event write to."""
    return Run(cfg, **kwargs)


def current() -> Run | None:
    return _current


def log(metrics: dict, step: int | None = None, **kwargs) -> None:
    if _current:
        _current.log(metrics, step, **kwargs)


def record(name: str, rows: list[dict], step: int | None = None, **kwargs) -> None:
    if _current:
        _current.record(name, rows, step, **kwargs)


def event(kind: str, **data) -> None:
    if _current:
        _current.event(kind, **data)


def finish(summary: dict | None = None, status: str = "finished") -> None:
    if _current:
        _current.finish(summary, status)


def provenance() -> dict:
    """What produced this run: code version, job, machine."""
    meta = {
        "created": _now(),
        "host": platform.node(),
        "python": platform.python_version(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git": _git_state(),
    }
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
        "SLURMD_NODENAME",
    )
    slurm = {
        k.removeprefix("SLURM_").lower(): os.environ[k]
        for k in slurm_keys
        if k in os.environ
    }
    if slurm:
        meta["slurm"] = slurm
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        meta["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except ImportError:
        pass
    return meta


def _git_state() -> str | None:
    """`git describe --always --dirty`, or the GIT_STATE file a non-git sync (rsync) leaves behind."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "GIT_STATE").exists():
            return (parent / "GIT_STATE").read_text().strip()
    return None


def _raise_on_sigterm() -> None:
    """Slurm sends SIGTERM at the time limit; turn it into SystemExit so the run closes as interrupted."""
    main = threading.current_thread() is threading.main_thread()
    if main and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, str, bool))


def _stringify(v):
    return v if _is_scalar(v) or v is None else json.dumps(v, default=str)
