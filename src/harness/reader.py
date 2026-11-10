"""
Read runs back for analysis: config, status, metrics as a DataFrame, records.

    run = harness.load("runs/grpo_A_20261029_120000")
    df = run.metrics()                       # one row per log() call
    df.plot(x="train/global_step", y="reward")
    for r in run.records("rollouts"): ...
    [r for r in harness.runs() if r.cfg("env.condition") == "C"]
"""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import yaml


class RunView:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.name = self.dir.name

    @cached_property
    def config(self) -> dict:
        return yaml.safe_load((self.dir / "config.yaml").read_text()) or {}

    @property
    def meta(self) -> dict:
        return _read_json(self.dir / "meta.json")

    @property
    def summary(self) -> dict:
        return _read_json(self.dir / "summary.json")

    @property
    def status(self) -> str:
        return self.summary.get("status") or self.meta.get("status", "unknown")

    @property
    def elapsed_sec(self) -> float | None:
        if "elapsed_sec" in self.summary:
            return self.summary["elapsed_sec"]
        created = self.meta.get("created")
        if created:
            return (datetime.now(UTC) - datetime.fromisoformat(created)).total_seconds()
        return None

    def cfg(self, key: str, default=None):
        """Dotted config lookup: run.cfg("env.condition")."""
        node = self.config
        for k in key.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def events(self, kind: str | None = None) -> Iterator[dict]:
        path = self.dir / "events.jsonl"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                if kind is None or event.get("event") == kind:
                    yield event

    def metrics(self):
        """pandas DataFrame, one row per logged metrics event."""
        import pandas as pd

        rows = [
            {k: v for k, v in e.items() if k not in ("event", "time")}
            for e in self.events("metrics")
        ]
        return pd.DataFrame(rows)

    def last(self, metric: str):
        """Summary value if present, else the last logged value."""
        if metric in self.summary:
            return self.summary[metric]
        value = None
        for e in self.events("metrics"):
            value = e.get(metric, value)
        return value

    def records(self, name: str) -> Iterator[dict]:
        path = self.dir / "records" / f"{name}.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def __repr__(self) -> str:
        return f"RunView({self.name!r}, status={self.status!r})"


def load(path: str | Path) -> RunView:
    return RunView(path)


def runs(root: str | Path | None = None) -> list[RunView]:
    """All runs under root (default $HARNESS_RUNS_DIR or runs/), oldest first."""
    root = Path(root or os.environ.get("HARNESS_RUNS_DIR", "runs"))
    if not root.exists():
        return []
    dirs = [d for d in root.iterdir() if (d / "meta.json").exists()]
    return [RunView(d) for d in sorted(dirs, key=lambda d: d.stat().st_mtime)]


def find(query: str, root: str | Path | None = None) -> RunView:
    """A run by path, or the newest run whose name contains query."""
    if Path(query).is_dir():
        return RunView(query)
    matches = [r for r in runs(root) if query in r.name]
    if not matches:
        raise FileNotFoundError(f"no run matching {query!r}")
    return matches[-1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
