"""
Experiment harness: one directory per run (config, JSONL metrics/events,
record streams, checkpoints) mirrored to W&B, plus a read side for analysis
and a CLI (`python -m harness ls|tail|compare`). Task-agnostic; HF/TRL
trainers plug in through harness.hf.HarnessCallback.
"""

from harness.config import get_config, load_config, override_config
from harness.metrics import MetricsStore
from harness.reader import RunView, find, load, runs
from harness.run import Run, current, event, finish, init, log, record
from harness.seed import set_seed

__all__ = [
    "MetricsStore",
    "Run",
    "RunView",
    "current",
    "event",
    "find",
    "finish",
    "get_config",
    "init",
    "load",
    "load_config",
    "log",
    "override_config",
    "record",
    "runs",
    "set_seed",
]
