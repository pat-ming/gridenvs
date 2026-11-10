"""
`harness` CLI (also `python -m harness`). RUN is a run directory or a substring
of a run name (newest match wins). --root defaults to $HARNESS_RUNS_DIR, else runs/.
"""

import csv
import itertools
import json
import shlex
import statistics
import time
from pathlib import Path
from typing import Annotated

import typer

from harness.reader import RunView, find, runs

app = typer.Typer(no_args_is_help=True, add_completion=False)
Root = Annotated[Path | None, typer.Option(help="Runs directory.")]


@app.command("ls")
def ls(root: Root = None, filter: str = "") -> None:
    """Runs with status, wall time, GPU-hours, GPU util/peak memory, best metric."""
    rows = [
        {
            "run": r.name,
            "status": r.status,
            "time": _duration(r.elapsed_sec),
            "gpu_h": r.summary.get("gpu_hours"),
            "util%": r.summary.get("gpu0_util_mean"),
            "mem_gb": r.summary.get("gpu0_mem_peak_gb"),
            "best": _best(r),
            "slurm": r.meta.get("slurm", {}).get("job_id"),
        }
        for r in runs(root)
        if filter in r.name
    ]
    typer.echo(
        markdown(
            rows, ["run", "status", "time", "gpu_h", "util%", "mem_gb", "best", "slurm"]
        )
    )


@app.command()
def show(run: str, root: Root = None) -> None:
    """Provenance, W&B link and summary of one run."""
    r = find(run, root)
    typer.echo(f"{r.name}  [{r.status}]  {r.dir}")
    typer.echo(
        json.dumps({"meta": r.meta, "summary": r.summary}, indent=2, default=str)
    )


@app.command()
def tail(
    run: str,
    metrics: Annotated[
        str,
        typer.Option(
            "-m", help="Comma-separated substrings, e.g. reward,sys/gpu0/util."
        ),
    ] = "",
    follow: Annotated[
        bool, typer.Option("-f", help="Keep printing until the run ends.")
    ] = False,
    interval: float = 5.0,
    root: Root = None,
) -> None:
    """Print events and metrics as they're logged."""
    r = find(run, root)
    wanted = [m for m in metrics.split(",") if m]
    typer.echo(f"# {r.name} ({r.status})")
    seen = 0
    while True:
        events = list(r.events())
        for e in events[seen:]:
            line = _event_line(e, wanted)
            if line:
                typer.echo(line)
        seen = len(events)
        if not follow or r.status != "running":
            break
        time.sleep(interval)


@app.command()
def compare(
    metrics: Annotated[str, typer.Option(help="Comma-separated metric names.")],
    group_by: Annotated[
        str | None,
        typer.Option(help="Comma-separated dotted config keys, e.g. env.mode."),
    ] = None,
    where: Annotated[
        list[str] | None,
        typer.Option(help="key=value on the config, e.g. tracking.wandb.group=x."),
    ] = None,
    filter: Annotated[
        str, typer.Option(help="Only runs whose name contains this.")
    ] = "",
    csv_path: Annotated[Path | None, typer.Option("--csv")] = None,
    root: Root = None,
) -> None:
    """Markdown table of finished runs; --group-by gives mean ± std over e.g. seeds."""
    names = metrics.split(",")
    conds = [w.split("=", 1) for w in where or []]
    selected = [
        r
        for r in runs(root)
        if r.summary and filter in r.name and all(str(r.cfg(k)) == v for k, v in conds)
    ]
    if group_by:
        keys = group_by.split(",")
        rows = group(selected, keys, names)
        columns = [*keys, "n", *names]
    else:
        rows = [
            {"run": r.name, "status": r.status, **{m: r.last(m) for m in names}}
            for r in selected
        ]
        columns = ["run", "status", *names]
    typer.echo(markdown(rows, columns))
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


@app.command()
def sweep(
    grid: Annotated[
        list[str], typer.Option(help="key=v1,v2,... (repeat for more axes).")
    ],
) -> None:
    """Print one line of overrides per point of the grid. The key `config` gives
    --config files (merged over the base config); other keys give --set."""
    axes = [g.split("=", 1) for g in grid]
    for combo in itertools.product(*(values.split(",") for _, values in axes)):
        args = []
        for (key, _), value in zip(axes, combo, strict=True):
            args += (
                ["--config", value] if key == "config" else ["--set", f"{key}={value}"]
            )
        typer.echo(shlex.join(args))


def group(selected: list[RunView], keys: list[str], metrics: list[str]) -> list[dict]:
    """mean ± std (and n) of each metric over runs sharing config values."""
    groups: dict = {}
    for r in selected:
        groups.setdefault(tuple(str(r.cfg(k)) for k in keys), []).append(r)
    out = []
    for values, members in sorted(groups.items()):
        row = {**dict(zip(keys, values, strict=True)), "n": len(members)}
        for m in metrics:
            vals = [v for r in members if _is_number(v := r.last(m))]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row[m] = f"{statistics.mean(vals):.4g} ± {sd:.2g}"
        out.append(row)
    return out


def markdown(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |" for row in rows
    ]
    return "\n".join(lines)


def _event_line(e: dict, wanted: list[str]) -> str | None:
    kind = e.get("event")
    clock = e.get("time", "")[11:19]
    if kind != "metrics":
        extra = {k: v for k, v in e.items() if k not in ("event", "time")}
        return f"{clock} [{kind}] " + " ".join(
            f"{k}={_fmt(v)}" for k, v in extra.items()
        )
    fields = {k: v for k, v in e.items() if k not in ("event", "time", "step")}
    if wanted:
        fields = {k: v for k, v in fields.items() if any(w in k for w in wanted)}
        if not fields:
            return None
    return f"{clock} step={e.get('step')} " + " ".join(
        f"{k}={_fmt(v)}" for k, v in fields.items()
    )


def _best(r: RunView) -> str | None:
    value = r.summary.get("best_value", r.meta.get("best_value"))
    metric = r.cfg("tracking.best.metric")
    return f"{metric}={_fmt(value)}" if metric and value is not None else None


def _duration(sec: float | None) -> str | None:
    if sec is None:
        return None
    h, rem = divmod(int(sec), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4g}" if isinstance(v, float) else str(v)
