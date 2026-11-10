import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

import harness
from harness import Run, override_config
from harness.cli import app


def cfg_for(tmp_path, condition="A"):
    return {
        "experiment": {
            "name": "t{env.condition}",
            "tags": ["x"],
            "output_dir": str(tmp_path),
        },
        "env": {"condition": condition},
        "tracking": {"best": {"metric": "eval/acc", "mode": "max"}},
    }


@pytest.mark.parametrize(
    "override, path, expected",
    [
        ("env.condition=C", ("env", "condition"), "C"),
        ("+grpo.beta=0.04", ("grpo", "beta"), 0.04),
        ('+experiment.tags=["a","b"]', ("experiment", "tags"), ["a", "b"]),
        (
            'env.condition=[[0, "true"], [200, none]]',
            ("env", "condition"),
            [[0, "true"], [200, "none"]],
        ),
        ("+model.lora={r: 8}", ("model", "lora"), {"r": 8}),
    ],
)
def test_overrides(tmp_path, override, path, expected):
    cfg = override_config(cfg_for(tmp_path), [override])
    assert cfg[path[0]][path[1]] == expected


def test_override_rejects_unknown_key_without_plus(tmp_path):
    with pytest.raises(KeyError):
        override_config(cfg_for(tmp_path), ["env.conditon=C"])


def test_run_roundtrip_best_and_resume(tmp_path):
    with harness.init(cfg_for(tmp_path)) as run:
        harness.log({"acc": 0.5}, step=1, prefix="eval/")
        harness.log({"loss": 1.0}, step=2, prefix="train/")
        assert run.is_best  # an unrelated metric doesn't clear it
        harness.log({"acc": 0.4}, step=3, prefix="eval/")
        assert not run.is_best
        harness.record("rollouts", [{"reward": 1.0, "actions": [1, 2]}], step=3)
    assert harness.current() is None

    view = harness.load(run.dir)
    assert (view.status, view.summary["best_value"], view.summary["best_step"]) == (
        "finished",
        0.5,
        1,
    )
    assert list(view.metrics()["eval/acc"].dropna()) == [0.5, 0.4]
    assert list(view.records("rollouts")) == [
        {"step": 3, "reward": 1.0, "actions": [1, 2]}
    ]
    assert view.cfg("env.condition") == "A"

    with pytest.raises(RuntimeError), Run(cfg_for(tmp_path), resume=run.dir) as resumed:
        assert resumed.best_value == 0.5
        raise RuntimeError("boom")
    assert view.status == "failed"
    assert [e["event"] for e in view.events()].count("start") == 2


def test_sigterm_marks_run_interrupted(tmp_path):
    script = textwrap.dedent(f"""
        import os, signal, harness
        with harness.init({cfg_for(tmp_path)!r}) as run:
            print(run.dir, flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
    """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert out.returncode == 143
    assert harness.load(out.stdout.strip()).status == "interrupted"


def test_nonzero_rank_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    with Run(cfg_for(tmp_path)) as run:
        run.log({"acc": 1.0}, step=1)
        run.record("x", [{"a": 1}])
    assert not any(tmp_path.iterdir())


def test_cli_ls_and_compare_group_by(tmp_path):
    for cond, acc in [("A", 0.2), ("A", 0.4), ("C", 0.9)]:
        with Run(cfg_for(tmp_path, cond)) as run:
            run.log({"acc": acc}, step=1, prefix="eval/")
        run.dir.rename(tmp_path / f"{cond}_{acc}")

    cli = CliRunner()
    out = cli.invoke(
        app,
        [
            "compare",
            "--root",
            str(tmp_path),
            "--metrics",
            "eval/acc",
            "--group-by",
            "env.condition",
        ],
    ).output
    assert "| A | 2 | 0.3 ± 0.14 |" in out
    assert "| C | 1 | 0.9 ± 0 |" in out

    out = cli.invoke(app, ["ls", "--root", str(tmp_path)]).output
    assert out.count("finished") == 3 and "eval/acc=0.9" in out


def test_cli_sweep_is_the_cartesian_product():
    out = (
        CliRunner()
        .invoke(
            app,
            [
                "sweep",
                "--grid",
                "config=m/a.yaml,m/b.yaml",
                "--grid",
                "env.mode=none,true",
            ],
        )
        .output.splitlines()
    )
    assert out == [
        "--config m/a.yaml --set env.mode=none",
        "--config m/a.yaml --set env.mode=true",
        "--config m/b.yaml --set env.mode=none",
        "--config m/b.yaml --set env.mode=true",
    ]
