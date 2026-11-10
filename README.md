# GridEnvs

Environments and training code for *Does RL Build a Circuit the Environment Already Runs?*: does a model
still learn to check constraints itself when the environment does it for it?

```text
src/gridenvs/   the 4x4 grid and Sudoku environments (docs/envs.md)
src/fbrl/       model <-> env play loop, eval, SFT, GRPO (docs/training.md)
src/harness/    experiment tracking, sweeps, comparisons (src/harness/README.md)
configs/        base configs and per-size model presets
cluster/        scripts for shell.engr.wustl.edu
results/runs/   small files from every cluster run
docs/runlog.md  what we ran and what came out
```

```sh
pip install -e . pytest && pytest tests/test_grid4.py tests/test_sudoku.py   # envs only
uv sync && uv run pytest                                                     # everything
cluster/remote.sh sync && cluster/remote.sh submit -- fbrl.evaluate --config configs/eval.yaml
```
