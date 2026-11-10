# harness

Experiment tracking for any training or eval script. Each run writes a directory with the config, metrics and events as JSONL, record streams, checkpoints, and a summary, and optionally mirrors metrics to W&B.

## Usage

```python
import harness

cfg = harness.get_config(
    "configs/smoke.yaml"
)  # also reads --config / --set key=val from argv
with harness.init(cfg) as run:
    for step in range(n):
        harness.log({"loss": loss, "lr": lr}, step=step, prefix="train/")
    harness.log({"acc": acc}, step=step, prefix="eval/")
    harness.record("predictions", [{"input": x, "pred": p, "label": y}], step=step)
    run.save_checkpoint({"model": model.state_dict()}, step=step)
```

`harness.log`, `harness.record` and `harness.event` write to the current run. On ranks other than 0 they do nothing.

Run directory:

```text
runs/<name>_<utc timestamp>[_<slurm job id>]/
  config.yaml      frozen config
  meta.json        git state, Slurm job, host, GPUs, versions, W&B URL, status
  events.jsonl     metrics and events
  records/         one JSONL file per record stream
  checkpoints/
  profiles/
  summary.json     written when the run ends
```

Status is `running`, then `finished`, `failed` (exception), or `interrupted` (Ctrl-C or SIGTERM, e.g. a Slurm time limit). `Run(cfg, resume=run_dir)` resumes into the same directory and W&B run.

## Config

All keys are optional.

```yaml
experiment: {name, tags, output_dir}        # output_dir defaults to $HARNESS_RUNS_DIR, else runs/
tracking:
  wandb: {enabled, project, entity, group, mode}
  best: {metric: eval/acc, mode: max}       # save_checkpoint also writes best.pt on improvement
  keep_last_n_checkpoints: 0                # 0 keeps all
  system_interval: 10                       # GPU sampling period in seconds, 0 disables
```

`--config` can be given more than once, and later files are merged over earlier ones, so a small preset (say, a model size) can sit on top of a base config. `--set` only accepts existing keys. Use `+` to add one: `--set +grpo.beta=0.04`. Values starting with `[` or `{` are parsed as YAML.

## GPU stats

If NVML is available, every visible GPU is sampled every `system_interval` seconds and logged as `sys/gpu<i>/{util,mem_util,mem_gb,power_w,temp_c,sm_mhz}`. These are written locally only, since W&B collects system metrics itself. `summary.json` gets `gpu<i>_util_mean`, `gpu<i>_mem_peak_gb`, `gpu<i>_energy_kwh`, and `gpu_hours`.

`profiling.vram_snapshot()` returns PyTorch allocator memory. `profiling.profile_steps()` writes a Chrome trace and a top-ops table to `profiles/`.

## HF / TRL trainers

```python
from harness.hf import HarnessCallback

run = harness.init(cfg, step_key="train/global_step")
trainer = GRPOTrainer(..., args=GRPOConfig(report_to="wandb", ...), callbacks=[HarnessCallback(run)])
```

The trainer's W&B logging goes to the run harness started. The callback copies trainer logs to `events.jsonl`, logs checkpoint saves, and adds token and FLOP totals to the summary. See `fbrl/train_grpo.py`.

## Reading runs

```python
run = harness.find("grpo_smoke")  # path, or newest run whose name contains this
df = run.metrics()  # pandas DataFrame, one row per log() call
rollouts = list(run.records("rollouts"))
c_runs = [r for r in harness.runs() if r.cfg("env.condition") == "C"]
```

## CLI

```text
harness ls                                   # all runs: status, time, GPU-hours, util, peak mem, best
harness show RUN                             # meta.json and summary.json
harness tail RUN [-f] [-m reward,sys/gpu0]   # print events, optionally follow
harness compare --metrics a,b [--group-by env.condition,model.name] [--where k=v] [--csv out.csv]
harness sweep --grid experiment.seed=0,1,2 --grid config=a.yaml,b.yaml
```

`RUN` is a path or a substring of a run name. `--group-by` reports mean ± std per combination of config values, and `--where` keeps runs whose config matches. Output is Markdown.

`sweep` prints one line of arguments per point of the grid (`--set key=value`, or `--config file` for the key `config`). A launcher runs the script once per line; on Slurm, `cluster/remote.sh sweep` submits the lines as an array job.
