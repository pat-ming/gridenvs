# Training and evaluation

`fbrl.grid` plays an environment in plain text. The observation goes in as a user message and the
model's reply goes to `env.step`, for exactly `H` turns. An episode is one token sequence with a mask
over the model's reply tokens, so eval, SFT, GRPO and probing all see the same tokens. Eval instances
use seeds from `EVAL_SEED_START` up, and SFT and GRPO seeds stay below it.

It also adds a `residual` mode to the 4x4 env: E_C's line plus how far each violated sum is from its
target (`violated: {0 (+3), 4}`). So `none`, `true` and `residual` hand the model nothing, the broken
constraint, or that plus the arithmetic.

| Module | What it does |
|---|---|
| `fbrl.evaluate` | per (instance params, H, mode): success, `mixed` (share of instances whose samples disagree, i.e. where GRPO gets a signal), parse rate, on-target edits, counterfactual score |
| `fbrl.train_sft` | warm start on scripted episodes with modes spread evenly; `demos.oracle` is the share of solved demos |
| `fbrl.train_grpo` | GRPO through TRL's `rollout_func`; `env.mode` can be a schedule like `[[0, "true"], [200, none]]`; runs `fbrl.evaluate` every `eval.every` steps |

Configs are layered: a base config, then optionally a preset from `configs/models/`, then `--set`.
Run names, tags and W&B notes come from `experiment.{name,tags,notes}` and can use config values
(`eval_{model.name}`). Say why a run exists with `--set experiment.notes="..."`.

`python -m fbrl.workspace` (on the cluster) saves the W&B "Overview" view: GRPO learning signal first,
then held-out eval over training, checkpoint evals as bars, then optimization and system panels.

```sh
cluster/remote.sh submit -- fbrl.evaluate --config configs/eval.yaml --set model.name=Qwen/Qwen3-1.7B
cluster/remote.sh submit -- fbrl.train_sft --config configs/sft.yaml --config configs/models/qwen3-1.7b.yaml \
    --set experiment.export='$FBRL/models/sft-qwen3-1.7b'
cluster/remote.sh sweep --gres=gpu:a40:1 -- fbrl.train_grpo --config configs/grpo.yaml \
    --grid env.mode=none,true,noise,residual --grid experiment.seed=0,1,2,3,4
cluster/remote.sh harness compare --where tracking.wandb.group=<sweep> --group-by env.mode \
    --metrics eval/spec_m1_H1_none/success,eval/spec_m1_H1_counterfactual/cf_score
cluster/remote.sh pull     # small run files -> results/runs/
```

## Cluster

Code, venv, caches and runs live in `/engrfs/project/netanel.raviv/motti_k/fbrl/`. Jobs run in
`nvcr.pytorch-25.11.sif` with a uv venv, on one A100-SXM4 on `general-gpu-long` by default. Pass
`--gres=gpu:a40:1` or `gpu:a6000:1` (48 GB) for anything that fits there, which is most eval and 0.6B
to 1.7B training. A sweep's runs share a W&B group named after the sweep (project `fbrl` on
forge.coreweave.com). `cluster/remote.sh` with no arguments lists its commands.
