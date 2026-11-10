"""
The W&B "Overview" view: the panels to look at first, top to bottom. Re-running
saves a new copy of the view (delete the old one in the UI).

    python -m fbrl.workspace   # entity/project from WANDB_ENTITY / WANDB_PROJECT
"""

import os

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

STEP = "train/global_step"


def line(title: str, y: str | None = None, regex: str | None = None, x: str = STEP):
    return wr.LinePlot(title=title, x=x, y=[y] if y else [], metric_regex=regex)


def bars(title: str, metrics: list[str]):
    return wr.BarPlot(title=title, metrics=metrics, orientation="v")


def cell(set_name: str, mode: str, stat: str = "success") -> str:
    return f"eval/{set_name}_m1_H1_{mode}/{stat}"


SECTIONS = [
    ws.Section(
        name="1. Is GRPO learning",
        is_open=True,
        panels=[
            line("Rollout success", "rollout/success"),
            line("Groups with a learning signal (mixed outcomes)", "rollout/mixed"),
            line("Groups with zero reward variance", "train/frac_reward_zero_std"),
            line("Edits inside a violated constraint", "rollout/on_target"),
            line("Parse rate", "rollout/parse"),
            line("Reply tokens per episode", "train/completions/mean_length"),
        ],
    ),
    ws.Section(
        name="2. Held-out eval during training",
        is_open=True,
        panels=[
            line("Success, no feedback line", regex=r"eval/.*_none/success"),
            line("Success, true feedback line", regex=r"eval/.*_true/success"),
            line(
                "Counterfactual score (> 0: follows the false line)",
                regex=r"eval/.*_counterfactual/cf_score",
            ),
            line("On-target edits, no feedback line", regex=r"eval/.*_none/on_target"),
        ],
    ),
    ws.Section(
        name="3. Checkpoint evals, m=1 H=1 (one bar per run)",
        is_open=True,
        panels=[
            bars(
                f"{s}: success by feedback mode",
                [cell(s, m) for m in ("none", "true", "residual")],
            )
            for s in ("spec", "rows", "ineq", "distinct")
        ]
        + [
            bars("Parse rate", [cell(s, "none", "parse") for s in ("spec", "distinct")])
        ],
    ),
    ws.Section(
        name="4. Optimization",
        panels=[
            line("Loss", "train/loss"),
            line("Grad norm", "train/grad_norm"),
            line("Entropy", "train/entropy"),
            line("SFT token accuracy", "train/mean_token_accuracy"),
            line(
                "vLLM vs trainer logprob gap",
                "train/sampling/sampling_logp_difference/mean",
            ),
            line("Clipped ratio", "train/clip_ratio/region_mean"),
            line("Learning rate", "train/learning_rate"),
            line("Seconds per step", "train/step_time"),
        ],
    ),
    ws.Section(
        name="5. System",
        panels=[
            line("GPU utilization (%)", regex=r"system/gpu\.\d+\.gpu$", x="_runtime"),
            line(
                "GPU memory allocated (%)",
                regex=r"system/gpu\.\d+\.memoryAllocated$",
                x="_runtime",
            ),
            line("GPU power (W)", regex=r"system/gpu\.\d+\.powerWatts$", x="_runtime"),
        ],
    ),
]


def main() -> None:
    view = ws.Workspace(
        entity=os.environ.get("WANDB_ENTITY", "cse-5100-fall-2026"),
        project=os.environ.get("WANDB_PROJECT", "fbrl"),
        name="Overview",
        sections=SECTIONS,
        settings=ws.WorkspaceSettings(x_axis=STEP, max_runs=30),
    )
    view.save()
    print(view.url)


if __name__ == "__main__":
    main()
