"""
Held-out evaluation of a model (hub name or checkpoint path) on a grid of cells:
instance params x slack x observation mode. GRPO runs the same `evaluate` every
`eval.every` steps on its colocated vLLM engine.

Per cell: success, any_success and mixed (share of instances whose samples
neither all fail nor all succeed, i.e. would give GRPO a signal), parse rate,
on_target (edits inside a violated constraint), and for counterfactual mode the
mean `counterfactual_score` (> 0 means edits follow the false line).

    python -m fbrl.evaluate --set model.name=Qwen/Qwen3-1.7B
"""

import importlib
import itertools

import numpy as np

from fbrl.grid import (
    ENVS,
    EVAL_SEED_START,
    ChatFormat,
    episode_row,
    load_env,
    play,
    start,
    summarize,
)
from harness import Run, get_config, set_seed


def cells_of(ev: dict) -> list[dict]:
    keys = list(ev["instances"])
    grid = itertools.product(
        itertools.product(*ev["instances"].values()), ev["slack"], ev["modes"]
    )
    return [
        {
            **dict(zip(keys, values, strict=True)),
            "H": values[keys.index("m")] + s,
            "mode": mode,
        }
        for values, s, mode in grid
    ]


def cell_name(cell: dict) -> str:
    """e.g. spec_m1_H1_true. Instance params come first, mode last."""
    keys = list(cell)[: list(cell).index("mode") + 1]
    return "_".join(f"{k}{cell[k]}" if k in ("m", "H") else str(cell[k]) for k in keys)


def evaluate(llm, fmt: ChatFormat, env_name: str, ev: dict, sampling):
    """Returns (cells with stats, one row per episode)."""
    env_cls, make_instance = load_env(env_name)
    cf_score = importlib.import_module(ENVS[env_name][0]).counterfactual_score
    k = ev["samples"]
    cells = cells_of(ev)
    episodes = []
    for cell in cells:
        params = {key: cell[key] for key in ev["instances"]}
        for i in range(ev["n_instances"]):
            inst = make_instance(**params, seed=EVAL_SEED_START + i)
            for j in range(k):
                env = env_cls(
                    mode=cell["mode"],
                    H=cell["H"],
                    seed=i * k + j,
                    noise_sizes=ev.get("noise_sizes"),
                )
                episodes.append(start(env, inst, fmt))
    play(llm, fmt, episodes, sampling)

    per_cell = ev["n_instances"] * k
    rows = []
    for c, cell in enumerate(cells):
        eps = episodes[c * per_cell : (c + 1) * per_cell]
        cell_rows = []
        for n, e in enumerate(eps):
            row = {**cell, "instance": n // k, **episode_row(e, env_name)}
            if ev.get("save_tokens"):
                row |= {
                    "ids": e.ids,
                    "prompt_len": e.prompt_len,
                    "turn_starts": e.turn_starts,
                }
            cell_rows.append(row)
        cell |= summarize(cell_rows, k)
        if cell["mode"] == "counterfactual":
            scores = [s["score"] for e in eps if (s := cf_score(e.env.log, e.env.cs))]
            cell["cf_score"] = float(np.mean(scores)) if scores else float("nan")
        rows += cell_rows
    return cells, rows


def sampling_params(ev: dict, max_reply_tokens: int, stop: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=ev["temperature"],
        max_tokens=max_reply_tokens,
        stop_token_ids=[stop],
    )


def log_cells(run: Run, cells: list[dict], step: int | None) -> None:
    stats = ("success", "mixed", "parse", "on_target", "cf_score")
    run.log(
        {f"{cell_name(c)}/{s}": c[s] for c in cells for s in stats if s in c},
        step=step,
        prefix="eval/",
    )
    run.record("eval_cells", cells, step=step, wandb_sample=len(cells))


def main() -> None:
    from vllm import LLM

    cfg = get_config("configs/eval.yaml")
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    ev = cfg["eval"]
    with Run(cfg) as run:
        llm = LLM(
            cfg["model"]["name"],
            gpu_memory_utilization=0.85,
            max_model_len=ev.get("max_model_len", 4096),
            seed=cfg["experiment"]["seed"],
        )
        fmt = ChatFormat(
            llm.get_tokenizer(), **cfg["model"].get("chat_template_kwargs", {})
        )
        sampling = sampling_params(ev, cfg["env"]["max_reply_tokens"], fmt.end_of_reply)
        cells, rows = evaluate(llm, fmt, cfg["env"]["name"], ev, sampling)
        run.record("eval_episodes", rows)
        log_cells(run, cells, step=0)
        for c in cells:
            print(c, flush=True)
        run.summary["cells"] = cells


if __name__ == "__main__":
    main()
