"""
SFT warm start on scripted episodes, in the same token format as fbrl.grid.
Teaches the reply format and turn-taking; how much it teaches about solving is
set by `demos.oracle`, the share of episodes played by the oracle (fix a wrong
cell with its solution value) rather than by random valid edits.
Observation modes are spread evenly so the warm start favors none of them.

    python -m fbrl.train_sft --config configs/sft.yaml --set model.name=Qwen/Qwen3-1.7B-Base
"""

import os
import shutil

import numpy as np
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from fbrl.grid import ChatFormat, advance, load_env, sample_params, start
from harness import Run, get_config, set_seed
from harness.hf import HarnessCallback

# SFT instances sit above GRPO's (seed * 1e7 + i) and below EVAL_SEED_START.
SFT_SEED_START = 900_000_000


def oracle_reply(env, inst: dict, env_name: str, rng) -> str:
    wrong = np.argwhere(env.g != inst["solution"])
    if len(wrong):
        r, c = map(int, wrong[rng.integers(len(wrong))])
    else:  # solved: rewrite a cell with its own value
        r, c = map(int, rng.integers(len(env.g), size=2))
    v = int(inst["solution"][r, c])
    return f"EDIT {r} {c} {v}" if env_name == "grid4" else f"WRITE {v} {r} {c}"


def random_reply(env, env_name: str, rng) -> str:
    n = len(env.g)
    r, c = map(int, rng.integers(n, size=2))
    if env_name == "grid4":
        return f"EDIT {r} {c} {int(rng.integers(10))}"
    return f"WRITE {int(rng.integers(1, 10))} {r} {c}"


def demo_dataset(cfg: dict, tok) -> tuple[Dataset, dict]:
    env_cfg, demos = cfg["env"], cfg["demos"]
    env_cls, make_instance = load_env(env_cfg["name"])
    fmt = ChatFormat(tok, **cfg["model"].get("chat_template_kwargs", {}))
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    rows, solved = [], 0
    for i in range(demos["n_episodes"]):
        params = sample_params(env_cfg["instances"], rng)
        mode = demos["modes"][i % len(demos["modes"])]
        env = env_cls(
            mode=mode,
            H=params["m"] + int(rng.choice(env_cfg["slack"])),
            seed=i,
            noise_sizes=env_cfg.get("noise_sizes"),
        )
        inst = make_instance(**params, seed=SFT_SEED_START + i)
        e = start(env, inst, fmt)
        oracle = rng.random() < demos["oracle"]
        while not e.done:
            text = (
                oracle_reply(env, inst, env_cfg["name"], rng)
                if oracle
                else random_reply(env, env_cfg["name"], rng)
            )
            reply = tok.encode(text, add_special_tokens=False) + [fmt.end_of_reply]
            advance(e, fmt, reply, [0.0] * len(reply))
        solved += e.reward
        rows.append(
            {"input_ids": e.ids, "assistant_masks": [0] * e.prompt_len + e.mask}
        )
    stats = {"episodes": len(rows), "demo_success": solved / len(rows)}
    return Dataset.from_list(rows), stats


def main() -> None:
    cfg = get_config("configs/sft.yaml")
    exp = cfg["experiment"]
    set_seed(exp["seed"], deterministic=False)
    with Run(cfg, step_key="train/global_step") as run:
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        dataset, stats = demo_dataset(cfg, tok)
        run.summary.update(stats)
        run.event("demos", **stats)
        args = SFTConfig(
            output_dir=str(run.dir / "checkpoints"),
            run_name=run.dir.name,
            seed=exp["seed"],
            report_to="wandb" if run.wandb else "none",
            **cfg["sft"],
        )
        trainer = SFTTrainer(
            model=cfg["model"]["name"],
            args=args,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=LoraConfig(**lora)
            if (lora := cfg["model"].get("lora"))
            else None,
            callbacks=[HarnessCallback(run)],
        )
        trainer.train()
        # Merged full weights, since GRPO loads the checkpoint into vLLM.
        final = run.dir / "checkpoints" / "final"
        model = trainer.model.merge_and_unload() if lora else trainer.model
        model.save_pretrained(str(final))
        tok.save_pretrained(str(final))
        run.summary["final_checkpoint"] = str(final)
        if export := exp.get("export"):
            shutil.copytree(final, os.path.expandvars(export), dirs_exist_ok=True)
            run.summary["exported_to"] = os.path.expandvars(export)


if __name__ == "__main__":
    main()
