"""
Multi-turn GRPO on the gridenvs environments through TRL's `rollout_func`, using the same
play loop as evaluation (fbrl.grid). Env tokens are masked out of the loss.

    python -m fbrl.train_grpo --config configs/grpo.yaml --set env.mode=none
    python -m fbrl.train_grpo --config configs/grpo.yaml --config configs/models/qwen3-4b.yaml

`env.mode` is a mode or a schedule [[step, mode], ...] (e.g. feedback removed
halfway). Everything under `grpo:` goes straight to GRPOConfig.
"""

import json

import numpy as np
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from fbrl.evaluate import evaluate, log_cells, sampling_params
from fbrl.grid import (
    EVAL_SEED_START,
    ChatFormat,
    episode_row,
    load_env,
    mode_at,
    play,
    sample_params,
    start,
    summarize,
)
from harness import Run, get_config, set_seed
from harness.hf import HarnessCallback


def instance_dataset(env_cfg: dict, n: int, seed: int) -> Dataset:
    """One row per training instance. The prompt column carries the instance spec;
    the rollout turns it into the real prompt."""
    rng = np.random.default_rng(seed)
    first = seed * 10_000_000
    assert first + n < EVAL_SEED_START, "training seeds would reach eval seeds"
    rows = []
    for i in range(n):
        spec = {
            "params": sample_params(env_cfg["instances"], rng),
            "slack": int(rng.choice(env_cfg["slack"])),
            "seed": first + i,
        }
        rows.append({"prompt": json.dumps(spec)})
    return Dataset.from_list(rows)


def make_rollout(cfg: dict, run: Run):
    env_cfg, track = cfg["env"], cfg["tracking"]
    env_cls, make_instance = load_env(env_cfg["name"])
    chat_kwargs = cfg["model"].get("chat_template_kwargs", {})
    fmt = None

    def rollout(prompts: list[str], trainer) -> dict:
        nonlocal fmt
        from vllm import SamplingParams

        fmt = fmt or ChatFormat(trainer.processing_class, **chat_kwargs)
        step = trainer.state.global_step
        mode = mode_at(env_cfg["mode"], step)
        args = trainer.args
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=env_cfg["max_reply_tokens"],
            stop_token_ids=[fmt.end_of_reply],
            logprobs=0,
        )
        episodes = []
        for i, prompt in enumerate(prompts):
            spec = json.loads(prompt)
            env = env_cls(
                mode=mode,
                H=spec["params"]["m"] + spec["slack"],
                seed=step * len(prompts) + i,
                noise_sizes=env_cfg.get("noise_sizes"),
            )
            inst = make_instance(**spec["params"], seed=spec["seed"])
            episodes.append(start(env, inst, fmt))
        play(trainer.vllm_generation.llm, fmt, episodes, sampling)

        rows = [
            {**json.loads(p), "mode": mode} | episode_row(e, env_cfg["name"])
            for p, e in zip(prompts, episodes, strict=True)
        ]
        run.log(summarize(rows, args.num_generations), step=step, prefix="rollout/")
        every = track.get("rollout_table_every", 0)
        sample = 8 if every and step % every == 0 else 0
        run.record("rollouts", rows, step=step, wandb_sample=sample)
        return {
            "prompt_ids": [e.prompt_ids for e in episodes],
            "completion_ids": [e.completion_ids for e in episodes],
            "logprobs": [e.logprobs for e in episodes],
            "env_mask": [e.mask for e in episodes],
            "env_reward": [e.reward for e in episodes],
        }

    return rollout


def env_reward(prompts, completions, env_reward, **kwargs) -> list[float]:
    return env_reward


class PeriodicEval(TrainerCallback):
    """Held-out eval (fbrl.evaluate) on the trainer's own vLLM engine every N steps."""

    def __init__(self, run: Run, cfg: dict):
        self.run, self.cfg, self.trainer = run, cfg, None

    def _eval(self, step: int) -> None:
        t = self.trainer
        if step != t._last_loaded_step:
            t.vllm_generation.sync_weights()
            t._last_loaded_step = step
        ev, env_cfg = self.cfg["eval"], self.cfg["env"]
        fmt = ChatFormat(
            t.processing_class, **self.cfg["model"].get("chat_template_kwargs", {})
        )
        sampling = sampling_params(ev, env_cfg["max_reply_tokens"], fmt.end_of_reply)
        cells, rows = evaluate(
            t.vllm_generation.llm, fmt, env_cfg["name"], ev, sampling
        )
        log_cells(self.run, cells, step)
        self.run.record("eval_episodes", rows, step=step)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.cfg["eval"].get("at_start", True):
            self._eval(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        every = self.cfg["eval"]["every"]
        if every and state.global_step % every == 0:
            self._eval(state.global_step)


def main() -> None:
    cfg = get_config("configs/grpo.yaml")
    exp = cfg["experiment"]
    set_seed(exp["seed"], deterministic=False)
    with Run(cfg, resume=exp.get("resume"), step_key="train/global_step") as run:
        args = GRPOConfig(
            output_dir=str(run.dir / "checkpoints"),
            run_name=run.dir.name,
            seed=exp["seed"],
            report_to="wandb" if run.wandb else "none",
            **cfg["grpo"],
        )
        n_instances = (
            args.max_steps * args.generation_batch_size // args.num_generations
        )
        lora = cfg["model"].get("lora")
        evaluator = PeriodicEval(run, cfg)
        trainer = GRPOTrainer(
            model=cfg["model"]["name"],
            args=args,
            reward_funcs=[env_reward],
            train_dataset=instance_dataset(cfg["env"], n_instances, exp["seed"]),
            rollout_func=make_rollout(cfg, run),
            peft_config=LoraConfig(**lora) if lora else None,
            callbacks=[HarnessCallback(run), evaluator],
        )
        evaluator.trainer = trainer
        trainer.train(resume_from_checkpoint=exp.get("resume_checkpoint"))
        trainer.save_model(str(run.dir / "checkpoints" / "final"))


if __name__ == "__main__":
    main()
