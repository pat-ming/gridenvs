"""
Plain-text multi-turn play between a chat model and the environments in
gridenvs (grid4, sudoku). One turn is one model reply, then the env's next observation as
a user message, for exactly H turns.

Each episode is a single append-only token sequence (first prompt, then reply,
observation, reply, ...), so eval, SFT, GRPO and interp all see the same tokens.
`mask` marks reply tokens (1) against env tokens (0) over the part after the prompt.
"""

import importlib
from dataclasses import dataclass, field

import numpy as np

ENVS = {
    "grid4": ("gridenvs.grid4.grid_env", "GridEnv"),
    "sudoku": ("gridenvs.sudoku.sudoku_env", "SudokuEnv"),
}
# Instance seeds. Training draws from below EVAL_SEED_START, evaluation from above.
EVAL_SEED_START = 1_000_000_000
REPLY, OBS = "@@REPLY@@", "@@OBS@@"


def load_env(name: str):
    """(env class, make_instance) for "grid4" or "sudoku"."""
    module, cls = ENVS[name]
    env_cls = getattr(importlib.import_module(module), cls)
    package = module.rsplit(".", 1)[0]
    make_instance = importlib.import_module(f"{package}.instances").make_instance
    return (with_residual(env_cls) if name == "grid4" else env_cls), make_instance


def with_residual(env_cls):
    class ResidualGridEnv(env_cls):
        """Adds mode "residual": E_C's line plus, for sum constraints, how much
        the sum is short of its target, e.g. `violated: {0 (+3), 4}`. Offloads
        the arithmetic as well as finding the broken constraint."""

        def __init__(self, mode: str, H: int, seed: int = 0, noise_sizes=None):
            self.residual = mode == "residual"
            super().__init__("true" if self.residual else mode, H, seed, noise_sizes)

        def _feedback_line(self):
            line = super()._feedback_line()
            if not self.residual:
                return line
            items = []
            for k in self.shown:
                c = self.cs[k]
                if hasattr(c, "target"):
                    total = sum(int(self.g[cell]) for cell in c.scope())
                    items.append(f"{k} ({c.target - total:+d})")
                else:
                    items.append(str(k))
            return "violated: {" + ", ".join(items) + "}"

    return ResidualGridEnv


def edited_cell(env_name: str, action: tuple) -> tuple[int, int]:
    return tuple(action[:2]) if env_name == "grid4" else tuple(action[1:])


def sample_params(spec: dict, rng: np.random.Generator) -> dict:
    """One draw from {key: [choices]}, e.g. {set_name: [spec], m: [1, 2]}."""
    return {k: v[rng.integers(len(v))] for k, v in spec.items()}


def mode_at(schedule, step: int) -> str:
    """`schedule` is a mode, or [[start_step, mode], ...] sorted by start_step."""
    if isinstance(schedule, str):
        return schedule
    return [mode for start, mode in schedule if start <= step][-1]


@dataclass
class Episode:
    env: object
    ids: list[int]
    prompt_len: int
    mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    turn_starts: list[int] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False

    @property
    def prompt_ids(self) -> list[int]:
        return self.ids[: self.prompt_len]

    @property
    def completion_ids(self) -> list[int]:
        return self.ids[self.prompt_len :]


class ChatFormat:
    """Token boundaries for the chat template, derived from the tokenizer itself."""

    def __init__(self, tok, **template_kwargs):
        self.tok = tok
        self.kwargs = template_kwargs
        msgs = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": REPLY},
            {"role": "user", "content": OBS},
        ]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        # e.g. "<|im_end|>\n<|im_start|>user\n@@OBS@@<|im_end|>\n<|im_start|>assistant\n"
        self.between = text[text.index(REPLY) + len(REPLY) :]
        self.end_of_reply = tok.convert_tokens_to_ids("<|im_end|>")

    def first(self, obs: str) -> list[int]:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": obs}],
            tokenize=False,
            add_generation_prompt=True,
            **self.kwargs,
        )
        return self.tok.encode(text, add_special_tokens=False)

    def next(self, obs: str, reply_ids: list[int]) -> list[int]:
        text = self.between.replace(OBS, obs)
        if reply_ids and reply_ids[-1] == self.end_of_reply:
            text = text.removeprefix("<|im_end|>")
        return self.tok.encode(text, add_special_tokens=False)


def start(env, inst: dict, fmt: ChatFormat) -> Episode:
    ids = fmt.first(env.reset(inst))
    return Episode(env=env, ids=ids, prompt_len=len(ids))


def advance(e: Episode, fmt: ChatFormat, reply: list[int], lps: list[float]) -> None:
    e.turn_starts.append(len(e.ids))
    e.ids += reply
    e.mask += [1] * len(reply)
    e.logprobs += lps
    obs, e.reward, e.done, _ = e.env.step(
        fmt.tok.decode(reply, skip_special_tokens=True)
    )
    if not e.done:
        nxt = fmt.next(obs, reply)
        e.ids += nxt
        e.mask += [0] * len(nxt)
        e.logprobs += [0.0] * len(nxt)


def play(llm, fmt: ChatFormat, episodes: list[Episode], sampling) -> None:
    """Run every episode to its horizon, batching one vLLM call per turn."""
    from vllm import TokensPrompt

    while active := [e for e in episodes if not e.done]:
        outs = llm.generate(
            [TokensPrompt(prompt_token_ids=e.ids) for e in active],
            sampling,
            use_tqdm=False,
        )
        for e, out in zip(active, outs, strict=True):
            gen = out.outputs[0]
            reply = list(gen.token_ids)
            lps = (
                [d[t].logprob for d, t in zip(gen.logprobs, reply, strict=True)]
                if gen.logprobs
                else [0.0] * len(reply)
            )
            advance(e, fmt, reply, lps)


def episode_row(e: Episode, env_name: str) -> dict:
    turns = []
    for t in e.env.log:
        on_target = None
        if t["action"] is not None and t["V"]:
            scope = frozenset().union(*(e.env.cs[k].scope() for k in t["V"]))
            on_target = edited_cell(env_name, t["action"]) in scope
        turns.append(
            {
                "V": t["V"],
                "shown": t["shown"],
                "text": t["text"],
                "action": t["action"],
                "on_target": on_target,
            }
        )
    return {"reward": e.reward, "turns": turns}


def summarize(rows: list[dict], group_size: int) -> dict:
    """Episode stats. Rows must come in groups of `group_size` per instance."""
    rewards = np.array([r["reward"] for r in rows]).reshape(-1, group_size)
    wins = rewards.sum(1)
    turns = [t for r in rows for t in r["turns"]]
    targeted = [t["on_target"] for t in turns if t["on_target"] is not None]
    return {
        "success": float(rewards.mean()),
        "any_success": float((wins > 0).mean()),
        "mixed": float(((wins > 0) & (wins < group_size)).mean()),
        "parse": float(np.mean([t["action"] is not None for t in turns])),
        "on_target": float(np.mean(targeted)) if targeted else float("nan"),
    }
