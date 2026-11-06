"""Connects GridEnv to a Hugging Face chat model: conversations, rollouts, token log-probs."""
import gc
import json
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from grid_env import GridEnv
from instances import make_instance

BASE_MODEL = "Qwen/Qwen3-0.6B"
HELDOUT_SEED_START = 1_000_000  # training seeds stay below this


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path: str = BASE_MODEL, device: str | None = None, eager: bool = False):
    """eager=True is required for attention ablation, which needs the attention weights."""
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation="eager" if eager else "sdpa"
    )
    return model.to(device).eval(), tok


class Conversation:
    """A chat kept as token ids, which is exactly what the model saw and generated.

    gen marks tokens the policy sampled. line marks feedback-line tokens.
    Thinking is off: reasoning in text would compute V(s) outside the residual stream, which
    is what the probes measure.
    """

    ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, tok):
        self.tok = tok
        self.inst = None  # the instance this conversation plays, when rollout created it
        self.ids: list[int] = []
        self.gen: list[bool] = []
        self.line: list[bool] = []
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def _extend(self, text: str, is_line: bool = False):
        new = self.tok(text, add_special_tokens=False).input_ids
        self.ids += new
        self.gen += [False] * len(new)
        self.line += [is_line] * len(new)

    def add_user(self, obs: str, line: str | None = None):
        if self.ids:
            text = f"\n<|im_start|>user\n{obs}<|im_end|>\n{self.ASSISTANT}"
        else:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": obs}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            assert text.endswith(self.ASSISTANT), "unexpected chat template"
        if line is None:
            self._extend(text)
        else:
            before, _, after = text.rpartition(line)
            self._extend(before)
            self._extend(line, is_line=True)
            self._extend(after)

    def add_reply(self, token_ids: list[int]) -> str:
        """Append generated tokens, closing the turn if generation hit the length limit."""
        kept = []
        for t in token_ids:
            if t == self.im_end:
                kept.append(t)
                break
            kept.append(t)
        self.ids += kept
        self.gen += [True] * len(kept)
        self.line += [False] * len(kept)
        if not kept or kept[-1] != self.im_end:
            self.ids.append(self.im_end)
            self.gen.append(False)
            self.line.append(False)
        return self.tok.decode(kept, skip_special_tokens=True)


def pad_left(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    ids = torch.tensor([[pad_id] * (n - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs], device=device)
    return ids, mask


def pad_right(seqs: list[list], pad, device, dtype=torch.long) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([list(s) + [pad] * (n - len(s)) for s in seqs], device=device, dtype=dtype)


def generation_config(tok, max_new_tokens: int, do_sample: bool) -> GenerationConfig:
    """Qwen's defaults (temperature 0.6, top_k 20) would sample from a distribution other than
    the softmax policy that GRPO's log-probs assume, so sampling is stated explicitly."""
    pad = tok.pad_token_id
    if do_sample:
        return GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                                max_new_tokens=max_new_tokens, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                                pad_token_id=pad)
    return GenerationConfig(do_sample=False, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                            eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=pad)


@dataclass
class Trajectory:
    inst: dict
    env: GridEnv
    conv: Conversation
    reward: float = 0.0
    solved: bool = False
    log: list = field(default_factory=list)


@torch.no_grad()
def rollout(model, tok, instances, mode, H, noise_sizes=None, max_new_tokens=16, do_sample=True,
            gen_batch=32, on_generate=None, turns=None) -> list[Trajectory]:
    """Play every instance for H turns (or stop after `turns` to study early decisions only).

    on_generate(batch_convs, input_ids, attention_mask) runs before each generate call, so an
    intervention can prepare per-batch state.
    """
    device = next(model.parameters()).device
    cfg = generation_config(tok, max_new_tokens, do_sample)
    trajs = []
    for i, inst in enumerate(instances):
        env = GridEnv(mode, H, seed=[int(inst["seed"]), i], noise_sizes=noise_sizes)
        obs = env.reset(inst)
        conv = Conversation(tok)
        conv.inst = inst
        conv.add_user(obs, env.line)
        trajs.append(Trajectory(inst, env, conv))

    for _ in range(H if turns is None else turns):
        for start in range(0, len(trajs), gen_batch):
            batch = trajs[start:start + gen_batch]
            convs = [t.conv for t in batch]
            ids, mask = pad_left([c.ids for c in convs], tok.pad_token_id, device)
            if on_generate is not None:
                on_generate(convs, ids, mask)
            out = model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)
            new = out[:, ids.shape[1]:].tolist()
            for t, gen in zip(batch, new):
                text = t.conv.add_reply(gen)
                obs, reward, done, info = t.env.step(text)
                t.reward, t.solved, t.log = reward, info["solved"], info["log"]
                if not done:
                    t.conv.add_user(obs, t.env.line)
    return trajs


def token_logprobs(model, seqs: list[list[int]], gens: list[list[bool]], pad_id: int) -> torch.Tensor:
    """log p(token_t | tokens before t) at generated tokens, and 0 elsewhere, for t >= 1.

    Right-padded so positions stay correct. The output head runs only on generated positions,
    which saves gigabytes: the vocabulary has about 150k entries and contexts are long.
    """
    device = next(model.parameters()).device
    ids = pad_right(seqs, pad_id, device)
    mask = pad_right([[1] * len(s) for s in seqs], 0, device)
    wanted = pad_right(gens, False, device, torch.bool)[:, 1:]
    hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]
    logp = torch.log_softmax(model.lm_head(hidden[wanted]).float(), dim=-1)
    out = torch.zeros(wanted.shape, device=device, dtype=logp.dtype)
    out[wanted] = logp.gather(-1, ids[:, 1:][wanted].unsqueeze(-1)).squeeze(-1)
    return out


def parse_ms(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",")]


def train_instance(set_name: str, ms: list[int], rng) -> dict:
    return make_instance(set_name, int(rng.choice(ms)), int(rng.integers(HELDOUT_SEED_START)))


def heldout_instances(set_name: str, ms: list[int], n: int, offset: int = 0) -> list[dict]:
    """The same list for every model, seed and condition, so comparisons are paired."""
    return [make_instance(set_name, ms[i % len(ms)], HELDOUT_SEED_START + offset + i) for i in range(n)]


def load_noise_sizes(path: str | None) -> list[int] | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)["sizes"]


def require_noise_sizes(mode: str, sizes):
    """Without calibration the line is uniform over 0..K, which is not length-matched."""
    if mode == "noise" and sizes is None:
        raise SystemExit("noise mode needs --noise-sizes (create the file with calibrate_noise.py)")
    return sizes


def release_memory():
    """Return cached GPU memory between phases. MPS otherwise holds generation caches and can run out."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
