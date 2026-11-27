"""Δ_M: what removing a set of attention heads does to behavior, under each line mode.

Exact first-edit probabilities, no sampling. For m = 1 states (one edit to fix), P(r, c, v) comes from
16 teacher-forced replies "EDIT r c " per prompt, and each (r, c, v) is applied to the grid to see
whether it solves it. Per mode this gives:
    success   P(the edit solves the grid)
    on_V      P(the edited cell is in a truly violated constraint)
    on_shown  P(the edited cell is in a constraint the line names), for lined modes

Heads are mean-ablated: their o_proj input is replaced, at every position, by its mean over prompts
and positions on line-free and lined prompts.

    python -m fbrl.ablate --model <path> --group movers=17.1,19.4 --group encoders=0.3,13.6 \
        --random 5 --out results/interp/ablate_theta_C_s0.json

--random k adds k random head sets the size of the largest group, as a control.

--knockout name=16-20 blocks attention from every later position to the line's index token at layers
16-20 inclusive; name=16-20+1 blocks the token one after it instead (the "}"), as a control. A
no-op on line-free prompts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.grid import EVAL_SEED_START, OBS, ChatFormat, load_env
from gridenvs.grid4.constraints import violations

MODES = ("none", "true", "counterfactual")


def states(set_name: str, mode: str, n: int, fmt: ChatFormat) -> dict:
    """n held-out m = 1 states (the same instances in every mode) with their 16 "EDIT r c " replies."""
    env_cls, make_instance = load_env("grid4")
    template = fmt.tok.apply_chat_template(
        [{"role": "user", "content": OBS}], tokenize=False, add_generation_prompt=True, **fmt.kwargs
    )
    replies = [fmt.tok.encode(f"EDIT {r} {c} ", add_special_tokens=False) for r in range(4) for c in range(4)]
    # tokens after the line's "}": the end of the user turn and the assistant header
    tail = len(fmt.tok.encode(template[template.index(OBS) + len(OBS) :], add_special_tokens=False))
    ids, solves, on_V, on_shown, line_pos = [], [], [], [], []
    for i in range(n):
        inst = make_instance(set_name, 1, EVAL_SEED_START + i)
        env = env_cls(mode, H=1, seed=i)
        prompt = fmt.tok.encode(template.replace(OBS, env.reset(inst)), add_special_tokens=False)
        ids.append([prompt + r for r in replies])
        g, cs = env.g, env.cs
        V, shown = violations(g, cs), env.shown or []
        pos = len(prompt) - tail - 2 if len(shown) == 1 else -1  # "{k}": k, then "}"
        assert pos < 0 or fmt.tok.decode(prompt[pos : pos + 2]) == f"{shown[0]}}}"
        line_pos.append(pos)
        s = np.zeros((4, 4, 10), bool)
        for r in range(4):
            for c in range(4):
                for v in range(10):
                    h = g.copy()
                    h[r, c] = v
                    s[r, c, v] = violations(h, cs) == []
        solves.append(s)
        on_V.append([[any((r, c) in cs[k].scope() for k in V) for c in range(4)] for r in range(4)])
        on_shown.append([[any((r, c) in cs[k].scope() for k in shown) for c in range(4)] for r in range(4)])
    # ineq prompts differ in length; left-pad so the reply tokens line up at the end
    width = max(len(x[0]) for x in ids)
    pad = fmt.tok.pad_token_id or fmt.tok.eos_token_id
    shift = [width - len(row[0]) for row in ids]
    return {
        "line_pos": torch.tensor([p + d if p >= 0 else -1 for p, d in zip(line_pos, shift)]),
        "ids": torch.tensor([[[pad] * (width - len(x)) + x for x in row] for row in ids]),
        "mask": torch.tensor([[[0] * (width - len(x)) + [1] * len(x) for x in row] for row in ids]),
        "solves": torch.tensor(np.array(solves)),
        "on_V": torch.tensor(on_V),
        "on_shown": torch.tensor(on_shown),
    }


class HeadAblation:
    """Replace chosen heads' o_proj input with a fixed mean vector while active."""

    def __init__(self, model):
        self.model, self.layers = model, model.model.layers
        self.hd = model.config.head_dim
        self.H = model.config.num_attention_heads
        self.means = None
        self.hooks = []

    @torch.no_grad()
    def fit_means(self, groups: list[dict], batch: int) -> None:
        """Mean o_proj input per layer over the real (unpadded) tokens of every group's prompts."""
        sums = torch.zeros(len(self.layers), self.H * self.hd, device=self.model.device)
        count = 0
        mask = None

        def add(_, args, l):
            sums[l] += (args[0].float() * mask[..., None]).sum((0, 1))

        hooks = [
            layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: add(m, a, l))
            for l, layer in enumerate(self.layers)
        ]
        try:
            for g in groups:
                for b in range(0, len(g["ids"]), batch):
                    x = g["ids"][b : b + batch, 0].to(self.model.device)  # one reply per prompt
                    mask = g["mask"][b : b + batch, 0].to(self.model.device)
                    self.model(input_ids=x, attention_mask=mask, logits_to_keep=1)
                    count += mask.sum().item()
        finally:
            for h in hooks:
                h.remove()
        self.means = sums / count

    def on(self, heads: list[tuple[int, int]]) -> None:
        by_layer = {}
        for l, h in heads:
            by_layer.setdefault(l, []).append(h)
        for l, hs in by_layer.items():
            idx = torch.cat([torch.arange(h * self.hd, (h + 1) * self.hd) for h in hs]).to(self.model.device)
            mean = self.means[l, idx]

            def put(_, args, idx=idx, mean=mean):
                x = args[0].clone()
                x[..., idx] = mean.to(x.dtype)
                return (x, *args[1:])

            self.hooks.append(self.layers[l].self_attn.o_proj.register_forward_pre_hook(put))

    def off(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []


class Knockout:
    """Block attention from every later position to one key position, at chosen layers."""

    def __init__(self, model):
        self.layers = model.model.layers
        self.hooks, self.mask = [], None

    def on(self, lo: int, hi: int) -> None:
        def put(_, args, kwargs):
            kwargs["attention_mask"] = self.mask
            return args, kwargs

        self.hooks = [
            self.layers[l].self_attn.register_forward_pre_hook(put, with_kwargs=True)
            for l in range(lo, hi + 1)
        ]

    def set(self, pad: torch.Tensor, key: torch.Tensor) -> None:
        """pad (N, S) 1 for real tokens; key (N,) the blocked position, -1 for none."""
        S = pad.shape[1]
        q = torch.arange(S, device=pad.device)
        allowed = (q[:, None] >= q[None, :])[None] & pad.bool()[:, None, :]
        k = key[:, None, None]
        allowed &= ~((q[None, :, None] > k) & (q[None, None, :] == k) & (k >= 0))
        allowed |= torch.eye(S, dtype=torch.bool, device=pad.device)  # pad rows see themselves
        self.mask = allowed[:, None]

    def off(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks, self.mask = [], None


@torch.no_grad()
def behavior(model, s: dict, digits: torch.Tensor, batch: int, knock=None, offset=0) -> dict:
    out = {"success": [], "on_V": [], "on_shown": []}
    for b in range(0, len(s["ids"]), batch):
        ids = s["ids"][b : b + batch].to(model.device)
        mask = s["mask"][b : b + batch].to(model.device)
        B = ids.shape[0]
        if knock:
            key = s["line_pos"][b : b + batch].to(model.device)
            key = torch.where(key >= 0, key + offset, key).repeat_interleave(16)
            knock.set(mask.flatten(0, 1), key)
        logits = model(
            input_ids=ids.flatten(0, 1), attention_mask=mask.flatten(0, 1), logits_to_keep=5
        ).logits.float()
        logits = logits.view(B, 4, 4, 5, -1)
        p_r = logits[:, 0, 0, 0, digits[:4]].softmax(-1)  # after "EDIT "
        p_c = logits[:, :, 0, 2, digits[:4]].softmax(-1)  # after "EDIT r "
        p_v = logits[:, :, :, 4, digits].softmax(-1)  # after "EDIT r c "
        joint = p_r[:, :, None, None] * p_c[:, :, :, None] * p_v  # (B, r, c, v)
        cell = joint.sum(-1)
        out["success"].append((joint * s["solves"][b : b + batch].to(joint.device)).sum((1, 2, 3)))
        out["on_V"].append((cell * s["on_V"][b : b + batch].to(cell.device)).sum((1, 2)))
        out["on_shown"].append((cell * s["on_shown"][b : b + batch].to(cell.device)).sum((1, 2)))
    return {k: torch.cat(v).mean().item() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group", action="append", default=[], help="name=layer.head,..., repeatable")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--knockout", action="append", default=[], help="name=lo-hi[+offset]")
    ap.add_argument("--sets", nargs="+", default=["distinct", "ineq"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # bf16 matmuls crawl on CPU
    tok = AutoTokenizer.from_pretrained(args.model)
    fmt = ChatFormat(tok, enable_thinking=False)
    # sdpa takes the boolean masks Knockout builds
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    digits = torch.tensor([tok.convert_tokens_to_ids(str(d)) for d in range(10)], device=device)
    abl = HeadAblation(model)
    rng = np.random.default_rng(args.seed)
    all_heads = [(l, h) for l in range(len(abl.layers)) for h in range(abl.H)]
    conditions = {"intact": []}
    for g in args.group:
        name, heads = g.split("=")
        conditions[name] = [tuple(map(int, h.split("."))) for h in heads.split(",")]
    size = max(map(len, conditions.values()))
    for i in range(args.random):
        pick = rng.choice(len(all_heads), size, replace=False)
        conditions[f"random{i}"] = [all_heads[j] for j in pick]

    knockouts = {}
    for k in args.knockout:
        name, spec = k.split("=")
        span, _, offset = spec.partition("+")
        lo, hi = map(int, span.split("-"))
        knockouts[name] = (lo, hi, int(offset or 0))
    knock = Knockout(model)

    res = {
        "model": args.model,
        "conditions": {k: [list(h) for h in v] for k, v in conditions.items()},
        "knockouts": knockouts,
    }
    for set_name in args.sets:
        data = {mode: states(set_name, mode, args.n, fmt) for mode in MODES}
        abl.fit_means(list(data.values()), args.batch * 4)
        for cond, hs in conditions.items():
            abl.on(hs)
            try:
                for mode, s in data.items():
                    r = behavior(model, s, digits, args.batch)
                    res.setdefault(set_name, {}).setdefault(cond, {})[mode] = r
                    print(set_name, cond, mode, {k: round(v, 3) for k, v in r.items()}, flush=True)
            finally:
                abl.off()
        for name, (lo, hi, offset) in knockouts.items():
            knock.on(lo, hi)
            try:
                for mode, s in data.items():
                    r = behavior(model, s, digits, args.batch, knock, offset)
                    res[set_name][name] = res[set_name].get(name, {}) | {mode: r}
                    print(set_name, name, mode, {k: round(v, 3) for k, v in r.items()}, flush=True)
            finally:
                knock.off()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
