"""Where does a model read the line? Activation patching on true/false line pairs, plus D(θ).

A pair is one held-out state with one violated constraint, shown once with the true line and once
with a false one (mode counterfactual). The two prompts differ only in the line's index token.

The first edit's cell distribution is P(r, c) = P(r) P(c | r), read from four teacher-forced replies
"EDIT r ". The metric is

    log P(edit in cells only the true constraint covers) - log P(edit in cells only the false one covers)

High under the true line, low under the false one, for a model that follows the line. Patching runs
the false-line prompt with one activation taken from the true-line run, from the line token on.

    python -m fbrl.patch --model theta_C_s0=<path> --sets distinct ineq --n 128 --out results/interp

Writes <out>/patch_<model>.pt with, per set:
    tv               (n,) total variation between P(r, c) under the true and the false line: D(θ)
    clean, corrupt   (n,) metric under the true and the false line
    resid            (layers + 1, P) mean metric after patching the residual at one layer and position
                     (row 0 is the embedding output; positions run from the line token to the end)
    heads            (layers, heads) mean metric after patching one head's output at all P positions
    tokens           the P position labels
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.grid import EVAL_SEED_START, OBS, ChatFormat, load_env
from gridenvs.grid4.constraints import violations

ROWS = range(4)


def pairs(set_name: str, n: int, fmt: ChatFormat) -> dict:
    """The first n held-out states with exactly one violated constraint, as true/false prompt pairs
    with the four "EDIT r " replies appended. Also returns the cell masks for the metric."""
    env_cls, make_instance = load_env("grid4")
    template = fmt.tok.apply_chat_template(
        [{"role": "user", "content": OBS}], tokenize=False, add_generation_prompt=True, **fmt.kwargs
    )
    replies = [fmt.tok.encode(f"EDIT {r} ", add_special_tokens=False) for r in ROWS]
    assert all(len(x) == 4 for x in replies), replies
    clean, corrupt, only_true, only_false = [], [], [], []
    i = 0
    while len(clean) < n:
        m = 1 + i % 2
        inst = make_instance(set_name, m, EVAL_SEED_START + i)
        ids, shown = {}, {}
        for mode in ("true", "counterfactual"):
            env = env_cls(mode, H=m, seed=i)
            ids[mode] = fmt.tok.encode(template.replace(OBS, env.reset(inst)), add_special_tokens=False)
            shown[mode] = env.shown
        i += 1
        V = violations(env.g, env.cs)
        a, b = ids["true"], ids["counterfactual"]
        if len(V) != 1 or len(a) != len(b) or sum(x != y for x, y in zip(a, b)) != 1:
            continue
        cs = inst["constraints"]
        t, f = cs[V[0]].scope(), cs[shown["counterfactual"][0]].scope()
        clean.append([a + r for r in replies])
        corrupt.append([b + r for r in replies])
        only_true.append([[(r, c) in t - f for c in range(4)] for r in ROWS])
        only_false.append([[(r, c) in f - t for c in range(4)] for r in ROWS])
    a, b = np.array(clean[0][0]), np.array(corrupt[0][0])
    first = int(np.flatnonzero(a != b)[0])
    P = len(a) - first  # line token .. end of reply prefix
    return {
        "clean": clean,
        "corrupt": corrupt,
        "only_true": torch.tensor(only_true),
        "only_false": torch.tensor(only_false),
        "P": P,
        "tokens": [fmt.tok.decode([x]) for x in a[first:]],
        "scanned": i,
    }


class Patcher:
    def __init__(self, model, digit_ids: list[int]):
        self.model = model
        self.digits = torch.tensor(digit_ids, device=model.device)
        self.layers = model.model.layers
        self.hooks = []

    def joint(self, ids: torch.Tensor) -> torch.Tensor:
        """ids (B, 4, S), the four row replies per prompt -> P(r, c) (B, 4, 4)."""
        B = ids.shape[0]
        logits = self.model(input_ids=ids.flatten(0, 1), logits_to_keep=3).logits.float()
        logits = logits[:, :, self.digits].view(B, 4, 3, 4)
        p_row = logits[:, 0, 0].softmax(-1)  # predicting r, after "EDIT "
        p_col = logits[:, :, 2].softmax(-1)  # predicting c, after "EDIT r "
        return p_row[:, :, None] * p_col

    def cache(self, ids: torch.Tensor, P: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean residuals (layers + 1, B*4, P, d) and head inputs to o_proj (layers, B*4, P, H*hd)."""
        heads = []
        hooks = [
            layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _, args: heads.append(args[0][:, -P:].clone())
            )
            for layer in self.layers
        ]
        try:
            out = self.model(input_ids=ids.flatten(0, 1), output_hidden_states=True, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        resid = torch.stack([h[:, -P:] for h in out.hidden_states])
        return resid, torch.stack(heads)

    def run_with(self, ids: torch.Tensor, module, fn, pre: bool = False) -> torch.Tensor:
        hook = (module.register_forward_pre_hook if pre else module.register_forward_hook)(fn)
        try:
            return self.joint(ids)
        finally:
            hook.remove()


def metric(joint: torch.Tensor, only_true: torch.Tensor, only_false: torch.Tensor) -> torch.Tensor:
    t = (joint * only_true).sum((1, 2)).clamp_min(1e-9)
    f = (joint * only_false).sum((1, 2)).clamp_min(1e-9)
    return t.log() - f.log()


@torch.no_grad()
def analyze(model, tok, set_name: str, n: int, batch: int, patch: bool) -> dict:
    fmt = ChatFormat(tok, enable_thinking=False)
    p = pairs(set_name, n, fmt)
    P, dev = p["P"], model.device
    pt = Patcher(model, [tok.convert_tokens_to_ids(str(d)) for d in range(4)])
    L, H = len(pt.layers), model.config.num_attention_heads
    hd = model.config.head_dim
    out = {k: [] for k in ("tv", "clean", "corrupt")}
    resid_sum = torch.zeros(L + 1, P)
    head_sum = torch.zeros(L, H)
    for b in range(0, n, batch):
        clean = torch.tensor(p["clean"][b : b + batch], device=dev)
        corrupt = torch.tensor(p["corrupt"][b : b + batch], device=dev)
        mt, mf = p["only_true"][b : b + batch].to(dev), p["only_false"][b : b + batch].to(dev)
        jc, jx = pt.joint(clean), pt.joint(corrupt)
        out["tv"].append(0.5 * (jc - jx).abs().sum((1, 2)).cpu())
        out["clean"].append(metric(jc, mt, mf).cpu())
        out["corrupt"].append(metric(jx, mt, mf).cpu())
        if not patch:
            continue
        resid, heads = pt.cache(clean, P)
        modules = [model.model.embed_tokens, *pt.layers]
        for l, module in enumerate(modules):
            for pos in range(P):
                s = -P + pos

                def put(_, __, output, l=l, pos=pos, s=s, resid=resid):
                    h = output[0] if isinstance(output, tuple) else output
                    h[:, s] = resid[l][:, pos]
                    return output

                j = pt.run_with(corrupt, module, put)
                resid_sum[l, pos] += metric(j, mt, mf).sum().cpu()
        for l, layer in enumerate(pt.layers):
            for h in range(H):
                sl = slice(h * hd, (h + 1) * hd)

                def put_head(_, args, l=l, sl=sl, heads=heads):
                    x = args[0].clone()
                    x[:, -P:, sl] = heads[l][:, :, sl]
                    return (x, *args[1:])

                j = pt.run_with(corrupt, layer.self_attn.o_proj, put_head, pre=True)
                head_sum[l, h] += metric(j, mt, mf).sum().cpu()
    res = {k: torch.cat(v) for k, v in out.items()}
    res |= {"tokens": p["tokens"], "scanned": p["scanned"]}
    if patch:
        res |= {"resid": resid_sum / n, "heads": head_sum / n}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--sets", nargs="+", default=["distinct", "ineq"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-patch", action="store_true", help="only D(θ) and the clean/corrupt metric")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # bf16 matmuls crawl on CPU
    for spec in args.model:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        res = {"model": path}
        for set_name in args.sets:
            r = analyze(model, tok, set_name, args.n, args.batch, not args.no_patch)
            res[set_name] = r
            print(
                f"{name} {set_name}: D={r['tv'].mean():.3f}"
                f" metric true {r['clean'].mean():+.2f} false {r['corrupt'].mean():+.2f}",
                flush=True,
            )
        torch.save(res, args.out / f"patch_{name}.pt")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
