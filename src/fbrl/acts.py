"""Residual-stream activations at the last prompt token, the position where the model starts its
first reply. Every mode sees the same instances, so line-free and lined prompts pair up row by row.

    python -m fbrl.acts --model theta_C_s0=$FBRL/runs/grpo_kl0.04_true_s0_*/checkpoints/final \\
        --sets distinct ineq --n 2000 --out $FBRL/interp

Saves <out>/<model>/<set>_<mode>.pt with
    resid  (n, layers + 1, d) float16, hidden_states[0] is the embedding output
    V      (n, K) bool, the truly violated constraints
    shown  (n, K) bool, the constraints the line names (all False for mode none)
    m      (n,) int, instance  (n,) int
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.grid import EVAL_SEED_START, OBS, ChatFormat, load_env
from gridenvs.grid4.constraints import violations

MODES = ("none", "true", "counterfactual")


def prompts(set_name: str, n: int, mode: str, fmt: ChatFormat) -> dict:
    """n held-out instances, half m=1 and half m=2, with H = m as in eval."""
    env_cls, make_instance = load_env("grid4")
    # fmt.first re-renders the chat template every call (~0.1 s); render it once and fill in
    template = fmt.tok.apply_chat_template(
        [{"role": "user", "content": OBS}], tokenize=False, add_generation_prompt=True, **fmt.kwargs
    )
    ids, V, shown, ms = [], [], [], []
    for i in range(n):
        m = 1 + i % 2
        inst = make_instance(set_name, m, EVAL_SEED_START + i)
        env = env_cls(mode, H=m, seed=i)
        ids.append(fmt.tok.encode(template.replace(OBS, env.reset(inst)), add_special_tokens=False))
        K = len(inst["constraints"])
        V.append(np.isin(np.arange(K), violations(env.g, env.cs)))
        shown.append(np.isin(np.arange(K), list(env.shown or [])))
        ms.append(m)
    return {"ids": ids, "V": np.array(V), "shown": np.array(shown), "m": np.array(ms)}


@torch.no_grad()
def residuals(model, ids: list[list[int]], pad_id: int, batch: int = 64) -> torch.Tensor:
    out = []
    for b in range(0, len(ids), batch):
        chunk = ids[b : b + batch]
        width = max(map(len, chunk))
        # left-pad so the last prompt token sits at position -1 for every row
        x = torch.tensor([[pad_id] * (width - len(s)) + s for s in chunk], device=model.device)
        lens = torch.tensor([len(s) for s in chunk], device=model.device)
        mask = (torch.arange(width, device=model.device)[None] >= width - lens[:, None]).long()
        hs = model(input_ids=x, attention_mask=mask, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1] for h in hs], 1).half().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--sets", nargs="+", default=["distinct", "ineq"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for spec in args.model:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        fmt = ChatFormat(tok, enable_thinking=False)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        for set_name in args.sets:
            for mode in args.modes:
                p = prompts(set_name, args.n, mode, fmt)
                resid = residuals(model, p.pop("ids"), tok.pad_token_id or tok.eos_token_id)
                dest = args.out / name / f"{set_name}_{mode}.pt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"resid": resid, **{k: torch.from_numpy(v) for k, v in p.items()},
                            "instance": torch.arange(args.n), "model": path}, dest)
                print(f"{name} {set_name} {mode}: {tuple(resid.shape)} -> {dest}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
