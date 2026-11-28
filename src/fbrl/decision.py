"""Probes at the position where the column is chosen, the " " after "EDIT r ".

fbrl.probe reads the last prompt token, but decodable there need not mean used: removing that
subspace barely moves θ_A (fbrl.steer). Here, on line-free m = 1 states, the residual at the column
decision position for every row r is probed per layer for two labels:
    broken   the truly violated column
    chosen   the column the model itself puts most probability on
A model that checks has `broken` decodable at the decision position and `chosen` tracking it. One
that computes the column upstream and never routes it would have `broken` low here.

    python -m fbrl.decision --model theta_C_s0=<path> --out results/interp/decision17.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.ablate import states
from fbrl.grid import ChatFormat
from fbrl.probe import L2_GRID, fit, scores

AT = -3  # reply "EDIT", " ", r, " ", c, " ": the logits for c come from the " " after r


@torch.no_grad()
def collect(model, s: dict, digits, batch: int):
    """Residuals (N, 4 rows, layers + 1, d) at AT, and the model's column probabilities (N, 4, 4)."""
    resid, p_c = [], []
    for b in range(0, len(s["ids"]), batch):
        # the 16 replies are r-major, so every 4th is "EDIT r 0 " for r = 0..3
        ids = s["ids"][b : b + batch, ::4].to(model.device)
        mask = s["mask"][b : b + batch, ::4].to(model.device)
        B = ids.shape[0]
        out = model(
            input_ids=ids.flatten(0, 1),
            attention_mask=mask.flatten(0, 1),
            output_hidden_states=True,
        )
        hs = torch.stack([h[:, AT] for h in out.hidden_states], 1)
        resid.append(hs.view(B, 4, *hs.shape[1:]).half().cpu())
        p_c.append(
            out.logits[:, AT, digits[:4]].float().softmax(-1).view(B, 4, 4).cpu()
        )
    return torch.cat(resid), torch.cat(p_c)


def probe(X: torch.Tensor, label: torch.Tensor, n_fit: int) -> list[dict]:
    """X (N, 4, L, d), label (N, 4) column ids. Split by state; L2 picked on the last 25% of fit."""
    Y = torch.nn.functional.one_hot(label, 4).bool()
    a = int(0.75 * n_fit)
    rows = []
    for layer in range(X.shape[2]):
        Z = X[:, :, layer].float().cuda()
        fl = lambda i, j, Z=Z: (Z[i:j].flatten(0, 1), Y[i:j].flatten(0, 1).cuda())
        (Xa, Ya), (Xv, Yv), (Xt, Yt) = fl(0, a), fl(a, n_fit), fl(n_fit, len(Z))
        l2 = max(
            L2_GRID, key=lambda l2: scores(fit(Xa, Ya, l2)(Xv), Yv)["balanced_acc"]
        )
        rows.append(
            {"layer": layer, "l2": l2, **scores(fit(*fl(0, n_fit), l2)(Xt), Yt)}
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--set", default="distinct")
    ap.add_argument("--n-fit", type=int, default=768)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    res = {"set": args.set, "n_fit": args.n_fit, "n_eval": args.n_eval}
    for spec in args.model:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        fmt = ChatFormat(tok, enable_thinking=False)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model = model.cuda().eval()
        digits = torch.tensor(
            [tok.convert_tokens_to_ids(str(d)) for d in range(10)], device="cuda"
        )
        # the eval states come first so they match fbrl.steer / fbrl.ablate's instances
        s = states(args.set, "none", args.n_eval + args.n_fit, fmt)
        order = torch.cat(
            [torch.arange(args.n_eval, len(s["ids"])), torch.arange(args.n_eval)]
        )
        s = {k: v[order] for k, v in s.items()}
        X, p_c = collect(model, s, digits, args.batch)
        broken = s["on_V"][:, 0].float().argmax(-1)[:, None].expand(-1, 4)
        chosen = p_c.argmax(-1)
        ev = slice(args.n_fit, None)
        res[name] = {
            "model": path,
            "p_broken": p_c[ev].gather(-1, broken[ev, :, None]).mean().item(),
            "chosen_is_broken": (chosen[ev] == broken[ev]).float().mean().item(),
            # (n_eval, row, column); eval state i is instance EVAL_SEED_START + i, m = 1
            "p_c_eval": p_c[ev].tolist(),
            "broken": probe(X, broken, args.n_fit),
            "chosen": probe(X, chosen, args.n_fit),
        }
        best = {
            k: max(res[name][k], key=lambda r: r["balanced_acc"])
            for k in ("broken", "chosen")
        }
        print(
            name,
            f"p_broken {res[name]['p_broken']:.3f}",
            *(
                f"{k} best L{r['layer']} bal {r['balanced_acc']:.3f}"
                for k, r in best.items()
            ),
            "| broken by layer",
            " ".join(f"{r['balanced_acc']:.2f}" for r in res[name]["broken"][::2]),
            flush=True,
        )
        del model, X
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
