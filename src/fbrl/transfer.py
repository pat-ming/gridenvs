"""Is the broken column written the same way before and after θ_C learns to use it?

Fit the decision-position probe (fbrl.decision) for the broken column on one model and score it,
unchanged, on another at the same layer. `base` is θ_C, `donor` its no-line continuation. If the
continuation only rewired how late layers read the column, a probe fit on either transfers; if it
changed how the column is written, the donor's probe fails on the base.

    python -m fbrl.transfer --base <θ_C final> --donor <continued final> --out transfer.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.ablate import states
from fbrl.decision import collect
from fbrl.grid import ChatFormat
from fbrl.probe import L2_GRID, fit, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--set", default="distinct")
    ap.add_argument("--n-fit", type=int, default=768)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    fmt = ChatFormat(tok, enable_thinking=False)
    digits = torch.tensor(
        [tok.convert_tokens_to_ids(str(d)) for d in range(10)], device="cuda"
    )
    s = states(args.set, "none", args.n_fit + args.n_eval, fmt)
    label = s["on_V"][:, 0].float().argmax(-1)[:, None].expand(-1, 4)
    Y = torch.nn.functional.one_hot(label, 4).bool().flatten(0, 1).cuda()
    X = {}
    for name, path in (("base", args.base), ("donor", args.donor)):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        X[name] = collect(model.cuda().eval(), s, digits, args.batch)[0]
        del model
        torch.cuda.empty_cache()

    n = args.n_fit * 4
    a = int(0.75 * n)
    rows = []
    for layer in range(X["base"].shape[2]):
        Z = {k: v[:, :, layer].float().flatten(0, 1).cuda() for k, v in X.items()}
        row = {"layer": layer}
        for src in Z:
            l2 = max(
                L2_GRID,
                key=lambda l2, src=src: scores(
                    fit(Z[src][:a], Y[:a], l2)(Z[src][a:n]), Y[a:n]
                )["balanced_acc"],
            )
            w = fit(Z[src][:n], Y[:n], l2)
            for dst in Z:
                row[f"{src}->{dst}"] = scores(w(Z[dst][n:]), Y[n:])["balanced_acc"]
        rows.append(row)
        print(
            " ".join(
                f"{k} {v:.3f}" if k != "layer" else f"L{v}" for k, v in row.items()
            ),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"base": args.base, "donor": args.donor, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
