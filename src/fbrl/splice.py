"""Which layers changed when θ_C started using its check? Weight splicing.

`base` is θ_C; `donor` is θ_C continued without the line, which checks on its own. Copy a subset
of the donor's decoder layers (whole layer, attention only, or MLP only) into the base and measure,
at the column decision position (the " " after "EDIT r "), on m = 1 `distinct` states:
    p_broken   probability on the truly broken column, no line
    p_shown    probability on the column a false line names
Splices: every suffix [L, 28), every prefix [0, L), and each layer alone.

    python -m fbrl.splice --base <θ_C final> --donor <continued final> --out splice.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.ablate import states
from fbrl.decision import AT
from fbrl.grid import ChatFormat

PARTS = {"layer": "", "attn": "self_attn.", "mlp": "mlp."}


@torch.no_grad()
def column_probs(model, s: dict, digits, batch: int) -> torch.Tensor:
    """(N, 4 rows, 4 columns) probabilities for the column after "EDIT r "."""
    out = []
    for b in range(0, len(s["ids"]), batch):
        ids = s["ids"][b : b + batch, ::4].to(model.device)
        mask = s["mask"][b : b + batch, ::4].to(model.device)
        B = ids.shape[0]
        logits = model(
            input_ids=ids.flatten(0, 1),
            attention_mask=mask.flatten(0, 1),
            logits_to_keep=-AT,
        ).logits[:, AT, digits[:4]]
        out.append(logits.float().softmax(-1).view(B, 4, 4).cpu())
    return torch.cat(out)


def mass(p_c: torch.Tensor, on: torch.Tensor) -> float:
    """Mean probability on the column marked in row 0 of the (N, 4, 4) cell mask `on`."""
    col = on[:, 0].float().argmax(-1)
    return p_c.gather(-1, col[:, None, None].expand(-1, 4, 1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--set", default="distinct")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    fmt = ChatFormat(tok, enable_thinking=False)
    load = lambda p: AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16)
    model = load(args.base).cuda().eval()
    base = {k: v.clone() for k, v in model.state_dict().items()}
    donor = load(args.donor).state_dict()
    n_layers = model.config.num_hidden_layers
    digits = torch.tensor(
        [tok.convert_tokens_to_ids(str(d)) for d in range(10)], device="cuda"
    )
    none = states(args.set, "none", args.n, fmt)
    false = states(args.set, "counterfactual", args.n, fmt)

    def measure(layers: list[int], part: str) -> dict:
        prefixes = [f"model.layers.{l}.{PARTS[part]}" for l in layers]
        sd = {
            k: (donor[k] if any(k.startswith(p) for p in prefixes) else v)
            for k, v in base.items()
        }
        model.load_state_dict(sd)
        return {
            "layers": layers,
            "part": part,
            "p_broken": mass(
                column_probs(model, none, digits, args.batch), none["on_V"]
            ),
            "p_shown": mass(
                column_probs(model, false, digits, args.batch), false["on_shown"]
            ),
        }

    L = range(n_layers + 1)
    plan = [([], "layer"), (list(range(n_layers)), "layer")]
    plan += [(list(range(n_layers)), p) for p in ("attn", "mlp")]
    plan += [(list(range(l, n_layers)), "layer") for l in L[1:-1]]
    plan += [(list(range(l)), "layer") for l in L[1:-1]]
    plan += [([l], p) for l in range(n_layers) for p in PARTS]
    rows = []
    for layers, part in plan:
        rows.append(measure(layers, part))
        r = rows[-1]
        span = f"{layers[0]}-{layers[-1]}" if layers else "none"
        print(
            f"{part:5s} {span:7s} p_broken {r['p_broken']:.3f} p_shown {r['p_shown']:.3f}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"base": args.base, "donor": args.donor, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
