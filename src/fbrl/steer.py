"""Does the model use the broken column it computes? Interventions on the column subspace.

At each chosen layer, the column subspace is spanned by the class means mu_k - mu (k = the violated
column) of the residuals `fbrl.acts` cached on line-free m = 1 states, rank 3 for 4 columns. From the
last prompt token on, on fresh line-free m = 1 states:
    ablate   set the subspace coordinates to the overall mean's (removes which column is broken)
    swap     set them to mu_{k+1}'s, the mean of states whose broken column is the next one
    random   ablate a random subspace of the same rank, as a control
Behavior is exact, as in fbrl.ablate: success, on_V (edit in the broken column) and, for every
condition, on_next (edit in column k+1, where swap points).

If a model acts on what the probe reads, ablate sends on_V toward 0.25 and swap moves edits to k+1.
If it computes the column but doesn't use it, neither changes much.

    python -m fbrl.steer --model theta_C_s0=<path> --acts $FBRL/interp17 --layers 16-27 \
        --out results/interp/steer/steer17.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fbrl.ablate import states
from fbrl.grid import ChatFormat

REPLY = 6  # "EDIT", " ", r, " ", c, " "


def subspaces(acts: Path, layers: range, device) -> dict:
    """Per layer: overall mean, per-column means (4, d) and an orthonormal basis Q (d, 3)."""
    d = torch.load(acts)
    one = d["m"] == 1
    out = {}
    for l in layers:
        X = (
            d["resid"][one, l + 1].float().to(device)
        )  # hidden_states[l + 1] is layer l's output
        V = d["V"][one].to(device)
        mu = X.mean(0)
        mus = torch.stack([X[V[:, k]].mean(0) for k in range(V.shape[1])])
        Q = torch.linalg.svd(mus - mu, full_matrices=False).Vh[: V.shape[1] - 1].T
        out[l] = {"mu": mu, "mus": mus, "Q": Q}
    return out


class Intervene:
    """Forward hooks that set the residual's coordinates in Q to a target, from position `start` on."""

    def __init__(self, model):
        self.layers, self.hooks = model.model.layers, []

    def on(self, spaces: dict, target) -> None:
        """target(l) -> (B, d) per-row point whose Q-coordinates to impose, or (d,)."""
        for l, s in spaces.items():

            def put(_, args, out, l=l, Q=s["Q"]):
                h = out[0] if isinstance(out, tuple) else out
                x = h[:, -REPLY - 1 :].float()
                t = target(l)
                t = t[:, None] if t.dim() == 2 else t
                x = x + ((t - x) @ Q) @ Q.T
                h = h.clone()
                h[:, -REPLY - 1 :] = x.to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h

            self.hooks.append(self.layers[l].register_forward_hook(put))

    def off(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []


@torch.no_grad()
def behavior(
    model,
    s: dict,
    col: torch.Tensor,
    digits,
    batch: int,
    iv=None,
    spaces=None,
    kind=None,
):
    out = {"success": [], "on_V": [], "on_next": []}
    nxt = (col + 1) % 4
    for b in range(0, len(s["ids"]), batch):
        ids = s["ids"][b : b + batch].to(model.device)
        mask = s["mask"][b : b + batch].to(model.device)
        B = ids.shape[0]
        if kind:
            k = nxt[b : b + batch].repeat_interleave(16).to(model.device)
            target = {
                "ablate": lambda l: spaces[l]["mu"],
                "random": lambda l: spaces[l]["mu"],
                "swap": lambda l, k=k: spaces[l]["mus"][k],
            }[kind]
            iv.on(spaces, target)
        try:
            logits = model(
                input_ids=ids.flatten(0, 1),
                attention_mask=mask.flatten(0, 1),
                logits_to_keep=5,
            ).logits.float()
        finally:
            iv and iv.off()
        logits = logits.view(B, 4, 4, 5, -1)
        p_r = logits[:, 0, 0, 0, digits[:4]].softmax(-1)
        p_c = logits[:, :, 0, 2, digits[:4]].softmax(-1)
        p_v = logits[:, :, :, 4, digits].softmax(-1)
        joint = p_r[:, :, None, None] * p_c[:, :, :, None] * p_v
        cell = joint.sum(-1)  # (B, r, c)
        colp = cell.sum(1)  # (B, c)
        i = torch.arange(B, device=colp.device)
        out["success"].append(
            (joint * s["solves"][b : b + batch].to(joint.device)).sum((1, 2, 3))
        )
        out["on_V"].append(colp[i, col[b : b + batch].to(colp.device)])
        out["on_next"].append(colp[i, nxt[b : b + batch].to(colp.device)])
    return {k: round(torch.cat(v).mean().item(), 4) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", action="append", required=True, help="name=path, repeatable"
    )
    ap.add_argument("--acts", type=Path, required=True, help="fbrl.acts output root")
    ap.add_argument("--layers", default="16-27")
    ap.add_argument("--set", default="distinct")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lo, hi = map(int, args.layers.split("-"))
    res = {"layers": [lo, hi], "set": args.set, "n": args.n}
    for spec in args.model:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        fmt = ChatFormat(tok, enable_thinking=False)
        model = (
            AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        digits = torch.tensor(
            [tok.convert_tokens_to_ids(str(d)) for d in range(10)], device="cuda"
        )
        s = states(args.set, "none", args.n, fmt)
        col = s["on_V"][:, 0].float().argmax(-1)  # row 0 of the broken column's mask
        spaces = subspaces(
            args.acts / name / f"{args.set}_none.pt", range(lo, hi + 1), "cuda"
        )
        g = torch.Generator(device="cuda").manual_seed(0)
        rand = {
            l: {
                **sp,
                "Q": torch.linalg.qr(
                    torch.randn(sp["Q"].shape, device="cuda", generator=g)
                )[0],
            }
            for l, sp in spaces.items()
        }
        iv = Intervene(model)
        res[name] = {
            "model": path,
            "intact": behavior(model, s, col, digits, args.batch),
        }
        for kind, sp in (("ablate", spaces), ("swap", spaces), ("random", rand)):
            res[name][kind] = behavior(model, s, col, digits, args.batch, iv, sp, kind)
        print(
            name,
            json.dumps({k: v for k, v in res[name].items() if k != "model"}),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
