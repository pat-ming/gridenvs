"""Summarize fbrl.patch output: recovery heatmaps and the heads that carry the line.

Recovery = (patched - false) / (true - false), so 0 is the false-line behavior and 1 is fully
restored true-line behavior. Only meaningful for models whose behavior moves with the line (θ_C).

    python -m fbrl.patch_report $FBRL/interp/patch/patch_theta_C_s0.pt --out results/interp/figs
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def recovery(r: dict, key: str) -> torch.Tensor:
    t, f = r["clean"].mean(), r["corrupt"].mean()
    return (r[key] - f) / (t - f)


def report(path: Path, out: Path, top: int) -> list[tuple[str, int, int, float]]:
    res = torch.load(path, weights_only=False)
    name = path.stem.removeprefix("patch_")
    rows = []
    sets = [s for s in res if isinstance(res[s], dict) and "resid" in res[s]]
    fig, axes = plt.subplots(2, len(sets), figsize=(6 * len(sets), 9), squeeze=False)
    for j, s in enumerate(sets):
        r = res[s]
        rec_resid, rec_heads = recovery(r, "resid"), recovery(r, "heads")
        ax = axes[0, j]
        im = ax.imshow(rec_resid.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_yticks(range(len(r["tokens"])), [repr(t)[1:-1] for t in r["tokens"]], fontsize=7)
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_title(f"{name} {s}: residual patch, recovery")
        fig.colorbar(im, ax=ax)
        ax = axes[1, j]
        im = ax.imshow(rec_heads.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_xlabel("layer")
        ax.set_ylabel("head")
        ax.set_title(f"{name} {s}: one head patched, recovery")
        fig.colorbar(im, ax=ax)
        flat = rec_heads.flatten().argsort(descending=True)[:top]
        H = rec_heads.shape[1]
        rows += [(s, int(i) // H, int(i) % H, rec_heads.flatten()[i].item()) for i in flat]
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"patch_{name}.png", dpi=120)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    for f in args.files:
        print(f.stem)
        for s, l, h, v in report(f, args.out, args.top):
            print(f"  {s:9s} L{l:02d}.H{h:02d}  recovery {v:+.2f}")


if __name__ == "__main__":
    main()
