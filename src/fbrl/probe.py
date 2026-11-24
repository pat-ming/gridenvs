"""Linear probes for the violated set V(s) from the residuals `fbrl.acts` saved.

One logistic probe per constraint bit and layer. Instances split 60/20/20 into fit / validation /
test: the L2 strength is picked on validation, and only test is reported. Bits are imbalanced (most
constraints hold), so the per-bit score is balanced accuracy, and the probe version of I(θ) is
exact-match accuracy of the decoded set. `--control` refits with V shuffled across instances, which
gives chance for the same pipeline.

    python -m fbrl.probe $FBRL/interp --sets distinct --mode none --control
"""

import argparse
import json
from pathlib import Path

import torch

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fit(X: torch.Tensor, Y: torch.Tensor, l2: float, steps: int = 100):
    """Multi-label logistic regression with balanced class weights. X (n, d), Y (n, K) bool."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    pos = Y.float().mean(0).clamp(1e-3, 1 - 1e-3)
    weight = torch.where(Y, 0.5 / pos, 0.5 / (1 - pos))
    W = torch.zeros(X.shape[1], Y.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Xs @ W + b, Y.float(), weight=weight
        ) + l2 * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda Z: ((Z - mu) / sd) @ W.detach() + b.detach() > 0


def scores(pred: torch.Tensor, Y: torch.Tensor) -> dict:
    tpr = (pred & Y).sum(0) / Y.sum(0).clamp(min=1)
    tnr = (~pred & ~Y).sum(0) / (~Y).sum(0).clamp(min=1)
    return {
        "balanced_acc": ((tpr + tnr) / 2).mean().item(),
        "exact": (pred == Y).all(1).float().mean().item(),
    }


def probe_layer(X: torch.Tensor, Y: torch.Tensor) -> dict:
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    best = max(L2_GRID, key=lambda l2: scores(fit(X[:a], Y[:a], l2)(X[a:b]), Y[a:b])["balanced_acc"])
    return {"l2": best, **scores(fit(X[:b], Y[:b], best)(X[b:]), Y[b:])}


def probe_file(path: Path, control: bool = False, seed: int = 0) -> list[dict]:
    d = torch.load(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = d["resid"].to(device).float(), d["V"].to(device).bool()
    if control:
        Y = Y[torch.randperm(len(Y), generator=torch.Generator().manual_seed(seed)).to(device)]
    return [{"layer": layer, **probe_layer(X[:, layer], Y)} for layer in range(X.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--models", nargs="*", help="default: every model under root")
    ap.add_argument("--sets", nargs="+", default=["distinct"])
    ap.add_argument("--mode", default="none")
    ap.add_argument("--control", action="store_true", help="also fit on shuffled V")
    ap.add_argument("--out", type=Path, help="write all rows here as JSON")
    args = ap.parse_args()

    models = args.models or sorted(p.name for p in args.root.iterdir() if p.is_dir())
    rows = []
    for model in models:
        for set_name in args.sets:
            path = args.root / model / f"{set_name}_{args.mode}.pt"
            for control in (False, True) if args.control else (False,):
                res = probe_file(path, control)
                best = max(res, key=lambda r: r["balanced_acc"])
                tag = "shuffled" if control else "real"
                print(f"{model:14s} {set_name:8s} {tag:8s} best layer {best['layer']:2d}  "
                      f"bal_acc {best['balanced_acc']:.3f}  exact {best['exact']:.3f}", flush=True)
                rows += [{"model": model, "set": set_name, "mode": args.mode, "control": control, **r}
                         for r in res]
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
