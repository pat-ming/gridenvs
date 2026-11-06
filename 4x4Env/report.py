"""Turn result files into the four-condition table, the probe table and recoverability scores."""
import argparse
import glob
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from grid_env import MODES

TRAINED_ON = {"theta_A": "none", "theta_C": "true", "theta_D": "noise"}
ORDER = ["theta_0", "theta_A", "theta_C", "theta_D"]


def load_all(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    out = []
    for path in paths:
        with open(path) as f:
            out.append(json.load(f))
    return out


def condition_table(results: list[dict]) -> str:
    """Mean success over seeds, with the seed range. * marks the primary, off-training conditions."""
    lines = ["| model | seeds | " + " | ".join(MODES) + " |", "|---|---|" + "---|" * len(MODES)]
    for label in ORDER:
        runs = [r for r in results if r["label"] == label]
        if not runs:
            continue
        cells = []
        for mode in MODES:
            rates = [np.mean([row["solved"] for row in r["conditions"][mode]]) for r in runs]
            star = "" if TRAINED_ON.get(label) == mode else "*"
            cells.append(f"{np.mean(rates):.3f}{star} ({min(rates):.3f}–{max(rates):.3f})")
        lines.append(f"| {label} | {len(runs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def probe_table(results: list[dict]) -> str:
    """Balanced accuracy per layer per model, averaged over seeds."""
    labels = [l for l in ORDER if any(r["label"] == l for r in results)]
    layers = len(results[0]["per_layer"])
    lines = ["| layer | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for layer in range(layers):
        cells = [f"{np.mean([r['per_layer'][layer]['mean'] for r in results if r['label'] == l]):.3f}" for l in labels]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def leave_one_seed_out(features, labels, seeds) -> dict:
    """Train on the other seeds and test on one, scored per instance. Seed is the nuisance variable."""
    features, labels, seeds = np.asarray(features, dtype=float), np.asarray(labels), np.asarray(seeds)
    if features.ndim == 1:
        features = features[:, None]
    correct, per_seed = [], {}
    for s in np.unique(seeds):
        test = seeds == s
        if len(np.unique(labels[~test])) < 2:
            raise SystemExit("recoverability needs usable episodes from at least two seeds of each model; "
                             "episodes count only if they have a parsed edit (check parse_rate)")
        mu, sd = features[~test].mean(0), features[~test].std(0) + 1e-9
        clf = LogisticRegression(max_iter=1000).fit((features[~test] - mu) / sd, labels[~test])
        hits = clf.predict((features[test] - mu) / sd) == labels[test]
        correct.append(hits)
        per_seed[int(s)] = float(hits.mean())
    correct = np.concatenate(correct)
    return {"accuracy": float(correct.mean()), "instances": len(correct), "per_seed": per_seed}


def recoverability(eval_results: list[dict], probe_results: list[dict] | None) -> dict:
    """Classify A vs C. Each fold holds out one seed number, for both models."""
    out = {}
    feats, labels, seeds, skipped = [], [], [], 0
    for r in eval_results:
        if r["label"] not in ("theta_A", "theta_C"):
            continue
        for row in r["conditions"]["counterfactual"]:
            if "counterfactual" not in row:
                skipped += 1
                continue
            feats.append(row["counterfactual"]["score"])
            labels.append(r["label"])
            seeds.append(r["seed"])
    out["behavior"] = leave_one_seed_out(feats, labels, seeds) | {"episodes_without_usable_edit": skipped}
    if probe_results:
        feats, labels, seeds = [], [], []
        for r in probe_results:
            if r["label"] not in ("theta_A", "theta_C"):
                continue
            for accuracies in r["per_instance"].values():
                feats.append(accuracies)
                labels.append(r["label"])
                seeds.append(r["seed"])
        out["probes"] = leave_one_seed_out(feats, labels, seeds)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["table", "probes", "recover"])
    p.add_argument("--evals", help="glob for evaluate.py outputs")
    p.add_argument("--probes", help="glob for probes.json files")
    a = p.parse_args()
    if a.command == "table":
        print(condition_table(load_all(a.evals)))
    elif a.command == "probes":
        print(probe_table(load_all(a.probes)))
    else:
        probes = load_all(a.probes) if a.probes else None
        print(json.dumps(recoverability(load_all(a.evals), probes), indent=2))


if __name__ == "__main__":
    main()
