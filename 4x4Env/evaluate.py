"""Behavioral measurement: success on held-out instances under all four prompt conditions."""
import argparse
import json
import os

import numpy as np

from grid_env import MODES, counterfactual_score
from llm import heldout_instances, load_model, load_noise_sizes, parse_ms, rollout, seed_everything


def episode_row(traj) -> dict:
    row = {
        "instance": int(traj.inst["seed"]),
        "m": int(traj.inst["m"]),
        "solved": bool(traj.reward),
        "parse_rate": float(np.mean([e["parse_ok"] for e in traj.log])),
    }
    cf = counterfactual_score(traj.log, traj.inst["constraints"])
    if cf is not None:
        row["counterfactual"] = cf
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, choices=["theta_0", "theta_A", "theta_C", "theta_D"])
    p.add_argument("--seed", type=int, default=0, help="the training seed of this checkpoint")
    p.add_argument("--set", default="spec")
    p.add_argument("--m", default="1,2,3")
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--noise-sizes", required=True)
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    seed_everything(a.seed)
    model, tok = load_model(a.model)
    instances = heldout_instances(a.set, parse_ms(a.m), a.n)
    noise_sizes = load_noise_sizes(a.noise_sizes)
    results = {"label": a.label, "seed": a.seed, "model": a.model, "set": a.set, "m": a.m, "H": a.H, "conditions": {}}
    for mode in MODES:
        trajs = rollout(model, tok, instances, mode, a.H, noise_sizes=noise_sizes,
                        max_new_tokens=a.max_new_tokens, gen_batch=a.gen_batch)
        rows = [episode_row(t) for t in trajs]
        results["conditions"][mode] = rows
        print(f"{a.label} seed={a.seed} {mode:15s} success={np.mean([r['solved'] for r in rows]):.3f} "
              f"parse={np.mean([r['parse_rate'] for r in rows]):.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
