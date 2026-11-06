"""Record |V| from theta_0's E_C rollouts, so E_D's noise line can match E_C's length."""
import argparse
import json

import numpy as np

from llm import BASE_MODEL, heldout_instances, load_model, parse_ms, rollout, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--set", default="spec")
    p.add_argument("--m", default="1,2,3")
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    seed_everything(a.seed)
    model, tok = load_model(a.model)
    # Seeds after the evaluation range, so calibration never sees evaluation instances.
    instances = heldout_instances(a.set, parse_ms(a.m), a.n, offset=100_000)
    trajs = rollout(model, tok, instances, "true", a.H, gen_batch=a.gen_batch)
    sizes = [len(e["shown"]) for t in trajs for e in t.log]
    with open(a.out, "w") as f:
        json.dump({"sizes": sizes, "set": a.set, "m": a.m, "H": a.H, "model": a.model}, f)
    print(f"{len(sizes)} turns, mean |V| = {np.mean(sizes):.2f}, saved to {a.out}")


if __name__ == "__main__":
    main()
