"""Where a GRPO run's first edits land at its last eval, from records/eval_episodes.jsonl.

follow_false: under a false line (mode counterfactual), the share of first edits inside a shown
constraint's scope. About chance for a model that ignores the line, 100% for one that obeys it.
find_alone: with no line, the share inside a truly violated constraint's scope.

    python -m fbrl.placement $FBRL/runs/grpo_kl0.04_*
"""

import json
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

from fbrl.grid import EVAL_SEED_START, edited_cell, load_env

_, make_instance = load_env("grid4")


@cache
def constraints(set_name: str, m: int, instance: int) -> list:
    return make_instance(set_name, m, EVAL_SEED_START + instance)["constraints"]


def placement(run: Path) -> dict[str, float]:
    episodes = [json.loads(line) for line in open(run / "records/eval_episodes.jsonl")]
    last = max(e["step"] for e in episodes)
    counts = defaultdict(lambda: [0, 0])  # name -> [hits, total]
    for e in episodes:
        first = e["turns"][0]
        if e["step"] != last or first["action"] is None:
            continue
        cell = edited_cell("grid4", first["action"])
        cs = constraints(e["set_name"], e["m"], e["instance"])
        if e["mode"] == "counterfactual":
            name, targets = "follow_false", first["shown"]
        elif e["mode"] == "none":
            name, targets = "find_alone", first["V"]
        else:
            continue
        c = counts[f"{e['set_name']}/{name}"]
        c[0] += any(cell in cs[k].scope() for k in targets)
        c[1] += 1
    return {k: 100 * hits / total for k, (hits, total) in sorted(counts.items())}


if __name__ == "__main__":
    for run in map(Path, sys.argv[1:]):
        print(run.name, json.dumps({k: round(v, 1) for k, v in placement(run).items()}))
