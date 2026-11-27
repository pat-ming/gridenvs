"""Where a GRPO run's first edits land at each eval, from records/eval_episodes.jsonl.

follow_false: under a false line (mode counterfactual), the share of first edits inside a shown
constraint's scope. About chance for a model that ignores the line, 100% for one that obeys it.
find_alone: with no line, the share inside a truly violated constraint's scope.

    python -m fbrl.placement $FBRL/runs/grpo_kl0.04_*          # last eval
    python -m fbrl.placement --steps $FBRL/runs/grpo_drop300_*  # every eval step
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


def placement(run: Path) -> dict[int, dict[str, float]]:
    """{eval step: {"<set>/<measure>": percent}}."""
    counts = defaultdict(lambda: [0, 0])  # (step, name) -> [hits, total]
    with open(run / "records/eval_episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    for e in episodes:
        first = e["turns"][0]
        if first["action"] is None:
            continue
        cell = edited_cell("grid4", first["action"])
        cs = constraints(e["set_name"], e["m"], e["instance"])
        if e["mode"] == "counterfactual":
            name, targets = "follow_false", first["shown"]
        elif e["mode"] == "none":
            name, targets = "find_alone", first["V"]
        else:
            continue
        c = counts[e["step"], f"{e['set_name']}/{name}"]
        c[0] += any(cell in cs[k].scope() for k in targets)
        c[1] += 1
    out = defaultdict(dict)
    for (step, name), (hits, total) in sorted(counts.items()):
        out[step][name] = round(100 * hits / total, 1)
    return dict(out)


if __name__ == "__main__":
    # --steps prints every eval step, otherwise only the last
    steps = "--steps" in sys.argv
    for run in (Path(a) for a in sys.argv[1:] if a != "--steps"):
        by_step = placement(run)
        for step in sorted(by_step) if steps else [max(by_step)]:
            print(run.name, step, json.dumps(by_step[step]))
