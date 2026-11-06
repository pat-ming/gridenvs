"""Representational measurement: per-layer linear probes for V(s_t) on feedback-free prompts.

Every model sees the same scripted prompts, so differences between models cannot come from
each policy visiting different states.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score

from constraints import violations
from grid_env import GridEnv
from instances import VMAX, VMIN
from llm import Conversation, heldout_instances, load_model, pad_right, parse_ms, seed_everything


def decoder_layers(model):
    return model.model.layers


def layer_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def scripted_states(tok, set_name, ms, H, n, seed=0, repair_prob=0.5):
    """Feedback-free contexts ending where the next edit is chosen, with labels V(s_t).

    The scripted player mixes repairs with random edits, so constraints are neither almost always
    violated nor almost always satisfied.
    """
    rng = np.random.default_rng(seed)
    states = []
    for inst in heldout_instances(set_name, ms, n, offset=200_000):
        env = GridEnv("none", H)
        conv = Conversation(tok)
        conv.add_user(env.reset(inst))
        solution = inst["solution"]
        for t in range(H):
            y = np.zeros(len(inst["constraints"]), dtype=np.float32)
            y[violations(env.g, inst["constraints"])] = 1
            states.append({"ids": list(conv.ids), "y": y, "instance": int(inst["seed"]), "t": t})
            wrong = np.argwhere(env.g != solution)
            if len(wrong) and rng.random() < repair_prob:
                r, c = map(int, wrong[rng.integers(len(wrong))])
                v = int(solution[r, c])
            else:
                r, c = map(int, rng.integers(4, size=2))
                v = int(rng.integers(VMIN, VMAX + 1))
            text = f"EDIT {r} {c} {v}"
            conv.add_reply(tok(text, add_special_tokens=False).input_ids)
            obs, _, done, _ = env.step(text)
            if not done:
                conv.add_user(obs)
    return states


@torch.no_grad()
def decision_activations(model, tok, id_lists, batch_size=16) -> np.ndarray:
    """Residual stream at each context's last token: the array is [states, layers, width]."""
    device = next(model.parameters()).device
    captured: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_mod, _inp, out):
            captured[i] = layer_tensor(out).detach()
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(decoder_layers(model))]
    out = []
    try:
        for s in range(0, len(id_lists), batch_size):
            chunk = id_lists[s:s + batch_size]
            ids = pad_right(chunk, tok.pad_token_id, device)
            mask = pad_right([[1] * len(x) for x in chunk], 0, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            last = torch.tensor([len(x) - 1 for x in chunk], device=device)
            rows = torch.arange(len(chunk), device=device)
            out.append(torch.stack([captured[i][rows, last].float().cpu()
                                    for i in range(len(captured))], dim=1).numpy())
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(out)


def balanced(y_true, score, threshold):
    """Balanced accuracy, or NaN when the test split holds only one class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, score > threshold))


def fit_layer(X_train, Y_train, X_test, Y_test, alphas=(10.0, 100.0, 1000.0, 10000.0), seed=0):
    """Ridge probes, one per constraint index, thresholded at the training prevalence.

    Returns the test scores and unit directions in raw activation space, which is the space
    the interventions act in.
    """
    mu, sigma = X_train.mean(0), X_train.std(0) + 1e-6
    A, B = (X_train - mu) / sigma, (X_test - mu) / sigma
    prevalence = Y_train.mean(0)

    # Pick the regularization on a slice of the training data.
    order = np.random.default_rng(seed).permutation(len(A))
    cut = int(0.8 * len(A))
    fit_i, val_i = order[:cut], order[cut:]
    best_alpha, best = alphas[0], -1.0
    for alpha in alphas:
        pred = Ridge(alpha=alpha).fit(A[fit_i], Y_train[fit_i]).predict(A[val_i])
        accs = [balanced(Y_train[val_i, k], pred[:, k], prevalence[k]) for k in range(Y_train.shape[1])]
        mean = np.nanmean(accs) if not np.all(np.isnan(accs)) else 0.0
        if mean > best:
            best_alpha, best = alpha, mean

    probe = Ridge(alpha=best_alpha).fit(A, Y_train)
    scores = probe.predict(B)
    directions = probe.coef_ / sigma  # [K, width]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    return scores, directions, prevalence, best_alpha


def fit_all(acts, Y, instances, seed=0, train_frac=0.8):
    """Splits by instance, so no instance contributes to both training and testing."""
    unique = np.unique(instances)
    shuffled = np.random.default_rng(seed).permutation(unique)
    train_ids = set(shuffled[:int(train_frac * len(unique))].tolist())
    is_train = np.array([i in train_ids for i in instances])
    per_layer, all_dirs, per_instance = [], [], {}
    for layer in range(acts.shape[1]):
        X = acts[:, layer]
        scores, dirs, prevalence, alpha = fit_layer(X[is_train], Y[is_train], X[~is_train], Y[~is_train], seed=seed)
        accs = [balanced(Y[~is_train][:, k], scores[:, k], prevalence[k]) for k in range(Y.shape[1])]
        per_layer.append({"layer": layer, "alpha": alpha, "per_constraint": accs, "mean": float(np.nanmean(accs))})
        all_dirs.append(dirs)
        correct = ((scores > prevalence) == (Y[~is_train] > 0.5)).mean(1)
        for inst, c in zip(instances[~is_train], correct):
            per_instance.setdefault(int(inst), {}).setdefault(layer, []).append(float(c))
    table = {str(i): [float(np.mean(by_layer[l])) for l in range(acts.shape[1])] for i, by_layer in per_instance.items()}
    return per_layer, np.stack(all_dirs), table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, choices=["theta_0", "theta_A", "theta_C", "theta_D"])
    p.add_argument("--seed", type=int, default=0, help="the training seed of this checkpoint")
    p.add_argument("--set", default="spec")
    p.add_argument("--m", default="1,2,3")
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True, help="output directory")
    a = p.parse_args()

    seed_everything(0)
    model, tok = load_model(a.model)
    states = scripted_states(tok, a.set, parse_ms(a.m), a.H, a.n)
    acts = decision_activations(model, tok, [s["ids"] for s in states], a.batch_size)
    Y = np.stack([s["y"] for s in states])
    instances = np.array([s["instance"] for s in states])
    per_layer, directions, per_instance = fit_all(acts, Y, instances)

    os.makedirs(a.out, exist_ok=True)
    best = max(per_layer, key=lambda r: r["mean"])
    np.savez(os.path.join(a.out, "probes.npz"), directions=directions)
    with open(os.path.join(a.out, "probes.json"), "w") as f:
        json.dump({"label": a.label, "seed": a.seed, "model": a.model, "set": a.set, "per_layer": per_layer,
                   "best_layer": best["layer"], "per_instance": per_instance, "violated_rate": Y.mean(0).tolist()}, f)
    print(f"{a.label} seed={a.seed}: best layer {best['layer']}, balanced accuracy {best['mean']:.3f}")


if __name__ == "__main__":
    main()
