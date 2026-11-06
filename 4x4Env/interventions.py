"""Causal measurement, with its controls.

  project  remove the probe directions from the residual stream, then measure success
  patch    put the true-feedback run's probe-direction component into a counterfactual run
  heads    block attention from the violated line to the edit position

Run each experiment on theta_0 as well: the same intervention on the untrained model is a
required control.
"""
import argparse
import json
import os
from contextlib import contextmanager

import numpy as np
import torch
from transformers.models.qwen3 import modeling_qwen3

from grid_env import GridEnv, counterfactual_score
from llm import (Conversation, heldout_instances, load_model, load_noise_sizes, pad_left, parse_ms,
                 rollout, seed_everything)
from probes import decoder_layers, layer_tensor

UNRELATED_PROMPTS = [
    "What is the capital of France?", "Name three primary colors.", "Translate 'good morning' into Spanish.",
    "What do bees make?", "Write a one-line poem about rain.", "Which planet is closest to the sun?",
    "What sound does a cat make?", "Give a synonym for happy.",
]


def orthonormal(directions, device) -> torch.Tensor:
    """Orthonormal basis [r, width] of the given vectors."""
    q, _ = np.linalg.qr(np.asarray(directions, dtype=np.float64).T)
    return torch.tensor(q.T, device=device, dtype=torch.float32)


def random_like(r: int, width: int, rng, device) -> torch.Tensor:
    return orthonormal(rng.standard_normal((r, width)), device)


def replace_output(out, new):
    return (new, *out[1:]) if isinstance(out, tuple) else new


@contextmanager
def project_out(model, layer: int, basis: torch.Tensor):
    """Remove the span of `basis` from this layer's output at every position."""
    def hook(_mod, _inp, out):
        h = layer_tensor(out)
        b = basis.to(h.dtype)
        return replace_output(out, h - (h @ b.T) @ b)
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patch_last_position(model, layer: int, basis: torch.Tensor, source_holder: dict):
    """On a prompt's first pass, set the span-of-basis component of the last token to the source's."""
    def hook(_mod, _inp, out):
        h = layer_tensor(out)
        if h.shape[1] == 1 or source_holder.get("source") is None:  # later tokens use the cache
            return out
        b = basis.to(h.dtype)
        new = h.clone()
        delta = source_holder["source"].to(h.dtype) - h[:, -1]
        new[:, -1] = h[:, -1] + (delta @ b.T) @ b
        return replace_output(out, new)
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def last_token_activation(model, layer: int, ids, mask) -> torch.Tensor:
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = layer_tensor(out)[:, -1].detach()

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    return captured["h"]


class LineAttention:
    """Measures or blocks attention from feedback-line tokens. The model must load with eager=True.

    query="decision" blocks attention from the edit position and generated tokens only, as in H2.
    The line can still reach them indirectly through tokens between it and the decision.
    query="all_after" blocks every later token, so all routes through attention are closed.
    heads maps layer -> head indices; None means every head in every layer.
    """

    def __init__(self, heads=None, query="decision", block=True):
        self.heads, self.query, self.block = heads, query, block
        self.source = None
        self.decision = 0
        self.mass = {}  # (layer, head) -> summed attention from the decision position to the line
        self.count = 0

    def set_batch(self, convs, ids, _mask):
        """Align each conversation's line mask with its left-padded row."""
        t = ids.shape[1]
        rows = [[False] * (t - len(c.line)) + c.line for c in convs]
        self.source = torch.tensor(rows, device=ids.device)
        self.decision = t - 1
        self.count += len(convs)

    def forward(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **_kw):
        key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask

        q_len, k_len = weights.shape[2], weights.shape[3]
        q_pos = torch.arange(k_len - q_len, k_len, device=weights.device)
        k_pos = torch.arange(k_len, device=weights.device)
        source = torch.zeros(weights.shape[0], k_len, dtype=torch.bool, device=weights.device)
        known = min(self.source.shape[1], k_len)
        source[:, :known] = self.source[:, :known]
        chosen = q_pos[:, None] > k_pos[None, :]
        if self.query == "decision":
            chosen = chosen & (q_pos[:, None] >= self.decision)
        heads = self.heads
        selected = None if heads is None else heads.get(module.layer_idx, [])
        if self.block and (selected is None or len(selected)):
            head_mask = torch.ones(weights.shape[1], dtype=torch.bool, device=weights.device)
            if selected is not None:
                head_mask[:] = False
                head_mask[selected] = True
            blocked = source[:, None, None, :] & chosen[None, None] & head_mask[None, :, None, None]
            weights = weights.masked_fill(blocked, float("-inf"))

        probs = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if not self.block and q_len == k_len:  # the prompt's first pass, where the decision is last
            on_line = (probs[:, :, -1, :] * source[:, None, :]).sum(-1).sum(0)
            for h, mass in enumerate(on_line.tolist()):
                self.mass[(module.layer_idx, h)] = self.mass.get((module.layer_idx, h), 0.0) + mass
        probs = torch.nn.functional.dropout(probs, p=dropout, training=module.training)
        output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
        return output, probs

    @contextmanager
    def active(self):
        original = modeling_qwen3.eager_attention_forward
        modeling_qwen3.eager_attention_forward = self.forward
        try:
            yield self
        finally:
            modeling_qwen3.eager_attention_forward = original


def success(trajs) -> float:
    return float(np.mean([t.reward for t in trajs]))


def direction_summary(trajs) -> dict:
    """Mean counterfactual statistics over the episodes where they are defined."""
    rows = [counterfactual_score(t.log, t.inst["constraints"]) for t in trajs]
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"episodes": 0}
    keys = ("toward_true", "toward_false", "chance_true", "chance_false", "score")
    return {"episodes": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}


@torch.no_grad()
def unrelated_kl(model, tok, context) -> float:
    """KL(clean || intervened) on the next token of unrelated prompts: what else does this break?"""
    device = next(model.parameters()).device
    seqs = []
    for prompt in UNRELATED_PROMPTS:
        conv = Conversation(tok)
        conv.add_user(prompt)
        seqs.append(conv.ids)
    ids, mask = pad_left(seqs, tok.pad_token_id, device)
    clean = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits[:, -1].float(), -1)
    with context():
        changed = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits[:, -1].float(), -1)
    return float((clean.exp() * (clean - changed)).sum(-1).mean())


def run_project(model, tok, a, layer, directions, instances, noise_sizes, rng) -> dict:
    device = next(model.parameters()).device
    basis = orthonormal(directions[layer], device)
    run = lambda: rollout(model, tok, instances, a.mode, a.H, noise_sizes=noise_sizes, gen_batch=a.gen_batch)
    out = {"baseline_success": success(run())}
    with project_out(model, layer, basis):
        out["probe_success"] = success(run())
    out["probe_unrelated_kl"] = unrelated_kl(model, tok, lambda: project_out(model, layer, basis))
    out["random"] = []
    for _ in range(a.random_reps):
        control = random_like(basis.shape[0], basis.shape[1], rng, device)
        with project_out(model, layer, control):
            ok = success(run())
        kl = unrelated_kl(model, tok, lambda: project_out(model, layer, control))
        out["random"].append({"success": ok, "unrelated_kl": kl})
    return out


def run_patch(model, tok, a, layer, directions, instances, rng) -> dict:
    """First-turn edits under a false line, with the true run's probe component patched in."""
    device = next(model.parameters()).device
    basis = orthonormal(directions[layer], device)
    holder = {}

    def capture_true_run(convs, _ids, _mask):
        seqs = []
        for conv in convs:
            env = GridEnv("true", a.H)
            true_conv = Conversation(tok)
            true_conv.add_user(env.reset(conv.inst), env.line)
            seqs.append(true_conv.ids)
        true_ids, true_mask = pad_left(seqs, tok.pad_token_id, device)
        holder["source"] = None  # the capture pass itself must stay unpatched
        holder["source"] = last_token_activation(model, layer, true_ids, true_mask)

    run = lambda cb=None: rollout(model, tok, instances, "counterfactual", a.H, gen_batch=a.gen_batch,
                                  turns=1, on_generate=cb)
    out = {"clean": direction_summary(run())}
    with patch_last_position(model, layer, basis, holder):
        out["probe_patched"] = direction_summary(run(capture_true_run))
    out["random_patched"] = []
    for _ in range(a.random_reps):
        control = random_like(basis.shape[0], basis.shape[1], rng, device)
        with patch_last_position(model, layer, control, holder):
            out["random_patched"].append(direction_summary(run(capture_true_run)))
    return out


def run_heads(model, tok, a, instances, rng) -> dict:
    if model.config._attn_implementation != "eager":
        raise SystemExit("the heads experiment needs a model loaded with eager attention")
    run = lambda cb=None, turns=None: rollout(model, tok, instances, "true", a.H, gen_batch=a.gen_batch,
                                              on_generate=cb, turns=turns)
    meter = LineAttention(block=False)
    with meter.active():
        run(meter.set_batch, turns=1)
    ranked = sorted(meter.mass, key=meter.mass.get, reverse=True)
    n_top = a.top_heads
    top = [(int(l), int(h)) for l, h in ranked[:n_top]]

    def group(pairs):
        out = {}
        for l, h in pairs:
            out.setdefault(l, []).append(h)
        return out

    def blocked(heads, query):
        block = LineAttention(heads=heads, query=query)
        with block.active():
            return success(run(block.set_batch))

    all_heads = list(meter.mass)
    picks = rng.choice(len(all_heads), size=n_top, replace=False)
    random_heads = [all_heads[int(i)] for i in picks]
    return {
        "baseline_success": success(run()),
        "top_heads": top,
        "top_heads_mean_attention_to_line": [meter.mass[p] / max(meter.count, 1) for p in top],
        "top_heads_success": blocked(group(top), "decision"),
        "random_heads_success": blocked(group(random_heads), "decision"),
        "all_heads_decision_success": blocked(None, "decision"),
        "all_heads_all_later_tokens_success": blocked(None, "all_after"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["project", "patch", "heads"])
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, choices=["theta_0", "theta_A", "theta_C", "theta_D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probes", help="directory written by probes.py (project and patch)")
    p.add_argument("--layer", type=int, default=None, help="default: the probes' best layer")
    p.add_argument("--mode", default="none", help="prompt condition for the project experiment")
    p.add_argument("--noise-sizes", default=None)
    p.add_argument("--set", default="spec")
    p.add_argument("--m", default="1,2,3")
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--random-reps", type=int, default=3)
    p.add_argument("--top-heads", type=int, default=16)
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    model, tok = load_model(a.model, eager=a.experiment == "heads")
    instances = heldout_instances(a.set, parse_ms(a.m), a.n)
    result = {"experiment": a.experiment, "label": a.label, "seed": a.seed, "model": a.model}

    if a.experiment == "heads":
        result |= run_heads(model, tok, a, instances, rng)
    else:
        if not a.probes:
            raise SystemExit(f"{a.experiment} needs --probes")
        directions = np.load(os.path.join(a.probes, "probes.npz"))["directions"]
        with open(os.path.join(a.probes, "probes.json")) as f:
            layer = json.load(f)["best_layer"] if a.layer is None else a.layer
        result["layer"] = layer
        if a.experiment == "project":
            result |= run_project(model, tok, a, layer, directions, instances, load_noise_sizes(a.noise_sizes), rng)
        else:
            result |= run_patch(model, tok, a, layer, directions, instances, rng)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
