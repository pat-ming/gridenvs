"""Multi-turn GRPO: theta_X = Train(theta_0, E_X, seed).

Each instance is played `group` times. An episode's advantage is its terminal reward,
normalized within its group, and it applies to every token the policy generated in that episode.
Use identical arguments for A, C and D so the optimization budget matches. There is no early
stopping, and each seed uses the same instance stream across environments.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from llm import (BASE_MODEL, load_model, load_noise_sizes, parse_ms, pad_right, release_memory, require_noise_sizes,
                 rollout, seed_everything, token_logprobs, train_instance)

ENV_TO_MODE = {"A": "none", "C": "true", "D": "noise"}


def group_advantages(rewards: torch.Tensor, group: int, eps: float = 1e-4) -> torch.Tensor:
    """(R - group mean) / group std. A group whose episodes all tie gets zero."""
    r = rewards.view(-1, group)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True, unbiased=False) + eps)).view(-1)


def grpo_loss(new_lp, old_lp, ref_lp, adv, mask, clip: float, beta: float):
    """Clipped policy objective plus a KL penalty toward theta_0, averaged per generated token."""
    ratio = torch.exp(new_lp - old_lp)
    a = adv.unsqueeze(1)
    objective = torch.min(a * ratio, a * ratio.clamp(1 - clip, 1 + clip))
    diff = ref_lp - new_lp
    kl = torch.exp(diff) - diff - 1  # sample estimate of KL(new || ref), always >= 0
    per_token = -objective + beta * kl
    denom = mask.sum(1).clamp(min=1)
    loss = ((per_token * mask).sum(1) / denom).mean()
    return loss, ((kl * mask).sum(1) / denom).mean().detach()


def grpo_step(model, ref, optim, tok, trajs, group, args) -> dict:
    """One policy update from a batch of finished episodes."""
    device = next(model.parameters()).device
    seqs = [t.conv.ids for t in trajs]
    gens = [t.conv.gen for t in trajs]
    pad = tok.pad_token_id
    rewards = torch.tensor([t.reward for t in trajs], device=device)
    adv = group_advantages(rewards, group)
    gen = pad_right([t.conv.gen for t in trajs], False, device, torch.bool)[:, 1:]

    n = len(trajs)
    mb = args.minibatch
    with torch.no_grad():
        old = [token_logprobs(model, seqs[i:i + mb], gens[i:i + mb], pad) for i in range(0, n, mb)]
        reference = [token_logprobs(ref, seqs[i:i + mb], gens[i:i + mb], pad) for i in range(0, n, mb)]

    kl_sum, loss_sum = 0.0, 0.0
    for _ in range(args.ppo_epochs):
        optim.zero_grad()
        for j, i in enumerate(range(0, n, mb)):
            idx = range(i, min(i + mb, n))
            new_lp = token_logprobs(model, seqs[i:i + mb], gens[i:i + mb], pad)
            width = new_lp.shape[1]
            mask = gen[list(idx)][:, :width].float()
            size = len(idx) / n  # weights minibatches so the sum is the batch mean
            loss, kl = grpo_loss(new_lp, old[j], reference[j], adv[list(idx)], mask, args.clip, args.beta)
            (loss * size).backward()
            loss_sum += loss.item() * size
            kl_sum += kl.item() * size
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
    return {"loss": loss_sum / args.ppo_epochs, "kl": kl_sum / args.ppo_epochs, "grad_norm": float(grad)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=list(ENV_TO_MODE))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--set", default="spec")
    p.add_argument("--m", default="1,2,3", help="corruption counts to sample from")
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--noise-sizes", default=None)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--instances", type=int, default=8, help="instances per iteration")
    p.add_argument("--group", type=int, default=8, help="episodes per instance")
    p.add_argument("--minibatch", type=int, default=8)
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.01, help="KL penalty toward theta_0")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=100)
    a = p.parse_args()

    ms = parse_ms(a.m)
    if max(ms) > a.H:
        raise SystemExit(f"H={a.H} is too small: instances with m={max(ms)} may be unsolvable")
    mode = ENV_TO_MODE[a.env]
    noise_sizes = require_noise_sizes(mode, load_noise_sizes(a.noise_sizes))

    seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)  # identical instance stream for A, C and D at this seed
    model, tok = load_model(a.model)
    ref, _ = load_model(a.model)
    for param in ref.parameters():
        param.requires_grad_(False)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    log_path = os.path.join(a.out, "metrics.jsonl")

    for it in range(a.iterations):
        t0 = time.time()
        base = [train_instance(a.set, ms, rng) for _ in range(a.instances)]
        repeated = [inst for inst in base for _ in range(a.group)]
        trajs = rollout(model, tok, repeated, mode, a.H, noise_sizes=noise_sizes,
                        max_new_tokens=a.max_new_tokens, gen_batch=a.gen_batch)
        release_memory()
        stats = grpo_step(model, ref, optim, tok, trajs, a.group, a)
        release_memory()
        edits = [e for t in trajs for e in t.log]
        stats |= {
            "iter": it,
            "success": float(np.mean([t.reward for t in trajs])),
            "parse_rate": float(np.mean([e["parse_ok"] for e in edits])),
            "seconds": time.time() - t0,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        print(json.dumps(stats), flush=True)
        if (it + 1) % a.save_every == 0 or it + 1 == a.iterations:
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
