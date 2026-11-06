"""LLM-layer checks on a tiny random Qwen3 and the real tokenizer (no model weights needed)."""
import argparse

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

import interventions as iv
from grid_env import GridEnv, NOISE_ITEM, format_feedback
from instances import make_instance
from llm import BASE_MODEL, Conversation, generation_config, pad_left, rollout, token_logprobs
from probes import decision_activations, fit_all, scripted_states
from report import condition_table, leave_one_seed_out
from train_grpo import group_advantages, grpo_loss, grpo_step


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer is not cached")


def tiny(tok, eager=False, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      head_dim=8, attn_implementation="eager" if eager else "sdpa")
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def model(tok):
    return tiny(tok)


def first_conv(tok, inst, mode="true"):
    env = GridEnv(mode, 4, seed=0)
    conv = Conversation(tok)
    conv.add_user(env.reset(inst), env.line)
    return conv, env


def test_noise_line_token_length_matches_true_line(tok):
    count = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    for n in range(11):
        assert count(format_feedback(list(range(10))[:n])) == count(format_feedback([NOISE_ITEM] * n))


def test_conversation_matches_the_chat_template(tok):
    conv, _ = first_conv(tok, make_instance("spec", 2, 0))
    full = tok.apply_chat_template(
        [{"role": "user", "content": tok.decode(conv.ids).split("<|im_start|>user\n")[1].split("<|im_end|>")[0]}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert conv.ids == list(full)


def test_later_turn_matches_template_structure(tok):
    conv, env = first_conv(tok, make_instance("spec", 2, 0))
    reply = tok("EDIT 0 0 1<|im_end|>", add_special_tokens=False).input_ids
    conv.add_reply(reply)
    obs, *_ = env.step("EDIT 0 0 1")
    conv.add_user(obs, env.line)
    text = tok.decode(conv.ids)
    assert text.count("<|im_start|>user\n") == 2
    assert "EDIT 0 0 1<|im_end|>\n<|im_start|>user\n" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_line_mask_covers_exactly_the_feedback_line(tok):
    conv, env = first_conv(tok, make_instance("spec", 3, 1))
    marked = [i for i, b in zip(conv.ids, conv.line) if b]
    assert tok.decode(marked) == env.line
    assert not any(conv.gen)


def test_reply_closes_a_truncated_turn(tok):
    conv = Conversation(tok)
    conv.add_user("hi")
    before = len(conv.ids)
    gen = tok("EDIT 0", add_special_tokens=False).input_ids
    assert conv.add_reply(gen) == "EDIT 0"
    assert conv.ids[-1] == conv.im_end and not conv.gen[-1] and sum(conv.gen) == len(gen)
    assert len(conv.ids) == before + len(gen) + 1


def test_reply_stops_at_the_first_end_token(tok):
    conv = Conversation(tok)
    conv.add_user("hi")
    gen = tok("EDIT 0 0 1", add_special_tokens=False).input_ids + [conv.im_end, tok.pad_token_id, tok.pad_token_id]
    assert conv.add_reply(gen) == "EDIT 0 0 1"
    assert conv.ids[-1] == conv.im_end and conv.gen[-1]


def test_sampling_config_is_the_plain_softmax(tok):
    cfg = generation_config(tok, 8, True)
    assert cfg.temperature == 1.0 and cfg.top_p == 1.0 and not cfg.top_k


def test_token_logprobs_match_full_logits_with_padding(tok, model):
    a, _ = first_conv(tok, make_instance("spec", 2, 0))
    b, _ = first_conv(tok, make_instance("spec", 2, 1), mode="none")
    for conv in (a, b):
        conv.add_reply(tok("EDIT 0 0 1<|im_end|>", add_special_tokens=False).input_ids)
    with torch.no_grad():
        out = token_logprobs(model, [a.ids, b.ids], [a.gen, b.gen], tok.pad_token_id)
        for row, conv in enumerate((a, b)):
            ids = torch.tensor([conv.ids])
            logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1], -1)
            expected = logp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            mask = torch.tensor(conv.gen[1:])
            assert torch.allclose(out[row, :len(conv.ids) - 1][mask], expected[mask], atol=1e-4)
            assert (out[row, :len(conv.ids) - 1][~mask] == 0).all()


def test_rollout_runs_all_modes_and_tracks_tokens(tok, model):
    instances = [make_instance("spec", 2, i) for i in range(3)]
    for mode in ["none", "true", "noise", "counterfactual"]:
        trajs = rollout(model, tok, instances, mode, H=2, noise_sizes=[2], max_new_tokens=4, gen_batch=2)
        for t in trajs:
            assert len(t.log) == 2 and t.env.t == 2 and sum(t.conv.gen) > 0
            assert any(t.conv.line) == (mode != "none")


def test_group_advantages():
    r = torch.tensor([1., 0., 0., 0., 1., 1., 1., 1.])
    adv = group_advantages(r, 4)
    assert adv[0] > 0 > adv[1] and adv[:4].sum().abs() < 1e-5
    assert (adv[4:] == 0).all()


def test_grpo_loss_pushes_toward_rewarded_tokens():
    """With no KL penalty and matching old log-probs, the gradient follows the advantage."""
    new = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]], requires_grad=True)
    old = new.detach().clone()
    mask = torch.ones_like(old)
    loss, kl = grpo_loss(new, old, old, torch.tensor([1.0, -1.0]), mask, clip=0.2, beta=0.0)
    loss.backward()
    assert (new.grad[0] < 0).all() and (new.grad[1] > 0).all() and kl == pytest.approx(0)


def test_grpo_loss_ignores_masked_tokens_and_kl_pulls_back():
    new = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    ref = torch.tensor([[-3.0, -9.0]])
    loss, kl = grpo_loss(new, new.detach(), ref, torch.zeros(1), torch.tensor([[1.0, 0.0]]), 0.2, 1.0)
    loss.backward()
    assert new.grad[0, 1] == 0 and new.grad[0, 0] != 0 and kl > 0


def test_grpo_step_updates_parameters(tok):
    m, ref = tiny(tok), tiny(tok)
    optim = torch.optim.AdamW(m.parameters(), lr=1e-2)
    instances = [make_instance("spec", 1, i) for i in range(2) for _ in range(2)]
    trajs = rollout(m, tok, instances, "true", H=1, max_new_tokens=4)
    trajs[0].reward = 1.0  # a random tiny model never solves; fake one success per group
    trajs[2].reward = 1.0
    before = m.lm_head.weight.clone()
    args = argparse.Namespace(minibatch=2, ppo_epochs=1, clip=0.2, beta=0.01, max_grad_norm=1.0)
    stats = grpo_step(m, ref, optim, tok, trajs, 2, args)
    assert stats["grad_norm"] > 0 and not torch.equal(before, m.lm_head.weight)


def test_probes_recover_a_planted_direction():
    rng = np.random.default_rng(0)
    instances = np.repeat(np.arange(200), 3)
    Y = (rng.random((600, 2)) < 0.5).astype(np.float32)
    secret = rng.standard_normal((2, 16))
    X = rng.standard_normal((600, 16)) * 0.2 + Y @ secret
    acts = np.stack([rng.standard_normal((600, 16)), X], axis=1).astype(np.float32)  # noise, then signal
    per_layer, dirs, per_instance = fit_all(acts, Y, instances)
    assert per_layer[0]["mean"] < 0.65 and per_layer[1]["mean"] > 0.95
    cosine = abs(dirs[1, 0] @ secret[0]) / np.linalg.norm(secret[0])
    assert cosine > 0.5
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1, atol=1e-5)
    assert len(per_instance) == 40 and len(next(iter(per_instance.values()))) == 2


def test_scripted_states_have_both_labels_and_no_feedback(tok, model):
    states = scripted_states(tok, "spec", [1, 2, 3], 4, 20)
    assert len(states) == 80
    Y = np.stack([s["y"] for s in states])
    assert 0.05 < Y.mean() < 0.95
    assert not any("violated" in tok.decode(s["ids"]) for s in states)
    acts = decision_activations(model, tok, [s["ids"] for s in states[:4]], 2)
    assert acts.shape == (4, 2, 32)


def test_activations_ignore_batch_padding(tok, model):
    states = scripted_states(tok, "spec", [1], 2, 2)
    ids = [states[0]["ids"], states[1]["ids"]]
    together = decision_activations(model, tok, ids, 2)
    alone = decision_activations(model, tok, ids[:1], 1)
    assert np.allclose(together[0], alone[0], atol=1e-3)


def test_project_out_removes_the_component(tok, model):
    basis = iv.random_like(3, 32, np.random.default_rng(0), "cpu")
    assert torch.allclose(basis @ basis.T, torch.eye(3), atol=1e-5)
    captured = {}
    ids = torch.tensor([[1, 2, 3, 4]])
    with iv.project_out(model, 0, basis):
        hook = model.model.layers[0].register_forward_hook(
            lambda _m, _i, out: captured.update(h=(out[0] if isinstance(out, tuple) else out).detach()))
        try:
            model.model(input_ids=ids)
        finally:
            hook.remove()
    # The projection hook ran first, so the later hook sees the edited output.
    assert (captured["h"] @ basis.T).abs().max() < 1e-4


def test_hooks_are_removed_after_each_context(tok, model):
    basis = iv.random_like(1, 32, np.random.default_rng(0), "cpu")
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        clean = model(input_ids=ids).logits
        with iv.project_out(model, 0, basis):
            changed = model(input_ids=ids).logits
        again = model(input_ids=ids).logits
    assert not torch.allclose(clean, changed) and torch.equal(clean, again)


def test_patch_with_true_source_only_changes_basis_component(tok, model):
    conv, _ = first_conv(tok, make_instance("spec", 2, 0), "counterfactual")
    ids, mask = pad_left([conv.ids], tok.pad_token_id, "cpu")
    basis = iv.random_like(2, 32, np.random.default_rng(0), "cpu")
    clean = iv.last_token_activation(model, 0, ids, mask)
    source = clean + 5.0
    holder = {"source": source}
    with iv.patch_last_position(model, 0, basis, holder):
        pass_through = {}
        hook = model.model.layers[0].register_forward_hook(
            lambda _m, _i, out: pass_through.update(h=(out[0] if isinstance(out, tuple) else out)[:, -1].detach()))
        try:
            model.model(input_ids=ids, attention_mask=mask)
        finally:
            hook.remove()
    got = pass_through["h"]
    assert torch.allclose(got @ basis.T, source @ basis.T, atol=1e-3)  # basis component = source's
    residual = lambda x: x - (x @ basis.T) @ basis
    assert torch.allclose(residual(got), residual(clean), atol=1e-3)   # the rest is unchanged


def line_pair(tok):
    """The same grid with two different, same-length violated lines."""
    inst = make_instance("spec", 2, 0)
    grid = GridEnv("none", 4).reset(inst)
    out = []
    for line in ("violated: {1, 2}", "violated: {7, 9}"):
        conv = Conversation(tok)
        conv.add_user(grid + "\n" + line, line)
        out.append(conv)
    assert len(out[0].ids) == len(out[1].ids)
    return out


def decision_logits(model, tok, conv, block):
    ids, mask = pad_left([conv.ids], tok.pad_token_id, "cpu")
    if block is None:
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask).logits[0, -1]
    block.set_batch([conv], ids, mask)
    with block.active(), torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits[0, -1]


def test_clean_run_is_reproduced_by_eager_patch_when_nothing_is_blocked(tok):
    m = tiny(tok, eager=True)
    conv = line_pair(tok)[0]
    clean = decision_logits(m, tok, conv, None)
    same = decision_logits(m, tok, conv, iv.LineAttention(heads={}, query="decision"))
    assert torch.allclose(clean, same, atol=1e-4)


def test_blocking_all_later_tokens_makes_the_decision_independent_of_the_line(tok):
    m = tiny(tok, eager=True)
    a, b = line_pair(tok)
    assert not torch.allclose(decision_logits(m, tok, a, None), decision_logits(m, tok, b, None), atol=1e-4)
    block = lambda: iv.LineAttention(heads=None, query="all_after")
    assert torch.allclose(decision_logits(m, tok, a, block()), decision_logits(m, tok, b, block()), atol=1e-4)


def test_decision_blocking_changes_output_but_leaves_indirect_routes(tok):
    m = tiny(tok, eager=True)
    a, b = line_pair(tok)
    clean = decision_logits(m, tok, a, None)
    block = lambda: iv.LineAttention(heads=None, query="decision")
    blocked_a, blocked_b = decision_logits(m, tok, a, block()), decision_logits(m, tok, b, block())
    assert not torch.allclose(clean, blocked_a, atol=1e-5)
    assert not torch.allclose(blocked_a, blocked_b, atol=1e-5)


def test_attention_meter_counts_every_head_and_blocking_works_with_cache(tok):
    m = tiny(tok, eager=True)
    instances = [make_instance("spec", 2, i) for i in range(3)]
    meter = iv.LineAttention(block=False)
    with meter.active():
        rollout(m, tok, instances, "true", H=1, on_generate=meter.set_batch, max_new_tokens=2)
    assert len(meter.mass) == 2 * 4 and all(v >= 0 for v in meter.mass.values())
    block = iv.LineAttention(heads={0: [0]}, query="decision")
    with block.active():
        trajs = rollout(m, tok, instances, "true", H=2, on_generate=block.set_batch, max_new_tokens=3)
    assert all(t.env.t == 2 for t in trajs)


def test_report_table_marks_off_training_conditions():
    def run(label, seed, rate):
        rows = [{"solved": i < rate * 10} for i in range(10)]
        return {"label": label, "seed": seed, "conditions": {m: rows for m in ["none", "true", "noise", "counterfactual"]}}
    table = condition_table([run("theta_A", 0, 0.5), run("theta_A", 1, 0.7), run("theta_C", 0, 0.2)])
    assert "0.600 (0.500–0.700)" in table  # theta_A on its own training condition has no star
    assert "0.600* (0.500–0.700)" in table
    assert "0.200*" in table


def test_recoverability_holds_out_seeds():
    feats = [0.0, 0.1, 1.0, 1.1] * 2
    labels = ["theta_A", "theta_A", "theta_C", "theta_C"] * 2
    seeds = [0, 0, 0, 0, 1, 1, 1, 1]
    out = leave_one_seed_out(feats, labels, seeds)
    assert out["accuracy"] == 1.0 and set(out["per_seed"]) == {0, 1} and out["instances"] == 8
    with pytest.raises(SystemExit):
        leave_one_seed_out(feats[:4], labels[:4], seeds[:4])
