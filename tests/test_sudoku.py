"""Environment checks. No model needed."""
import numpy as np
import pytest

from gridenvs.sudoku.constraints import Box, Col, Row, all_constraints, violations
from gridenvs.sudoku.instances import make_instance
from gridenvs.sudoku.sudoku_env import (
    NOISE_ITEM,
    SudokuEnv,
    counterfactual_score,
    format_feedback,
)

MODES = ["none", "true", "noise", "counterfactual"]


def test_constraint_indices_and_scopes():
    cs = all_constraints()
    assert len(cs) == 27
    assert cs[0] == Row(0) and cs[9] == Col(0) and cs[18] == Box(0)
    assert all(len(c.scope()) == 9 for c in cs)
    assert Box(5).scope() == {(r, c) for r in range(3, 6) for c in range(6, 9)}
    assert (8, 8) in Box(8).scope() and (0, 0) in Box(0).scope()
    # Every cell lies in exactly one row, one column and one box.
    for r in range(9):
        for c in range(9):
            assert sum((r, c) in u.scope() for u in cs) == 3


def test_a_swap_inside_a_row_breaks_only_the_columns_and_boxes_it_touches():
    inst = make_instance(1, 0)
    g, cs = inst["solution"].copy(), inst["constraints"]
    g[0, 0], g[0, 1] = g[0, 1], g[0, 0]  # same row and same box, two columns
    assert violations(g, cs) == [9, 10]


def test_duplicate_is_a_violation_even_with_the_right_sum():
    g = make_instance(1, 0)["solution"].copy()
    row = np.array([1, 1, 1, 1, 5, 9, 9, 9, 9])  # sums to 45, as a valid row does
    g[0] = row
    assert not Row(0).holds(g)


def test_solutions_are_valid_and_varied():
    grids = set()
    for seed in range(300):
        inst = make_instance(2, seed)
        assert violations(inst["solution"], inst["constraints"]) == []
        assert violations(inst["grid"], inst["constraints"]) != []
        assert ((inst["grid"] >= 1) & (inst["grid"] <= 9)).all()
        grids.add(inst["solution"].tobytes())
    assert len(grids) == 300


def test_instances_are_reproducible():
    a, b = make_instance(3, 7), make_instance(3, 7)
    assert (a["grid"] == b["grid"]).all() and (a["solution"] == b["solution"]).all()


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", MODES)
def test_solvable_in_m_writes_and_reward_only_at_horizon(m, mode):
    """Replaying the solution earns R = 1, and only on the final turn."""
    for seed in range(10):
        inst = make_instance(m, seed)
        env = SudokuEnv(mode, H=4, seed=seed)
        env.reset(inst)
        wrong = np.argwhere(inst["grid"] != inst["solution"])
        assert len(wrong) <= m
        writes = [(int(inst["solution"][r, c]), int(r), int(c)) for r, c in wrong]
        writes += [(int(inst["solution"][0, 0]), 0, 0)] * (4 - len(writes))
        for t, (v, r, c) in enumerate(writes):
            _, reward, done, info = env.step(f"WRITE {v} {r} {c}")
            assert done == (t == 3)
            assert reward == (1.0 if t == 3 else 0.0)
        assert info["solved"]


def test_reward_zero_if_broken_at_horizon_even_when_solved_earlier():
    inst = make_instance(1, 0)
    env = SudokuEnv("none", H=2)
    env.reset(inst)
    r, c = map(int, np.argwhere(inst["grid"] != inst["solution"])[0])
    env.step(f"WRITE {inst['solution'][r, c]} {r} {c}")
    bad = inst["solution"][0, 0] % 9 + 1
    _, reward, _, info = env.step(f"WRITE {bad} 0 0")
    assert reward == 0.0 and not info["solved"]


def test_action_order_is_value_then_row_then_column():
    env = SudokuEnv("none", H=4)
    env.reset(make_instance(2, 0))
    _, _, _, info = env.step("WRITE 5 1 2")
    assert env.g[1, 2] == 5 and info["log"][-1]["action"] == (5, 1, 2)


def test_parser_takes_the_last_write():
    env = SudokuEnv("none", H=4)
    env.reset(make_instance(2, 0))
    _, _, _, info = env.step("Maybe WRITE 1 0 0, but better: WRITE 3 1 2")
    assert info["log"][-1]["action"] == (3, 1, 2) and env.g[1, 2] == 3


@pytest.mark.parametrize("text", ["nonsense", "WRITE 0 0 0", "WRITE 10 0 0", "WRITE 1 9 0", "WRITE 1 0 9",
                                  "WRITE 1 0 12", "write 1 0 0", "REWRITE 1 0 0", ""])
def test_invalid_action_is_a_no_op_that_uses_a_turn(text):
    env = SudokuEnv("none", H=4)
    inst = make_instance(2, 0)
    env.reset(inst)
    _, _, _, info = env.step(text)
    assert (env.g == inst["grid"]).all() and env.t == 1 and not info["log"][-1]["parse_ok"]


def test_episode_cannot_run_past_horizon():
    env = SudokuEnv("none", H=1)
    env.reset(make_instance(1, 0))
    env.step("x")
    with pytest.raises(AssertionError):
        env.step("x")


def test_observations_differ_only_in_the_feedback_line():
    inst = make_instance(2, 3)
    obs = {m: SudokuEnv(m, H=4, seed=0).reset(inst) for m in MODES}
    assert "violated" not in obs["none"]
    V = violations(inst["grid"], inst["constraints"])
    assert obs["true"] == obs["none"] + "\n" + format_feedback(V)
    for m in ["noise", "counterfactual"]:
        assert obs[m].startswith(obs["none"] + "\nviolated: {")


def test_true_feedback_tracks_state_after_a_write():
    inst = make_instance(2, 3)
    env = SudokuEnv("true", H=4)
    env.reset(inst)
    obs, *_ = env.step("WRITE 9 0 0")
    assert obs.endswith(format_feedback(violations(env.g, inst["constraints"])))


def test_counterfactual_differs_from_truth_and_keeps_size_when_possible():
    env = SudokuEnv("counterfactual", H=4, seed=0)
    env.reset(make_instance(2, 0))
    for V in ([], [1], [0, 14], list(range(26))):
        Vp = env._counterfactual(V)
        assert set(Vp) != set(V)
        if V:
            assert len(Vp) == len(V)
    assert env._counterfactual(list(range(27))) != list(range(27))


def test_noise_line_ignores_state_and_carries_no_indices():
    inst = make_instance(2, 0)
    env = SudokuEnv("noise", H=4, seed=0, noise_sizes=[3])
    env.reset(inst)
    assert env.line.count(",") == 2 and not any(ch.isdigit() for ch in env.line)
    a = SudokuEnv("noise", H=4, seed=5).reset(inst).splitlines()[-1]
    b = SudokuEnv("noise", H=4, seed=5).reset(make_instance(3, 99)).splitlines()[-1]
    assert a == b  # same seed, different state, same line


def test_counterfactual_score_reads_direction():
    cs = all_constraints()  # V - V' = row 0, V' - V = row 1
    out = counterfactual_score([{"V": [0], "shown": [1], "action": (5, 1, 2)}], cs)
    assert out["toward_false"] == 1 and out["toward_true"] == 0 and out["score"] == pytest.approx(1.0)
    out = counterfactual_score([{"V": [0], "shown": [1], "action": (5, 0, 2)}], cs)
    assert out["score"] == pytest.approx(-1.0)
    assert counterfactual_score([{"V": [0], "shown": [1], "action": None}], cs) is None
    assert counterfactual_score([{"V": [0], "shown": None, "action": (5, 0, 0)}], cs) is None


def test_noise_line_token_length_matches_true_line():
    """Skipped unless transformers and the Qwen3 tokenizer are available locally."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer is not cached")
    count = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    rng = np.random.default_rng(0)
    for _ in range(300):
        indices = sorted(int(k) for k in rng.choice(27, size=int(rng.integers(0, 28)), replace=False))
        noise = [NOISE_ITEM * len(str(k)) for k in indices]
        assert count(format_feedback(indices)) == count(format_feedback(noise))
