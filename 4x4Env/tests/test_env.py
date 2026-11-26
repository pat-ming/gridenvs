"""Environment checks. No model or tokenizer needed."""
import numpy as np
import pytest

from constraints import AllDistinctCol, PairwiseLess, violations
from grid_env import NOISE_ITEM, GridEnv, counterfactual_score, format_feedback
from instances import CONSTRAINT_SETS, CURRICULUM, make_instance, make_level

SETS = list(CONSTRAINT_SETS)


def test_pairwise_less():
    g = np.zeros((4, 4), dtype=int)
    g[0, 1] = 5
    assert PairwiseLess((0, 0), (0, 1)).holds(g)
    assert not PairwiseLess((0, 1), (0, 0)).holds(g)
    assert not PairwiseLess((0, 0), (1, 1)).holds(g)  # strict
    assert PairwiseLess((0, 0), (0, 1)).scope() == {(0, 0), (0, 1)}


def test_all_distinct_col():
    g = np.array([[1, 1, 0, 0], [2, 1, 0, 0], [3, 2, 0, 0], [4, 3, 0, 0]])
    assert AllDistinctCol(0).holds(g)
    assert not AllDistinctCol(1).holds(g)
    assert len(AllDistinctCol(1).scope()) == 4


@pytest.mark.parametrize("set_name", SETS)
def test_solution_satisfies_and_start_is_broken(set_name):
    for seed in range(50):
        inst = make_instance(set_name, 2, seed)
        assert violations(inst["solution"], inst["constraints"]) == []
        assert violations(inst["grid"], inst["constraints"]) != []
        assert ((inst["grid"] >= 0) & (inst["grid"] <= 9)).all()


def test_spec_set_has_every_constraint_type_and_single_digit_indices():
    inst = make_instance("spec", 2, 0)
    assert len(inst["constraints"]) == 10
    assert {c.name for c in inst["constraints"]} == {"row_sum", "all_distinct_col", "pairwise_less"}


def test_instances_are_reproducible():
    a, b = make_instance("spec", 3, 7), make_instance("spec", 3, 7)
    assert (a["grid"] == b["grid"]).all() and a["constraints"] == b["constraints"]


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["none", "true", "noise", "counterfactual"])
def test_solvable_in_m_edits_and_reward_only_at_horizon(m, mode):
    """Replaying the solution earns R = 1, and only on the final turn."""
    for seed in range(10):
        inst = make_instance("spec", m, seed)
        env = GridEnv(mode, H=4, seed=seed)
        env.reset(inst)
        wrong = np.argwhere(inst["grid"] != inst["solution"])
        assert len(wrong) <= m
        edits = [(int(r), int(c), int(inst["solution"][r, c])) for r, c in wrong]
        edits += [(0, 0, int(inst["solution"][0, 0]))] * (4 - len(edits))
        for t, (r, c, v) in enumerate(edits):
            _, reward, done, info = env.step(f"EDIT {r} {c} {v}")
            assert done == (t == 3)
            assert reward == (1.0 if t == 3 else 0.0)
        assert info["solved"]


def test_reward_zero_if_broken_at_horizon_even_when_solved_earlier():
    inst = make_instance("rows", 1, 0)
    env = GridEnv("none", H=2)
    env.reset(inst)
    r, c = map(int, np.argwhere(inst["grid"] != inst["solution"])[0])
    env.step(f"EDIT {r} {c} {inst['solution'][r, c]}")
    bad = (inst["solution"][0, 0] + 1) % 10
    _, reward, _, info = env.step(f"EDIT 0 0 {bad}")
    assert reward == 0.0 and not info["solved"]


def test_observations_differ_only_in_the_feedback_line():
    inst = make_instance("spec", 2, 3)
    obs = {m: GridEnv(m, H=4, seed=0).reset(inst) for m in ["none", "true", "noise", "counterfactual"]}
    assert "violated" not in obs["none"]
    V = violations(inst["grid"], inst["constraints"])
    assert obs["true"] == obs["none"] + "\n" + format_feedback(V)
    for m in ["noise", "counterfactual"]:
        assert obs[m].startswith(obs["none"] + "\nviolated: {")


def test_true_feedback_tracks_state_after_an_edit():
    inst = make_instance("spec", 2, 3)
    env = GridEnv("true", H=4)
    env.reset(inst)
    obs, *_ = env.step("EDIT 0 0 9")
    assert obs.endswith(format_feedback(violations(env.g, inst["constraints"])))


def test_counterfactual_differs_from_truth_and_keeps_size_when_possible():
    env = GridEnv("counterfactual", H=4, seed=0)
    env.reset(make_instance("spec", 2, 0))
    for V in ([], [1], [0, 4], list(range(9))):
        Vp = env._counterfactual(V)
        assert set(Vp) != set(V)
        if V:
            assert len(Vp) == len(V)
    assert env._counterfactual(list(range(10))) != list(range(10))


def test_noise_line_ignores_state_and_carries_no_indices():
    inst = make_instance("spec", 2, 0)
    env = GridEnv("noise", H=4, seed=0, noise_sizes=[3])
    env.reset(inst)
    assert env.line == format_feedback([NOISE_ITEM] * 3)
    assert not any(ch.isdigit() for ch in env.line)
    a = GridEnv("noise", H=4, seed=5).reset(inst).splitlines()[-1]
    other = make_instance("spec", 3, 99)
    b = GridEnv("noise", H=4, seed=5).reset(other).splitlines()[-1]
    assert a == b  # same seed, different state, same line


def test_noise_rejects_two_digit_indices():
    inst = make_instance("spec", 2, 0)
    inst["constraints"] = inst["constraints"] + inst["constraints"]
    with pytest.raises(AssertionError):
        GridEnv("noise", H=4).reset(inst)


def test_parser_takes_the_last_edit():
    env = GridEnv("none", H=4)
    env.reset(make_instance("spec", 2, 0))
    _, _, _, info = env.step("Maybe EDIT 0 0 1, but better: EDIT 1 2 3")
    assert info["log"][-1]["action"] == (1, 2, 3) and env.g[1, 2] == 3


@pytest.mark.parametrize("text", ["nonsense", "EDIT 4 0 1", "EDIT 0 4 1", "EDIT 0 0 10", "edit 0 0 1", ""])
def test_invalid_action_is_a_no_op_that_uses_a_turn(text):
    env = GridEnv("none", H=4)
    inst = make_instance("spec", 2, 0)
    env.reset(inst)
    _, _, _, info = env.step(text)
    assert (env.g == inst["grid"]).all() and env.t == 1 and not info["log"][-1]["parse_ok"]


def test_episode_cannot_run_past_horizon():
    env = GridEnv("none", H=1)
    env.reset(make_instance("spec", 1, 0))
    env.step("x")
    with pytest.raises(AssertionError):
        env.step("x")


def test_counterfactual_score_reads_direction():
    cs = CONSTRAINT_SETS["rows"]  # V - V' = row 0, V' - V = row 1
    log = [{"V": [0], "shown": [1], "action": (1, 2, 3)}]
    out = counterfactual_score(log, cs)
    assert out["toward_false"] == 1 and out["toward_true"] == 0 and out["score"] == pytest.approx(1.0)
    out = counterfactual_score([{"V": [0], "shown": [1], "action": (0, 2, 3)}], cs)
    assert out["score"] == pytest.approx(-1.0)
    assert counterfactual_score([{"V": [0], "shown": [1], "action": None}], cs) is None
    assert counterfactual_score([{"V": [0], "shown": None, "action": (0, 0, 0)}], cs) is None


def test_noise_line_token_length_matches_true_line():
    """Skipped unless transformers and the Qwen3 tokenizer are available locally."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer is not cached")
    count = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    for n in range(11):
        assert count(format_feedback(list(range(10))[:n])) == count(format_feedback([NOISE_ITEM] * n))


# ------------------------------------------------------------ difficulty settings
@pytest.mark.parametrize("set_name,k", [("spec", 4), ("spec", 8), ("rows", 1), ("ineq", 3)])
def test_k_keeps_a_prefix_and_corrupts_only_cells_it_can_see(set_name, k):
    for seed in range(30):
        full = make_instance(set_name, 2, seed)
        inst = make_instance(set_name, 2, seed, k=k)
        assert inst["constraints"] == full["constraints"][:k]
        assert violations(inst["solution"], inst["constraints"]) == []
        assert violations(inst["grid"], inst["constraints"]) != []
        seen = frozenset().union(*(c.scope() for c in inst["constraints"]))
        assert {tuple(map(int, rc)) for rc in np.argwhere(inst["grid"] != inst["solution"])} <= seen


@pytest.mark.parametrize("set_name", ["rows", "spec", "distinct", "ineq", "desc"])
def test_vmax_narrows_the_value_range(set_name):
    for seed in range(30):
        inst = make_instance(set_name, 2, seed, vmax=5)
        assert inst["solution"].max() <= 5 and inst["grid"].max() <= 5
        assert violations(inst["solution"], inst["constraints"]) == []
    env = GridEnv("none", H=2)
    obs = env.reset(make_instance("rows", 1, 0, vmax=5))
    assert "Values are integers 0-5." in obs and "Example reply: EDIT 2 1 5" in obs
    _, _, _, info = env.step("EDIT 0 0 6")  # outside the narrowed range
    assert not info["log"][-1]["parse_ok"]


def test_vmax_rejects_sum_sets_that_cannot_reach_ten():
    with pytest.raises(AssertionError):
        make_instance("rows", 1, 0, vmax=4)


def test_default_prompt_is_unchanged():
    obs = GridEnv("none", H=4).reset(make_instance("spec", 2, 0))
    assert "Values are integers 0-9." in obs and "Example reply: EDIT 2 1 7" in obs


@pytest.mark.parametrize("ladder", list(CURRICULUM))
def test_every_curriculum_level_is_solvable_within_its_horizon(ladder):
    for level in range(len(CURRICULUM[ladder])):
        for seed in range(10):
            inst, H = make_level(ladder, level, seed)
            assert H >= inst["m"]
            env = GridEnv("true", H=H)
            env.reset(inst)
            edits = [(int(r), int(c), int(inst["solution"][r, c]))
                     for r, c in np.argwhere(inst["grid"] != inst["solution"])]
            edits += [edits[0]] * (H - len(edits))
            for r, c, v in edits:
                _, reward, _, _ = env.step(f"EDIT {r} {c} {v}")
            assert reward == 1.0
    # The last level of each ladder is the full set at its usual settings.
    last = dict(CURRICULUM[ladder][-1])
    assert last.get("k") is None and last.get("vmax", 9) == 9
