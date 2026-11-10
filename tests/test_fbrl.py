import pytest

from fbrl.grid import load_env

GridEnv, make_instance = load_env("grid4")


def test_residual_line_reports_how_far_each_sum_is_off():
    inst = make_instance(set_name="rows", m=2, seed=3)
    env = GridEnv("residual", H=2)
    env.reset(inst)
    want = [f"{k} ({10 - int(env.g[k].sum()):+d})" for k in env.shown]
    assert env.line == "violated: {" + ", ".join(want) + "}"
    assert env.mode == "true"  # everything else behaves as E_C


def test_oracle_demos_solve_and_only_replies_are_trained():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3 tokenizer is not cached")
    from fbrl.train_sft import demo_dataset

    cfg = {
        "experiment": {"seed": 0},
        "model": {"chat_template_kwargs": {"enable_thinking": False}},
        "env": {
            "name": "grid4",
            "instances": {"set_name": ["spec"], "m": [1, 2]},
            "slack": [0, 1],
        },
        "demos": {
            "n_episodes": 40,
            "oracle": 1.0,
            "modes": ["none", "true", "noise", "residual"],
        },
    }
    data, stats = demo_dataset(cfg, tok)
    assert stats["demo_success"] == 1.0
    row = data[0]
    trained = [t for t, m in zip(row["input_ids"], row["assistant_masks"]) if m]
    text = tok.decode(trained)
    assert text.count("EDIT") == text.count("<|im_end|>") >= 1
    assert "violated" not in text and "<|im_start|>" not in text
