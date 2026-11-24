# Prompt: the Jacobian lens on our trained models

Paste everything below the line into a Claude Code session started in the repo checkout
(on the cluster, where the checkpoints are).

---

I'm working on the interpretability side of our CSE 5100 project with the Jacobian lens (J-lens), which
maps a model's residual stream to words the model could report about that state. Motti is doing the
same questions with probes, activation patching and neuron-level analysis, so we can check each other.

Read these first:
- `docs/interp.md`, the shared notes: what we have, the shared groundwork, the questions, and who owns
  each;
- the proposal tex in the repo root, §3 and §5;
- `docs/next.md` and the top two entries of `docs/runlog.md`;
- the J-lens reference implementation, github.com/anthropics/jacobian-lens (its README and Qwen
  examples).

**Where things stand.** We trained Qwen3-0.6B-Base with GRPO on 4x4 puzzles under three conditions,
4 seeds each:
- θ_A: no feedback line;
- θ_C: the true violated-constraint line;
- θ_D: a noise line.

θ_C ended up following the line about 100% of the time, even when it's false. What we don't know is
whether any of them work out the broken constraint themselves (H1, the proposal's I(θ)), or how θ_C
uses the line.

The work, in order:
1. **Get the J-lens running on θ_0** (`$FBRL/models/sft-qwen3-0.6b`).
   - Reproduce one of the repo's Qwen examples first, so we know the setup is right.
   - Then run it on our prompts, built with `src/fbrl/grid.py`. Use the shared prompt set if it
     exists (see `docs/interp.md`), otherwise build it and tell Motti.
2. **Question 9 on `distinct`.** At the last prompt token, on line-free states, does the readout
   name the broken column or a duplicate value?
   - Compare θ_0 against `final` of θ_A, θ_C and θ_D at seed 0.
   - Also run θ_D seed 1 from `grpo_kl0.1`, the one model that learned to find the broken column on
     its own. If the lens reads anything about violations, it should be there.
   - Then put the line in, true and false: at θ_C's line token, does the readout carry the line's
     index, and does it change the readout at the last token?
3. **Check the readouts against known answers.** V(s) is known for every state, and Motti's probes
   give a second view. A readout that doesn't match either is a finding about the lens, not the
   model, and is worth writing down as such.
4. **Question 6.** Sweep θ_C seed 0 across `checkpoint-100` to `checkpoint-600`: when does the readout
   change, compared with when following the line switches on in the eval records?
5. **Iterate.** Change what's read out (which token, which layer, which vocabulary), add a second
   seed, and try `ineq`, where the line names a constraint the model has to look up.

Use a GPU node interactively (`srun --gres=gpu:a40:1 --pty bash` on `general-gpu-long`), not the login
node. Keep results in `results/interp/jlens/`, write a dated entry in `docs/runlog.md` when something
comes out, and commit as you go. Plain prose, say what didn't work, and don't overclaim from one seed.
