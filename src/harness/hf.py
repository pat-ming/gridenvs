"""
Hook any HF/TRL Trainer (GRPOTrainer, SFTTrainer, Trainer) into a harness Run.

Create the Run with step_key="train/global_step" before building the trainer
and pass report_to="wandb": HF's WandbCallback reuses the wandb run that Run
started, so W&B sees one run. This callback adds the JSONL copy, checkpoint
events, and token/FLOP totals in summary.json.
"""

from transformers import TrainerCallback


class HarnessCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            self.run.log(logs, step=state.global_step, wandb=False)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = f"{args.output_dir}/checkpoint-{state.global_step}"
            self.run.event("checkpoint", step=state.global_step, path=path)

    def on_train_end(self, args, state, control, **kwargs):
        self.run.summary.update(
            global_step=state.global_step,
            total_flos=state.total_flos,
            num_input_tokens_seen=state.num_input_tokens_seen,
        )
