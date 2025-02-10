import lightning as L
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_warn


class EvenlySpacedModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, n_checkpoints: int = 10, **kwargs):
        assert "every_n_epochs" not in kwargs
        assert "every_n_steps" not in kwargs
        assert "train_time_interval" not in kwargs

        kwargs.setdefault("save_top_k", -1)
        super().__init__(*args, **kwargs)

        self.n_checkpoints = n_checkpoints

    def setup(self, trainer: L.Trainer, *args, **kwargs):
        total_steps = (
            trainer.max_steps
            if trainer.max_steps != -1
            else int(trainer.estimated_stepping_batches)
        )

        if self.n_checkpoints > total_steps:
            self.n_checkpoints = total_steps
            rank_zero_warn(
                f"n_checkpoints is larger than total_steps. Setting to {total_steps}"
            )

        self.steps_to_save = self._calculate_steps_to_save(
            total_steps, self.n_checkpoints
        )

        super().setup(trainer, *args, **kwargs)

    def on_train_start(self, *args, **kwargs):
        self.current_checkpoint_index = 0
        super().on_train_start(*args, **kwargs)

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs):
        if self._should_skip_saving_checkpoint(trainer):
            return

        if self.current_checkpoint_index >= self.n_checkpoints:
            return

        if trainer.global_step >= self.steps_to_save[self.current_checkpoint_index]:
            self.current_checkpoint_index += 1
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    @staticmethod
    def _calculate_steps_to_save(total_steps: int, n_checkpoints: int) -> list[int]:
        steps = np.linspace(0, total_steps, n_checkpoints + 1)[1:]
        return steps.astype(int).tolist()
