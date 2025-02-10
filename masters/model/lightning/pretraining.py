import einops
import lightning as L
from torch import LongTensor, nn, optim
from transformers import get_scheduler

from ..model import BertConfig, BertPretraining


class LightningPretraining(L.LightningModule):
    def __init__(
        self,
        config: BertConfig,
        ignore_index: int = -100,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        initialization_range: float = 0.02,
        lr_scheduler: str = "cosine",
        warmup_steps_or_ratio: int | float = 0.1,
    ):
        super(LightningPretraining, self).__init__()
        self.model = BertPretraining(**config)
        self.model.reset_weights(initialization_range)

        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.lr_scheduler = lr_scheduler
        self.warmup_steps_or_ratio = warmup_steps_or_ratio

        self.save_hyperparameters()

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def training_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        input_ids, labels, attn_mask = batch

        # labels: [batch_size, seq_len], token_probs: [batch_size, seq_len, vocab_size]
        token_probs = self.model.forward(input_ids, attn_mask)
        token_probs = einops.rearrange(token_probs, "b n v -> (b n) v")
        labels = einops.rearrange(labels, "b n -> (b n)")
        loss = self.loss(token_probs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(self.warmup_steps_or_ratio, float):
            assert (
                0.0 < self.warmup_steps_or_ratio < 1.0
            ), "Warmup ratio should be in (0, 1)"
            warmup_steps = int(self.total_steps * self.warmup_steps_or_ratio)
        else:
            assert (
                0 < self.warmup_steps_or_ratio < self.total_steps
            ), "Warmup steps should be in (0, total_steps)"
            warmup_steps = self.warmup_steps_or_ratio

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @property
    def total_steps(self) -> int:
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps
        else:
            return int(self.trainer.estimated_stepping_batches)
