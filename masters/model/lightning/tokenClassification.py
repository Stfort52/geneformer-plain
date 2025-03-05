from pathlib import Path

import einops
import lightning as L
from torch import LongTensor, nn, optim
from transformers import get_scheduler

from ..model import BertConfig, BertTokenClassification
from ..utils import continuous_metrics, threshold_metrics
from . import LightningPretraining


class LightningTokenClassification(L.LightningModule):
    def __init__(
        self,
        model_path_or_config: str | Path | BertConfig,
        n_classes: int,
        cls_dropout: float | None = None,
        ignore_index: int = -100,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        warmup_steps_or_ratio: int | float = 0.1,
        freeze_first_n_layers: int = 0,
        **_,  # log additional arguments as needed
    ):
        super().__init__()

        match model_path_or_config:
            case str() | Path():
                pretrained = LightningPretraining.load_from_checkpoint(
                    model_path_or_config
                )
                self.config = pretrained.model.config
                self.model_path = model_path_or_config
            case BertConfig():
                pretrained = None
                self.config = model_path_or_config
                self.model_path = None

        self.config.n_classes = n_classes
        if cls_dropout is not None:
            self.config.cls_dropout = cls_dropout

        self.model = BertTokenClassification(self.config)
        if pretrained is not None:
            self.model.bert.load_state_dict(pretrained.model.bert.state_dict())

        if freeze_first_n_layers > 0:
            assert (
                freeze_first_n_layers < self.config["num_layers"]
            ), "Number of layers to freeze should be less than total number of layers"

            for i in range(freeze_first_n_layers):
                for param in self.model.bert.encoder.layers[i].parameters():
                    param.requires_grad = False

        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.lr_scheduler = lr_scheduler
        self.warmup_steps_or_ratio = warmup_steps_or_ratio

        self.save_hyperparameters(
            {
                "model_path": self.model_path,
                "config": self.config.asdict(),
                "lr": lr,
                "weight_decay": weight_decay,
                "lr_scheduler": lr_scheduler,
                "warmup_steps_or_ratio": warmup_steps_or_ratio,
                "freeze_first_n_layers": freeze_first_n_layers,
                **_,
            }
        )

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.threshold_metrics = threshold_metrics(
            num_classes=self.config.n_classes, ignore_index=ignore_index
        )
        self.continueous_metrics = continuous_metrics(
            num_classes=self.config.n_classes, ignore_index=ignore_index
        )

    def training_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self.model(inputs, mask=padding_mask)
        logits = einops.rearrange(logits, "b n v -> (b n) v")
        loss = self.loss(logits, labels.flatten())

        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self.model(inputs, mask=padding_mask)
        logits = einops.rearrange(logits, "b n v -> (b n) v")
        labels = labels.flatten()
        loss = self.loss(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

        probablities = nn.functional.softmax(logits, dim=-1)
        if probablities.size(-1) == 2:
            probablities = probablities[:, 1]

        predictions = logits.argmax(dim=-1)

        self.log_dict(self.threshold_metrics(predictions, labels))
        self.log_dict(self.continueous_metrics(probablities, labels))

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
