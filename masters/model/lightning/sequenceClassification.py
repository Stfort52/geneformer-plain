from pathlib import Path
from typing import cast

import lightning as L
from torch import LongTensor, nn, optim
from transformers import get_scheduler

from ..model import BertConfig, BertSequenceClassification
from ..utils import continuous_metrics, threshold_metrics
from . import LightningPretraining


class LightningSequenceClassification(L.LightningModule):
    def __init__(
        self,
        model_path_or_config: str | Path | BertConfig,
        n_classes: int,
        cls_dropout: float = 0.0,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        initialization_range: float = 0.02,
        lr_scheduler: str = "cosine",
        warmup_steps_or_ratio: int | float = 0.1,
        freeze_first_n_layers: int = 0,
    ):
        super(LightningSequenceClassification, self).__init__()

        if isinstance(model_path_or_config, (str, Path)):
            pretrained = LightningPretraining.load_from_checkpoint(model_path_or_config)
            base_model = pretrained.model.bert
            config = cast(BertConfig, pretrained.hparams["config"])
            self.model = BertSequenceClassification(
                **config, n_classes=n_classes, cls_dropout=cls_dropout
            )
            self.model.bert = base_model
        else:
            config = model_path_or_config
            self.model = BertSequenceClassification(
                **config, n_classes=n_classes, cls_dropout=cls_dropout
            )

        self.model.reset_weights(initialization_range)

        if freeze_first_n_layers > 0:
            assert (
                freeze_first_n_layers < config["num_layers"]
            ), "Number of layers to freeze should be less than total number of layers"

            for i in range(freeze_first_n_layers):
                for param in self.model.bert.encoder.layers[i].parameters():
                    param.requires_grad = False

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_steps_or_ratio = warmup_steps_or_ratio

        self.save_hyperparameters()

        self.loss = nn.CrossEntropyLoss()
        self.threshold_metrics = threshold_metrics(num_classes=n_classes)
        self.continueous_metrics = continuous_metrics(num_classes=n_classes)

    def training_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self.model(inputs, mask=padding_mask)
        loss = self.loss(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self.model(inputs, mask=padding_mask)
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
