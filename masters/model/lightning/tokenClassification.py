from pathlib import Path
from typing import Any, Self, cast

import einops
import lightning as L
from torch import LongTensor, Tensor, nn, optim
from transformers import BertConfig as HFBertConfig
from transformers import BertForTokenClassification, get_scheduler

from ..model import BertConfig, BertTokenClassification
from ..utils import continuous_metrics, threshold_metrics
from . import LightningPretraining


class LightningTokenClassification(L.LightningModule):
    def __init__(
        self,
        config: BertConfig | HFBertConfig | dict[str, Any],
        n_classes: int,
        cls_dropout: float = 0.0,
        ignore_index: int = -100,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        warmup_steps_or_ratio: int | float = 0.1,
        freeze_first_n_layers: int | None = None,
        **_,  # log additional arguments as needed
    ):
        super().__init__()

        if isinstance(config, dict):
            if "model_type" in config:
                config = HFBertConfig(**config)
            else:
                config = BertConfig(**config)

        match config:
            case HFBertConfig():
                config.num_labels = n_classes
                config.classifier_dropout = cls_dropout
                self.model = BertForTokenClassification(config)
                self.forward = self.hf_forward
                self.is_hf = True
            case BertConfig():
                config.n_classes = n_classes
                config.cls_dropout = cls_dropout
                self.model = BertTokenClassification(config)
                self.forward = self.native_forward
                self.is_hf = False
            case _:
                raise ValueError("Configuration not recognized")

        if freeze_first_n_layers is not None:
            self.freeze_layers(freeze_first_n_layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.lr_scheduler = lr_scheduler
        self.warmup_steps_or_ratio = warmup_steps_or_ratio
        self.model_path = None

        self.save_hyperparameters(
            {
                "model_path": self.model_path,
                "config": self.model.config.to_dict(),
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
            num_classes=n_classes, ignore_index=ignore_index
        )
        self.continueous_metrics = continuous_metrics(
            num_classes=n_classes, ignore_index=ignore_index
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path, n_classes: int, **kwargs) -> Self:
        pretrained = LightningPretraining.load_from_checkpoint(model_path)
        config = pretrained.model.config
        model = cls(config, n_classes, **kwargs)
        model.model.bert.load_state_dict(pretrained.model.bert.state_dict())
        # should do model.model.reset_weights()?
        model.model_path = str(model_path)
        return model

    @classmethod
    def from_hf_model(cls, model_path: str | Path, n_classes: int, **kwargs) -> Self:
        pretrained = BertForTokenClassification.from_pretrained(
            model_path, num_labels=n_classes
        )
        config = cast(HFBertConfig, pretrained.config)
        model = cls(config, n_classes, **kwargs)
        model.model.load_state_dict(pretrained.state_dict())
        model.model_path = str(model_path)
        return model

    def native_forward(self, inputs: LongTensor, mask: LongTensor) -> Tensor:
        return self.model(inputs, mask)

    def hf_forward(self, inputs: LongTensor, mask: LongTensor) -> Tensor:
        return self.model(inputs, mask).logits

    def freeze_layers(self, n_layers: int):
        if isinstance(self.model, BertTokenClassification):
            for param in self.model.bert.embedder.parameters():
                param.requires_grad = False
            for param in self.model.bert.encoder.layers[:n_layers].parameters():
                param.requires_grad = False
        else:
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            for param in self.model.bert.encoder.layer[:n_layers].parameters():
                param.requires_grad = False

    def training_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self(inputs, mask=padding_mask)
        logits = einops.rearrange(logits, "b n c -> (b n) c")
        loss = self.loss(logits, labels.flatten())

        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[LongTensor, LongTensor, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self(inputs, mask=padding_mask)
        logits = einops.rearrange(logits, "b n c -> (b n) c")
        labels = labels.flatten()
        loss = self.loss(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

        probabilities = nn.functional.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        if probabilities.size(-1) == 2:
            probabilities = probabilities[:, 1]

        self.log_dict(self.threshold_metrics(predictions, labels))
        self.log_dict(self.continueous_metrics(probabilities, labels))

        return loss

    def predict_step(self, batch: tuple[LongTensor, LongTensor | None, LongTensor], _):
        inputs, labels, padding_mask = batch
        logits = self(inputs, mask=padding_mask)

        if labels is not None and labels.numel() > 0:
            valid_idx = labels != self.ignore_index
            logits = logits[valid_idx]
            labels = labels[valid_idx]

        probabilities = nn.functional.softmax(logits, dim=-1)
        if probabilities.size(-1) == 2:
            probabilities = probabilities[:, 1]

        predictions = logits.argmax(dim=-1)

        return probabilities, predictions, labels

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
