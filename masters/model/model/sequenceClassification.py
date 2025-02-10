import warnings
from typing import Any, Literal

from torch import LongTensor, Tensor, nn

from ..unembedder import Pooling, SequenceClassification
from ..utils import reset_weights
from . import BertBase


class BertSequenceClassification(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        attn_dropout: float,
        ff_dropout: float,
        norm: Literal["pre", "post"],
        absolute_pe_strategy: str | None,
        absolute_pe_kwargs: dict[str, Any],
        relative_pe_strategy: str | None,
        relative_pe_kwargs: dict[str, Any],
        relative_pe_shared: bool,
        act_fn: str,
        n_classes: int,
        cls_dropout: float,
    ):
        super(BertSequenceClassification, self).__init__()

        self.bert = BertBase(
            n_vocab,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            attn_dropout,
            ff_dropout,
            norm,
            absolute_pe_strategy,
            absolute_pe_kwargs,
            relative_pe_strategy,
            relative_pe_kwargs,
            relative_pe_shared,
            act_fn,
        )

        self.pool = Pooling("mean")
        self.cls = SequenceClassification(d_model, n_classes, cls_dropout)

    def reset_weights(
        self, initialization_range: float = 0.02, reset_all: bool = False
    ) -> None:
        if reset_all:
            reset_weights(self.bert, initialization_range)
            warnings.warn("Resetting entire model including pretrained weights")
        reset_weights(self.cls, initialization_range)

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        return self.cls(self.pool(self.bert(x, mask)))
