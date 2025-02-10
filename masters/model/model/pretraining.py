from typing import Any, Literal

from torch import LongTensor, Tensor, nn

from ..unembedder import LanguageModeling
from ..utils import reset_weights
from . import BertBase


class BertPretraining(nn.Module):
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
    ):
        super(BertPretraining, self).__init__()

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

        self.lm = LanguageModeling(d_model, n_vocab)
        self.lm.tie_weights(self.bert.embedder.embed)

    def reset_weights(
        self, initialization_range: float = 0.02, reset_all: bool = True
    ) -> None:
        reset_weights(self.lm, initialization_range)
        if reset_all:
            reset_weights(self.bert, initialization_range)

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        return self.lm(self.bert(x, mask))
