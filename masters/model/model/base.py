import warnings
from typing import Any, Literal

from torch import LongTensor, Tensor, nn

from ..embedder import WordEmbedding
from ..transformer import Encoder
from ..utils import pe_from_name


class BertBase(nn.Module):
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
        absolute_pe_strategy: str | None = None,
        absolute_pe_kwargs: dict[str, Any] = {},
        relative_pe_strategy: str | None = None,
        relative_pe_kwargs: dict[str, Any] = {},
        relative_pe_shared: bool = False,
        act_fn: str = "relu",
    ):
        super(BertBase, self).__init__()
        self.absolute_pe_strategy = absolute_pe_strategy
        self.relative_pe_strategy = relative_pe_strategy
        self._check_args()

        self.embedder = WordEmbedding(n_vocab, d_model, dropout_p=ff_dropout)

        if absolute_pe_strategy is not None:
            absolute_pe_kwargs.setdefault("embed_size", d_model)
            self.absolute_pe = pe_from_name("absolute", absolute_pe_strategy)(
                **absolute_pe_kwargs
            )
        else:
            self.absolute_pe = None

        if relative_pe_strategy is not None and relative_pe_shared:
            assert d_model % num_heads == 0, ":("
            relative_pe_kwargs.setdefault("embed_size", d_model // num_heads)
            self.relative_pe = pe_from_name("relative", relative_pe_strategy)(
                **relative_pe_kwargs
            )
        else:
            self.relative_pe = None

        self.encoder = Encoder(
            d_model,
            num_heads,
            num_layers,
            d_ff,
            attn_dropout,
            ff_dropout,
            norm,
            relative_pe_strategy,
            relative_pe_kwargs,
            relative_pe_shared,
            act_fn,
        )

    def _check_args(self) -> None:
        if self.absolute_pe_strategy is None and self.relative_pe_strategy is None:
            warnings.warn("No positional encoding is used.")
        if (
            self.absolute_pe_strategy is not None
            and self.relative_pe_strategy is not None
        ):
            warnings.warn("Both positional encoding strategies are used.")

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        if self.absolute_pe is not None:
            x = self.embedder(x) + self.absolute_pe(x)
        else:
            x = self.embedder(x)

        if self.relative_pe is not None:
            return self.encoder(x, mask, self.relative_pe(x))
        else:
            return self.encoder(x, mask)
