from typing import Literal

from torch import LongTensor, Tensor, nn

from ..utils import activation_from_name
from . import MHA


class Block(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        attn_dropout: float,
        intermidiate_size: int,
        ff_dropout: float,
        norm: Literal["pre", "post"],
        relative_pe: str | None = None,
        relative_pe_kwargs: dict = {},
        relative_pe_shared: bool = False,
        ln_eps: float = 1e-12,
        act_fn: str = "relu",
    ):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.intermidiate_size = intermidiate_size
        self.norm = norm
        self.act_fn = activation_from_name(act_fn)

        self.attn = MHA(
            embed_size,
            num_heads,
            attn_dropout,
            ff_dropout,
            relative_pe,
            relative_pe_kwargs,
            relative_pe_shared,
        )
        self.ln1 = nn.LayerNorm(embed_size, ln_eps)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, intermidiate_size),
            self.act_fn(),
            nn.Linear(intermidiate_size, embed_size),
            nn.Dropout(ff_dropout),
        )
        self.ln2 = nn.LayerNorm(embed_size, ln_eps)

    def forward(
        self,
        x: Tensor,
        mask: LongTensor | None = None,
        rpe_mtx: Tensor | None = None,
    ) -> Tensor:
        if self.norm == "pre":
            x = x + self.attn(self.ln1(x), mask, rpe_mtx)
            x = x + self.ff(self.ln2(x))
        elif self.norm == "post":
            x = self.ln1(x + self.attn(x, mask, rpe_mtx))
            x = self.ln2(x + self.ff(x))
        else:
            raise ValueError(":(")

        return x
