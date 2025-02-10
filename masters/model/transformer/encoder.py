from typing import Literal

from torch import LongTensor, Tensor, nn

from . import Block


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        intermidiate_size: int,
        attn_dropout: float,
        ff_dropout: float,
        norm: Literal["pre", "post"],
        relative_pe: str | None,
        relative_pe_kwargs: dict,
        relative_pe_shared: bool,
        act_fn: str,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermidiate_size = intermidiate_size
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.layers = nn.ModuleList(
            Block(
                embed_size=d_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                intermidiate_size=intermidiate_size,
                ff_dropout=ff_dropout,
                norm=norm,
                relative_pe=relative_pe,
                relative_pe_kwargs=relative_pe_kwargs,
                relative_pe_shared=relative_pe_shared,
                act_fn=act_fn,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        mask: LongTensor | None = None,
        rpe_mtx: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask, rpe_mtx)

        return x
