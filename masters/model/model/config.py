from typing import Any, Literal, TypedDict


class BertConfig(TypedDict):
    n_vocab: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    attn_dropout: float
    ff_dropout: float
    norm: Literal["pre", "post"]
    absolute_pe_strategy: str | None
    absolute_pe_kwargs: dict[str, Any]
    relative_pe_strategy: str | None
    relative_pe_kwargs: dict[str, Any]
    relative_pe_shared: bool
    act_fn: str
