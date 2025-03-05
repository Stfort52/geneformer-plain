from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Self


@dataclass
class BertConfig:
    n_vocab: int = 25_426
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    d_ff: int = 512
    attn_dropout: float = 0.02
    ff_dropout: float = 0.02
    norm: Literal["pre", "post"] = "post"
    initialization_range: float = 0.02
    ln_eps: float = 1e-12
    absolute_pe_strategy: str | None = "trained"
    absolute_pe_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"max_len": 2048}
    )
    relative_pe_strategy: str | None = None
    relative_pe_kwargs: dict[str, Any] = field(default_factory=dict)
    relative_pe_shared: bool = False
    act_fn: str = "relu"
    n_classes: int = 2
    cls_dropout: float = 0.0

    def keys(self):
        return asdict(self).keys()

    def __getitem__(self, key):
        return asdict(self)[key]

    @classmethod
    def from_setting(cls, setting: str) -> Self:
        match setting:
            # v1: CV Theodoris, Transfer learning enables predictions in network biology
            case "base" | "v1" | "v1-base":
                return cls()
            case "large" | "v1-large":
                return cls(
                    d_model=512,
                    num_heads=8,
                    num_layers=12,
                    d_ff=1024,
                    absolute_pe_kwargs={"max_len": 2048},
                )
            # v2: H Chen, Quantized multi-task learning for context-specific representations of gene network dynamics
            case "v2" | "v2-base":
                return cls(
                    n_vocab=20_275,
                    d_model=512,
                    num_heads=8,
                    num_layers=12,
                    d_ff=1024,
                    absolute_pe_kwargs={"max_len": 4096},
                )
            case "v2-large":
                return cls(
                    n_vocab=20_275,
                    d_model=896,
                    num_heads=14,
                    num_layers=20,
                    d_ff=1792,
                    absolute_pe_kwargs={"max_len": 4096},
                )
            case _:
                raise ValueError(f"Invalid setup: {setting}")
