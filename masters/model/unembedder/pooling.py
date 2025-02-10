from typing import Literal

from torch import Tensor, nn


class Pooling(nn.Module):
    def __init__(self, strategy: Literal["mean", "first", "max"]):
        super(Pooling, self).__init__()
        self.strategy = strategy

    def forward(self, x: Tensor) -> Tensor:
        # x: [b, n, d]
        match self.strategy:
            case "mean":
                return x.mean(dim=1)
            case "first":
                return x[:, 0]
            case "max":
                return x.max(dim=1).values
            case _:
                raise ValueError(f":(")
