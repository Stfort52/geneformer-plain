import torch
from torch import LongTensor, Tensor

from . import BasePE


class SinusoidalPE(BasePE):
    def __init__(self, max_len: int, embed_size: int):
        super(SinusoidalPE, self).__init__()

        # precompute the sin and cos values
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2)
            * -(torch.log(torch.tensor(10000.0)) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        self.pe: Tensor

    def forward(self, x: LongTensor) -> Tensor:
        return self.pe[:, : x.size(1)]

    @property
    def max_len(self):
        return self.pe.size(1)
