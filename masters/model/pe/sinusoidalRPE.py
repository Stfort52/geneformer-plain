import torch
from torch import LongTensor, Tensor

from . import BaseRPE


class SinusoidalRPE(BaseRPE):
    coupled = True
    shape = "i j d"

    def __init__(self, max_len: int, embed_size: int):
        super(SinusoidalRPE, self).__init__()
        self._max_len = max_len

        # precompute the sin and cos values
        pe = torch.zeros(2 * max_len - 1, embed_size)
        position = torch.arange(0, 2 * max_len - 1).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2)
            * -(torch.log(torch.tensor(10000.0)) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        indices = torch.arange(max_len)
        distances = indices.unsqueeze(0) - indices.unsqueeze(1)
        self.register_buffer("pe", pe[distances + max_len - 1])

    def forward(self, x: LongTensor) -> Tensor:
        return self.pe[: x.size(1), : x.size(1)]

    @property
    def max_len(self):
        return self._max_len
