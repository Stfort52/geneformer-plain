import torch
from torch import LongTensor, Tensor, nn

from . import BaseRPE


class TrainedRPE(BaseRPE):
    coupled = True
    shape = "i j d"

    def __init__(self, max_len: int, embed_size: int):
        super(TrainedRPE, self).__init__()
        self._max_len = max_len
        self.pe = nn.Embedding(2 * max_len - 1, embed_size)
        self.distances: LongTensor

        indices = torch.arange(max_len)
        distances = indices.unsqueeze(0) - indices.unsqueeze(1)
        self.register_buffer("distances", distances + max_len - 1)

    def forward(self, x: LongTensor) -> Tensor:
        return self.pe(self.distances[: x.size(1), : x.size(1)])

    @property
    def max_len(self) -> int:
        return self._max_len
