import torch
from torch import LongTensor, Tensor, nn

from .basePE import BasePE


class TrainedPE(BasePE):
    def __init__(self, max_len: int, embed_size: int):
        super(TrainedPE, self).__init__()
        self.pe = nn.Embedding(max_len, embed_size)

    def forward(self, x: LongTensor) -> Tensor:
        return self.pe(torch.arange(x.size(1), device=x.device))

    @property
    def max_len(self) -> int:
        return self.pe.num_embeddings
