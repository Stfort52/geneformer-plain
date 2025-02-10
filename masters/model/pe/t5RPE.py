import torch
from torch import LongTensor, Tensor, nn
from transformers.models.t5.modeling_t5 import T5Attention

from . import BaseRPE


class T5RPE(BaseRPE):
    coupled = False
    shape = "h i j"

    def __init__(self, num_heads: int, num_buckets: int, max_distance: int, **kwargs):
        super(T5RPE, self).__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, x: LongTensor) -> Tensor:
        dist = torch.arange(x.size(1), device=x.device)
        dist = dist.unsqueeze(0) - dist.unsqueeze(1)

        buckets = T5Attention._relative_position_bucket(
            dist, True, self.num_buckets, self.max_distance
        )

        return self.bias(buckets).permute(2, 0, 1)  # [i j h] -> [h i j]

    @property
    def max_len(self) -> None:
        return None
