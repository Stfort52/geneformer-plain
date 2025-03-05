import warnings

from torch import LongTensor, Tensor, nn

from ..unembedder import TokenClassification
from ..utils import reset_weights
from . import BertBase, BertConfig


class BertTokenClassification(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config
        self.bert = BertBase(**config)

        self.cls = TokenClassification(
            config.d_model, config.n_classes, config.cls_dropout
        )

    def reset_weights(
        self, initialization_range: float | None = None, reset_all: bool = False
    ) -> None:
        if initialization_range is None:
            initialization_range = self.config.initialization_range
        if reset_all:
            reset_weights(self.bert, initialization_range)
            warnings.warn("Resetting entire model including pretrained weights")
        reset_weights(self.cls, initialization_range)

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        return self.cls(self.bert(x, mask))
