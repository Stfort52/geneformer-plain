from torch import LongTensor, Tensor, nn

from ..unembedder import LanguageModeling
from ..utils import reset_weights
from . import BertBase, BertConfig


class BertPretraining(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config
        self.bert = BertBase(**config)

        self.lm = LanguageModeling(
            config.d_model, config.n_vocab, config.ln_eps, config.act_fn
        )
        self.lm.tie_weights(self.bert.embedder.embed)

    def reset_weights(
        self, initialization_range: float | None = None, reset_all: bool = True
    ) -> None:
        if initialization_range is None:
            initialization_range = self.config.initialization_range
        reset_weights(self.lm, initialization_range)
        if reset_all:
            reset_weights(self.bert, initialization_range)

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        return self.lm(self.bert(x, mask))
