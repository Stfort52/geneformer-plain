from torch import Tensor, nn

from ..utils import activation_from_name


class LanguageModeling(nn.Module):
    def __init__(
        self, d_model: int, n_vocab: int, ln_eps: float = 1e-12, act_fn: str = "relu"
    ):
        super(LanguageModeling, self).__init__()

        self.dense = nn.Sequential(
            nn.Linear(d_model, d_model),
            activation_from_name(act_fn)(),
            nn.LayerNorm(d_model, ln_eps),
        )
        self.unembedding = nn.Linear(d_model, n_vocab, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: [b, n, d]
        return self.unembedding(self.dense(x))

    def tie_weights(self, embeddings: nn.Embedding):
        self.unembedding.weight = embeddings.weight
