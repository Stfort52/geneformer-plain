from torch import Tensor, nn


class TokenClassification(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout_p: float = 0.0):
        super(TokenClassification, self).__init__()

        self.dense = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(self.dropout(x))
