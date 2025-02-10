from pathlib import Path

import gensim
import torch
from torch import Tensor


def load_gensim_model_or_kv(
    model_path: str,
    token_dict: dict[str, int] | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    model = gensim.utils.SaveLoad.load(model_path)

    if isinstance(model, gensim.models.KeyedVectors):
        pass
    elif isinstance(model, gensim.models.Word2Vec):
        model = model.wv
    else:
        raise ValueError("Unsupported model type")

    if token_dict is None:
        return torch.tensor(model.vectors, dtype=dtype)

    # map the token_dict to the model's vocabulary
    indices = list(map(model.key_to_index.get, token_dict.keys()))
    assert all(index is not None for index in indices), "Some tokens are missing"
    return torch.tensor(
        model.vectors[indices],  # pyright: ignore[reportCallIssue, reportArgumentType]
        dtype=dtype,
    )
