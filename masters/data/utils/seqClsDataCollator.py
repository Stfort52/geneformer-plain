from typing import Literal, cast

import torch
from torch import LongTensor

from . import Cell


class SeqClsDataCollator:
    def __init__(
        self,
        token_dict: dict[str, int],
        pad_token_or_index: str | int,
        max_length: int | Literal["longest"] = 2_048,
    ):
        self.token_dict = token_dict
        self.max_length = max_length
        self.vocab_size = len(token_dict)

        if isinstance(pad_token_or_index, str):
            self.pad_index = token_dict[pad_token_or_index]
        else:
            self.pad_index = pad_token_or_index

    def __call__(self, batch: list[Cell]) -> tuple[LongTensor, LongTensor, LongTensor]:

        input_ids = [example["input_ids"] for example in batch]
        labels = torch.tensor(
            [example["cell_label"] for example in batch]  # pyright: ignore
        )
        lengths = torch.tensor([example["length"] for example in batch])

        if self.max_length == "longest":
            max_length = int(lengths.max().item())
        else:
            max_length = cast(int, self.max_length)

        # pad the inputs to the max length
        inputs = input_ids[0].new_full((len(input_ids), max_length), self.pad_index)
        padding_mask = torch.zeros_like(inputs)

        for i, (input_id, length) in enumerate(zip(input_ids, lengths)):
            inputs[i, :length] = input_id[:length]
            padding_mask[i, :length] = 1

        return inputs, labels, padding_mask  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_length={self.max_length}, "
            f"pad_index={self.pad_index})"
        )
