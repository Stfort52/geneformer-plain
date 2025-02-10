from typing import Literal

import torch
from torch import LongTensor

from . import Cell


class NerDataCollator:
    def __init__(
        self,
        token_dict: dict[str, int],
        pad_token_or_index: str | int,
        ignore_index: int = -100,
        max_length: int | Literal["longest"] = 2_048,
    ):
        self.max_length = max_length
        self.ignore_index = ignore_index

        if isinstance(pad_token_or_index, str):
            self.pad_index = token_dict[pad_token_or_index]
        else:
            self.pad_index = pad_token_or_index

    def __call__(self, batch: list[Cell]) -> tuple[LongTensor, LongTensor, LongTensor]:
        input_ids = [example["input_ids"] for example in batch]
        lengths = torch.tensor([example["length"] for example in batch])
        gene_labels = [
            example["gene_labels"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            for example in batch
        ]

        if self.max_length == "longest":
            max_length = int(lengths.max().item())
        else:
            max_length = int(self.max_length)

        # pad the inputs to the max length
        inputs = input_ids[0].new_full((len(input_ids), max_length), self.pad_index)
        labels = torch.full_like(inputs, self.ignore_index)
        padding_mask = torch.zeros_like(inputs)

        for i, (input_id, length, gene_label) in enumerate(
            zip(input_ids, lengths, gene_labels)
        ):
            inputs[i, :length] = input_id[:length]
            labels[i, :length] = gene_label[:length]
            padding_mask[i, :length] = 1

        return inputs, labels, padding_mask  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_length={self.max_length}, "
            f"pad_index={self.pad_index}, "
            f"ignore_index={self.ignore_index})"
        )
