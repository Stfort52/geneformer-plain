from typing import Literal, cast

import torch
from torch import LongTensor

from . import Cell


class MlmDataCollator:
    def __init__(
        self,
        token_dict: dict[str, int],
        pad_token_or_index: str | int,
        mask_token_or_index: str | int,
        mask_prob: float = 0.15,
        ignore_index: int = -100,
        max_length: int | Literal["longest"] = 2_048,
    ):
        self.token_dict = token_dict
        self.max_length = max_length
        self.vocab_size = len(token_dict)
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

        if isinstance(pad_token_or_index, str):
            self.pad_index = token_dict[pad_token_or_index]
        else:
            self.pad_index = pad_token_or_index

        if isinstance(mask_token_or_index, str):
            self.mask_index = token_dict[mask_token_or_index]
        else:
            self.mask_index = mask_token_or_index

        self.word_pool = torch.tensor(
            list(set(token_dict.values()) - {self.pad_index, self.mask_index}),
            dtype=torch.long,
        )

    def __call__(self, batch: list[Cell]) -> tuple[LongTensor, LongTensor, LongTensor]:

        input_ids = [example["input_ids"] for example in batch]
        lengths = torch.tensor([example["length"] for example in batch])

        if self.max_length == "longest":
            max_length = int(lengths.max().item())
        else:
            max_length = cast(int, self.max_length)

        # pad the inputs to the max length
        inputs = input_ids[0].new_full((len(input_ids), max_length), self.pad_index)
        labels = torch.full_like(inputs, self.ignore_index)
        padding_mask = torch.zeros_like(inputs)

        for i, (input_id, length) in enumerate(zip(input_ids, lengths)):
            inputs[i, :length] = input_id[:length]
            labels[i, :length] = input_id[:length]
            padding_mask[i, :length] = 1

        # mask `mask_prob` of the tokens
        should_mask = torch.bernoulli(torch.full(inputs.shape, self.mask_prob)).bool()
        # change 80% of the tokens to mask tokens
        should_replace = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool()
        indices_replaced = should_mask & should_replace
        # change 50% of the remaining tokens, 10% in total, to random tokens
        should_randomize = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
        indices_randomized = should_mask & ~should_replace & should_randomize
        random_words = self.word_pool[torch.randint_like(inputs, self.vocab_size - 2)]

        inputs[indices_replaced] = self.mask_index
        inputs[indices_randomized] = random_words[indices_randomized]
        labels[~should_mask] = self.ignore_index

        return inputs, labels, padding_mask  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_length={self.max_length}, "
            f"pad_index={self.pad_index}, "
            f"mask_index={self.mask_index}, "
            f"mask_prob={self.mask_prob}, "
            f"ignore_index={self.ignore_index})"
        )
