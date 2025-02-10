from typing import NotRequired, TypedDict

from torch import LongTensor


class Cell(TypedDict):
    input_ids: LongTensor
    length: int
    gene_labels: NotRequired[LongTensor]
    cell_label: NotRequired[int]
