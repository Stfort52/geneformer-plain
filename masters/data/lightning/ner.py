from pathlib import Path
from typing import Any, cast

import datasets
import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils import Cell, NerDataCollator


class NerDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str | Path,
        token_dict: dict[str, int],
        entity_labels: pd.Series,
        train_cell_count_or_ratio: int | float = 10_000,
        test_cell_count_or_ratio: int | float = 2_000,
        test_gene_ratio: float = 0.2,
        ignore_index: int = -100,
        batch_size: int = 32,
        dataset_shuffle: int | bool = 42,
        label_shuffle: int | bool = 42,
        num_workers: int = 16,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.token_dict = token_dict
        self.ignore_index = ignore_index
        self.entity_labels = entity_labels
        self.train_cell_count_or_ratio = train_cell_count_or_ratio
        self.test_cell_count_or_ratio = test_cell_count_or_ratio
        self.test_gene_Ratio = test_gene_ratio
        self.batch_size = batch_size
        self.dataset_shuffle = dataset_shuffle
        self.label_shuffle = label_shuffle
        self.num_workers = num_workers

        self.train_labels: pd.Series
        self.test_labels: pd.Series

    def prepare_data(self) -> None:
        self.dataset = cast(datasets.Dataset, datasets.load_from_disk(self.dataset_dir))

        # shuffle the cells
        match self.dataset_shuffle:
            case True:
                self.dataset = self.dataset.shuffle(seed=None)
            case False:
                pass
            case int():
                self.dataset = self.dataset.shuffle(seed=self.dataset_shuffle)

        # shuffle the genes
        match self.label_shuffle:
            case bool():
                self.train_labels, self.test_labels = train_test_split(
                    self.entity_labels,
                    test_size=self.test_gene_Ratio,
                    shuffle=self.label_shuffle,
                    random_state=None,
                    stratify=self.entity_labels.to_list(),
                )
            case int():
                self.train_labels, self.test_labels = train_test_split(
                    self.entity_labels,
                    test_size=self.test_gene_Ratio,
                    shuffle=True,
                    random_state=self.label_shuffle,
                    stratify=self.entity_labels.to_list(),
                )

        # pick cells with at least one gene label
        train_label_list = self.train_labels.index.to_list()
        self.train_dataset = self.dataset.filter(
            lambda cell: not set(cell["input_ids"]).isdisjoint(train_label_list),
            num_proc=self.num_workers,
        )
        test_label_list = self.test_labels.index.to_list()
        self.test_dataset = self.dataset.filter(
            lambda cell: not set(cell["input_ids"]).isdisjoint(test_label_list),
            num_proc=self.num_workers,
        )

        # cut the dataset
        if isinstance(self.train_cell_count_or_ratio, int):
            n_train_cells = self.train_cell_count_or_ratio
            if n_train_cells > len(self.train_dataset):
                self.print(
                    f"Requested {n_train_cells} train cells, "
                    f"but only {len(self.train_dataset)} are available. "
                    "Using all available cells."
                )
                n_train_cells = len(self.train_dataset)
        else:
            n_train_cells = int(
                len(self.train_dataset) * self.train_cell_count_or_ratio
            )

        if isinstance(self.test_cell_count_or_ratio, int):
            n_test_cells = self.test_cell_count_or_ratio
            if n_test_cells > len(self.test_dataset):
                self.print(
                    f"Requested {n_test_cells} test cells, "
                    f"but only {len(self.test_dataset)} are available. "
                    "Using all available cells."
                )
                n_test_cells = len(self.test_dataset)
        else:
            n_test_cells = int(len(self.test_dataset) * self.test_cell_count_or_ratio)

        self.train_dataset = self.train_dataset.select(range(n_train_cells))
        self.test_dataset = self.test_dataset.select(range(n_test_cells))

        # annotate the gene labels
        self.train_dataset = self.train_dataset.map(
            self._add_labels,
            num_proc=self.num_workers,
            fn_kwargs={
                "label_map": self.train_labels,
                "ignore_index": self.ignore_index,
            },
        )
        self.test_dataset = self.test_dataset.map(
            self._add_labels,
            num_proc=self.num_workers,
            fn_kwargs={
                "label_map": self.test_labels,
                "ignore_index": self.ignore_index,
            },
        )

        # set the format
        self.train_dataset.set_format(
            type="torch", columns=["input_ids", "gene_labels"], output_all_columns=True
        )
        self.test_dataset.set_format(
            type="torch", columns=["input_ids", "gene_labels"], output_all_columns=True
        )

        self.collator = NerDataCollator(
            token_dict=self.token_dict,
            pad_token_or_index="<pad>",
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self):
        assert self.train_dataset is not None

        return DataLoader(
            self.train_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self.collator,  # pyright: ignore[reportArgumentType]
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        assert self.test_dataset is not None

        return DataLoader(
            self.test_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self.collator,  # pyright: ignore[reportArgumentType]
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _add_labels(
        examples: dict[str, Any], label_map: dict[str, int], ignore_index: int
    ) -> dict[str, Any]:
        examples["gene_labels"] = [
            int(label_map.get(label, ignore_index)) for label in examples["input_ids"]
        ]
        return examples

    def print(self, *args, **kwargs):
        if self.trainer is not None:
            self.trainer.print(*args, **kwargs)
