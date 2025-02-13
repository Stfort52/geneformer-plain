from pathlib import Path
from typing import Any, cast

import lightning as L
import pandas as pd
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from ..utils import NerDataCollator


class NerDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str | Path,
        token_dict: dict[str, int],
        train_gene_labels: pd.Series,
        test_gene_labels: pd.Series,
        train_cell_count_or_ratio: int | float = 10_000,
        test_cell_count_or_ratio: int | float = 2_000,
        ignore_index: int = -100,
        batch_size: int = 32,
        dataset_shuffle: int | bool = 42,
        num_workers: int = 16,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.token_dict = token_dict
        self.train_gene_labels = train_gene_labels
        self.test_gene_labels = test_gene_labels
        self.train_cell_count_or_ratio = train_cell_count_or_ratio
        self.test_cell_count_or_ratio = test_cell_count_or_ratio
        self.ignore_index = ignore_index
        self.batch_size = batch_size
        self.dataset_shuffle = dataset_shuffle
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.dataset = cast(Dataset, load_from_disk(self.dataset_dir))
        # shuffle the cells
        match self.dataset_shuffle:
            case True:
                self.dataset = self.dataset.shuffle(seed=None)
            case False:
                pass
            case int():
                self.dataset = self.dataset.shuffle(seed=self.dataset_shuffle)

        self.collator = NerDataCollator(
            token_dict=self.token_dict,
            pad_token_or_index="<pad>",
            ignore_index=self.ignore_index,
        )

        if stage == "fit":
            self.train_dataset = self._prepare_split(
                gene_labels=self.train_gene_labels,
                cell_count_or_ratio=self.train_cell_count_or_ratio,
            )
        if stage in ("fit", "validate", "test"):
            self.test_dataset = self._prepare_split(
                gene_labels=self.test_gene_labels,
                cell_count_or_ratio=self.test_cell_count_or_ratio,
            )

    def _prepare_split(
        self, gene_labels: pd.Series, cell_count_or_ratio: int | float
    ) -> Dataset:
        # pick cells with at least one gene label
        label_list = gene_labels.index.to_list()
        dataset = self.dataset.filter(
            lambda cell: not set(cell["input_ids"]).isdisjoint(label_list),
            num_proc=self.num_workers,
        )

        # cut the dataset
        match cell_count_or_ratio:
            case int():
                n_cells = cell_count_or_ratio
                if n_cells > len(dataset):
                    self.print(
                        f"Requested {n_cells} cells, "
                        f"but only {len(dataset)} are available. "
                        "Using all available cells."
                    )
                    n_cells = len(dataset)
            case float():
                n_cells = int(len(dataset) * cell_count_or_ratio)
        dataset = dataset.select(range(n_cells))

        # annotate the gene labels
        dataset = dataset.map(
            self._add_labels,
            num_proc=self.num_workers,
            fn_kwargs={
                "label_map": gene_labels,
                "ignore_index": self.ignore_index,
            },
        )

        # set the format
        dataset.set_format(
            type="torch",
            columns=["input_ids", "gene_labels"],
            output_all_columns=True,
        )

        return dataset

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
