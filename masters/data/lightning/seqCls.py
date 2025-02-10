from pathlib import Path
from typing import cast

import datasets
import lightning as L
from torch.utils.data import DataLoader

from ..utils import SeqClsDataCollator


class SeqClsDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str | Path,
        token_dict: dict[str, int],
        batch_size: int = 32,
        test_cell_ratio: float = 0.2,
        cell_label_column: str = "cell_type",
        shuffle: int | bool = 42,
        num_workers: int = 16,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.token_dict = token_dict
        self.batch_size = batch_size
        self.test_cell_ratio = test_cell_ratio
        self.cell_type_column = cell_label_column
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.dataset = cast(datasets.Dataset, datasets.load_from_disk(self.dataset_dir))
        self.dataset = self.dataset.rename_column(self.cell_type_column, "cell_label")
        self.dataset.set_format(
            type="torch", columns=["input_ids"], output_all_columns=True
        )

        if not isinstance(self.dataset.features["cell_label"], datasets.ClassLabel):
            self.print(
                f"label column {self.cell_type_column} is not a ClassLabel\n"
                f"Automatically converting it to ClassLabel"
            )
            cell_types = self.dataset.unique("cell_label")
            cell_type_class = datasets.ClassLabel(names=cell_types)
            self.dataset = self.dataset.cast_column("cell_label", cell_type_class)

        if self.test_cell_ratio > 0:
            match self.shuffle:
                case bool():
                    self.datasets = self.dataset.train_test_split(
                        test_size=self.test_cell_ratio,
                        shuffle=self.shuffle,
                        stratify_by_column="cell_label",
                        seed=None,
                    )
                case int():
                    self.datasets = self.dataset.train_test_split(
                        test_size=self.test_cell_ratio,
                        shuffle=True,
                        stratify_by_column="cell_label",
                        seed=self.shuffle,
                    )
            self.train_dataset, self.test_dataset = (
                self.datasets["train"],
                self.datasets["test"],
            )
        else:
            self.train_dataset, self.test_dataset = self.dataset, None

        self.collator = SeqClsDataCollator(
            token_dict=self.token_dict,
            pad_token_or_index="<pad>",
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

    def print(self, *args, **kwargs):
        if self.trainer is not None:
            self.trainer.print(*args, **kwargs)
