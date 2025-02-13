from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import NerDataModule


class NerSplitsDataModule(NerDataModule):
    def __init__(
        self,
        dataset_dir: str | Path,
        token_dict: dict[str, int],
        gene_labels: pd.Series,
        train_cell_count_or_ratio: int | float = 10_000,
        test_cell_count_or_ratio: int | float = 2_000,
        test_gene_ratio: float = 0.2,
        ignore_index: int = -100,
        batch_size: int = 32,
        num_workers: int = 16,
        dataset_shuffle: int | bool = 42,
        gene_label_shuffle: int | bool = 42,
    ):
        # split and shuffle the genes automatically
        match gene_label_shuffle:
            case bool():
                train_gene_labels, test_gene_labels = train_test_split(
                    gene_labels,
                    test_size=test_gene_ratio,
                    stratify=gene_labels.to_list(),
                    shuffle=gene_label_shuffle,
                    random_state=None,
                )
            case int():
                train_gene_labels, test_gene_labels = train_test_split(
                    gene_labels,
                    test_size=test_gene_ratio,
                    stratify=gene_labels.to_list(),
                    shuffle=True,
                    random_state=gene_label_shuffle,
                )

        super().__init__(
            dataset_dir=dataset_dir,
            token_dict=token_dict,
            train_gene_labels=train_gene_labels,
            test_gene_labels=test_gene_labels,
            train_cell_count_or_ratio=train_cell_count_or_ratio,
            test_cell_count_or_ratio=test_cell_count_or_ratio,
            ignore_index=ignore_index,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_shuffle=dataset_shuffle,
        )
