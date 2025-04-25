import argparse
import os
import pickle
from pathlib import Path
from typing import cast

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from masters.data.lightning import NerSplitsDataModule
from masters.model.lightning import LightningTokenClassification
from masters.train.utils import GeneClassificationTask, training_setup

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_TASKS_FILE = BASE_DIR / "data" / "gene_labeling_tasks.csv"


def main(
    model_name: str,
    task_name: str,
    epochs: int = 5,
    batch_size: int = 16,
    grad_accumul: int = 1,
    seed: int = 42,
    precision: str = "32",
    tasks_file: str | Path | None = None,
    experiment_name: str | None = None,
):
    training_setup(seed)

    world_size = int(os.getenv("WORLD_SIZE", 1))
    batch_per_gpu = batch_size // world_size

    data_dir = BASE_DIR / "data"
    model_dir = BASE_DIR / f"checkpoints/lightning_logs/{model_name}"

    if tasks_file is None:
        tasks_file = DEFAULT_TASKS_FILE

    tasks = pd.read_csv(tasks_file).set_index("task")
    task = cast(GeneClassificationTask, tasks.loc[task_name].to_dict())
    labels = pd.read_csv(data_dir / "gene_labels" / task["label_file"]).set_index(
        task["id_column"]
    )[task["target_column"]]

    dataset_dir = data_dir / "datasets/panglao_SRA553822-SRS2119548.dataset"
    token_dict = pickle.load((data_dir / "token_dictionary.pkl").open("rb"))

    data = NerSplitsDataModule(
        dataset_dir=dataset_dir,
        token_dict=token_dict,
        gene_labels=labels,
        batch_size=batch_per_gpu,
        train_cell_count_or_ratio=1.0,
        test_cell_count_or_ratio=1.0,
    )

    ckpt_dir = model_dir / "checkpoints" / "last.ckpt"
    save_dir = model_dir / "finetune"

    model = LightningTokenClassification.from_pretrained(
        model_path=ckpt_dir,
        n_classes=labels.nunique(),
        lr=5e-5,
        weight_decay=1e-3,
        lr_scheduler="linear",
        warmup_steps_or_ratio=0.1,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", every_n_epochs=1, save_last="link"
    )
    csv_logger = CSVLogger(save_dir, name=task_name, version=experiment_name)
    tb_logger = TensorBoardLogger(save_dir, name=task_name, version=csv_logger.version)

    trainer = L.Trainer(
        strategy="ddp" if world_size > 1 else "auto",
        max_epochs=epochs,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=grad_accumul,
        precision=precision,  # pyright: ignore[reportArgumentType]
        num_nodes=world_size,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model_name",
        required=True,
        help="Model to fine-tune, relative to `checkpoints/lightning_logs/`",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task_name",
        type=str,
        required=True,
        help="Fine-tuning task to run, as defined in TASKS_FILE",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=5, help="Number of epochs to train (5)"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="Total batch size for training (16)",
    )
    parser.add_argument(
        "-g",
        "--grad_accumul",
        type=int,
        default=1,
        help="Gradient accumulation steps (1)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (42)",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="32",
        help="Precision for training (32)",
    )
    parser.add_argument(
        "-T",
        "--tasks-file",
        type=str,
        default=None,
        help="Override for tasks definition file",
    )
    parser.add_argument(
        "-N",
        "--name",
        dest="experiment_name",
        type=str,
        default=None,
        help="Override experiment name",
    )
    args = parser.parse_args()
    main(**vars(args))
