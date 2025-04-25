import argparse
import os
import pickle
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from masters.data.lightning import GenecorpusDataModule
from masters.data.utils import load_gensim_model_or_kv
from masters.model.lightning import LightningPretraining
from masters.model.model import BertConfig
from masters.model.utils import EvenlySpacedModelCheckpoint
from masters.model.utils.hf_interface import config_to_hf_config
from masters.train.utils import training_setup


def main(
    embed_path: str | None,
    dataset_path: str,
    batch_size: int = 12,
    epochs: int = 1,
    grad_accumul: int = 1,
    precision: str = "32",
    seed: int = 42,
    name: str | None = None,
):
    training_setup(seed)

    if name is not None and (Path("checkpoints/lightning_logs") / name).exists():
        raise ValueError(f"Experiment of name {name} already exists")

    world_size = int(os.getenv("WORLD_SIZE", 1))
    batch_per_gpu = batch_size // world_size

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_absolute():
        dataset_dir = DATA_DIR / "datasets" / dataset_path

    token_dict = pickle.load((DATA_DIR / "token_dictionary.pkl").open("rb"))

    data = GenecorpusDataModule(
        dataset_dir, token_dict=token_dict, batch_size=batch_per_gpu
    )

    config = BertConfig.from_setting("v1-base")
    hf_config = config_to_hf_config(config)

    model = LightningPretraining(
        hf_config,
        lr=1e-3,
        weight_decay=1e-3,
        warmup_steps_or_ratio=0.1,
        lr_scheduler="linear",
        embed_path=str(embed_path),
        batch_size=batch_size,
        grad_accumul=grad_accumul,
        precision=precision,
        seed=seed,
    )

    if embed_path is not None:
        word_embed = load_gensim_model_or_kv(str(embed_path), token_dict)
        model.load_embedding(word_embed)

    checkpoint_callback = EvenlySpacedModelCheckpoint(
        save_last="link", n_checkpoints=10
    )
    csv_logger = CSVLogger("checkpoints", version=name)
    tb_logger = TensorBoardLogger("checkpoints", version=csv_logger.version)

    trainer = L.Trainer(
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        strategy="ddp" if world_size > 1 else "auto",
        num_nodes=world_size,
        gradient_clip_val=1.0,
        accumulate_grad_batches=grad_accumul,
        precision=precision,  # pyright: ignore[reportArgumentType]
    )

    trainer.print("Start training")
    trainer.print(repr(model))

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-E", "--embed-path", type=str, default=None, help="Path to the embedding file"
    )
    parser.add_argument(
        "-D",
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset, relative to default dataset dir or absolute",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=12, help="Total batch size"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "-g", "--grad-accumul", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "-p", "--precision", type=str, default="32", help="Precision for training"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-N", "--name", type=str, default=None, help="Name of the experiment"
    )
    args = parser.parse_args()
    main(**vars(args))
