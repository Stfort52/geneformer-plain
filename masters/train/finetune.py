import os
import pickle
from pathlib import Path

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from masters.data.lightning import NerSplitsDataModule
from masters.model.lightning import LightningTokenClassification
from masters.model.utils import training_setup

if __name__ == "__main__":
    training_setup(42)

    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    BATCH_SIZE = 32
    BATCH_PER_GPU = BATCH_SIZE // WORLD_SIZE

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    MODEL_DIR = DATA_DIR.parent / f"checkpoints/lightning_logs/version_{10}"
    TASK_NAME = "tf_range_prediction"

    labels = pd.read_csv(DATA_DIR / "is_longrange_tf.csv").set_index("id")[
        "is_longrange"
    ]

    dataset_dir = DATA_DIR / "datasets/iCM_diff_dropseq.dataset"
    token_dict = pickle.load((DATA_DIR / "token_dictionary.pkl").open("rb"))

    data = NerSplitsDataModule(
        dataset_dir=dataset_dir,
        token_dict=token_dict,
        gene_labels=labels,
        batch_size=BATCH_PER_GPU,
        train_cell_count_or_ratio=1.0,
        test_cell_count_or_ratio=1.0,
    )

    ckpt_dir = MODEL_DIR / "checkpoints" / "last.ckpt"
    save_dir = MODEL_DIR / "finetune"

    model = LightningTokenClassification(
        model_path_or_config=ckpt_dir,
        n_classes=labels.nunique(),
        lr=5e-5,
        weight_decay=1e-3,
        lr_scheduler="linear",
        warmup_steps_or_ratio=0.1,
        freeze_first_n_layers=0,
    )
    model.model.reset_weights()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", every_n_epochs=1
    )
    csv_logger = CSVLogger(save_dir, name=TASK_NAME)
    tb_logger = TensorBoardLogger(save_dir, name=TASK_NAME, version=csv_logger.version)

    trainer = L.Trainer(
        strategy="ddp" if WORLD_SIZE > 1 else "auto",
        max_epochs=5,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback],
        num_nodes=WORLD_SIZE,
    )

    trainer.fit(model, data)
