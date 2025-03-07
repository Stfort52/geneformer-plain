# geneformer-plain

This repository contains the implementation of the Geneformer model, without huggingface magic.
Should be useful if you're looking to understand the model or modify it deeply.

It still has the `transformers` library as a dependency, but it's not used for the model itself but for:

- the learning rate scheduler with warmup
- the bucket calculation in T5 positional bias (which is not the default)

And yes, I didn't make a decision on the name yet.
Why is the package named `masters`? Because this was originally developed as a basis for my master's project.
Please don't bother.

## Install

```bash
git clone https://github.com/Stfort52/geneformer-plain.git
cd geneformer-plain
pip install -e .
```

It's highly recommended to use a virtual environment. To also install the dev dependencies, run `pip install -e .[dev]` instead.

## Usage

### Get the data

Clone the [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M) repository to get the data.
You'll likely need git-lfs to clone the repository.
Then, set up a symlink to the required files in the `data` directory like below:
You should be able to easily locate the required files in the GeneCorpus-30M repository.

```bash
data
├── word_embeddings/
├── datasets/
│   ├── genecorpus_30M_2048.dataset -> /path/to/30M/dataset
│   ├── iCM_diff_dropseq.dataset -> /path/to/dropseq/dataset
│   └── panglao_SRA553822-SRS2119548.dataset -> /path/to/panglao/dataset
├── is_bivalent.csv
└── token_dictionary.pkl -> /path/to/token/dictionary
```

#### Optional: subset the data

The full GeneCorpus-30M dataset is quite large. You can subset it by running the notebook at `notebooks/subset_genecorpus.ipynb`.

### launch pretraining

```bash
python -m masters.train.pretrain
```

Alternatively, Visual Studio Code users can launch the task `Launch Pretraining` under the command `Tasks: Run Task`.

This will create a new version of the model and save it to the `checkpoints` directory.

#### pretrain with DDP

To launch pretraining with DDP, run the following command:

```bash
bash -c masters/train/ddp.sh <master_port> <hosts> pretrain
```

Alternatively, Visual Studio Code users can launch the task `Distributed Pretraining` under the command `Tasks: Run Task`.

### launch finetuning

```bash
python -m masters.train.finetune
```

Alternatively, Visual Studio Code users can launch the task `Launch Fine-tuning` under the command `Tasks: Run Task`.

#### fine-tune with DDP

To launch finetuning with DDP, run the following command:

```bash
bash -c masters/train/ddp.sh <master_port> <hosts> finetune
```

Alternatively, Visual Studio Code users can launch the task `Distributed Fine-tuning` under the command `Tasks: Run Task`.

### Configure the model

The `BertConfig` is a dataclass that can be used to configure the model. The default configuration respects the configuration used in the original 6 layer model from the 2023 Geneformer paper.

```python
from masters.model.model import BertConfig

# All load the same default configuration
config = BertConfig()
config = BertConfig.from_setting("v1")
config = BertConfig.from_setting("base")
config = BertConfig.from_setting("v1-base")

```

The 12 layer model from the 2023 Geneformer paper can be configured as follows:

```python
from masters.model.model import BertConfig

# All load the same configuration
config = BertConfig.from_setting("v1-large")
config = BertConfig.from_setting("large")
```

The 2024 Geneformer paper can be configured as follows:

```python
from masters.model.model import BertConfig

# All load the same configuration
config = BertConfig.from_setting("v2")
config = BertConfig.from_setting("v2-base")

# And the larger model
config = BertConfig.from_setting("v2-large")
```

But note that: the 2024 Geneformer uses a `[CLS]` token to generate cell embeddeings, which is different from the 2023 Geneformer which mean pools.
