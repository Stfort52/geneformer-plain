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

## Advanced Usage

### Configure the model

The base model has the following configurations, respecting the original paper "Transfer learning enables predictions in network biology" (<https://doi.org/10.1038/s41586-023-06139-9>)

```yaml
config:
  absolute_pe_kwargs:
    embed_size: 256
    max_len: 2048
  absolute_pe_strategy: trained
  act_fn: relu
  attn_dropout: 0.02
  d_ff: 512
  d_model: 256
  ff_dropout: 0.02
  n_vocab: 25426
  norm: post
  num_heads: 4
  num_layers: 6
  relative_pe_kwargs: {}
  relative_pe_shared: true
  relative_pe_strategy: null
ignore_index: -100
initialization_range: 0.02
lr: 0.001
lr_scheduler: linear
warmup_steps_or_ratio: 0.1
weight_decay: 0.001
batch_size: 12
```

The `config` key contains the model configuration. Anything else is a hyperparameter used for training.
