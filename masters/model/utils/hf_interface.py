from pathlib import Path
from typing import Any, cast

from transformers import BertConfig as HFBertConfig

from ..model import BertConfig


def config_from_hf_config(hf_config: HFBertConfig) -> BertConfig:
    match hf_config.position_embedding_type:
        case "absolute":
            absolute_pe_strategy = "trained"
            absolute_pe_kwargs = {"max_len": hf_config.max_position_embeddings}
            relative_pe_strategy = None
            relative_pe_kwargs = {}
        case "relative_key":
            absolute_pe_strategy = None
            absolute_pe_kwargs = {}
            relative_pe_strategy = "trained"
            relative_pe_kwargs = {"max_len": hf_config.max_position_embeddings}
        case "relative_key_query":
            raise NotImplementedError("Relative key query not implemented")
        case _:
            raise ValueError("Unknown position embedding type")

    config = BertConfig(
        n_vocab=hf_config.vocab_size,
        d_model=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        num_layers=hf_config.num_hidden_layers,
        d_ff=hf_config.intermediate_size,
        attn_dropout=hf_config.attention_probs_dropout_prob,
        ff_dropout=hf_config.hidden_dropout_prob,
        norm="post",
        initialization_range=hf_config.initializer_range,
        ln_eps=hf_config.layer_norm_eps,
        absolute_pe_strategy=absolute_pe_strategy,
        absolute_pe_kwargs=absolute_pe_kwargs,
        relative_pe_strategy=relative_pe_strategy,
        relative_pe_kwargs=relative_pe_kwargs,
        relative_pe_shared=False,
        act_fn=hf_config.hidden_act,
        cls_dropout=cast(float, hf_config.classifier_dropout),
    )

    return config


def config_to_hf_config(config: BertConfig) -> HFBertConfig:
    if config["absolute_pe_strategy"] == "trained":
        position_embedding_type = "absolute"
        max_position_embeddings = config["absolute_pe_kwargs"]["max_len"]
        if config["relative_pe_strategy"] == "trained":
            raise ValueError(
                "Cannot have both absolute and relative position embeddings"
            )
    elif config["relative_pe_strategy"] == "trained":
        position_embedding_type = "relative_key"
        max_position_embeddings = config["relative_pe_kwargs"]["max_len"]
    else:
        raise ValueError("Unknown position embedding type")

    hf_config = HFBertConfig(
        vocab_size=config.n_vocab,
        hidden_size=config.d_model,
        num_attention_heads=config.num_heads,
        num_hidden_layers=config.num_layers,
        intermediate_size=config.d_ff,
        attention_probs_dropout_prob=config.attn_dropout,
        hidden_dropout_prob=config.ff_dropout,
        hidden_act=config.act_fn,
        position_embedding_type=position_embedding_type,
        max_position_embeddings=max_position_embeddings,
        initializer_range=config.initialization_range,
        layer_norm_eps=config.ln_eps,
        classifier_dropout=config.cls_dropout,
    )

    return hf_config


def is_hf(
    model_config_or_path: str | Path | HFBertConfig | BertConfig | dict[str, Any],
) -> bool:
    match model_config_or_path:
        case HFBertConfig():
            return True
        case BertConfig():
            return False
        case dict():
            return "hidden_size" in model_config_or_path
        case str() | Path():
            return (Path(model_config_or_path) / "config.json").exists()
        case _:
            raise ValueError("Unable to determine")
