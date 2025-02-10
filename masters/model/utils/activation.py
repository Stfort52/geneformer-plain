from typing import Type

from torch import nn


def activation_from_name(name: str) -> Type[nn.Module]:
    match name.lower():
        case "relu":
            return nn.ReLU
        case "gelu":
            return nn.GELU
        case "tanh":
            return nn.Tanh
        case "sigmoid":
            return nn.Sigmoid
        case "leakyrelu":
            return nn.LeakyReLU
        case _:
            raise ValueError(f"Activation {name} is not supported.")
