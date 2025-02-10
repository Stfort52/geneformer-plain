from typing import Literal, Type, overload

from ..pe import *

MODEL_MAP = {
    "absolute": {
        "trained": TrainedPE,
        "learned": TrainedPE,  # alias
        "sinusoidal": SinusoidalPE,
        "trigonometric": SinusoidalPE,  # alias
    },
    "relative": {
        "trained": TrainedRPE,
        "learned": TrainedRPE,  # alias
        "sinusoidal": SinusoidalRPE,
        "trigonometric": SinusoidalRPE,  # alias
        "t5": T5RPE,
    },
}


@overload
def pe_from_name(category: Literal["absolute"], name: str) -> Type[BasePE]: ...


@overload
def pe_from_name(category: Literal["relative"], name: str) -> Type[BaseRPE]: ...


def pe_from_name(category: str, name: str) -> Type[BasePE]:
    return MODEL_MAP[category][name]
