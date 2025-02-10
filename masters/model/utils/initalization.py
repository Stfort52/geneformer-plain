from torch import nn


def reset_weights(model: nn.Module, initializer_range: float = 0.02):
    for module in model.modules():
        match module:
            case nn.Linear():
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            case nn.Embedding():
                module.weight.data.normal_(mean=0.0, std=initializer_range)
            case nn.LayerNorm():
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            case _:
                pass
