from abc import ABC as AbstractBaseClass

import torch


class Projector(AbstractBaseClass, torch.nn.Module):
    """Abstract projector

    Args:
        normalize: whether to normalize after feed-forward
    """

    def __init__(self, net: torch.nn.Module, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.net = net
        self.add_module("Projector", net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.reshape(input, (input.shape[0], -1))
        out = self.net(input)
        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out
