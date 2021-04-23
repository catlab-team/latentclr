from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import torch


class Generator(AbstractBaseClass, torch.nn.Module):
    """Abstract generator

    Args:
        feature_layer: targeted layer to extract features from
    """

    def __init__(self, feature_layer: str) -> None:
        super(Generator, self).__init__()
        self.feature_layer = feature_layer

    def get_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.partial_forward(z, self.feature_layer)

    @abstractmethod
    def n_latent(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def partial_forward(self, z: torch.Tensor, layer_name: str):
        raise NotImplementedError

    @abstractmethod
    def sample_latent(self, batch_size: int) -> torch.Tensor:
        """Samples random codes from the latent space"""
        raise NotImplementedError

    @abstractmethod
    def set_output_class(self, new_class):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Retrieves desired features associated with the given latent codes"""
        raise NotImplementedError
