# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

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
