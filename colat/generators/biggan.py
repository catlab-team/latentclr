import os
import re
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_biggan import (
    model,
    one_hot_from_int,
    one_hot_from_names,
    truncated_noise_sample,
)
from pytorch_pretrained_biggan.file_utils import http_get, s3_get

from colat.generators.abstract import Generator

logger = logging.getLogger(__name__)


class GeneratorMultiZ(model.Generator):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, cond_vector, truncation):
        z = self.gen_z(cond_vector[0])

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        cond_idx = 1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, model.GenBlock):
                z = layer(z, cond_vector[cond_idx], truncation)
                cond_idx += 1
            else:
                z = layer(z)

        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = self.tanh(z)
        return z


class BigGANMultiZ(nn.Module):
    """BigGAN Generator."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs
    ):
        if pretrained_model_name_or_path in model.PRETRAINED_MODEL_ARCHIVE_MAP:
            model_file = model.PRETRAINED_MODEL_ARCHIVE_MAP[
                pretrained_model_name_or_path
            ]
            config_file = model.PRETRAINED_CONFIG_ARCHIVE_MAP[
                pretrained_model_name_or_path
            ]
        else:
            model_file = os.path.join(pretrained_model_name_or_path, model.WEIGHTS_NAME)
            config_file = os.path.join(pretrained_model_name_or_path, model.CONFIG_NAME)

        try:
            resolved_model_file = model.cached_path(model_file, cache_dir=cache_dir)
            resolved_config_file = model.cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Wrong model name, should be a valid path to a folder containing "
                "a {} file and a {} file or a model name in {}".format(
                    model.WEIGHTS_NAME,
                    model.CONFIG_NAME,
                    model.PRETRAINED_MODEL_ARCHIVE_MAP.keys(),
                )
            )
            raise

        logger.info(
            "loading model {} from cache at {}".format(
                pretrained_model_name_or_path, resolved_model_file
            )
        )

        # Load config
        config = model.BigGANConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))

        # Instantiate model.
        _model = cls(config, *inputs, **kwargs)
        state_dict = torch.load(
            resolved_model_file,
            map_location="cpu" if not torch.cuda.is_available() else None,
        )
        _model.load_state_dict(state_dict, strict=False)
        return _model

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
        self.generator = GeneratorMultiZ(config)
        self.n_latents = len(config.layers) + 1  # one for gen_z + one per layer

    def forward(self, z, class_label, truncation):
        assert 0 < truncation <= 1

        if not isinstance(z, list):
            z = self.n_latents * [z]

        if isinstance(class_label, list):
            embed = [self.embeddings(l) for l in class_label]
        else:
            embed = self.n_latents * [self.embeddings(class_label)]

        assert (
            len(z) == self.n_latents
        ), f"Expected {self.n_latents} latents, got {len(z)}"
        assert (
            len(embed) == self.n_latents
        ), f"Expected {self.n_latents} class vectors, got {len(class_label)}"

        cond_vectors = [torch.cat((z, e), dim=1) for (z, e) in zip(z, embed)]
        z = self.generator(cond_vectors, truncation)
        return z


class BigGANGenerator(Generator):
    def __init__(
        self,
        resolution: str,
        device: str,
        truncation: float = 0.4,
        class_name: str = "husky",
        feature_layer: str = "generator.layers.0",
    ) -> None:
        super(BigGANGenerator, self).__init__(feature_layer=feature_layer)

        self.device = device
        self.truncation = truncation
        self.set_output_class(class_name)
        self.load_model(f"biggan-deep-{resolution}")

    def n_latent(self):
        return self.model.n_latents

    def set_output_class(self, class_id):
        if isinstance(class_id, int):
            self.v_class = torch.from_numpy(one_hot_from_int([class_id])).to(
                self.device
            )
            self.outclass = f"class{class_id}"
        elif isinstance(class_id, str):
            self.outclass = class_id.replace(" ", "_")
            self.v_class = torch.from_numpy(one_hot_from_names([class_id])).to(
                self.device
            )
        else:
            raise RuntimeError(f"Unknown class identifier {class_id}")

    # Default implementaiton fails without an internet
    # connection, even if the model has been cached
    def load_model(self, name):
        if name not in model.PRETRAINED_MODEL_ARCHIVE_MAP:
            raise RuntimeError("Unknown BigGAN model name", name)

        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        model_path = Path(checkpoint_root) / name

        os.makedirs(model_path, exist_ok=True)

        model_file = model_path / model.WEIGHTS_NAME
        config_file = model_path / model.CONFIG_NAME
        model_url = model.PRETRAINED_MODEL_ARCHIVE_MAP[name]
        config_url = model.PRETRAINED_CONFIG_ARCHIVE_MAP[name]

        for filename, url in ((model_file, model_url), (config_file, config_url)):
            if not filename.is_file():
                print("Downloading", url)
                with open(filename, "wb") as f:
                    if url.startswith("s3://"):
                        s3_get(url, f)
                    else:
                        http_get(url, f)

        self.model = BigGANMultiZ.from_pretrained(model_path).to(self.device)
        self.model.n_latents = (
            len(self.model.config.layers) + 1
        )  # one for gen_z + one per layer

    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        """Samples random codes from the latent space"""
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        noise_vector = truncated_noise_sample(
            truncation=self.truncation, batch_size=batch_size, seed=seed
        )
        noise = torch.from_numpy(noise_vector)
        return noise.to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Retrieves desired features associated with the given latent codes"""
        if isinstance(z, list):
            c = self.v_class.repeat(z[0].shape[0], 1)
            class_vector = len(z) * [c]
        else:
            class_vector = self.v_class.repeat(z.shape[0], 1)
        out = self.model.forward(
            z, class_vector, self.truncation
        )  # [N, 3, 128, 128], in [-1, 1]
        return 0.5 * (out + 1)

    # Run model only until given layer
    def partial_forward(self, x, layer_name: str) -> torch.Tensor:
        if layer_name in ["embeddings", "generator.gen_z"]:
            n_layers = 0
        elif "generator.layers" in layer_name:
            layer_base = re.match("^generator\.layers\.[0-9]+", layer_name)[0]
            n_layers = int(layer_base.split(".")[-1]) + 1
        else:
            n_layers = len(self.model.config.layers)

        if not isinstance(x, list):
            x = self.model.n_latents * [x]

        class_label = self.v_class.repeat(x[0].shape[0], 1)
        embed = len(x) * [self.model.embeddings(class_label)]

        assert (
            len(x) == self.model.n_latents
        ), f"Expected {self.model.n_latents} latents, got {len(x)}"
        assert (
            len(embed) == self.model.n_latents
        ), f"Expected {self.model.n_latents} class vectors, got {len(class_label)}"

        cond_vectors = [torch.cat((z, e), dim=1) for (z, e) in zip(x, embed)]

        # Generator forward
        z = self.model.generator.gen_z(cond_vectors[0])
        z = z.view(-1, 4, 4, 16 * self.model.generator.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        cond_idx = 1
        for i, layer in enumerate(self.model.generator.layers[:n_layers]):
            if isinstance(layer, model.GenBlock):
                z = layer(z, cond_vectors[cond_idx], self.truncation)
                cond_idx += 1
            else:
                z = layer(z)

        return z
