import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from colat.generators.abstract import Generator

module_path = Path(__file__).parent / "stylegan2-pytorch"
sys.path.insert(1, str(module_path.resolve()))

from model import Generator as StyleGAN2Model

from colat.utils.model_utils import download_ckpt


class StyleGAN2Generator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation: float = 1.0,
        use_w: bool = False,
        feature_layer: str = "generator.layers.0",
    ):
        super(StyleGAN2Generator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w  # use W as primary latent space?
        self.outclass = class_name

        # Image widths
        configs = {
            # Converted NVIDIA official
            "ffhq": 1024,
            "car": 512,
            "cat": 256,
            "church": 256,
            "horse": 256,
            # Tuomas
            "bedrooms": 256,
            "kitchen": 256,
            "places": 256,
        }

        assert (
            self.outclass in configs
        ), f'Invalid StyleGAN2 class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = configs[self.outclass]
        self.name = f"StyleGAN2-{self.outclass}"
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def n_latent(self):
        return self.model.n_latent

    def latent_space_name(self):
        return "W" if self.w_primary else "Z"

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    # URLs created with https://sites.google.com/site/gdocs2direct/
    def download_checkpoint(self, outfile):
        checkpoints = {
            "horse": "https://drive.google.com/uc?export=download&id=18SkqWAkgt0fIwDEf2pqeaenNi4OoCo-0",
            "ffhq": "https://drive.google.com/uc?export=download&id=1FJRwzAkV-XWbxgTwxEmEACvuqF5DsBiV",
            "church": "https://drive.google.com/uc?export=download&id=1HFM694112b_im01JT7wop0faftw9ty5g",
            "car": "https://drive.google.com/uc?export=download&id=1iRoWclWVbDBAy5iXYZrQnKYSbZUqXI6y",
            "cat": "https://drive.google.com/uc?export=download&id=15vJP8GDr0FlRYpE8gD7CdeEz2mXrQMgN",
            "places": "https://drive.google.com/uc?export=download&id=1X8-wIH3aYKjgDZt4KMOtQzN1m4AlCVhm",
            "bedrooms": "https://drive.google.com/uc?export=download&id=1nZTW7mjazs-qPhkmbsOLLA_6qws-eNQu",
            "kitchen": "https://drive.google.com/uc?export=download&id=15dCpnZ1YLAnETAPB0FGmXwdBclbwMEkZ",
        }

        url = checkpoints[self.outclass]
        download_ckpt(url, outfile)

    def load_model(self):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = (
            Path(checkpoint_root)
            / f"stylegan2/stylegan2_{self.outclass}_{self.resolution}.pt"
        )

        self.model = StyleGAN2Model(self.resolution, 512, 8).to(self.device)

        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            self.download_checkpoint(checkpoint)

        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt["latent_avg"].to(self.device)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        z = (
            torch.from_numpy(
                rng.standard_normal(512 * n_samples).reshape(n_samples, 512)
            )
            .float()
            .to(self.device)
        )  # [N, 512]

        if self.w_primary:
            z = self.model.style(z)

        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError(
                "StyleGAN2: cannot change output class without reloading"
            )

    def forward(self, x):
        x = x if isinstance(x, list) else [x]
        out, _ = self.model(
            x,
            noise=self.noise,
            truncation=self.truncation,
            truncation_latent=self.latent_avg,
            input_is_w=self.w_primary,
        )
        return 0.5 * (out + 1)

    def partial_forward(self, x, layer_name):
        styles = x if isinstance(x, list) else [x]
        inject_index = None
        noise = self.noise

        if not self.w_primary:
            styles = [self.model.style(s) for s in styles]

        if len(styles) == 1:
            # One global latent
            inject_index = self.model.n_latent
            latent = self.model.strided_style(
                styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            )  # [N, 18, 512]
        elif len(styles) == 2:
            # Latent mixing with two latents
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = (
                styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)
            )

            latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        else:
            # One latent per layer
            assert (
                len(styles) == self.model.n_latent
            ), f"Expected {self.model.n_latents} latents, got {len(styles)}"
            styles = torch.stack(styles, dim=1)  # [N, 18, 512]
            latent = self.model.strided_style(styles)

        if "style" in layer_name:
            return latent

        out = self.model.input(latent)
        if "input" == layer_name:
            return out

        out = self.model.conv1(out, latent[:, 0], noise=noise[0])
        if "conv1" in layer_name:
            return out

        skip = self.model.to_rgb1(out, latent[:, 1])
        if "to_rgb1" in layer_name:
            return skip

        i = 1
        noise_i = 1

        for conv1, conv2, to_rgb in zip(
            self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[noise_i])
            if f"convs.{i-1}" in layer_name:
                return out 

            out = conv2(out, latent[:, i + 1], noise=noise[noise_i + 1])
            if f"convs.{i}" in layer_name:
                return out

            skip = to_rgb(out, latent[:, i + 2], skip)
            if f"to_rgbs.{i//2}" in layer_name:
                return out

            i += 2
            noise_i += 2

        image = skip

        raise RuntimeError(f"Layer {layer_name} not encountered in partial_forward")

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
        self.noise = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=self.device)]

        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.device))
