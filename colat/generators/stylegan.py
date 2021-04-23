import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import colat.utils.stylegan_helper as stylegan
from colat.generators.abstract import Generator
from colat.utils.model_utils import download_ckpt


# PyTorch port of StyleGAN 1
class StyleGANGenerator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation: float = 1.0,
        use_w: bool = False,
        feature_layer: str = "generator.layers.0",
    ):
        super(StyleGANGenerator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.w_primary = use_w  # is W primary latent space?
        self.outclass = class_name

        configs = {
            # Official
            "ffhq": 1024,
            "celebahq": 1024,
            "bedrooms": 256,
            "cars": 512,
            "cats": 256,
            # From https://github.com/justinpinkney/awesome-pretrained-stylegan
            "vases": 1024,
            "wikiart": 512,
            "fireworks": 512,
            "abstract": 512,
            "anime": 512,
            "ukiyo-e": 512,
        }

        assert (
            self.outclass in configs
        ), f'Invalid StyleGAN class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = configs[self.outclass]
        self.load_model()
        self.set_noise_seed(0)

    def n_latent(self):
        return 18

    def latent_space_name(self):
        return "W" if self.w_primary else "Z"

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    def load_model(self):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = (
            Path(checkpoint_root)
            / f"stylegan/stylegan_{self.outclass}_{self.resolution}.pt"
        )

        self.model = stylegan.StyleGAN_G(self.resolution).to(self.device)

        urls_tf = {
            "vases": "https://thisvesseldoesnotexist.s3-us-west-2.amazonaws.com/public/network-snapshot-008980.pkl",
            "fireworks": "https://mega.nz/#!7uBHnACY!quIW-pjdDa7NqnZOYh1z5UemWwPOW6HkYSoJ4usCg9U",
            "abstract": "https://mega.nz/#!vCQyHQZT!zdeOg3VvT4922Z2UfxO51xgAfJD-NAK2nW7H_jMlilU",
            "anime": "https://mega.nz/#!vawjXISI!F7s13yRicxDA3QYqYDL2kjnc2K7Zk3DwCIYETREmBP4",
            "ukiyo-e": "https://drive.google.com/uc?id=1CHbJlci9NhVFifNQb3vCGu6zw4eqzvTd",
        }

        urls_torch = {
            "celebahq": "https://drive.google.com/uc?export=download&id=1lGcRwNoXy_uwXkD6sy43aAa-rMHRR7Ad",
            "bedrooms": "https://drive.google.com/uc?export=download&id=1r0_s83-XK2dKlyY3WjNYsfZ5-fnH8QgI",
            "ffhq": "https://drive.google.com/uc?export=download&id=1GcxTcLDPYxQqcQjeHpLUutGzwOlXXcks",
            "cars": "https://drive.google.com/uc?export=download&id=1aaUXHRHjQ9ww91x4mtPZD0w50fsIkXWt",
            "cats": "https://drive.google.com/uc?export=download&id=1JzA5iiS3qPrztVofQAjbb0N4xKdjOOyV",
            "wikiart": "https://drive.google.com/uc?export=download&id=1fN3noa7Rsl9slrDXsgZVDsYFxV0O08Vx",
        }

        if not checkpoint.is_file():
            os.makedirs(checkpoint.parent, exist_ok=True)
            if self.outclass in urls_torch:
                download_ckpt(urls_torch[self.outclass], checkpoint)
            else:
                checkpoint_tf = checkpoint.with_suffix(".pkl")
                if not checkpoint_tf.is_file():
                    download_ckpt(urls_tf[self.outclass], checkpoint_tf)
                print("Converting TensorFlow checkpoint to PyTorch")
                self.model.export_from_tf(checkpoint_tf)

        self.model.load_weights(checkpoint)

    def sample_latent(self, batch_size: int, seed: int = None) -> torch.Tensor:
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        noise = (
            torch.from_numpy(
                rng.standard_normal(512 * batch_size).reshape(batch_size, 512)
            )
            .float()
            .to(self.device)
        )  # [N, 512]

        if self.w_primary:
            noise = self.model._modules["g_mapping"].forward(noise)
        return noise

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError("StyleGAN: cannot change output class without reloading")

    def forward(self, x):
        out = self.model.forward(x, latent_is_w=self.w_primary)
        return 0.5 * (out + 1)

    # Run model only until given layer
    def partial_forward(self, x, layer_name):
        mapping = self.model._modules["g_mapping"]
        G = self.model._modules["g_synthesis"]
        trunc = self.model._modules.get("truncation", lambda x: x)

        if not self.w_primary:
            x = mapping.forward(x)  # handles list inputs

        if isinstance(x, list):
            x = torch.stack(x, dim=1)
        else:
            x = x.unsqueeze(1).expand(-1, 18, -1)

        # Whole mapping
        if "g_mapping" in layer_name:
            return x

        x = trunc(x)
        if layer_name == "truncation":
            return x

        # Get names of children
        def iterate(m, name, seen):
            children = getattr(m, "_modules", [])
            if len(children) > 0:
                for child_name, module in children.items():
                    seen += iterate(module, f"{name}.{child_name}", seen)
                return seen
            else:
                return [name]

        # Generator
        batch_size = x.size(0)
        for i, (n, m) in enumerate(G.blocks.items()):  # InputBlock or GSynthesisBlock
            if i == 0:
                r = m(x[:, 2 * i : 2 * i + 2])
            else:
                r = m(r, x[:, 2 * i : 2 * i + 2])
            children = iterate(m, f"g_synthesis.blocks.{n}", [])
            for c in children:
                if layer_name in c:  # substring
                    return r

        raise RuntimeError(f"Layer {layer_name} not encountered in partial_forward")

    def set_noise_seed(self, seed):
        G = self.model._modules["g_synthesis"]

        def for_each_child(this, name, func):
            children = getattr(this, "_modules", [])
            for child_name, module in children.items():
                for_each_child(module, f"{name}.{child_name}", func)
            func(this, name)

        def modify(m, name):
            if isinstance(m, stylegan.NoiseLayer):
                H, W = [int(s) for s in name.split(".")[2].split("x")]
                torch.random.manual_seed(seed)
                m.noise = torch.randn(
                    1, 1, H, W, device=self.device, dtype=torch.float32
                )
                # m.noise = 1.0 # should be [N, 1, H, W], but this also works

        for_each_child(G, "g_synthesis", modify)
