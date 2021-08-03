import logging
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import tqdm
from PIL import Image, ImageDraw, ImageFont

from colat.generators import Generator

sign = lambda x: math.copysign(1, x)


class Visualizer:
    """Model evaluator

    Args:
        model: model to be evaluated
        generator: pretrained generator
        projector: pretrained projector
        device: device on which to evaluate model
        n_samples: number of samples
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generator: Generator,
        projector: torch.nn.Module,
        device: torch.device,
        n_samples: Union[int, str],
        n_dirs: Union[int, List[int]],
        alphas: List[int],
        iterative: bool = True,
        feed_layers: Optional[List[int]] = None,
        image_size: Optional[Union[int, List[int]]] = None,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Model
        self.model = model
        self.generator = generator
        self.projector = projector

        # Set to eval
        self.generator.eval()
        self.projector.eval()
        self.model.eval()

        # N Samples
        if isinstance(n_samples, int):
            self.samples = self.generator.sample_latent(n_samples)
            self.samples = self.samples.to(self.device)
        else:
            print(f"Loading input file {n_samples}...")
            self.samples = np.load(n_samples)

        #  Sub-sample Dirs
        if n_dirs == -1:
            self.dirs = list(range(self.model.k))
        elif isinstance(n_dirs, int):
            self.dirs = np.random.choice(self.model.k, n_dirs, replace=False)
        else:
            assert isinstance(n_dirs, list)
            self.dirs = n_dirs

        # Alpha
        alphas = sorted(alphas)
        i = 0
        while alphas[i] < 0:
            i += 1
        self.neg_alphas = alphas[:i]

        if alphas[i] == 0:
            i += 1
        self.pos_alphas = alphas[i:]

        # Iterative
        self.iterative = iterative

        # Image Size
        if image_size:
            self.image_transform = T.Resize(image_size)
        else:
            self.image_transform = torch.nn.Identity()

        # Feed Layers
        self.feed_layers = feed_layers

    def visualize(self) -> float:
        """Generates images from the trained model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """
        # Progress bar
        pbar = tqdm.tqdm(total=self.samples.shape[0], leave=False)
        pbar.set_description("Generating... ")

        # Set to eval
        self.generator.eval()
        self.projector.eval()
        self.model.eval()

        #  Helper function to edit latent codes
        def _edit(z, alpha, ks):
            #  check if only one latent code is given
            assert z.shape[0] == 1 or z.shape[0] == len(
                ks
            ), """Only able to apply all directions to single latent code or
                apply each direction to single code"""
            self.model.alpha = alpha

            # Apply Directions
            zs = []
            for i, k in enumerate(ks):
                _i = i if z.shape[0] > 1 else 0
                zs.append(self.model.forward_single(z[_i : _i + 1, ...], k=k))
            zs = torch.cat(zs, dim=0)
            return zs

        # Helper function to generate images
        def _generate(zs, z=None):
            # Manipulate only asked layers
            if self.feed_layers is not None and z is not None:
                n_latent = self.generator.n_latent()

                zs_layers = []
                for i in range(n_latent):
                    if i in self.feed_layers:
                        zs_layers.append(zs)
                    else:
                        zs_layers.append(z.expand(zs.shape[0], -1))
                zs = zs_layers

            images = self.generator(zs).detach().cpu()
            return self.image_transform(images)

        # Loop
        with torch.no_grad():
            for i in range(self.samples.shape[0]):
                # Take a single sample
                z = self.samples[i : i + 1, ...]

                # Get original image
                orj_img = _generate(z)

                # Accumulator for images
                images = []

                #  Start with z and alpha = 0
                z_orig = z
                prev_alpha = 0
                for alpha in reversed(self.neg_alphas):
                    #  if iterative use last z and d(alpha)
                    _z = z if self.iterative else z_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z = _edit(_z, _alpha, ks=self.dirs)
                    images.append(_generate(z, z_orig))
                    prev_alpha = alpha

                # Reverse images
                images = list(reversed(images))

                # Reset z and alpha
                z = z_orig
                prev_alpha = 0
                for alpha in self.pos_alphas:
                    #  if iterative use last z and d(alpha)
                    _z = z if self.iterative else z_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z = _edit(_z, _alpha, ks=self.dirs)
                    images.append(_generate(z, z_orig))
                    prev_alpha = alpha

                #  Prepare final image
                images = torch.stack(images, dim=0)
                images = images.transpose(1, 0)
                col_orj_img = orj_img.repeat((images.size(0), 1, 1, 1))  # .unsqueeze(1)

                titles = []
                before_sign = -1
                imgs = []
                for ind, alpha in enumerate(self.neg_alphas + self.pos_alphas):
                    # append orijinal image
                    if sign(alpha) != before_sign:
                        imgs.append(col_orj_img)
                        titles.append("α=0")
                        before_sign = sign(alpha)

                    titles.append(f"α= {alpha:.3f}")
                    imgs.append(images[:, ind, ...])
                images = torch.stack(imgs).transpose(1, 0)

                images = images.reshape(
                    -1, images.size(-3), images.size(-2), images.size(-1)
                )

                imgs_grid = torchvision.utils.make_grid(
                    images,
                    nrow=len(self.neg_alphas) + len(self.pos_alphas) + 1,
                    padding=2,
                    pad_value=255,
                )

                fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)

                # get a drawing context
                img_alpha = Image.new("RGB", (imgs_grid.size(-1), 50), (255, 255, 255))
                d = ImageDraw.Draw(img_alpha)

                # draw alpha text
                for ind, text in enumerate(titles):
                    d.multiline_text(
                        (40 + ind * (images.size(-2) + 2), 10),
                        text,
                        font=fnt,
                        fill=(0, 0, 0),
                    )

                # get a drawing context
                img_k = Image.new(
                    "RGB", (100, imgs_grid.size(-2) + 50), (255, 255, 255)
                )
                d = ImageDraw.Draw(img_k)

                #  draw direction text
                for ind in range(len(self.dirs)):
                    d.multiline_text(
                        (10, 100 + ind * (images.size(-1) + 2)),
                        f"k={self.dirs[ind]}",
                        font=fnt,
                        fill=(0, 0, 0),
                    )

                img_alpha = T.ToTensor()(img_alpha)
                img_k = T.ToTensor()(img_k)

                imgs_grid = torch.cat([img_alpha, imgs_grid], dim=-2)
                imgs_grid = torch.cat([img_k, imgs_grid], dim=-1)

                torchvision.utils.save_image(imgs_grid, f"sample_{i}.png")

        pbar.close()
