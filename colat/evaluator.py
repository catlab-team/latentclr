import logging
from typing import Optional, List

import torch
import tqdm
from torch.utils.data import DataLoader

from colat.generators import Generator
from colat.metrics import LossMetric


class Evaluator:
    """Model evaluator

    Args:
        model: model to be evaluated
        loss_fn: loss function
        generator: pretrained generator
        projector: pretrained projector
        device: device on which to evaluate model
        iterations: number of iterations
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        generator: Generator,
        projector: torch.nn.Module,
        device: torch.device,
        batch_size: int,
        iterations: int,
        feed_layers: Optional[List[int]] = None
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.generator = generator
        self.projector = projector
        self.feed_layers = feed_layers

        # Iterations & batch size
        self.iterations = iterations
        self.batch_size = batch_size

        # Metrics
        self.loss_metric = LossMetric()

    def evaluate(self) -> float:
        """Evaluates the model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """

        # Progress bar
        pbar = tqdm.tqdm(total=len(self.loader), leave=False)
        pbar.set_description("Evaluating... ")

        # Set to eval
        self.generator.eval()
        self.projector.eval()
        self.model.eval()

        # Loop
        for i in range(self.iterations):
            with torch.no_grad():
                # To device
                z = self.generator.sample_latent(self.batch_size)
                z = z.to(self.device)

                # Original features
                with torch.no_grad():
                    orig_feats = self.generator.get_features(z)
                    orig_feats = self.projector(orig_feats)

                # Apply Directions
                z = self.model(z)

                # Forward
                features = []
                for j in range(z.shape[0] // self.batch_size):
                    # Prepare batch
                    start, end = j * self.batch_size, (j + 1) * self.batch_size
                    z_batch = z[start:end, ...]

                    # Manipulate only asked layers
                    if self.feed_layers is not None:
                        n_latent = self.generator.n_latent()

                        z_batch_layers = []
                        for i in range(n_latent):
                            if i in self.feed_layers:
                                z_batch_layers.append(z_batch)
                            else:
                                z_batch_layers.append(z_orig)
                        z_batch = z_batch_layers

                    # Get features
                    feats = self.generator.get_features(z_batch)
                    feats = self.projector(feats)

                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)
                self.acc_metric.update(acc.item(), z.shape[0])
                self.loss_metric.update(loss.item(), z.shape[0])

                # Update progress bar
                pbar.update()

        pbar.close()

        acc = self.acc_metric.compute()
        loss = self.loss_metric.compute()
        self.logger.info(f"Acc: {acc:.4f} Loss: {loss:.4f}\n")

        return loss
