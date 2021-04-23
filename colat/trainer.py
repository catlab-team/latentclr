import logging
import os
import time
from typing import Optional, List

import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from colat.generators import Generator
from colat.metrics import LossMetric


class Trainer:
    """Model trainer

    Args:
        model: model to train
        loss_fn: loss function
        optimizer: model optimizer
        generator: pretrained generator
        projector: pretrained projector
        device: device to train the model on
        batch_size: number of batch elements
        iterations: number of iterations
        scheduler: learning rate scheduler
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training
        mixed_precision: enable mixed precision training

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: Generator,
        projector: torch.nn.Module,
        batch_size: int,
        iterations: int,
        device: torch.device,
        eval_freq: int = 1000,
        eval_iters: int = 100,
        scheduler: Optional = None,  # Type: torch.optim.lr_scheduler._LRScheduler
        grad_clip_max_norm: Optional[float] = None,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        mixed_precision: bool = False,
        train_projector: bool = True,
        feed_layers: Optional[List[int]] = None
    ) -> None:

        # Logging
        self.logger = logging.getLogger()
        self.writer = writer

        # Saving
        self.save_path = save_path

        # Device
        self.device = device

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.generator = generator
        self.projector = projector
        self.train_projector = train_projector
        self.feed_layers = feed_layers

        #  Eval
        self.eval_freq = eval_freq
        self.eval_iters = eval_iters

        # Scheduler
        self.scheduler = scheduler
        self.grad_clip_max_norm = grad_clip_max_norm

        # Batch & Iteration
        self.batch_size = batch_size
        self.iterations = iterations
        self.start_iteration = 0

        # Floating-point precision
        self.mixed_precision = (
            True if self.device.type == "cuda" and mixed_precision else False
        )
        self.scaler = GradScaler() if self.mixed_precision else None

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.train_acc_metric = LossMetric()
        self.train_loss_metric = LossMetric()

        self.val_acc_metric = LossMetric()
        self.val_loss_metric = LossMetric()

        # Best
        self.best_loss = -1

    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        epoch = 0
        iteration = self.start_iteration
        while iteration < self.iterations:
            if iteration + self.eval_freq < self.iterations:
                num_iters = self.eval_freq
            else:
                num_iters = self.iterations - iteration

            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(epoch, num_iters)
            else:
                self._train_loop(epoch, num_iters)

            self._val_loop(epoch, self.eval_iters)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time, iteration)

            iteration += num_iters
            epoch += 1

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(
            os.path.join(self.save_path, "final_model.pt"), self.iterations
        )

    def _train_loop(self, epoch: int, iterations: int) -> None:
        """
        Regular train loop

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Set to eval
        self.generator.eval()

        if self.train_projector:
            self.projector.train()
        else:
            self.projector.eval()

        for i in range(iterations):
            # To device
            z = self.generator.sample_latent(self.batch_size)
            z = z.to(self.device)
            z_orig = z

            # Original features
            with torch.no_grad():
                orig_feats = self.generator.get_features(z)
                orig_feats = self.projector(orig_feats)

            # Apply Directions
            self.optimizer.zero_grad()
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
            loss.backward()

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
            )

        pbar.close()

    def _train_loop_amp(self, epoch: int, iterations: int) -> None:
        """
        Train loop with Automatic Mixed Precision

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(iterations), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Loop
        for i in range(iterations):
            # To device
            z = self.generator.sample_latent(self.batch_size)
            z = z.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()

            # Use amp in forward pass
            with autocast():
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

                    # Get features
                    feats = self.generator.get_features(z[start:end, ...])
                    feats = self.projector(feats)

                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Update optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            # Update metrics
            self.train_acc_metric.update(acc.item(), z.shape[0])
            self.train_loss_metric.update(loss.item(), z.shape[0])

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(
                f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
            )

        pbar.close()

    def _val_loop(self, epoch: int, iterations: int) -> None:
        """
        Standard validation loop

        Args:
            epoch: current epoch
            iterations: iterations to run model
        """
        # Progress bar
        pbar = tqdm.tqdm(total=iterations, leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # Set to eval
        self.model.eval()
        self.generator.eval()
        self.projector.eval()

        # Loop
        for i in range(iterations):
            with torch.no_grad():
                # To device
                z = self.generator.sample_latent(self.batch_size)
                z = z.to(self.device)

                # Original features
                orig_feats = self.generator.get_features(z)
                orig_feats = self.projector(orig_feats)

                # Apply Directions
                z = self.model(z)

                # Forward
                features = []
                for j in range(z.shape[0] // self.batch_size):
                    # Prepare batch
                    start, end = j * self.batch_size, (j + 1) * self.batch_size

                    # Get features
                    feats = self.generator.get_features(z[start:end, ...])
                    feats = self.projector(feats)

                    # Take feature divergence
                    feats = feats - orig_feats
                    feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

                    features.append(feats)
                features = torch.cat(features, dim=0)

                # Loss
                acc, loss = self.loss_fn(features)
                self.val_acc_metric.update(acc.item(), z.shape[0])
                self.val_loss_metric.update(loss.item(), z.shape[0])

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(
                    f"Acc: {acc.item():.3f} Loss: {loss.item():.3f}", refresh=False
                )

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float, iteration: int):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # Save model
        if self.save_path is not None:
            self._save_model(os.path.join(self.save_path, "most_recent.pt"), iteration)

        eval_loss = self.val_loss_metric.compute()
        if self.best_loss == -1 or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self._save_model(os.path.join(self.save_path, "best_model.pt"), iteration)

        # Clear metrics
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()
        self.val_loss_metric.reset()
        self.val_acc_metric.reset()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train acc: {self.train_acc_metric.compute():.3f} "
        s += f"| Train loss: {self.train_loss_metric.compute():.3f} "
        s += f"| Val acc: {self.val_acc_metric.compute():.3f} "
        s += f"| Val loss: {self.val_loss_metric.compute():.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"

        return s

    def _write_to_tb(self, iteration):
        self.writer.add_scalar(
            "Loss/train", self.train_loss_metric.compute(), iteration
        )
        self.writer.add_scalar("Acc/train", self.train_acc_metric.compute(), iteration)
        self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), iteration)
        self.writer.add_scalar("Acc/val", self.val_acc_metric.compute(), iteration)

    def _save_model(self, path, iteration):
        obj = {
            "iteration": iteration + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "projector": self.projector.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict() if self.mixed_precision else None,
        }
        torch.save(obj, os.path.join(self.save_path, path))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_iteration = checkpoint["iteration"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.mixed_precision and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scheduler"])

        if self.start_iteration > self.iterations:
            raise ValueError("Starting iteration is larger than total iterations")

        self.logger.info(
            f"Checkpoint loaded, resuming from iteration {self.start_iteration}"
        )
