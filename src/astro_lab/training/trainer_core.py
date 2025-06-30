"""
Core Trainer Module for AstroLab
===============================

Handles the main training loop, validation, and optimization logic.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from astro_lab.training.trainer_utils import (
    setup_cuda_graphs,
    setup_device,
    setup_mixed_precision,
    setup_torch_compile,
)

logger = logging.getLogger(__name__)


class TrainingCore:
    """
    Core training functionality for AstroLab models.

    Handles the main training loop, validation, and optimization logic
    with support for mixed precision, CUDA graphs, and torch.compile.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize training core.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}

        # Setup device
        self.device = setup_device(device)
        self.model.to(self.device)

        # Setup optimizations
        self.setup_optimizations()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        logger.info("Training core initialized")

    def setup_optimizations(self):
        """Setup training optimizations based on hardware and config."""
        # Mixed precision
        self.use_mixed_precision, self.grad_scaler = setup_mixed_precision(
            self.model, self.device, self.config.get("use_mixed_precision")
        )

        # Torch compile
        self.model = setup_torch_compile(
            self.model, self.device, self.config.get("use_torch_compile")
        )

        # CUDA graphs
        self.use_cuda_graphs = setup_cuda_graphs(
            self.model, self.device, self.config.get("use_cuda_graphs")
        )

        # CUDA graph state
        if self.use_cuda_graphs:
            self.cuda_graph = None
            self.static_input = None
            self.static_output = None

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (train_loss, train_acc)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            if self.use_mixed_precision:
                with torch.autocast(device_type="cuda"):
                    loss, acc, num_samples = self._forward_step(batch)
            else:
                loss, acc, num_samples = self._forward_step(batch)

            # Backward pass
            if self.use_mixed_precision:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            total_correct += acc * num_samples
            total_samples += num_samples

            # Logging
            if batch_idx % self.config.get("log_every_n_steps", 100) == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Acc: {acc:.4f}"
                )

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s, "
            f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
        )

        return avg_loss, avg_acc

    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (val_loss, val_acc)
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                if self.use_mixed_precision:
                    with torch.autocast(device_type="cuda"):
                        loss, acc, num_samples = self._forward_step(batch)
                else:
                    loss, acc, num_samples = self._forward_step(batch)

                # Update metrics
                total_loss += loss.item()
                total_correct += acc * num_samples
                total_samples += num_samples

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples

        logger.info(f"Validation - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        return avg_loss, avg_acc

    def _forward_step(self, batch: Any) -> Tuple[torch.Tensor, float, int]:
        """
        Perform forward step and return loss, accuracy, and number of samples.

        Args:
            batch: Input batch

        Returns:
            Tuple of (loss, accuracy, num_samples)
        """
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            outputs = self.model(*batch)
            targets = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            outputs = self.model(**batch)
            targets = batch.get("y", batch.get("target", None))
        else:
            outputs = self.model(batch)
            targets = None

        # Calculate loss
        if targets is not None:
            if outputs.dim() == 2 and targets.dim() == 1:
                loss = F.cross_entropy(outputs, targets)
                acc = (outputs.argmax(dim=1) == targets).float().mean().item()
            else:
                loss = F.mse_loss(outputs, targets)
                acc = 0.0  # MSE doesn't have accuracy
        else:
            loss = outputs.mean() if outputs.numel() > 0 else torch.tensor(0.0)
            acc = 0.0

        num_samples = (
            batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
        )

        return loss, acc, num_samples

    def _move_batch_to_device(self, batch: Any) -> Any:
        """
        Move batch to device.

        Args:
            batch: Input batch

        Returns:
            Batch moved to device
        """
        if hasattr(batch, "to"):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [b.to(self.device) if hasattr(b, "to") else b for b in batch]
        elif isinstance(batch, dict):
            return {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        else:
            return batch

    def update_scheduler(self, val_loss: float):
        """Update learning rate scheduler."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "training_history": self.training_history,
            "config": self.config,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.training_history = checkpoint["training_history"]

        logger.info(f"Checkpoint loaded from {filepath}")

    def test(self) -> Dict[str, float]:
        """
        Run test evaluation.

        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                batch = self._move_batch_to_device(batch)

                if self.use_mixed_precision:
                    with torch.autocast(device_type="cuda"):
                        loss, acc, num_samples = self._forward_step(batch)
                else:
                    loss, acc, num_samples = self._forward_step(batch)

                total_loss += loss.item()
                total_correct += acc * num_samples
                total_samples += num_samples

        test_loss = total_loss / len(self.test_loader)
        test_acc = total_correct / total_samples

        results = {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

        logger.info(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        return results
