"""
Optimization Mixin for AstroLab Models
=====================================

Advanced optimization strategies for astronomical models.
"""

import math
from typing import Any, Dict, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn


class OptimizationMixin(nn.Module):
    """Advanced optimization strategies for astronomical models."""

    def __init__(self):
        super().__init__()
        self.trainer: L.Trainer = None  # type: ignore

    def configure_astro_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers with astronomical domain knowledge."""
        learning_rate = getattr(self, "learning_rate", 1e-3)
        weight_decay = getattr(self, "weight_decay", 1e-5)

        # Use AdamW with modern hyperparameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Calculate total steps for OneCycleLR
        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            # Fallback calculation
            epochs = int(getattr(self.trainer, "max_epochs", 100))
            dataloader = self.trainer.train_dataloader
            if dataloader is not None and hasattr(dataloader, "__len__"):
                steps_per_epoch = len(dataloader)
            else:
                steps_per_epoch = 100  # Default fallback
            total_steps = epochs * steps_per_epoch

        # OneCycleLR for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_cosine_annealing(self, T_max: int = 100) -> Dict[str, Any]:
        """Configure cosine annealing scheduler."""
        learning_rate = getattr(self, "learning_rate", 1e-3)
        weight_decay = getattr(self, "weight_decay", 1e-5)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_warmup_scheduler(self, warmup_steps: int = 1000) -> Dict[str, Any]:
        """Configure learning rate warmup with cosine decay."""
        learning_rate = getattr(self, "learning_rate", 1e-3)
        weight_decay = getattr(self, "weight_decay", 1e-5)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - warmup_steps) / (10000 - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_astronomical_optimizer(self) -> Dict[str, Any]:
        """Configure optimizer specifically for astronomical data patterns."""
        learning_rate = getattr(self, "learning_rate", 1e-3)
        weight_decay = getattr(self, "weight_decay", 1e-5)

        # Use Lion optimizer for better performance on astronomical data
        try:
            from lion_pytorch import Lion

            optimizer = Lion(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                use_triton=False,
            )
        except ImportError:
            # Fallback to AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )

        # Cosine annealing with restarts for astronomical data
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=learning_rate * 0.001,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get current optimization configuration."""
        return {
            "learning_rate": getattr(self, "learning_rate", 1e-3),
            "weight_decay": getattr(self, "weight_decay", 1e-5),
            "optimizer_type": "AdamW",
            "scheduler_type": "OneCycleLR",
        }
