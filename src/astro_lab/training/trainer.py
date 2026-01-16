"""
AstroLab Trainer - TensorDict Optimized
=======================================

Main trainer class using PyTorch Lightning with native TensorDict support
for seamless integration with modern PyTorch data pipelines.
"""

import logging
from typing import Dict, List, Optional, Union, cast

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.trainer.trainer import _PRECISION_INPUT

from astro_lab.config import get_data_paths
from astro_lab.training import BatchProcessingMixin, MemoryEfficientMixin

logger = logging.getLogger(__name__)


class AstroTrainer(L.Trainer, BatchProcessingMixin, MemoryEfficientMixin):
    """
    Extension of PyTorch Lightning Trainer with config-driven defaults, MLflow integration,
    and batch/memory utilities via Mixins.
    """

    def __init__(
        self,
        experiment_name: str = "astro_gnn",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        max_epochs: Optional[int] = None,
        devices: Union[int, str, List[int]] = "auto",
        accelerator: str = "auto",
        precision: Union[int, str] = "32-true",
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_monitor: str = "val_loss",
        early_stopping_mode: str = "min",
        checkpoint_monitor: str = "val_loss",
        checkpoint_mode: str = "min",
        checkpoint_save_top_k: int = 3,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        log_every_n_steps: int = 10,
        val_check_interval: Union[int, float] = 1.0,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        scheduler_config: Optional[Dict] = None,
        config: Optional[dict] = None,
        **kwargs,
    ):
        if "config" in kwargs:
            kwargs.pop("config")

        # Store scheduler config for model initialization
        self.scheduler_config = scheduler_config or {}

        if config is not None:
            experiment_name = experiment_name or config.get(
                "experiment_name", "astro_gnn"
            )
            run_name = run_name or config.get("run_name")
            max_epochs = max_epochs or config.get("max_epochs", 100)
            tracking_uri = tracking_uri or get_data_paths()["mlruns_dir"]
            checkpoint_dir = checkpoint_dir or get_data_paths()["checkpoint_dir"]
            devices = devices or config.get("devices", "auto")
            accelerator = accelerator or config.get("accelerator", "auto")
            precision = precision or config.get("precision", "32-true")
            gradient_clip_val = gradient_clip_val or config.get(
                "gradient_clip_val", 1.0
            )
            accumulate_grad_batches = accumulate_grad_batches or config.get(
                "accumulate_grad_batches", 1
            )
            early_stopping = (
                early_stopping
                if early_stopping is not None
                else config.get("early_stopping", True)
            )
            early_stopping_patience = early_stopping_patience or config.get(
                "early_stopping_patience", 10
            )
            early_stopping_monitor = early_stopping_monitor or config.get(
                "early_stopping_monitor", "val_loss"
            )
            early_stopping_mode = early_stopping_mode or config.get(
                "early_stopping_mode", "min"
            )
            checkpoint_monitor = checkpoint_monitor or config.get(
                "checkpoint_monitor", "val_loss"
            )
            checkpoint_mode = checkpoint_mode or config.get("checkpoint_mode", "min")
            checkpoint_save_top_k = checkpoint_save_top_k or config.get(
                "checkpoint_save_top_k", 3
            )
            enable_progress_bar = (
                enable_progress_bar
                if enable_progress_bar is not None
                else config.get("enable_progress_bar", True)
            )
            enable_model_summary = (
                enable_model_summary
                if enable_model_summary is not None
                else config.get("enable_model_summary", True)
            )
            log_every_n_steps = log_every_n_steps or config.get("log_every_n_steps", 10)
            val_check_interval = val_check_interval or config.get(
                "val_check_interval", 1.0
            )
            limit_train_batches = limit_train_batches or config.get(
                "limit_train_batches", 1.0
            )
            limit_val_batches = limit_val_batches or config.get(
                "limit_val_batches", 1.0
            )

            # Update scheduler config from config
            if not self.scheduler_config:
                self.scheduler_config = {
                    "scheduler": config.get("scheduler", "cosine"),
                    "warmup_epochs": config.get("warmup_epochs", 5),
                }

        # Callbacks
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="model",
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_top_k=checkpoint_save_top_k,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        if early_stopping:
            early_stop_callback = EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                mode=early_stopping_mode,
                verbose=True,
            )
            callbacks.append(early_stop_callback)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        if enable_progress_bar:
            progress_bar = RichProgressBar()
            callbacks.append(progress_bar)

        logger_instance = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            log_model=True,
        )

        int_to_str_precision = {16: "16-true", 32: "32-true", 64: "64-true"}
        allowed_str_precisions = {
            "16-true",
            "16-mixed",
            "32-true",
            "64-true",
            "bf16-mixed",
            "bf16-true",
            "transformer-engine",
            "transformer-engine-float16",
        }
        if isinstance(precision, int) and precision in int_to_str_precision:
            trainer_precision = int_to_str_precision[precision]
        elif isinstance(precision, str) and precision in allowed_str_precisions:
            trainer_precision = precision
        else:
            trainer_precision = "32-true"

        super().__init__(
            max_epochs=max_epochs,
            devices=devices,
            accelerator=accelerator,
            precision=cast(_PRECISION_INPUT, trainer_precision),
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            logger=logger_instance,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=val_check_interval,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            **kwargs,
        )

        logger.info(f"AstroTrainer initialized with {devices} devices")
        if self.scheduler_config:
            logger.info(f"Scheduler config: {self.scheduler_config}")

    def fit(
        self,
        model,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        ckpt_path=None,
    ):
        """Override fit to pass scheduler config to model if needed."""
        # Pass scheduler config to model if it supports it
        if hasattr(model, "scheduler_config") and self.scheduler_config:
            object.__setattr__(model, "scheduler_config", self.scheduler_config)
            logger.info(f"Passed scheduler config to model: {self.scheduler_config}")

        return super().fit(
            model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )
