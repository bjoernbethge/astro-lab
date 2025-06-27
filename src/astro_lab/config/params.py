"""
Simple Parameter Management for AstroLab
=======================================

Basic parameter handling for training configurations.
"""

from typing import Any, Dict


def get_trainer_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trainer parameters from config."""
    trainer_params = {}

    # Map common parameters
    if "epochs" in config:
        trainer_params["max_epochs"] = config["epochs"]
    if "max_epochs" in config:
        trainer_params["max_epochs"] = config["max_epochs"]

    # Direct trainer parameters
    for key in [
        "accelerator",
        "devices",
        "precision",
        "gradient_clip_val",
        "accumulate_grad_batches",
        "log_every_n_steps",
        "val_check_interval",
        "check_val_every_n_epoch",
        "num_sanity_val_steps",
    ]:
        if key in config:
            trainer_params[key] = config[key]

    return trainer_params


def get_lightning_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Lightning model parameters from config."""
    lightning_params = {}

    # Direct Lightning parameters
    for key in [
        "learning_rate",
        "weight_decay",
        "optimizer",
        "scheduler",
        "warmup_steps",
        "beta1",
        "beta2",
        "eps",
    ]:
        if key in config:
            lightning_params[key] = config[key]

    # Model parameters
    if "model" in config:
        lightning_params["model"] = config["model"]

    return lightning_params


def get_data_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data parameters from config."""
    data_params = {}

    # Direct data parameters
    for key in [
        "dataset",
        "data_dir",
        "batch_size",
        "num_workers",
        "max_samples",
        "return_tensor",
        "split_ratios",
        "shuffle",
        "pin_memory",
    ]:
        if key in config:
            data_params[key] = config[key]

    return data_params


def get_mlflow_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MLflow parameters from config."""
    mlflow_params = {}

    if "mlflow" in config:
        mlflow_params.update(config["mlflow"])

    return mlflow_params
