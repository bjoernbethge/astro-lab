"""
Centralized Parameter Distribution for AstroLab
===============================================

Unifies parameter passing between CLI, Trainer, Lightning and Optuna.
Resolves inconsistencies and parameter conflicts.
"""

from typing import Any, Dict, Set, Tuple

# Define parameter categories
TRAINER_PARAMS: Set[str] = {
    "max_epochs",
    "accelerator",
    "devices",
    "precision",
    "gradient_clip_val",
    "accumulate_grad_batches",
    "enable_swa",
    "patience",
    "monitor",
    "mode",
    "log_every_n_steps",
    "check_val_every_n_epoch",
    "num_sanity_val_steps",
    "model",
}

LIGHTNING_PARAMS: Set[str] = {
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "warmup_steps",
    "beta1",
    "beta2",
    "eps",
    "model",
}

OPTUNA_PARAMS: Set[str] = {
    "n_trials",
    "timeout",
    "study_name",
    "sampler",
    "pruner",
    "storage",
}

MLFLOW_PARAMS: Set[str] = {
    "experiment_name",
    "tracking_uri",
    "run_name",
    "tags",
    "log_model",
}

DATA_PARAMS: Set[str] = {
    "dataset",
    "data_dir",
    "batch_size",
    "num_workers",
    "max_samples",
    "return_tensor",
    "split_ratios",
    "shuffle",
    "pin_memory",
}

# Parameters that should not be passed through
EXCLUDED_PARAMS: Set[str] = {
    "direction",  # Only for Optuna Study, not for Trainer
    "search_space",  # Only for Optuna, not for Trainer
}


def distribute_config_parameters(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Distributes configuration parameters to the appropriate components.

    Args:
        config: Complete configuration dictionary

    Returns:
        Dictionary with categorized parameters
    """
    distributed = {
        "trainer": {},
        "lightning": {},
        "optuna": {},
        "mlflow": {},
        "data": {},
        "excluded": {},
    }

    # Training section -> Trainer + Lightning
    if "training" in config:
        training_config = config["training"]
        for key, value in training_config.items():
            if key in TRAINER_PARAMS:
                distributed["trainer"][key] = value
            elif key in LIGHTNING_PARAMS:
                distributed["lightning"][key] = value
            # Ignore unknown training params

    # MLflow section
    if "mlflow" in config:
        distributed["mlflow"].update(config["mlflow"])

    # Data section
    if "data" in config:
        distributed["data"].update(config["data"])

    # Optimization section -> Optuna + Excluded
    if "optimization" in config:
        opt_config = config["optimization"]
        for key, value in opt_config.items():
            if key in EXCLUDED_PARAMS:
                distributed["excluded"][key] = value
            elif key in OPTUNA_PARAMS:
                distributed["optuna"][key] = value

    # Model section -> Lightning (model parameters)
    if "model" in config:
        model_config = config["model"]
        # Model type and params go to Lightning
        if "type" in model_config:
            distributed["lightning"]["model_type"] = model_config["type"]
        if "params" in model_config:
            distributed["lightning"].update(model_config["params"])

    # Modellname auf Top-Level
    if "model" in config and isinstance(config["model"], str):
        distributed["trainer"]["model"] = config["model"]
        distributed["lightning"]["model"] = config["model"]

    return distributed


def _flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flattens nested configuration structure.

    Args:
        config: Nested configuration dictionary
        prefix: Prefix for keys

    Returns:
        Flattened configuration dictionary
    """
    flat = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict) and key not in [
            "search_space",
            "tags",
            "model",
            "callbacks",
        ]:
            # Recursively flatten, except for special dicts
            flat.update(_flatten_config(value, full_key))
        else:
            flat[key] = value

    return flat


def get_trainer_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts only Trainer parameters."""
    distributed = distribute_config_parameters(config)
    return distributed["trainer"]


def get_lightning_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts only Lightning parameters."""
    distributed = distribute_config_parameters(config)
    return distributed["lightning"]


def get_optuna_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts only Optuna parameters."""
    distributed = distribute_config_parameters(config)
    return distributed["optuna"]


def get_mlflow_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts only MLflow parameters."""
    distributed = distribute_config_parameters(config)
    return distributed["mlflow"]


def get_data_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts only Data parameters."""
    distributed = distribute_config_parameters(config)
    return distributed["data"]


def validate_parameter_conflicts(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates configuration for parameter conflicts.

    Args:
        config: Configuration to validate

    Returns:
        (is_valid, error_message)
    """
    try:
        distributed = distribute_config_parameters(config)

        # Check for missing critical parameters
        if "learning_rate" not in distributed["lightning"]:
            return False, "learning_rate missing in configuration"

        # Check for Optuna-specific conflicts
        if "optimization" in config:
            opt_config = config["optimization"]
            if "search_space" in opt_config:
                search_space = opt_config["search_space"]

                # Validate learning_rate range
                if "learning_rate" in search_space:
                    lr_config = search_space["learning_rate"]
                    if lr_config.get("low", 0) >= lr_config.get("high", 1):
                        return (
                            False,
                            f"learning_rate: low ({lr_config.get('low')}) >= high ({lr_config.get('high')})",
                        )

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def print_parameter_distribution(config: Dict[str, Any]) -> None:
    """Debug function: Shows parameter distribution."""
    distributed = distribute_config_parameters(config)

    print("🔧 Parameter Distribution:")
    print("=" * 40)

    for category, params in distributed.items():
        if params:
            print(f"\n📋 {category.upper()}:")
            for key, value in params.items():
                print(f"   • {key}: {value}")

    # Validation
    is_valid, error = validate_parameter_conflicts(config)
    if not is_valid:
        print(f"\n❌ Validation Error: {error}")
    else:
        print("\n✅ Configuration is valid")


__all__ = [
    "distribute_config_parameters",
    "get_trainer_params",
    "get_lightning_params",
    "get_optuna_params",
    "get_mlflow_params",
    "get_data_params",
    "validate_parameter_conflicts",
    "print_parameter_distribution",
]
