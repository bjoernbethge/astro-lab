"""
AstroLab Model Factory
======================

Factory for creating astronomical ML models.
Supports various model architectures and survey types.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple, Type, Callable

from astro_lab.models.base_gnn import BaseAstroGNN, BaseTemporalGNN, BaseTNGModel
from astro_lab.models.output_heads import OutputHeadRegistry, create_output_head
from .astro import AstroSurveyGNN
from .config import ModelConfig
from astro_lab.utils.config.surveys import get_survey_config, get_available_surveys

class ModelRegistry:
    """Central registry for all available models."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        """Decorator to register models."""

        def decorator(model_class: Type) -> Type:
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> nn.Module:
        """Create model by type."""
        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {available}"
            )

        return cls._models[model_type](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available model types."""
        return list(cls._models.keys())

class ModelFactory:
    """Centralized model factory with survey-specific configurations."""

    # Task-specific configurations
    TASK_CONFIGS = {
        "stellar_classification": {
            "output_head": "classification",
            "output_dim": None,  # Will be determined automatically
            "pooling": "mean",
        },
        "galaxy_property_prediction": {
            "output_head": "regression",
            "output_dim": 5,  # Mass, SFR, metallicity, size, morphology
            "pooling": "attention",
        },
        "transient_detection": {
            "output_head": "classification",
            "output_dim": None,  # Will be determined automatically
            "pooling": "max",
        },
        "period_detection": {
            "output_head": "period_detection",
            "output_dim": 1,
            "pooling": "mean",
        },
        "shape_modeling": {
            "output_head": "shape_modeling",
            "output_dim": 6,
            "pooling": "mean",
        },
        "cosmological_inference": {
            "output_head": "cosmological",
            "output_dim": 6,
            "pooling": "attention",
        },
    }

    @classmethod
    def infer_num_classes_from_data(cls, data_loader: Any, target_key: str = "target") -> int:
        """
        Automatically infer the number of classes from the dataset.
        
        Args:
            data_loader: PyTorch DataLoader or similar iterable
            target_key: Key for target tensor in batch dict
            
        Returns:
            Number of unique classes found in the data
        """
        try:
            all_targets = []
            
            # Sample a few batches to determine class count
            for i, batch in enumerate(data_loader):
                if i >= 10:  # Limit to first 10 batches for efficiency
                    break
                    
                if isinstance(batch, dict):
                    targets = batch.get(target_key)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    targets = batch[1]  # Assume (data, target) format
                else:
                    targets = batch
                
                if targets is not None:
                    if isinstance(targets, torch.Tensor):
                        all_targets.append(targets.flatten())
                    else:
                        all_targets.append(torch.tensor(targets).flatten())
            
            if not all_targets:
                raise ValueError("No targets found in data loader")
            
            # Concatenate all targets and find unique values
            all_targets = torch.cat(all_targets)
            unique_classes = torch.unique(all_targets)
            num_classes = len(unique_classes)
            
            # Ensure classes are 0-indexed
            min_class = unique_classes.min().item()
            max_class = unique_classes.max().item()
            
            if min_class != 0:
                print(f"⚠️  Warning: Classes start from {min_class}, not 0")
            
            print(f"🔍 Automatically detected {num_classes} classes: {unique_classes.tolist()}")
            return num_classes
            
        except Exception as e:
            print(f"❌ Error inferring classes from data: {e}")
            # Fallback to default
            return 7

    @classmethod
    def create_survey_model(
        cls, 
        survey: str, 
        task: str = "stellar_classification", 
        data_loader: Optional[Any] = None,
        **kwargs: Any
    ) -> nn.Module:
        """Create model optimized for specific survey with automatic class detection."""

        # Get survey configuration
        survey_config = get_survey_config(survey)
        if not survey_config:
            available_surveys = get_available_surveys()
            raise ValueError(
                f"Unknown survey: {survey}. Available: {available_surveys}"
            )

        # Get task configuration
        task_config = cls.TASK_CONFIGS.get(task, {})
        if not task_config:
            available_tasks = list(cls.TASK_CONFIGS.keys())
            raise ValueError(f"Unknown task: {task}. Available: {available_tasks}")

        # Determine output dimension
        output_dim = None
        
        # Check if explicitly provided
        if 'output_dim' in kwargs:
            output_dim = kwargs['output_dim']
            print(f"🎯 Using explicitly provided output_dim: {output_dim}")
        elif data_loader is not None:
            # Auto-detect from data
            output_dim = cls.infer_num_classes_from_data(data_loader)
            print(f"🔍 Auto-detected output_dim: {output_dim} from data")
        else:
            # Use default from task config or fallback to 7
            output_dim = task_config.get('output_dim', 7)
            if output_dim is None:
                output_dim = 7  # Fallback for None values
            print(f"⚠️ No data_loader provided, using default {output_dim} classes")
        
        # Override task config with detected/explicit output_dim
        task_config['output_dim'] = output_dim

        # Merge configurations (kwargs override defaults)
        config = {**survey_config, **task_config, **kwargs}

        # Import here to avoid circular imports
        from astro_lab.models.astro import AstroSurveyGNN

        return AstroSurveyGNN(task=task, **config)

    @classmethod
    def create_temporal_model(
        cls, model_type: str = "alcdef", task: str = "period_detection", **kwargs
    ) -> nn.Module:
        """Create temporal model for time-series analysis."""

        temporal_configs = {
            "alcdef": {
                "hidden_dim": 128,
                "num_layers": 3,
                "recurrent_type": "lstm",
                "recurrent_layers": 2,
            },
            "lightcurve": {
                "hidden_dim": 96,
                "num_layers": 2,
                "recurrent_type": "gru",
                "recurrent_layers": 1,
            },
            "transient": {
                "hidden_dim": 256,
                "num_layers": 4,
                "recurrent_type": "lstm",
                "recurrent_layers": 3,
            },
        }

        config = temporal_configs.get(model_type, temporal_configs["alcdef"])
        task_config = cls.TASK_CONFIGS.get(task, {})

        # Merge configurations
        final_config = {**config, **task_config, **kwargs}

        # Import here to avoid circular imports
        from astro_lab.models.tgnn import ALCDEFTemporalGNN

        return ALCDEFTemporalGNN(**final_config)

    @classmethod
    def create_3d_stellar_model(
        cls,
        model_type: str = "point_cloud",
        num_stars: int = 1024,
        radius: float = 0.1,
        scales: Optional[List[float]] = None,
        **kwargs,
    ) -> nn.Module:
        """Create specialized model for 3D stellar data."""

        from astro_lab.models.point_cloud_models import create_stellar_point_cloud_model

        return create_stellar_point_cloud_model(
            model_type=model_type,
            num_stars=num_stars,
            radius=radius,
            scales=scales,
            **kwargs,
        )

    @classmethod
    def create_tng_model(
        cls, model_type: str = "cosmic_evolution", **kwargs
    ) -> nn.Module:
        """Create TNG simulation model."""

        tng_configs = {
            "cosmic_evolution": {
                "cosmological_features": True,
                "redshift_encoding": True,
                "hidden_dim": 256,
                "num_layers": 4,
            },
            "galaxy_formation": {
                "cosmological_features": False,
                "redshift_encoding": True,
                "hidden_dim": 192,
                "num_layers": 3,
            },
            "halo_merger": {
                "cosmological_features": True,
                "redshift_encoding": True,
                "conv_type": "gat",
                "hidden_dim": 256,
                "num_layers": 4,
            },
            "environmental_quenching": {
                "cosmological_features": False,
                "redshift_encoding": False,
                "hidden_dim": 128,
                "num_layers": 3,
            },
        }

        config = tng_configs.get(model_type, tng_configs["cosmic_evolution"])
        final_config = {**config, **kwargs}

        # Import here to avoid circular imports
        from astro_lab.models.tng_models import (
            CosmicEvolutionGNN,
            EnvironmentalQuenchingGNN,
            GalaxyFormationGNN,
            HaloMergerGNN,
        )

        model_classes = {
            "cosmic_evolution": CosmicEvolutionGNN,
            "galaxy_formation": GalaxyFormationGNN,
            "halo_merger": HaloMergerGNN,
            "environmental_quenching": EnvironmentalQuenchingGNN,
        }

        model_class = model_classes.get(model_type, CosmicEvolutionGNN)
        return model_class(**final_config)

    @classmethod
    def create_multi_survey_model(
        cls,
        surveys: List[str],
        task: str = "stellar_classification",
        fusion_strategy: str = "attention",
        **kwargs,
    ) -> nn.Module:
        """Create model that can handle multiple surveys."""

        # Combine configurations from multiple surveys
        combined_config = {}
        for survey in surveys:
            survey_config = get_survey_config(survey)
            for key, value in survey_config.items():
                if (
                    key == "use_photometry"
                    or key == "use_astrometry"
                    or key == "use_spectroscopy"
                ):
                    combined_config[key] = combined_config.get(key, False) or value
                elif key == "hidden_dim":
                    combined_config[key] = max(combined_config.get(key, 0), value)
                elif key == "num_layers":
                    combined_config[key] = max(combined_config.get(key, 0), value)
                else:
                    combined_config[key] = value

        # Add multi-survey specific configurations
        combined_config.update(
            {"fusion_strategy": fusion_strategy, "multi_survey": True, **kwargs}
        )

        task_config = cls.TASK_CONFIGS.get(task, {})
        final_config = {**combined_config, **task_config}

        # Import here to avoid circular imports
        from astro_lab.models.astro import AstroSurveyGNN

        return AstroSurveyGNN(task=task, **final_config)

# Convenience functions for common use cases
def create_gaia_classifier(
    num_classes: int = 7, hidden_dim: int = 128, **kwargs
) -> nn.Module:
    """Create Gaia stellar classifier."""
    return ModelFactory.create_survey_model(
        survey="gaia",
        task="stellar_classification",
        output_dim=num_classes,
        hidden_dim=hidden_dim,
        **kwargs,
    )

def create_sdss_galaxy_model(
    task: str = "galaxy_property_prediction", **kwargs
) -> nn.Module:
    """Create SDSS galaxy model."""
    return ModelFactory.create_survey_model(survey="sdss", task=task, **kwargs)

def create_lsst_transient_detector(**kwargs) -> nn.Module:
    """Create LSST transient detector."""
    return ModelFactory.create_survey_model(
        survey="lsst", task="transient_detection", **kwargs
    )

def create_asteroid_period_detector(**kwargs) -> nn.Module:
    """Create asteroid period detection model."""
    return ModelFactory.create_temporal_model(
        model_type="alcdef", task="period_detection", **kwargs
    )

def create_lightcurve_classifier(num_classes: int = 5, **kwargs) -> nn.Module:
    """Create lightcurve classification model."""
    return ModelFactory.create_temporal_model(
        model_type="lightcurve",
        task="stellar_classification",
        output_dim=num_classes,
        **kwargs,
    )

def create_stellar_cluster_analyzer(**kwargs) -> nn.Module:
    """Create stellar cluster analysis model."""
    return ModelFactory.create_3d_stellar_model(model_type="cluster", **kwargs)

def create_galactic_structure_model(**kwargs) -> nn.Module:
    """Create galactic structure analysis model."""
    return ModelFactory.create_3d_stellar_model(model_type="galactic", **kwargs)

# Model compilation utilities
def compile_astro_model(
    model: nn.Module,
    mode: str = "default",
    dynamic: bool = True,
) -> nn.Module:
    """Compile model for optimized inference."""
    try:
        return torch.compile(model, mode=mode, dynamic=dynamic)
    except (ImportError, AttributeError):
        # Fallback if torch.compile not available
        return model

# Model information utilities
def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "has_temporal": hasattr(model, "rnn"),
        "has_attention": any("GAT" in str(type(m)) for m in model.modules()),
        "conv_type": getattr(model, "conv_type", "unknown"),
        "hidden_dim": getattr(model, "hidden_dim", "unknown"),
    }

def list_available_models() -> Dict[str, List[str]]:
    """List all available models and configurations."""
    return {
        "surveys": get_available_surveys(),
        "tasks": list(ModelFactory.TASK_CONFIGS.keys()),
        "output_heads": OutputHeadRegistry.list_available(),
        "registered_models": ModelRegistry.list_available(),
    }
