"""
Combined Mixins for Common Use Cases
====================================

Pre-configured combinations of mixins for typical astronomical ML scenarios.
Enhanced with production-ready defaults and adaptive configurations.
"""

from typing import Dict, Any, Optional
import torch
from ....config.adaptive import get_adaptive_config
from .base import CombinedMixin


class StandardModelMixin(CombinedMixin):
    """
    Standard model with basic functionality and adaptive configuration.
    
    Features:
    - Automatic hardware adaptation
    - MLflow tracking
    - Basic metrics and visualization
    - Memory-efficient defaults
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin", 
        "VisualizationMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, **kwargs):
        """Initialize with adaptive defaults."""
        super().__init__(**kwargs)
        
        # Get adaptive configuration
        adaptive_config = get_adaptive_config(
            task=getattr(self, 'task', 'node_classification'),
            model_type=self.__class__.__name__,
        )
        
        # Apply adaptive settings if not explicitly set
        for key, value in adaptive_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        
        # Setup MLflow if enabled
        if getattr(self, 'mlflow_logging', True):
            self.setup_mlflow(
                experiment_name=kwargs.get('experiment_name', 'astrolab_experiments'),
                run_name=kwargs.get('run_name'),
                tags=kwargs.get('tags'),
            )


class ProductionModelMixin(CombinedMixin):
    """
    Production-ready model with all optimizations enabled.
    
    Features:
    - torch.compile enabled by default (if supported)
    - Mixed precision training
    - Gradient checkpointing for memory efficiency
    - Robust error handling
    - Comprehensive logging
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "VisualizationMixin",
        "MLflowMixin",
        "EfficientTrainingMixin",
    ]
    
    def __init__(self, **kwargs):
        """Initialize with production defaults."""
        super().__init__(**kwargs)
        
        # Production defaults
        self.compile_model = kwargs.get('compile_model', torch.cuda.is_available())
        self.compile_mode = kwargs.get('compile_mode', 'default')
        self.compile_dynamic = kwargs.get('compile_dynamic', True)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        self.mixed_precision = kwargs.get('mixed_precision', True)
        self.enable_profiling = kwargs.get('enable_profiling', False)
        
        # Error handling
        self.nan_detection = kwargs.get('nan_detection', True)
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        
        # Setup comprehensive MLflow tracking
        if getattr(self, 'mlflow_logging', True):
            self.setup_mlflow(
                experiment_name=kwargs.get('experiment_name', 'production_models'),
                tags={'model_type': 'production', 'optimized': 'true'},
            )


class HPOModelMixin(CombinedMixin):
    """
    Model optimized for hyperparameter optimization with efficient resets.
    
    Features:
    - Efficient parameter reset without recreating model
    - Memory-efficient HPO tracking
    - Adaptive search space based on hardware
    - Built-in pruning support
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "HPOResetMixin",
        "HPOMemoryMixin",
        "EfficientTrainingMixin",
        "ArchitectureSearchMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, **kwargs):
        """Initialize with HPO-friendly defaults."""
        super().__init__(**kwargs)
        
        # HPO-specific settings
        self.enable_pruning = kwargs.get('enable_pruning', True)
        self.pruning_patience = kwargs.get('pruning_patience', 5)
        self.reset_optimizer_on_reset = kwargs.get('reset_optimizer_on_reset', True)
        
        # Lighter logging for HPO
        self.mlflow_logging = kwargs.get('mlflow_logging', False)  # Disable by default
        self.log_every_n_steps = kwargs.get('log_every_n_steps', 50)
        
        # Memory efficiency
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.compile_model = False  # Disable compilation for faster iteration


class AstronomicalModelMixin(CombinedMixin):
    """
    Model with astronomical domain-specific features and optimizations.
    
    Features:
    - Survey-specific defaults
    - Astronomical augmentations
    - Coordinate system handling
    - Physical constraints
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "VisualizationMixin",
        "AstronomicalAugmentationMixin",
        "AstronomicalLossMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, survey: Optional[str] = None, **kwargs):
        """Initialize with astronomical defaults."""
        super().__init__(**kwargs)
        
        # Get survey-specific configuration
        if survey:
            adaptive_config = get_adaptive_config(
                survey=survey,
                task=getattr(self, 'task', 'node_classification'),
            )
            
            # Apply survey-specific settings
            for key, value in adaptive_config.items():
                if key not in kwargs:  # Don't override explicit settings
                    setattr(self, key, value)
        
        # Astronomical defaults
        self.coordinate_system = kwargs.get('coordinate_system', 'icrs')
        self.distance_metric = kwargs.get('distance_metric', 'angular')
        self.magnitude_system = kwargs.get('magnitude_system', 'ab')
        
        # Physical constraints
        self.enforce_physical_constraints = kwargs.get('enforce_physical_constraints', True)
        self.min_stellar_mass = kwargs.get('min_stellar_mass', 0.08)  # Solar masses
        self.max_stellar_mass = kwargs.get('max_stellar_mass', 150.0)
        
        # Setup MLflow with astronomical tags
        if getattr(self, 'mlflow_logging', True):
            self.setup_mlflow(
                experiment_name=f"astronomical_{survey}" if survey else "astronomical_models",
                tags={
                    'survey': survey or 'generic',
                    'coordinate_system': self.coordinate_system,
                    'astronomical': 'true',
                },
            )


class LargeScaleModelMixin(CombinedMixin):
    """
    Model optimized for large-scale astronomical datasets.
    
    Features:
    - Automatic sampling strategy selection
    - Dynamic batching
    - Memory-efficient processing
    - Distributed training support
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "EfficientTrainingMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, num_nodes: Optional[int] = None, **kwargs):
        """Initialize with large-scale defaults."""
        super().__init__(**kwargs)
        
        # Determine optimal configuration based on dataset size
        if num_nodes:
            dataset_stats = {'num_nodes': num_nodes, 'num_edges': num_nodes * 10}  # Estimate
            adaptive_config = get_adaptive_config(dataset_stats=dataset_stats)
            
            # Apply adaptive settings
            self.sampling_strategy = adaptive_config.get('sampling_strategy', 'neighbor')
            self.neighbor_sizes = adaptive_config.get('neighbor_sizes', [25, 10])
            self.enable_dynamic_batching = adaptive_config.get('enable_dynamic_batching', True)
        else:
            # Conservative defaults
            self.sampling_strategy = kwargs.get('sampling_strategy', 'neighbor')
            self.neighbor_sizes = kwargs.get('neighbor_sizes', [25, 10])
            self.enable_dynamic_batching = kwargs.get('enable_dynamic_batching', True)
        
        # Large-scale specific settings
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        self.compile_model = kwargs.get('compile_model', True)
        self.compile_dynamic = True  # Always use dynamic shapes
        self.accumulate_grad_batches = kwargs.get('accumulate_grad_batches', 4)
        
        # Distributed training readiness
        self.sync_batchnorm = kwargs.get('sync_batchnorm', True)
        self.find_unused_parameters = kwargs.get('find_unused_parameters', False)


class ResearchModelMixin(CombinedMixin):
    """
    Model for research with comprehensive analysis and interpretability.
    
    Features:
    - Full explainability suite
    - Extensive visualization
    - Detailed logging and debugging
    - Experimental features enabled
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "VisualizationMixin",
        "ExplainabilityMixin",
        "AstronomicalAugmentationMixin",
        "AstronomicalLossMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, **kwargs):
        """Initialize with research-friendly defaults."""
        super().__init__(**kwargs)
        
        # Research settings
        self.enable_attention_visualization = kwargs.get('enable_attention_visualization', True)
        self.log_embeddings = kwargs.get('log_embeddings', True)
        self.log_gradients = kwargs.get('log_gradients', True)
        self.profile_memory = kwargs.get('profile_memory', True)
        
        # Comprehensive logging
        self.log_every_n_steps = kwargs.get('log_every_n_steps', 10)
        self.log_batch_statistics = kwargs.get('log_batch_statistics', True)
        
        # Experimental features
        self.enable_experimental = kwargs.get('enable_experimental', True)
        self.stochastic_depth = kwargs.get('stochastic_depth', 0.1)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        
        # Setup detailed MLflow tracking
        if getattr(self, 'mlflow_logging', True):
            self.setup_mlflow(
                experiment_name=kwargs.get('experiment_name', 'research_experiments'),
                tags={
                    'model_type': 'research',
                    'explainable': 'true',
                    'experimental': str(self.enable_experimental),
                },
            )


class LightweightModelMixin(CombinedMixin):
    """
    Lightweight model for edge deployment or quick experiments.
    
    Features:
    - Minimal dependencies
    - Reduced memory footprint
    - Fast inference
    - Mobile/edge ready
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
    ]
    
    def __init__(self, **kwargs):
        """Initialize with lightweight defaults."""
        super().__init__(**kwargs)
        
        # Lightweight settings
        self.compile_model = False  # Avoid compilation overhead
        self.mlflow_logging = False  # No external dependencies
        self.precision = kwargs.get('precision', '32-true')  # Full precision for compatibility
        
        # Model compression
        self.pruning_amount = kwargs.get('pruning_amount', 0.3)
        self.quantization = kwargs.get('quantization', False)
        
        # Minimal features
        self.hidden_dim = min(getattr(self, 'hidden_dim', 64), 64)
        self.num_layers = min(getattr(self, 'num_layers', 2), 2)
        self.dropout = 0.0  # No dropout for inference


class FullAstronomicalModelMixin(CombinedMixin):
    """
    Complete model with all astronomical features for production use.
    
    Combines:
    - Production optimizations
    - Astronomical domain features
    - Large-scale support
    - Comprehensive tracking
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "VisualizationMixin",
        "ExplainabilityMixin",
        "AstronomicalAugmentationMixin",
        "AstronomicalLossMixin",
        "EfficientTrainingMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, survey: Optional[str] = None, **kwargs):
        """Initialize with full astronomical capabilities."""
        super().__init__(**kwargs)
        
        # Get optimal configuration
        adaptive_config = get_adaptive_config(
            survey=survey,
            task=getattr(self, 'task', 'node_classification'),
            model_type=self.__class__.__name__,
        )
        
        # Apply all adaptive settings
        for key, value in adaptive_config.items():
            if key not in kwargs:
                setattr(self, key, value)
        
        # Enable all features
        self.enable_all_features = True
        
        # Setup comprehensive tracking
        if getattr(self, 'mlflow_logging', True):
            self.setup_mlflow(
                experiment_name=f"astrolab_full_{survey}" if survey else "astrolab_full",
                tags={
                    'model_type': 'full_astronomical',
                    'survey': survey or 'multi',
                    'production': 'true',
                },
            )


class ExplainableModelMixin(CombinedMixin):
    """
    Model with focus on interpretability and explainability.
    
    Features:
    - Attention visualization
    - Feature importance tracking
    - Decision path analysis
    - Uncertainty quantification
    """
    
    _included_mixins = [
        "MetricsMixin",
        "OptimizationMixin",
        "VisualizationMixin",
        "ExplainabilityMixin",
        "MLflowMixin",
    ]
    
    def __init__(self, **kwargs):
        """Initialize with explainability focus."""
        super().__init__(**kwargs)
        
        # Explainability settings
        self.track_attention_weights = kwargs.get('track_attention_weights', True)
        self.compute_feature_importance = kwargs.get('compute_feature_importance', True)
        self.enable_uncertainty = kwargs.get('enable_uncertainty', True)
        
        # Use interpretable architectures
        self.use_attention = kwargs.get('use_attention', True)
        self.residual_connections = kwargs.get('residual_connections', False)  # Cleaner gradients
        
        # Detailed logging
        self.log_decision_paths = kwargs.get('log_decision_paths', True)
        self.log_confidence_scores = kwargs.get('log_confidence_scores', True)
        
        # Setup MLflow with explainability tracking
        if getattr(self, 'mlflow_logging', True):
            self.setup_mlflow(
                experiment_name=kwargs.get('experiment_name', 'explainable_models'),
                tags={
                    'model_type': 'explainable',
                    'interpretable': 'true',
                    'uncertainty': str(self.enable_uncertainty),
                },
            )


# Aliases for backward compatibility
FullFeaturedModelMixin = FullAstronomicalModelMixin
FastTrainingModelMixin = HPOModelMixin
