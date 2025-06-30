"""
HPO (Hyperparameter Optimization) Mixins
========================================

Efficient HPO support with Optuna integration and memory optimizations.
"""

import gc
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, GCNConv, GINConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool
)

logger = logging.getLogger(__name__)


class HPOResetMixin:
    """
    Efficient parameter reset for HPO without model recreation.
    
    Based on 2025 best practices:
    - In-place parameter reset
    - Optimizer state clearing
    - Memory-efficient operations
    """
    
    def reset_all_parameters(self) -> None:
        """
        Reset all parameters efficiently for HPO.
        
        This method is optimized for:
        - Minimal memory allocation
        - Fast execution
        - Complete state reset
        """
        if not isinstance(self, nn.Module):
            raise TypeError("HPOResetMixin must be used with nn.Module")
        
        # Reset all modules with reset_parameters
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                # Manual reset for Linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Manual reset for Conv layers
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Reset batch norm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.reset_running_stats()
        
        # Clear gradients
        self.zero_grad(set_to_none=True)
        
        # Reset internal states if any
        if hasattr(self, '_reset_internal_states'):
            self._reset_internal_states()
        
        # Clear compiled model cache if exists
        if hasattr(self, '_compiled_forward'):
            self._compiled_forward = None
        
        logger.debug("All parameters reset successfully")
    
    def reset_optimizer_states(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Reset optimizer states for fresh training."""
        if optimizer is not None:
            optimizer.state.clear()
            # Reset learning rate scheduler if attached
            if hasattr(optimizer, '_lr_scheduler'):
                optimizer._lr_scheduler = None


class HPOMemoryMixin:
    """
    Memory optimization for HPO trials.
    
    Features:
    - Shared memory tensors
    - Efficient cleanup
    - GPU memory management
    """
    
    def __init__(self):
        """Initialize memory management."""
        self._shared_buffers = {}
        self._memory_stats = {
            'peak_memory_mb': 0,
            'current_memory_mb': 0,
            'num_resets': 0,
        }
    
    def cleanup_memory(self) -> None:
        """Clean up memory after trial."""
        # Clear shared buffers
        self._shared_buffers.clear()
        
        # Garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log memory stats
            self._memory_stats['current_memory_mb'] = torch.cuda.memory_allocated() / 1e6
            self._memory_stats['peak_memory_mb'] = max(
                self._memory_stats['peak_memory_mb'],
                torch.cuda.max_memory_allocated() / 1e6
            )
        
        self._memory_stats['num_resets'] += 1
    
    def share_buffer(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Share buffer across trials to save memory."""
        if name in self._shared_buffers:
            # Reuse existing buffer
            buffer = self._shared_buffers[name]
            if buffer.shape == tensor.shape:
                buffer.copy_(tensor)
                return buffer
        
        # Create new shared buffer
        self._shared_buffers[name] = tensor.clone()
        return self._shared_buffers[name]


class EfficientTrainingMixin:
    """
    Efficient training strategies for HPO.
    
    Features:
    - Gradient checkpointing
    - Mixed precision
    - Fused operations
    - torch.compile support
    """
    
    def enable_memory_efficient_training(self) -> None:
        """Enable memory-efficient training features."""
        if not isinstance(self, nn.Module):
            return
        
        # Enable gradient checkpointing for transformer-like modules
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        
        # Set modules to use less memory
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.inplace = True
            elif isinstance(module, nn.ReLU):
                module.inplace = True
    
    def compile_for_training(self, mode: str = "default") -> None:
        """
        Compile model with torch.compile for faster training.
        
        Optimized for RTX 4070 and similar GPUs.
        """
        if not isinstance(self, nn.Module):
            return
        
        try:
            # Check if compilation is beneficial
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()[0]
                if compute_capability >= 8:  # Ampere and newer (RTX 30xx, 40xx)
                    # Compile with dynamic shapes for GNN
                    compiled_model = torch.compile(
                        self,
                        mode=mode,
                        dynamic=True,  # Important for variable graph sizes
                        fullgraph=False,  # Allow Python control flow
                    )
                    # Store reference
                    self._original_forward = self.forward
                    self.forward = compiled_model.forward
                    logger.info(f"Model compiled with mode={mode}")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")


class ArchitectureSearchMixin:
    """
    Dynamic architecture search for HPO.
    
    Features:
    - Modular layer construction
    - Efficient architecture switching
    - Pruning support
    """
    
    def build_dynamic_architecture(self, config: Dict[str, Any]) -> None:
        """Build architecture based on HPO config."""
        if not hasattr(self, 'conv_type'):
            self.conv_type = config.get('conv_type', 'gcn')
        
        # Clear existing layers
        if hasattr(self, 'convs'):
            del self.convs
        if hasattr(self, 'norms'):
            del self.norms
        if hasattr(self, 'dropouts'):
            del self.dropouts
        
        # Build new architecture
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_channels = config.get('num_features', 128)
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config.get('num_layers', 3)
        dropout = config.get('dropout', 0.1)
        
        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else config.get('num_classes', 10)
            
            # Create appropriate conv layer
            if self.conv_type == 'gcn':
                conv = GCNConv(
                    in_channels, 
                    out_channels,
                    improved=config.get('improved', True),
                    cached=config.get('cached', False),  # Disable for dynamic graphs
                    add_self_loops=config.get('add_self_loops', True),
                    normalize=config.get('normalize', True),
                )
            elif self.conv_type == 'gat':
                heads = config.get('heads', 4)
                conv = GATConv(
                    in_channels,
                    out_channels // heads if i < num_layers - 1 else out_channels,
                    heads=heads if i < num_layers - 1 else 1,
                    concat=i < num_layers - 1,
                    dropout=config.get('attention_dropout', 0.1),
                    add_self_loops=config.get('add_self_loops', True),
                )
                if i < num_layers - 1:
                    out_channels = out_channels  # heads * out_channels/heads
            elif self.conv_type == 'sage':
                conv = SAGEConv(
                    in_channels,
                    out_channels,
                    normalize=config.get('normalize', True),
                    aggr=config.get('aggr', 'mean'),
                )
            elif self.conv_type == 'gin':
                nn_gin = nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_channels),
                )
                conv = GINConv(nn_gin, eps=config.get('gin_eps', 0.0))
            else:
                raise ValueError(f"Unknown conv_type: {self.conv_type}")
            
            self.convs.append(conv)
            
            # Add normalization
            if config.get('use_batch_norm', True):
                self.norms.append(nn.BatchNorm1d(out_channels))
            else:
                self.norms.append(nn.LayerNorm(out_channels))
            
            # Add dropout
            self.dropouts.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        # Pooling layer for graph-level tasks
        pooling = config.get('pooling', 'mean')
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool
        
        logger.info(f"Built {self.conv_type} architecture with {num_layers} layers")
    
    def prune_architecture(self, importance_scores: Dict[str, float]) -> None:
        """Prune less important parts of the architecture."""
        # This is a placeholder for structured pruning
        # In practice, you would implement magnitude-based or gradient-based pruning
        pass


class OptunaMixin:
    """
    Direct Optuna integration for PyTorch Lightning models.
    
    Features:
    - Automatic trial pruning
    - Metric reporting
    - Hyperparameter suggestions
    """
    
    def __init__(self):
        """Initialize Optuna integration."""
        self._optuna_trial = None
        self._pruned = False
    
    def set_optuna_trial(self, trial) -> None:
        """Set Optuna trial for this model."""
        self._optuna_trial = trial
    
    def report_to_optuna(self, metric: float, step: int) -> None:
        """Report intermediate metric to Optuna."""
        if self._optuna_trial is not None and not self._pruned:
            self._optuna_trial.report(metric, step)
            
            # Check if should prune
            if self._optuna_trial.should_prune():
                self._pruned = True
                import optuna
                raise optuna.TrialPruned()
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial."""
        if self._optuna_trial is None:
            raise ValueError("Optuna trial not set")
        
        trial = self._optuna_trial
        
        # Architecture parameters
        config = {
            'conv_type': trial.suggest_categorical('conv_type', ['gcn', 'gat', 'sage', 'gin']),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
        }
        
        # Conv-specific parameters
        if config['conv_type'] == 'gat':
            config['heads'] = trial.suggest_categorical('heads', [1, 2, 4, 8])
            config['attention_dropout'] = trial.suggest_float('attention_dropout', 0.0, 0.3)
        elif config['conv_type'] == 'sage':
            config['aggr'] = trial.suggest_categorical('aggr', ['mean', 'max', 'lstm'])
        elif config['conv_type'] == 'gin':
            config['gin_eps'] = trial.suggest_float('gin_eps', 0.0, 0.5)
        
        # Training parameters
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        config['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        
        # Regularization
        config['use_batch_norm'] = trial.suggest_categorical('use_batch_norm', [True, False])
        
        return config


# Combined HPO mixin for convenience
class FullHPOMixin(
    HPOResetMixin,
    HPOMemoryMixin,
    EfficientTrainingMixin,
    ArchitectureSearchMixin,
    OptunaMixin
):
    """Complete HPO support with all optimizations."""
    
    def __init__(self):
        HPOMemoryMixin.__init__(self)
        OptunaMixin.__init__(self)
