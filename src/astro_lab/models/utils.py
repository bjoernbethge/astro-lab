"""Pure utility functions - no factories or classes."""

import torch
import torch.nn as nn
from typing import Optional, Union


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'mish': nn.Mish(),
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        
    return activations[name]


def get_pooling(pooling_type: str) -> str:
    """Get pooling type string (for compatibility)."""
    valid_types = ['mean', 'max', 'add', 'attention']
    pooling_type = pooling_type.lower()
    
    if pooling_type not in valid_types:
        raise ValueError(f"Unknown pooling: {pooling_type}. Available: {valid_types}")
        
    return pooling_type


def initialize_weights(module: nn.Module) -> None:
    """Initialize model weights using Xavier/Kaiming initialization."""
    for name, param in module.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Use Kaiming for ReLU networks, Xavier otherwise
                if hasattr(module, 'activation') and 'relu' in str(module.activation).lower():
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get torch device, with automatic CUDA detection."""
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to device."""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            moved_batch[key] = [v.to(device) for v in value]
        else:
            moved_batch[key] = value
    return moved_batch


def get_model_summary(model: nn.Module, input_shape: Optional[tuple] = None) -> str:
    """Get a simple model summary string."""
    total_params = count_parameters(model)
    
    summary = f"{model.__class__.__name__}\n"
    summary += f"Total trainable parameters: {total_params:,}\n"
    
    if input_shape is not None:
        summary += f"Expected input shape: {input_shape}\n"
        
    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type in ['Module', 'Sequential', 'ModuleList', 'ModuleDict']:
            continue
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
        
    summary += "\nLayer counts:\n"
    for layer_type, count in sorted(layer_counts.items()):
        summary += f"  {layer_type}: {count}\n"
        
    return summary


def set_dropout_rate(model: nn.Module, dropout_rate: float) -> None:
    """Set dropout rate for all dropout layers in model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate


def freeze_layers(model: nn.Module, layer_names: Optional[list[str]] = None) -> None:
    """Freeze specific layers or all layers if layer_names is None."""
    if layer_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: Optional[list[str]] = None) -> None:
    """Unfreeze specific layers or all layers if layer_names is None."""
    if layer_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True


# Keep AttentionPooling for backward compatibility
class AttentionPooling(nn.Module):
    """Simple attention pooling module."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention pooling."""
        # Compute attention scores
        scores = self.attention(x)
        
        if batch is not None:
            # Masked softmax for batched graphs
            from torch_geometric.utils import softmax
            alpha = softmax(scores, batch, dim=0)
        else:
            # Simple softmax for single graph
            alpha = torch.softmax(scores, dim=0)
            
        # Weighted sum
        return (x * alpha).sum(dim=0, keepdim=True)
