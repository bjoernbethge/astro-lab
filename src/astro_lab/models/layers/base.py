"""
Base Layer Components for AstroLab
==================================

Core building blocks for all astronomical neural network layers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex


class BaseGraphLayer(nn.Module, ABC):
    """Abstract base class for all graph layers."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, EdgeIndex],
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Forward pass through the layer."""
        pass

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module != self:
                module.reset_parameters()

    def cache_edge_index(self, edge_index: Union[Tensor, EdgeIndex]) -> Union[Tensor, EdgeIndex]:
        """Pass through edge_index without caching (torch.compile compatible)."""
        # 2025 Best Practice: No manual EdgeIndex wrapping for torch.compile compatibility
        return edge_index


class BasePoolingLayer(nn.Module, ABC):
    """Abstract base class for pooling layers."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Pool node features to graph-level representation."""
        pass


class BaseAttentionLayer(nn.Module, ABC):
    """Abstract base class for attention mechanisms."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    @abstractmethod
    def compute_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute attention weights and optionally apply to values."""
        pass


class TensorDictLayer(nn.Module, ABC):
    """Base class for layers that work with TensorDict."""

    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        pass_through: bool = True,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.pass_through = pass_through

    @abstractmethod
    def process_tensordict(self, td: TensorDict) -> Dict[str, Tensor]:
        """Process input tensordict and return output dict."""
        pass

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass with TensorDict."""
        # Extract inputs
        inputs = {key: td[key] for key in self.in_keys if key in td}

        # Process
        outputs = self.process_tensordict(inputs)

        # Update tensordict
        if self.pass_through:
            td = td.clone()
        else:
            td = TensorDict({}, batch_size=td.batch_size)

        for key, value in outputs.items():
            td[key] = value

        return td
