"""
Base Dataset for AstroLab
========================

Base classes for astronomical datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch_geometric.data import Dataset


class AstroDatasetBase(Dataset, ABC):
    """
    Base class for astronomical datasets.

    Provides common functionality for survey data loading and processing.
    """

    def __init__(
        self,
        root: str,
        survey: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
    ):
        self.survey = survey
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    @abstractmethod
    def raw_file_names(self) -> List[str]:
        """Raw file names."""
        pass

    @property
    @abstractmethod
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        pass

    @abstractmethod
    def download(self):
        """Download raw data if needed."""
        pass

    @abstractmethod
    def process(self):
        """Process raw data into graph format."""
        pass

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        if len(self) > 0:
            graph = self[0]
            if hasattr(graph, "x") and graph.x is not None:
                return graph.x.size(1)
        return 0

    def get_num_classes(self) -> int:
        """Get number of classes."""
        if len(self) > 0:
            graph = self[0]
            if hasattr(graph, "y") and graph.y is not None:
                if graph.y.dim() == 1:
                    return graph.y.max().item() + 1
                else:
                    return graph.y.size(1)
        return 0

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "survey": self.survey,
            "num_samples": len(self),
            "feature_dim": self.get_feature_dim(),
            "num_classes": self.get_num_classes(),
        }
