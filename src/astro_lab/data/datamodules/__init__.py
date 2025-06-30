"""
AstroLab DataModules
===================

Unified interface for data loading with PyTorch Geometric.
"""

from .lightning import (
    AstroLightningDataset,
    AstroLightningNodeData,
    create_lightning_datamodule
)


def create_datamodule(survey: str, **kwargs):
    """
    Create a Lightning DataModule for astronomical data.
    
    Args:
        survey: Survey name (e.g., 'gaia', 'sdss')
        **kwargs: Additional arguments passed to create_lightning_datamodule
        
    Returns:
        Lightning DataModule instance (AstroLightningDataset or AstroLightningNodeData)
        
    Examples:
        >>> # Graph-level task with point clouds
        >>> dm = create_datamodule("gaia", task="graph")
        
        >>> # Node-level task with neighbor sampling
        >>> dm = create_datamodule("gaia", task="node", num_neighbors=[25, 10])
    """
    return create_lightning_datamodule(survey, **kwargs)


__all__ = [
    "create_datamodule",
    "AstroLightningDataset", 
    "AstroLightningNodeData",
    "create_lightning_datamodule",
]
