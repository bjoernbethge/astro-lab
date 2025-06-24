"""Simple factory functions for common models."""

from typing import Optional, Union
import torch

from .core import AstroSurveyGNN, ALCDEFTemporalGNN, AstroPhotGNN, TemporalGCN
from .config import get_predefined_config


def create_gaia_classifier(
    num_classes: int = 7,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> AstroSurveyGNN:
    """Create Gaia stellar classifier.
    
    Args:
        num_classes: Number of stellar classes (default: 7)
        hidden_dim: Hidden dimension size
        device: Device to use
        **kwargs: Additional arguments passed to AstroSurveyGNN
        
    Returns:
        Configured AstroSurveyGNN for Gaia classification
    """
    config = get_predefined_config('gaia_classifier')
    
    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        conv_type=config.conv_type,
        task='classification',
        use_photometry=config.use_photometry,
        use_astrometry=config.use_astrometry,
        use_spectroscopy=config.use_spectroscopy,
        device=device,
        **kwargs
    )


def create_sdss_galaxy_model(
    output_dim: int = 5,
    hidden_dim: int = 256,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> AstroSurveyGNN:
    """Create SDSS galaxy property predictor.
    
    Args:
        output_dim: Number of galaxy properties to predict
        hidden_dim: Hidden dimension size
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured AstroSurveyGNN for SDSS galaxy modeling
    """
    config = get_predefined_config('sdss_galaxy')
    
    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        conv_type=config.conv_type,
        task='regression',
        use_photometry=config.use_photometry,
        use_astrometry=config.use_astrometry,
        use_spectroscopy=config.use_spectroscopy,
        pooling=config.pooling,
        device=device,
        **kwargs
    )


def create_lsst_transient_detector(
    hidden_dim: int = 192,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> AstroSurveyGNN:
    """Create LSST transient detector.
    
    Args:
        hidden_dim: Hidden dimension size
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured AstroSurveyGNN for transient detection
    """
    config = get_predefined_config('lsst_transient')
    
    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=2,  # Binary classification
        conv_type=config.conv_type,
        task='classification',
        use_photometry=config.use_photometry,
        use_astrometry=config.use_astrometry,
        pooling=config.pooling,
        device=device,
        **kwargs
    )


def create_asteroid_period_detector(
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> ALCDEFTemporalGNN:
    """Create asteroid period detector from lightcurves.
    
    Args:
        hidden_dim: Hidden dimension size
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured ALCDEFTemporalGNN for period detection
    """
    return ALCDEFTemporalGNN(
        hidden_dim=hidden_dim,
        task='period_detection',
        device=device,
        **kwargs
    )


def create_lightcurve_classifier(
    num_classes: int = 2,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> ALCDEFTemporalGNN:
    """Create lightcurve classifier.
    
    Args:
        num_classes: Number of classes
        hidden_dim: Hidden dimension size
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured ALCDEFTemporalGNN for classification
    """
    return ALCDEFTemporalGNN(
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        task='classification',
        num_classes=num_classes,
        device=device,
        **kwargs
    )


def create_galaxy_modeler(
    model_components: Optional[list[str]] = None,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> AstroPhotGNN:
    """Create galaxy morphology modeler.
    
    Args:
        model_components: List of components ['sersic', 'disk', 'bulge']
        hidden_dim: Hidden dimension size
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured AstroPhotGNN for galaxy modeling
    """
    if model_components is None:
        model_components = ['sersic', 'disk']
        
    return AstroPhotGNN(
        hidden_dim=hidden_dim,
        model_components=model_components,
        device=device,
        **kwargs
    )


def create_temporal_graph_model(
    input_dim: int,
    output_dim: int = 1,
    hidden_dim: int = 128,
    task: str = 'regression',
    rnn_type: str = 'lstm',
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> TemporalGCN:
    """Create temporal graph model for time-series analysis.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension size
        task: Task type ('regression', 'classification')
        rnn_type: RNN type ('lstm', 'gru')
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Configured TemporalGCN
    """
    return TemporalGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        task=task,
        rnn_type=rnn_type,
        device=device,
        **kwargs
    ) 