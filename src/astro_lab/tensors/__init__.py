"""
AstroLab TensorDict Module
==========================

This module provides specialized TensorDict classes for astronomical data processing.
All classes inherit from AstroTensorDict and provide domain-specific functionality.

The module has been refactored to:
- Use proper astropy APIs without fantasy imports
- Leverage PyTorch Geometric efficiently for graph operations
- Integrate real astrophot functionality where appropriate
- Maintain clean separation of concerns via mixins
"""

# Import refactored core classes
# Import other specialized classes (to be refactored)
from .analysis import AnalysisTensorDict
from .base import AstroTensorDict
from .cosmology import (
    CosmologyTensorDict,
    create_cosmology_from_parameters,
    validate_cosmology_parameters,
)
from .crossmatch import CrossMatchTensorDict
from .factories import (
    create_crossmatch_example,
    create_generic_survey,
    create_nbody_simulation,
    create_survey_from_pyg_data,
)
from .image import ImageTensorDict
from .lightcurve import LightcurveTensorDict
from .maneuver import ManeuverTensorDict, create_hohmann_transfer
from .mixins import (
    CoordinateConversionMixin,
    FeatureExtractionMixin,
    GraphConstructionMixin,
    NormalizationMixin,
    ValidationMixin,
)
from .orbital import OrbitTensorDict, create_asteroid_population, from_kepler_elements
from .photometric import PhotometricTensorDict
from .satellite import EarthSatelliteTensorDict
from .simulation import SimulationTensorDict

# Import fully refactored classes
from .spatial import SpatialTensorDict
from .spectral import SpectralTensorDict
from .survey import SurveyTensorDict

__all__ = [
    # Base classes
    "AstroTensorDict",
    # Core refactored classes
    "SpatialTensorDict",
    "PhotometricTensorDict",
    "SpectralTensorDict",
    "LightcurveTensorDict",
    "ImageTensorDict",
    "CosmologyTensorDict",
    # Mixins for common functionality
    "NormalizationMixin",
    "FeatureExtractionMixin",
    "CoordinateConversionMixin",
    "ValidationMixin",
    "GraphConstructionMixin",
    # Analysis class
    "AnalysisTensorDict",
    # Domain-specific classes (to be refactored)
    "CrossMatchTensorDict",
    "ManeuverTensorDict",
    "OrbitTensorDict",
    "EarthSatelliteTensorDict",
    "SimulationTensorDict",
    "SurveyTensorDict",
    # Factory functions
    "create_nbody_simulation",
    "create_crossmatch_example",
    "create_generic_survey",
    "create_survey_from_pyg_data",
    "create_hohmann_transfer",
    "from_kepler_elements",
    "create_asteroid_population",
    # Cosmology helper functions
    "create_cosmology_from_parameters",
    "validate_cosmology_parameters",
]

# Version info for the tensors module
__version__ = "0.3.0"
__author__ = "AstroLab Team"

# Module-level documentation
__doc__ += """

Recent Changes in v0.3.0:
========================

1. **Comprehensive Refactoring**:
   - SpatialTensorDict: Proper PyG integration, efficient cosmic web analysis
   - PhotometricTensorDict: Real astropy.units integration, proper magnitude systems
   - SpectralTensorDict: Complete spectral analysis with line fitting
   - LightcurveTensorDict: variability analysis and period detection
   - ImageTensorDict: Full photutils integration for source detection
   - CosmologyTensorDict: Proper astropy.cosmology integration

2. **Mixins**:
   - CoordinateConversionMixin: Astronomical coordinate transformations
   - GraphConstructionMixin: Efficient PyG graph building
   - ValidationMixin: Astronomical data validation
   - NormalizationMixin: Astronomical-specific normalization methods
   - FeatureExtractionMixin: Domain-specific feature extraction

3. **Real API Integration**:
   - No more fantasy imports or anti-patterns
   - Proper astropy.units.photometric usage
   - Real photutils source detection and photometry
   - Efficient PyG graph operations for spatial analysis
   - Proper astropy.cosmology distance calculations

4. **Features**:
   - Lomb-Scargle periodogram analysis for lightcurves
   - DAOStarFinder source detection in images
   - Proper WCS coordinate transformations
   - Multi-scale cosmic web clustering algorithms
   - Spectral line detection and equivalent width measurements

Performance Features:
====================

1. **GPU Acceleration**: All tensor operations run efficiently on GPU
2. **Memory Management**: Proper cleanup and memory optimization
3. **Batch Processing**: Vectorized operations across multiple objects
4. **Graph Operations**: Efficient PyG integration for cosmic web analysis
5. **Astropy Integration**: Seamless coordinate transformations and units
6. **History Tracking**: Complete audit trail of all operations

The refactored module provides a solid foundation for astronomical data analysis
while maintaining compatibility with the broader AstroLab ecosystem.
"""
