"""
AstroLab CLI Package
===================

Command-line interface for AstroLab astronomical machine learning framework.

Available Commands:
------------------
- preprocess: Preprocess raw data files into training-ready format
- train: Train ML models with Lightning
- hpo: Hyperparameter optimization
- config: Configuration management
- cosmic-web: Analyze cosmic web structure
- download: Download survey data

Available Surveys:
-----------------
- gaia: Gaia DR3 stellar data
- sdss: SDSS galaxy data
- nsa: NASA-Sloan Atlas galaxy data (includes images)
- tng50: TNG50 simulation data
- exoplanet: Exoplanet catalog data
- twomass: 2MASS infrared survey
- wise: WISE infrared survey
- panstarrs: Pan-STARRS optical survey
- des: Dark Energy Survey
- euclid: Euclid mission data

# Show available surveys
astro-lab config surveys
"""

__version__ = "0.1.0"
__author__ = "AstroLab Team"

# Export main CLI function
from .__main__ import main

__all__ = ["main"]
