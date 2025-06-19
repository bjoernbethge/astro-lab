"""
AstroLab Data Module - Pure Clean API 🌟
========================================

Saubere Polars-First Implementation ohne Legacy-Wrapper.
Perfekt für Prototyping und moderne astronomische ML-Pipelines.

Quick Start:
    from astro_lab.data import load_gaia_data
    dataset = load_gaia_data(max_samples=5000)  # Done!
"""

# 🌟 PURE CLEAN API - Polars-First Only
from .core import (
    # Survey Configuration (for advanced users)
    SURVEY_CONFIGS,
    AstroDataModule,
    # Core Classes
    AstroDataset,
    # Factory Functions
    create_astro_dataloader,
    create_astro_datamodule,
    # Convenience Functions for Common Surveys
    load_gaia_data,  # Stellar catalogs with astrometry
    load_lightcurve_data,  # Variable stars/lightcurves
    load_nsa_data,  # NASA Sloan Atlas galaxies
    load_sdss_data,  # Galaxy photometry & spectroscopy
)

# Clean exports - no legacy bloat
__all__ = [
    # 🎯 CORE CLASSES
    "AstroDataset",  # Universal dataset for all surveys
    "AstroDataModule",  # Lightning integration
    # 🏭 FACTORY FUNCTIONS
    "create_astro_dataloader",  # Universal loader factory
    "create_astro_datamodule",  # Universal datamodule factory
    # 🚀 CONVENIENCE FUNCTIONS (Most Common)
    "load_gaia_data",  # One-liner for Gaia
    "load_sdss_data",  # One-liner for SDSS
    "load_nsa_data",  # One-liner for NSA
    "load_lightcurve_data",  # One-liner for lightcurves
    # 🔧 CONFIGURATION
    "SURVEY_CONFIGS",  # Survey definitions (DRY)
]

# Feature flags
HAS_CLEAN_API = True
HAS_LEGACY_API = False

# Supported surveys
SUPPORTED_SURVEYS = list(SURVEY_CONFIGS.keys())

# API version
__version__ = "2.0.0-clean"
