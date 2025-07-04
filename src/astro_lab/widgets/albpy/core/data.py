"""
Core Data API for AlbPy
=======================

Centralized data access and conversion for astronomical surveys using the AstroLab DataModule API and tensor bridges.
"""

from astro_lab.data.dataset.astrolab import create_dataset
from astro_lab.widgets.enhanced import ZeroCopyTensorConverter

converter = ZeroCopyTensorConverter()


def list_available_surveys():
    """
    List all available survey names.
    """
    return ["gaia", "sdss", "nsa", "tng50", "exoplanet"]


def load_survey_data(survey: str, max_samples: int = 10000):
    """
    Load data for a given survey using the central DataModule API.
    Only already processed data is loaded for visualization (no processing or downloading).
    Returns a batch or dataset object.
    """
    dataset = create_dataset(survey_name=survey)
    # Optionally, you can use dataset.get_loader() for batching
    loader = dataset.get_loader(batch_size=max_samples, shuffle=False)
    batch = next(iter(loader))
    return batch


def get_coordinates_and_features(survey: str, max_samples: int = 10000):
    """
    Load survey data and extract coordinates and features as numpy arrays for visualization.

    Returns:
        coords: Numpy array of shape (N, 3)
        features: Dict of additional features (magnitudes, cluster_labels, etc.)
    """
    batch = load_survey_data(survey, max_samples)
    coords = converter.extract_coordinates(batch)
    features = converter.extract_features(batch)
    return coords, features
