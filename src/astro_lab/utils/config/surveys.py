"""
Survey Configurations
====================

Extensible survey configurations for astronomical datasets.
Can be easily extended with new surveys.
"""

from typing import Any, Dict, List

# Survey configurations - easily extensible
SURVEY_CONFIGS = {
    "gaia": {
        "name": "Gaia DR3 (Minimal)",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
        "extra_cols": [],
        "color_pairs": [
            ("phot_g_mean_mag", "phot_bp_mean_mag"),
            ("phot_bp_mean_mag", "phot_rp_mean_mag"),
        ],
        "default_limit": 12.0,
        "filter_system": "gaia",
        "data_release": "DR3",
        "coordinate_system": "icrs",
        "photometric_bands": ["G", "BP", "RP"],
    },
    "sdss": {
        "name": "SDSS DR17",
        "coord_cols": ["ra", "dec", "z"],
        "mag_cols": [
            "modelMag_u",
            "modelMag_g",
            "modelMag_r",
            "modelMag_i",
            "modelMag_z",
        ],
        "extra_cols": ["petroRad_r", "fracDeV_r"],
        "color_pairs": [("modelMag_g", "modelMag_r"), ("modelMag_r", "modelMag_i")],
        "default_limit": 20.0,
        "filter_system": "sdss",
        "data_release": "DR17",
        "coordinate_system": "icrs",
        "photometric_bands": ["u", "g", "r", "i", "z"],
    },
    "nsa": {
        "name": "NASA Sloan Atlas",
        "coord_cols": ["ra", "dec", "z"],
        "mag_cols": ["mag_g", "mag_r", "mag_i"],
        "extra_cols": ["sersic_n", "sersic_ba", "mass"],
        "color_pairs": [("mag_g", "mag_r"), ("mag_r", "mag_i")],
        "default_limit": 18.0,
        "filter_system": "sdss",
        "data_release": "v1_0_1",
        "coordinate_system": "icrs",
        "photometric_bands": ["g", "r", "i"],
    },
    "linear": {
        "name": "LINEAR Survey",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_mean", "mag_std"],
        "extra_cols": ["period", "amplitude"],
        "color_pairs": [],
        "default_limit": 15.0,
        "filter_system": "linear",
        "data_release": "final",
        "coordinate_system": "icrs",
        "photometric_bands": ["V"],
    },
    "tng50": {
        "name": "TNG50 Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": ["mass"],
        "extra_cols": ["vx", "vy", "vz", "particle_type"],
        "color_pairs": [],
        "default_limit": None,
        "filter_system": "simulation",
        "data_release": "TNG50-4",
        "coordinate_system": "simulation",
        "photometric_bands": [],
    },
    "exoplanet": {
        "name": "Confirmed Exoplanets Archive",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["sy_vmag", "sy_kmag"],
        "extra_cols": ["pl_rade", "pl_masse", "sy_dist", "pl_orbper"],
        "color_pairs": [],
        "default_limit": 15.0,
        "filter_system": "mixed",
        "data_release": "current",
        "coordinate_system": "icrs", 
        "photometric_bands": ["V", "K"],
    },
    "rrlyrae": {
        "name": "RR Lyrae Variable Stars",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_mean", "mag_amp"],
        "extra_cols": ["period", "phase", "metallicity"],
        "color_pairs": [],
        "default_limit": 18.0,
        "filter_system": "linear",
        "data_release": "cleaned",
        "coordinate_system": "icrs",
        "photometric_bands": ["V"],
    },
}


def get_survey_config(survey: str) -> Dict[str, Any]:
    """Get configuration for a specific survey."""
    if survey not in SURVEY_CONFIGS:
        raise ValueError(
            f"Unknown survey: {survey}. Available: {list(SURVEY_CONFIGS.keys())}"
        )
    return SURVEY_CONFIGS[survey]


def get_available_surveys() -> List[str]:
    """Get list of available surveys."""
    return list(SURVEY_CONFIGS.keys())


def register_survey(name: str, config: Dict[str, Any]) -> None:
    """Register a new survey configuration."""
    required_keys = ["name", "coord_cols", "mag_cols", "extra_cols"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Survey config missing required key: {key}")

    SURVEY_CONFIGS[name] = config
    print(f"✅ Registered new survey: {name}")


def get_survey_features(survey: str) -> List[str]:
    """Get all feature columns for a survey."""
    config = get_survey_config(survey)
    features = config.get("coord_cols", []) + config.get("mag_cols", []) + config.get("extra_cols", [])
    return features


def get_survey_coordinates(survey: str) -> List[str]:
    """Get coordinate columns for a survey."""
    config = get_survey_config(survey)
    return config["coord_cols"]


def get_survey_magnitudes(survey: str) -> List[str]:
    """Get magnitude columns for a survey."""
    config = get_survey_config(survey)
    return config["mag_cols"]
