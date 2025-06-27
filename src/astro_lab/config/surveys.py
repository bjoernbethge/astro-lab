"""
Survey Configurations
====================

Simple survey configurations for astronomical datasets.
"""

from typing import Any, Dict, List

# Survey configurations - simplified
SURVEY_CONFIGS = {
    "gaia": {
        "name": "Gaia DR3",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
        "extra_cols": ["parallax", "pmra", "pmdec"],
        "color_pairs": [
            ("phot_g_mean_mag", "phot_bp_mean_mag"),
            ("phot_bp_mean_mag", "phot_rp_mean_mag"),
        ],
    },
    "sdss": {
        "name": "SDSS DR17",
        "coord_cols": ["ra", "dec"],
        "mag_cols": [
            "modelMag_u",
            "modelMag_g",
            "modelMag_r",
            "modelMag_i",
            "modelMag_z",
        ],
        "extra_cols": ["z", "petroRad_r"],
        "color_pairs": [("modelMag_g", "modelMag_r"), ("modelMag_r", "modelMag_i")],
    },
    "nsa": {
        "name": "NASA Sloan Atlas",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_g", "mag_r", "mag_i"],
        "extra_cols": ["z", "sersic_n", "mass"],
        "color_pairs": [("mag_g", "mag_r"), ("mag_r", "mag_i")],
    },
    "linear": {
        "name": "LINEAR Survey",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_mean"],
        "extra_cols": ["period", "amplitude"],
        "color_pairs": [],
    },
    "tng50": {
        "name": "TNG50 Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": ["mass"],
        "extra_cols": ["vx", "vy", "vz", "particle_type"],
        "color_pairs": [],
    },
    "exoplanet": {
        "name": "Confirmed Exoplanets",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["sy_vmag", "sy_kmag"],
        "extra_cols": ["pl_rade", "pl_masse", "sy_dist", "pl_orbper"],
        "color_pairs": [],
    },
    "rrlyrae": {
        "name": "RR Lyrae Variables",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_mean"],
        "extra_cols": ["period", "amplitude", "metallicity"],
        "color_pairs": [],
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
    features = (
        config.get("coord_cols", [])
        + config.get("mag_cols", [])
        + config.get("extra_cols", [])
    )
    return features


def get_survey_coordinates(survey: str) -> List[str]:
    """Get coordinate columns for a survey."""
    config = get_survey_config(survey)
    return config["coord_cols"]


def get_survey_magnitudes(survey: str) -> List[str]:
    """Get magnitude columns for a survey."""
    config = get_survey_config(survey)
    return config["mag_cols"]
