"""
Survey Configurations
====================

Complete survey configurations for all available astronomical datasets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Complete survey configurations - all available surveys
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
        "coordinate_system": "icrs",
        "data_release": "DR3",
        "filter_system": "Vega",
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
        "coordinate_system": "icrs",
        "data_release": "DR17",
        "filter_system": "AB",
    },
    "nsa": {
        "name": "NASA Sloan Atlas",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_g", "mag_r", "mag_i"],
        "extra_cols": ["z", "sersic_n", "mass"],
        "color_pairs": [("mag_g", "mag_r"), ("mag_r", "mag_i")],
        "coordinate_system": "icrs",
        "data_release": "v1_0_1",
        "filter_system": "AB",
        "has_images": True,
    },
    "tng50": {
        "name": "TNG50 Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": ["mass"],
        "extra_cols": ["vx", "vy", "vz", "particle_type"],
        "color_pairs": [],
        "coordinate_system": "comoving",
        "data_release": "TNG50-1",
        "filter_system": "simulation",
    },
    "exoplanet": {
        "name": "Confirmed Exoplanets",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["sy_vmag", "sy_kmag"],
        "extra_cols": ["pl_rade", "pl_masse", "sy_dist", "pl_orbper"],
        "color_pairs": [],
        "coordinate_system": "icrs",
        "data_release": "2024",
        "filter_system": "Vega",
    },
    "twomass": {
        "name": "2MASS All-Sky Survey",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["j_m", "h_m", "ks_m"],
        "extra_cols": ["j_msigcom", "h_msigcom", "ks_msigcom"],
        "color_pairs": [("j_m", "h_m"), ("h_m", "ks_m")],
        "coordinate_system": "icrs",
        "data_release": "All-Sky",
        "filter_system": "Vega",
    },
    "wise": {
        "name": "WISE All-Sky Survey",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["w1mpro", "w2mpro", "w3mpro", "w4mpro"],
        "extra_cols": ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"],
        "color_pairs": [("w1mpro", "w2mpro"), ("w2mpro", "w3mpro")],
        "coordinate_system": "icrs",
        "data_release": "AllWISE",
        "filter_system": "Vega",
    },
    "panstarrs": {
        "name": "Pan-STARRS DR2",
        "coord_cols": ["ra", "dec"],
        "mag_cols": [
            "g_mean_psf_mag",
            "r_mean_psf_mag",
            "i_mean_psf_mag",
            "z_mean_psf_mag",
            "y_mean_psf_mag",
        ],
        "extra_cols": [
            "g_mean_psf_mag_err",
            "r_mean_psf_mag_err",
            "i_mean_psf_mag_err",
        ],
        "color_pairs": [
            ("g_mean_psf_mag", "r_mean_psf_mag"),
            ("r_mean_psf_mag", "i_mean_psf_mag"),
        ],
        "coordinate_system": "icrs",
        "data_release": "DR2",
        "filter_system": "AB",
    },
    "des": {
        "name": "Dark Energy Survey",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["g", "r", "i", "z", "Y"],
        "extra_cols": ["g_err", "r_err", "i_err", "z_err", "Y_err"],
        "color_pairs": [("g", "r"), ("r", "i"), ("i", "z")],
        "coordinate_system": "icrs",
        "data_release": "Y6",
        "filter_system": "AB",
    },
    "euclid": {
        "name": "Euclid Mission",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["VIS", "Y", "J", "H"],
        "extra_cols": ["VIS_err", "Y_err", "J_err", "H_err"],
        "color_pairs": [("VIS", "Y"), ("Y", "J"), ("J", "H")],
        "coordinate_system": "icrs",
        "data_release": "Early Release",
        "filter_system": "AB",
    },
    "linear": {
        "name": "LINEAR Survey",
        "coord_cols": ["raLIN", "decLIN"],
        "mag_cols": ["r"],
        "extra_cols": [
            "ug",
            "gr",
            "ri",
            "iz",
            "JK",
            "<mL>",
            "std",
            "rms",
            "Lchi2",
            "LP1",
            "phi1",
            "S",
            "prior",
        ],
        "color_pairs": [],
        "coordinate_system": "icrs",
        "data_release": "unknown",
        "filter_system": "AB",
    },
    "rrlyrae": {
        "name": "RR Lyrae Survey",
        "coord_cols": ["RAJ2000", "DEJ2000", "Dist"],
        "mag_cols": ["umag", "gmag", "rmag", "imag", "zmag", "Vmag"],
        "extra_cols": [
            "__SIG2010_",
            "Type",
            "Per",
            "uAmp",
            "T0_u",
            "gAmp",
            "T0_g",
            "rAmp",
            "T0_r",
            "iAmp",
            "T0_i",
            "T0_z",
            "Ar",
        ],
        "color_pairs": [],
        "coordinate_system": "icrs",
        "data_release": "unknown",
        "filter_system": "AB",
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


def get_survey_info(survey: str) -> Dict[str, Any]:
    """Get comprehensive survey information."""
    config = get_survey_config(survey)
    return {
        "name": config["name"],
        "data_release": config.get("data_release", "unknown"),
        "coordinate_system": config.get("coordinate_system", "icrs"),
        "filter_system": config.get("filter_system", "unknown"),
        "has_images": config.get("has_images", False),
        "n_features": len(get_survey_features(survey)),
        "n_coordinates": len(config["coord_cols"]),
        "n_magnitudes": len(config["mag_cols"]),
        "n_extra": len(config["extra_cols"]),
    }


# Training optimization configurations
@dataclass
class SurveyOptimizationConfig:
    """Optimization settings for training and HPO."""

    # Batch size settings
    batch_size: int = 256
    batch_size_range: Tuple[int, int] = (32, 1024)

    # Graph construction
    k_neighbors: int = 8
    k_neighbors_range: Tuple[int, int] = (5, 30)

    # Data characteristics
    graph_size: str = "medium"  # small, medium, large
    typical_nodes: int = 10000
    typical_edges: int = 80000

    # Memory estimates (GB)
    memory_per_sample: float = 0.001
    recommended_gpu_memory: float = 8.0

    # Training optimizations
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1


# Survey-specific optimization configs
SURVEY_OPTIMIZATIONS = {
    "gaia": SurveyOptimizationConfig(
        batch_size=512,
        batch_size_range=(128, 2048),
        k_neighbors=10,
        k_neighbors_range=(5, 20),
        graph_size="large",
        typical_nodes=1000000,
        typical_edges=10000000,
        memory_per_sample=0.0005,
        recommended_gpu_memory=16.0,
    ),
    "sdss": SurveyOptimizationConfig(
        batch_size=256,
        batch_size_range=(64, 1024),
        k_neighbors=12,
        k_neighbors_range=(8, 25),
        graph_size="medium",
        typical_nodes=100000,
        typical_edges=1200000,
        memory_per_sample=0.001,
        recommended_gpu_memory=12.0,
    ),
    "nsa": SurveyOptimizationConfig(
        batch_size=128,
        batch_size_range=(32, 512),
        k_neighbors=15,
        k_neighbors_range=(10, 30),
        graph_size="medium",
        typical_nodes=50000,
        typical_edges=750000,
        memory_per_sample=0.002,
        recommended_gpu_memory=8.0,
    ),
    "tng50": SurveyOptimizationConfig(
        batch_size=64,
        batch_size_range=(16, 256),
        k_neighbors=20,
        k_neighbors_range=(10, 50),
        graph_size="large",
        typical_nodes=1000000,
        typical_edges=20000000,
        memory_per_sample=0.001,
        recommended_gpu_memory=24.0,
        use_mixed_precision=False,  # Simulation data needs precision
    ),
    "exoplanet": SurveyOptimizationConfig(
        batch_size=128,
        batch_size_range=(32, 512),
        k_neighbors=8,
        k_neighbors_range=(5, 15),
        graph_size="small",
        typical_nodes=5000,
        typical_edges=40000,
        memory_per_sample=0.0008,
        recommended_gpu_memory=4.0,
    ),
}


def get_survey_optimization(survey: str) -> SurveyOptimizationConfig:
    """Get optimization config for a survey."""
    if survey in SURVEY_OPTIMIZATIONS:
        return SURVEY_OPTIMIZATIONS[survey]
    # Return default config for unknown surveys
    return SurveyOptimizationConfig()
