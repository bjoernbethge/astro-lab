"""
Survey-Specific Preprocessing
============================

Centralized preprocessing functions for different astronomical surveys.
Consolidates functionality from various modules.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from scipy.integrate import quad

logger = logging.getLogger(__name__)


class SurveyPreprocessor:
    """Base class for survey-specific preprocessing."""
    
    def __init__(self, survey_name: str):
        self.survey_name = survey_name
        self.config = self._get_survey_config()
    
    def _get_survey_config(self) -> Dict:
        """Get survey-specific configuration."""
        configs = {
            "gaia": {
                "coord_system": "icrs",
                "distance_unit": "pc",
                "magnitude_system": "Vega",
                "position_cols": ["ra", "dec", "parallax"],
                "feature_cols": ["phot_g_mean_mag", "bp_rp", "pmra", "pmdec"],
            },
            "sdss": {
                "coord_system": "icrs", 
                "distance_unit": "Mpc",
                "magnitude_system": "AB",
                "position_cols": ["ra", "dec", "z"],
                "feature_cols": ["modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z"],
            },
            "lsst": {
                "coord_system": "icrs",
                "distance_unit": "Mpc",
                "magnitude_system": "AB",
                "position_cols": ["ra", "dec", "z"],
                "feature_cols": ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "mag_y"],
            },
            "euclid": {
                "coord_system": "icrs",
                "distance_unit": "Mpc",
                "magnitude_system": "AB",
                "position_cols": ["ra", "dec", "photo_z"],
                "feature_cols": ["mag_vis", "mag_y", "mag_j", "mag_h"],
            },
            "tng50": {
                "coord_system": "cartesian",
                "distance_unit": "kpc",
                "position_cols": ["x", "y", "z"],
                "feature_cols": ["masses", "density", "temperature", "velocities"],
            },
        }
        return configs.get(self.survey_name, configs["gaia"])
    
    def extract_3d_positions(self, df: pl.DataFrame) -> np.ndarray:
        """Extract 3D positions with survey-specific logic."""
        raise NotImplementedError
    
    def extract_features(self, df: pl.DataFrame) -> np.ndarray:
        """Extract features with survey-specific logic."""
        raise NotImplementedError
    
    def create_labels(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Create astronomical labels."""
        raise NotImplementedError


class GaiaPreprocessor(SurveyPreprocessor):
    """Preprocessor for Gaia survey data."""
    
    def __init__(self):
        super().__init__("gaia")
    
    def extract_3d_positions(self, df: pl.DataFrame) -> np.ndarray:
        """Extract 3D positions using parallax with Bayesian distance estimation."""
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        
        # Bayesian distance estimation
        if "parallax" in df.columns:
            parallax = df["parallax"].to_numpy()
            
            # Handle parallax uncertainties
            if "parallax_error" in df.columns:
                parallax_error = df["parallax_error"].to_numpy()
                reliable_mask = (parallax > 0) & (parallax / parallax_error > 5)
            else:
                reliable_mask = parallax > 0.1
            
            distance = np.zeros(len(parallax))
            
            # Direct inversion for reliable parallaxes
            distance[reliable_mask] = 1000.0 / parallax[reliable_mask]
            
            # Bayesian prior for unreliable parallaxes
            unreliable_mask = ~reliable_mask
            if unreliable_mask.any():
                # Exponentially decreasing space density prior
                L = 1350.0  # Length scale in pc (Bailer-Jones et al. 2018)
                distance[unreliable_mask] = L
                
                # Improved estimate if errors available
                if "parallax_error" in df.columns:
                    plx_unreliable = parallax[unreliable_mask]
                    pos_mask = plx_unreliable > 0
                    if pos_mask.any():
                        r_est = 1000.0 / plx_unreliable[pos_mask]
                        r_est = np.clip(r_est, 10, 10000)
                        distance[unreliable_mask][pos_mask] = r_est
        else:
            # Photometric distance fallback
            distance = self._estimate_photometric_distance(df)
        
        # Convert to Cartesian coordinates
        return self._spherical_to_cartesian(ra, dec, distance)
    
    def _estimate_photometric_distance(self, df: pl.DataFrame) -> np.ndarray:
        """Estimate distance from photometry."""
        if "phot_g_mean_mag" in df.columns:
            g_mag = df["phot_g_mean_mag"].to_numpy()
            # Assume main sequence stars with M_G ≈ 4.5
            M_G = 4.5
            distance = 10 ** ((g_mag - M_G + 5) / 5)
            return distance
        return np.full(len(df), 100.0)  # Default 100 pc
    
    def _spherical_to_cartesian(self, ra: np.ndarray, dec: np.ndarray, 
                               distance: np.ndarray) -> np.ndarray:
        """Convert spherical to Cartesian coordinates."""
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)
        
        return np.stack([x, y, z], axis=1)
    
    def extract_features(self, df: pl.DataFrame) -> np.ndarray:
        """Extract Gaia-specific features."""
        features = []
        feature_names = []
        
        # Photometry - vectorized column selection
        phot_cols = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
        available_phot = [col for col in phot_cols if col in df.columns]
        if available_phot:
            features.extend([df[col].to_numpy() for col in available_phot])
            feature_names.extend(available_phot)
        
        # Colors
        if "bp_rp" in df.columns:
            features.append(df["bp_rp"].to_numpy())
            feature_names.append("bp_rp")
        elif "phot_bp_mean_mag" in df.columns and "phot_rp_mean_mag" in df.columns:
            bp_rp = df["phot_bp_mean_mag"].to_numpy() - df["phot_rp_mean_mag"].to_numpy()
            features.append(bp_rp)
            feature_names.append("bp_rp_computed")
        
        # Proper motions - vectorized column selection
        pm_cols = ["pmra", "pmdec"]
        available_pm = [col for col in pm_cols if col in df.columns]
        if available_pm:
            features.extend([df[col].to_numpy() for col in available_pm])
            feature_names.extend(available_pm)
        
        # Astrophysical parameters - vectorized column selection
        astro_cols = ["teff_gspphot", "logg_gspphot", "mh_gspphot"]
        available_astro = [col for col in astro_cols if col in df.columns]
        if available_astro:
            features.extend([df[col].to_numpy() for col in available_astro])
            feature_names.extend(available_astro)
        
        if not features:
            logger.warning("No features found in Gaia data")
            return np.zeros((len(df), 1))
        
        # Stack and handle NaN values
        features = np.column_stack(features)
        features = np.nan_to_num(features, nan=0.0)
        
        self.feature_names = feature_names
        return features
    
    def create_labels(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Create stellar classification labels."""
        # HR diagram-based classification
        if "bp_rp" in df.columns and "phot_g_mean_mag" in df.columns:
            bp_rp = df["bp_rp"].to_numpy()
            g_mag = df["phot_g_mean_mag"].to_numpy()
            
            labels = np.zeros(len(df), dtype=int)
            
            # Calculate absolute magnitude if parallax available
            if "parallax" in df.columns:
                parallax = df["parallax"].to_numpy()
                valid_plx = parallax > 0.1
                
                M_G = np.full(len(g_mag), np.nan)
                M_G[valid_plx] = g_mag[valid_plx] + 5 + 5 * np.log10(parallax[valid_plx] / 1000)
                
                # Main sequence
                ms_mask = (M_G > 1.5 * bp_rp + 1) & (M_G < 1.5 * bp_rp + 6) & valid_plx
                labels[ms_mask] = 0
                
                # Giants
                giant_mask = (M_G < 1.5 * bp_rp + 1) & (bp_rp > 0.8) & valid_plx
                labels[giant_mask] = 1
                
                # White dwarfs
                wd_mask = (M_G > 10) & (bp_rp < 0.5) & valid_plx
                labels[wd_mask] = 2
                
                # Hot stars (O/B)
                ob_mask = (bp_rp < -0.1) & (M_G < 2) & valid_plx
                labels[ob_mask] = 3
                
                # Cool stars (M dwarfs)
                m_mask = (bp_rp > 2.0) & (M_G > 8) & valid_plx
                labels[m_mask] = 4
                
                # Unclassified
                labels[~valid_plx] = 5
            else:
                # Color-based classification only
                bins = [-np.inf, -0.3, 0.0, 0.3, 0.58, 0.82, 1.41, 2.0, np.inf]
                labels = np.digitize(bp_rp, bins) - 1
                labels = np.clip(labels, 0, 7)
            
            return labels
        
        return None


class SDSSPreprocessor(SurveyPreprocessor):
    """Preprocessor for SDSS survey data."""
    
    def __init__(self):
        super().__init__("sdss")
        # Cosmological parameters (Planck 2018)
        self.H0 = 67.66  # km/s/Mpc
        self.Omega_m = 0.3111
        self.Omega_Lambda = 0.6889
        self.c = 299792.458  # km/s
    
    def extract_3d_positions(self, df: pl.DataFrame) -> np.ndarray:
        """Extract 3D positions using cosmological distance."""
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        
        if "z" in df.columns:
            redshift = df["z"].to_numpy()
            distance = self._compute_luminosity_distance(redshift)
        else:
            distance = np.full(len(df), 100.0)  # Default 100 Mpc
        
        return self._spherical_to_cartesian(ra, dec, distance)
    
    def _compute_luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        """Compute luminosity distance with proper cosmology."""
        distance = np.zeros_like(z)
        
        # Valid redshift mask
        valid_mask = (z > 0) & (z < 5.0)
        
        # Small z: Hubble law
        small_z = valid_mask & (z < 0.1)
        distance[small_z] = (self.c * z[small_z]) / self.H0
        
        # Large z: Numerical integration
        large_z = valid_mask & (z >= 0.1)
        if large_z.any():
            def E(z_val):
                return np.sqrt(self.Omega_m * (1 + z_val)**3 + self.Omega_Lambda)
            
            for i in np.where(large_z)[0]:
                D_c, _ = quad(lambda zz: 1/E(zz), 0, z[i])
                D_c *= self.c / self.H0
                distance[i] = D_c * (1 + z[i])
        
        # Invalid z
        distance[~valid_mask] = 100.0  # Default 100 Mpc
        
        return distance
    
    def _spherical_to_cartesian(self, ra: np.ndarray, dec: np.ndarray,
                               distance: np.ndarray) -> np.ndarray:
        """Convert spherical to Cartesian coordinates."""
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)
        
        return np.stack([x, y, z], axis=1)
    
    def extract_features(self, df: pl.DataFrame) -> np.ndarray:
        """Extract SDSS-specific features."""
        features = []
        feature_names = []
        
        # Magnitudes (ugriz) - vectorized column selection
        mag_bands = ["u", "g", "r", "i", "z"]
        mag_cols = [f"modelMag_{band}" for band in mag_bands]
        available_mags = [col for col in mag_cols if col in df.columns]
        if available_mags:
            features.extend([df[col].to_numpy() for col in available_mags])
            feature_names.extend(available_mags)
        
        # Colors
        if "modelMag_g" in df.columns and "modelMag_r" in df.columns:
            g_r = df["modelMag_g"].to_numpy() - df["modelMag_r"].to_numpy()
            features.append(g_r)
            feature_names.append("g_r_color")
        
        # Morphological parameters - vectorized column selection
        morph_params = ["fracDeV_r", "petroR50_r", "petroR90_r"]
        available_morph = [param for param in morph_params if param in df.columns]
        if available_morph:
            features.extend([df[param].to_numpy() for param in available_morph])
            feature_names.extend(available_morph)
        
        # Spectroscopic parameters if available - vectorized column selection
        spec_params = ["z", "zErr", "velDisp"]
        available_spec = [param for param in spec_params if param in df.columns]
        if available_spec:
            features.extend([df[param].to_numpy() for param in available_spec])
            feature_names.extend(available_spec)
        
        if not features:
            logger.warning("No features found in SDSS data")
            return np.zeros((len(df), 1))
        
        features = np.column_stack(features)
        features = np.nan_to_num(features, nan=0.0)
        
        self.feature_names = feature_names
        return features
    
    def create_labels(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Create galaxy classification labels."""
        n_objects = len(df)
        labels = np.zeros(n_objects, dtype=int)
        
        # Color-based classification
        if "modelMag_g" in df.columns and "modelMag_r" in df.columns:
            g_r_color = df["modelMag_g"].to_numpy() - df["modelMag_r"].to_numpy()
            
            # Red sequence: Elliptical/S0
            labels[g_r_color > 0.7] = 0
            
            # Blue cloud: Spiral
            labels[g_r_color < 0.5] = 1
            
            # Green valley: Transition
            labels[(g_r_color >= 0.5) & (g_r_color <= 0.7)] = 2
        
        # Morphology-based refinement
        if "fracDeV_r" in df.columns:
            fracDev = df["fracDeV_r"].to_numpy()
            
            # High deV fraction → Elliptical
            labels[fracDev > 0.8] = 0
            
            # Low deV fraction → Spiral
            labels[fracDev < 0.2] = 1
        
        # Concentration index
        if "petroR50_r" in df.columns and "petroR90_r" in df.columns:
            r50 = df["petroR50_r"].to_numpy()
            r90 = df["petroR90_r"].to_numpy()
            
            valid_radii = (r50 > 0) & (r90 > 0)
            concentration = np.ones(n_objects) * 2.5
            concentration[valid_radii] = r90[valid_radii] / r50[valid_radii]
            
            # High concentration → Elliptical
            labels[concentration > 3.0] = 0
            
            # Low concentration → Spiral/Irregular
            labels[concentration < 2.3] = 1
        
        # AGN/QSO identification
        if "class" in df.columns:
            obj_class = df["class"].to_numpy()
            agn_mask = np.isin(obj_class, ["QSO", "AGN"])
            labels[agn_mask] = 3
        
        return labels


# Factory function
def get_survey_preprocessor(survey_name: str) -> SurveyPreprocessor:
    """Get appropriate preprocessor for survey."""
    preprocessors = {
        "gaia": GaiaPreprocessor,
        "sdss": SDSSPreprocessor,
        # Add more as needed
    }
    
    if survey_name in preprocessors:
        return preprocessors[survey_name]()
    else:
        logger.warning(f"No specific preprocessor for {survey_name}, using base class")
        return SurveyPreprocessor(survey_name)


# Convenience functions (maintain backward compatibility)
def extract_3d_positions(df: pl.DataFrame, survey: str, 
                        position_cols: Optional[List[str]] = None) -> np.ndarray:
    """Extract 3D positions for any survey."""
    preprocessor = get_survey_preprocessor(survey)
    return preprocessor.extract_3d_positions(df)


def extract_features(df: pl.DataFrame, survey: str,
                    feature_cols: Optional[List[str]] = None) -> np.ndarray:
    """Extract features for any survey."""
    preprocessor = get_survey_preprocessor(survey)
    return preprocessor.extract_features(df)


def create_astronomical_labels(df: pl.DataFrame, survey: str) -> Optional[np.ndarray]:
    """Create astronomical labels for any survey."""
    preprocessor = get_survey_preprocessor(survey)
    return preprocessor.create_labels(df)
