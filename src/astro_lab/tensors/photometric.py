"""
TensorDict for photometric data with proper astronomical handling.

Provides photometric operations using astropy units and astronomical
magnitude systems (AB, Vega, ST).
"""

from typing import Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.units import Unit
from tensordict import TensorDict

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class PhotometricTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    TensorDict for photometric data with astropy integration.

    Features:
    - AB/Vega/ST magnitude systems with proper astropy units
    - Magnitude and flux conversions
    - K-corrections for cosmological sources
    - Color transformations between photometric systems
    - Error propagation
    """

    # Standard zero points for Vega to AB conversion (AB - Vega)
    VEGA_AB_OFFSETS = {
        # Johnson-Cousins
        "U": 0.79,
        "B": -0.09,
        "V": 0.02,
        "R": 0.21,
        "I": 0.45,
        # 2MASS
        "J": 0.91,
        "H": 1.39,
        "Ks": 1.85,
        # SDSS (already in AB)
        "u": 0.0,
        "g": 0.0,
        "r": 0.0,
        "i": 0.0,
        "z": 0.0,
        # Gaia
        "G": 0.0,
        "BP": 0.0,
        "RP": 0.0,
        # WISE
        "W1": 2.699,
        "W2": 3.339,
        "W3": 5.174,
        "W4": 6.620,
        # HST
        "F435W": 0.0,
        "F606W": 0.0,
        "F814W": 0.0,
        "F160W": 0.0,
    }

    # Effective wavelengths in Angstroms
    BAND_WAVELENGTHS = {
        "U": 3650,
        "B": 4450,
        "V": 5510,
        "R": 6580,
        "I": 8060,
        "u": 3543,
        "g": 4770,
        "r": 6231,
        "i": 7625,
        "z": 9134,
        "J": 12350,
        "H": 16620,
        "Ks": 21590,
        "G": 6730,
        "BP": 5320,
        "RP": 7970,
        "W1": 33526,
        "W2": 46028,
        "W3": 115608,
        "W4": 220883,
        "F435W": 4350,
        "F606W": 6060,
        "F814W": 8140,
        "F160W": 16000,
    }

    def __init__(
        self,
        magnitudes: torch.Tensor,
        bands: List[str],
        errors: Optional[torch.Tensor] = None,
        filter_system: str = "AB",
        is_magnitude: bool = True,
        wavelengths: Optional[Dict[str, float]] = None,
        redshift: Optional[Union[float, torch.Tensor]] = None,
        coordinates: Optional[SkyCoord] = None,
        **kwargs,
    ):
        """
        Initialize PhotometricTensorDict.

        Args:
            magnitudes: [N, B] Tensor with magnitudes/fluxes
            bands: List of band names
            errors: Optional measurement errors
            filter_system: 'AB', 'Vega', or 'ST'
            is_magnitude: True for magnitudes, False for fluxes
            wavelengths: Custom wavelengths for bands (Angstroms)
            redshift: Source redshift(s) for K-corrections
            coordinates: Sky coordinates for extinction calculations
        """
        if magnitudes.shape[-1] != len(bands):
            raise ValueError(
                f"Number of bands ({len(bands)}) doesn't match data columns "
                f"({magnitudes.shape[-1]})"
            )

        # Merge custom wavelengths
        band_wavelengths = self.BAND_WAVELENGTHS.copy()
        if wavelengths:
            band_wavelengths.update(wavelengths)

        data = {
            "magnitudes": magnitudes,
            "meta": {
                "bands": bands,
                "filter_system": filter_system,
                "is_magnitude": is_magnitude,
                "n_bands": len(bands),
                "wavelengths": {b: band_wavelengths.get(b, None) for b in bands},
                "zero_points": {b: self.VEGA_AB_OFFSETS.get(b, 0.0) for b in bands},
            },
        }

        if errors is not None:
            data["errors"] = errors

        if redshift is not None:
            if isinstance(redshift, (int, float)):
                redshift = torch.tensor(redshift, dtype=torch.float32)
            data["redshift"] = redshift

        if coordinates is not None:
            data["coordinates"] = coordinates

        # Don't pass batch_size separately if it's already in kwargs
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = magnitudes.shape[:-1]
        super().__init__(data, **kwargs)

    def to_ab_system(self) -> "PhotometricTensorDict":
        """Convert magnitudes to AB system."""
        if self._metadata["filter_system"] == "AB":
            return self.clone()

        if not self.is_magnitude:
            raise ValueError("Conversion requires magnitude data")

        # Get offsets for conversion
        offsets = torch.tensor(
            [self.VEGA_AB_OFFSETS.get(b, 0.0) for b in self.bands],
            dtype=torch.float32,
            device=self["magnitudes"].device,
        )

        # Apply conversion: m_AB = m_Vega + offset
        ab_mags = self["magnitudes"] + offsets

        result = PhotometricTensorDict(
            ab_mags,
            self.bands,
            self.get("errors", None),
            filter_system="AB",
            is_magnitude=True,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history(f"to_ab_system from {self._metadata['filter_system']}")
        return result

    def to_vega_system(self) -> "PhotometricTensorDict":
        """Convert magnitudes to Vega system."""
        if self._metadata["filter_system"] == "Vega":
            return self.clone()

        if not self.is_magnitude:
            raise ValueError("Conversion requires magnitude data")

        # Get offsets for conversion
        offsets = torch.tensor(
            [self.VEGA_AB_OFFSETS.get(b, 0.0) for b in self.bands],
            dtype=torch.float32,
            device=self["magnitudes"].device,
        )

        # Apply conversion: m_Vega = m_AB - offset
        vega_mags = self["magnitudes"] - offsets

        result = PhotometricTensorDict(
            vega_mags,
            self.bands,
            self.get("errors", None),
            filter_system="Vega",
            is_magnitude=True,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history(f"to_vega_system from {self._metadata['filter_system']}")
        return result

    def compute_k_corrections(
        self, sed_type: str = "elliptical", z_ref: float = 0.0
    ) -> torch.Tensor:
        """
        Compute K-corrections for cosmological sources.

        Args:
            sed_type: 'elliptical', 'spiral', 'starburst', or 'qso'
            z_ref: Reference redshift (usually 0)

        Returns:
            K-corrections to add to observed magnitudes
        """
        if "redshift" not in self:
            raise ValueError("Redshift required for K-corrections")

        z = self["redshift"]

        # Simplified K-correction templates
        k_corrections = {}

        if sed_type == "elliptical":
            k_corrections = {
                "g": 2.5 * torch.log10(1 + z) + 1.5 * z,
                "r": 2.5 * torch.log10(1 + z) + 1.0 * z,
                "i": 2.5 * torch.log10(1 + z) + 0.8 * z,
                "z": 2.5 * torch.log10(1 + z) + 0.6 * z,
            }
        elif sed_type == "spiral":
            k_corrections = {
                "g": 2.5 * torch.log10(1 + z) + 0.8 * z,
                "r": 2.5 * torch.log10(1 + z) + 0.5 * z,
                "i": 2.5 * torch.log10(1 + z) + 0.3 * z,
                "z": 2.5 * torch.log10(1 + z) + 0.2 * z,
            }
        elif sed_type == "starburst":
            k_corrections = {
                "g": 2.5 * torch.log10(1 + z) + 0.5 * z,
                "r": 2.5 * torch.log10(1 + z) + 0.3 * z,
                "i": 2.5 * torch.log10(1 + z) + 0.2 * z,
                "z": 2.5 * torch.log10(1 + z) + 0.1 * z,
            }

        # Build K-correction tensor
        k_corr = torch.zeros_like(self["magnitudes"])
        for i, band in enumerate(self.bands):
            if band in k_corrections:
                k_corr[..., i] = k_corrections[band]

        return k_corr

    def compute_colors(
        self,
        band_pairs: Optional[List[Tuple[str, str]]] = None,
        rest_frame: bool = False,
    ) -> TensorDict:
        """
        Calculate color indices with optional rest-frame correction.

        Args:
            band_pairs: List of (band1, band2) tuples. If None, compute standard colors
            rest_frame: Apply K-corrections for rest-frame colors
        """
        if not self.is_magnitude:
            raise ValueError("Color computation requires magnitude data")

        # Default color pairs if not specified
        if band_pairs is None:
            available_bands = set(self.bands)
            band_pairs = []

            # Standard optical colors
            optical_pairs = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
            band_pairs.extend(
                [p for p in optical_pairs if all(b in available_bands for b in p)]
            )

            # NIR colors
            nir_pairs = [("J", "H"), ("H", "Ks"), ("J", "Ks")]
            band_pairs.extend(
                [p for p in nir_pairs if all(b in available_bands for b in p)]
            )

            # Optical-NIR
            if "r" in available_bands and "Ks" in available_bands:
                band_pairs.append(("r", "Ks"))

        # Get magnitudes (apply K-corrections if rest-frame)
        mags = self["magnitudes"]
        if rest_frame and "redshift" in self:
            k_corr = self.compute_k_corrections()
            mags = mags - k_corr

        colors = {}
        color_errors = {}

        for band1, band2 in band_pairs:
            try:
                idx1 = self.bands.index(band1)
                idx2 = self.bands.index(band2)

                color_name = f"{band1}-{band2}"
                colors[color_name] = mags[..., idx1] - mags[..., idx2]

                # Error propagation
                if "errors" in self:
                    err1 = self["errors"][..., idx1]
                    err2 = self["errors"][..., idx2]
                    color_errors[color_name] = torch.sqrt(err1**2 + err2**2)
            except ValueError:
                # Skip if band not available
                continue

        result = TensorDict(colors, batch_size=self.batch_size)
        if color_errors:
            result["errors"] = TensorDict(color_errors, batch_size=self.batch_size)

        return result

    def to_flux_density(self) -> "PhotometricTensorDict":
        """Convert magnitudes to flux densities."""
        if not self.is_magnitude:
            return self.clone()

        # device = self["magnitudes"].device  # Removed unused variable
        flux_data = torch.zeros_like(self["magnitudes"])

        for i, band in enumerate(self.bands):
            mags = self["magnitudes"][:, i]

            # Standard flux conversion: f = 10^(-0.4 * (m - zp))
            if self.filter_system == "AB":
                # AB system: f_nu [Jy] = 10^(-0.4 * (m_AB + 48.6))
                flux_data[:, i] = 10 ** (-0.4 * (mags + 48.6))
            elif self.filter_system == "ST":
                # ST system
                flux_data[:, i] = 10 ** (-0.4 * (mags + 21.1))
            else:
                # Vega system - convert to AB first
                offset = self.VEGA_AB_OFFSETS.get(band, 0.0)
                ab_mags = mags + offset
                flux_data[:, i] = 10 ** (-0.4 * (ab_mags + 48.6))

        # Error propagation
        new_errors = None
        if "errors" in self:
            # For flux: sigma_f = f * ln(10) / 2.5 * sigma_m
            ln10_over_2p5 = torch.log(torch.tensor(10.0)) / 2.5
            new_errors = flux_data * ln10_over_2p5 * self["errors"]

        result = PhotometricTensorDict(
            flux_data,
            self.bands,
            new_errors,
            filter_system=self._metadata["filter_system"],
            is_magnitude=False,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history("to_flux_density")
        return result

    def to_magnitude(
        self, zeropoint: Optional[float] = None
    ) -> "PhotometricTensorDict":
        """Convert fluxes to magnitudes."""
        if self.is_magnitude:
            return self.clone()

        # Apply appropriate zero point based on filter system
        if self._metadata["filter_system"] == "AB":
            mag_data = (
                -2.5 * torch.log10(torch.clamp(self["magnitudes"], min=1e-30)) - 48.60
            )
        elif self._metadata["filter_system"] == "ST":
            mag_data = (
                -2.5 * torch.log10(torch.clamp(self["magnitudes"], min=1e-30)) - 21.10
            )
        else:
            zp = zeropoint if zeropoint is not None else 0.0
            mag_data = (
                -2.5 * torch.log10(torch.clamp(self["magnitudes"], min=1e-30)) + zp
            )

        # Error propagation
        new_errors = None
        if "errors" in self:
            factor = 2.5 / torch.log(torch.tensor(10.0))
            new_errors = (
                factor * self["errors"] / torch.clamp(self["magnitudes"], min=1e-30)
            )

        result = PhotometricTensorDict(
            mag_data,
            self.bands,
            new_errors,
            filter_system=self._metadata["filter_system"],
            is_magnitude=True,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history("to_magnitude")
        return result

    def compute_absolute_magnitudes(
        self, distance: Union[torch.Tensor, u.Quantity]
    ) -> torch.Tensor:
        """
        Compute absolute magnitudes from apparent magnitudes.

        Args:
            distance: Luminosity distance or distance modulus

        Returns:
            Absolute magnitudes
        """
        if not self.is_magnitude:
            raise ValueError("Absolute magnitude requires magnitude data")

        # Handle different distance inputs
        if isinstance(distance, u.Quantity):
            if distance.unit.is_equivalent(Unit("pc")):
                distance_modulus = 5 * torch.log10(distance.to(Unit("pc")).value / 10)
            elif distance.unit.is_equivalent(Unit("Mpc")):
                distance_modulus = 5 * torch.log10(
                    distance.to(Unit("Mpc")).value * 1e6 / 10
                )
            else:
                raise ValueError(f"Unsupported distance unit: {distance.unit}")
        else:
            # Assume distance in pc or distance modulus
            if distance.max() < 100:
                distance_modulus = distance
            else:
                distance_modulus = 5 * torch.log10(distance / 10)

        # Apply K-corrections if we have redshift
        k_corrections = torch.zeros_like(self["magnitudes"])
        if "redshift" in self:
            k_corrections = self.compute_k_corrections()

        # M = m - DM - K
        absolute_mags = (
            self["magnitudes"] - distance_modulus.unsqueeze(-1) - k_corrections
        )

        return absolute_mags

    def sigma_clip_outliers(
        self, sigma: float = 3.0, maxiters: int = 5, band_index: Optional[int] = None
    ) -> torch.Tensor:
        """
        Identify outliers using sigma clipping.

        Args:
            sigma: Sigma threshold for clipping
            maxiters: Maximum iterations
            band_index: Specific band to analyze (if None, use all bands)

        Returns:
            Boolean mask of good (non-outlier) objects
        """
        if band_index is not None:
            data = self["magnitudes"][:, band_index].detach().cpu().numpy()
        else:
            # Use magnitude range across all bands
            data = torch.ptp(self["magnitudes"], dim=1).detach().cpu().numpy()

        # Use astropy's sigma clipping
        mean, median, std = sigma_clipped_stats(data, sigma=sigma, maxiters=maxiters)

        # Create mask for good objects
        good_mask = np.abs(data - median) < (sigma * std)

        return torch.tensor(
            good_mask, dtype=torch.bool, device=self["magnitudes"].device
        )

    @property
    def bands(self) -> List[str]:
        """List of photometric bands."""
        return self._metadata["bands"]

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return self._metadata["n_bands"]

    @property
    def is_magnitude(self) -> bool:
        """Whether data represents magnitudes (True) or fluxes (False)."""
        return self._metadata["is_magnitude"]

    @property
    def filter_system(self) -> str:
        """Photometric system (AB, Vega, or ST)."""
        return self._metadata["filter_system"]

    @property
    def wavelengths(self) -> Dict[str, float]:
        """Effective wavelengths for all bands in Angstroms."""
        return self._metadata["wavelengths"]

    def get_band(self, band: str) -> torch.Tensor:
        """Extract data for a specific band."""
        try:
            band_idx = self.bands.index(band)
            return self["magnitudes"][..., band_idx]
        except ValueError:
            raise ValueError(f"Band '{band}' not found in {self.bands}")

    def get_band_errors(self, band: str) -> Optional[torch.Tensor]:
        """Extract errors for a specific band."""
        if "errors" not in self:
            return None
        try:
            band_idx = self.bands.index(band)
            return self["errors"][..., band_idx]
        except ValueError:
            raise ValueError(f"Band '{band}' not found in {self.bands}")

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract photometric features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('photometric', 'colors', 'magnitudes')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted photometric features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add photometric-specific computed features
        if feature_types is None or "photometric" in feature_types:
            # Basic photometric properties
            mags = self["magnitudes"]
            features["mean_magnitude"] = torch.mean(mags, dim=-1)
            features["magnitude_range"] = (
                torch.max(mags, dim=-1)[0] - torch.min(mags, dim=-1)[0]
            )
            features["magnitude_std"] = torch.std(mags, dim=-1)

            # Brightest and faintest magnitudes
            features["brightest_magnitude"] = torch.min(mags, dim=-1)[0]
            features["faintest_magnitude"] = torch.max(mags, dim=-1)[0]

        if feature_types is None or "colors" in feature_types:
            # Add color features if multiple bands available
            if len(self.bands) >= 2:
                colors = self.compute_colors()
                for color_name, color_values in colors.items():
                    if isinstance(color_values, torch.Tensor):
                        features[f"color_{color_name}"] = color_values

        return features

    def extract_photometric_features(self) -> torch.Tensor:
        """
        Extract comprehensive photometric features for classification.

        Returns:
            [N, F] Feature tensor with photometric properties
        """
        features = []
        mags = self["magnitudes"]

        # Basic magnitude statistics
        features.extend(
            [
                torch.mean(mags, dim=-1),  # Mean magnitude
                torch.std(mags, dim=-1),  # Magnitude scatter
                torch.min(mags, dim=-1)[0],  # Brightest magnitude
                torch.max(mags, dim=-1)[0],  # Faintest magnitude
                torch.median(mags, dim=-1)[0],  # Median magnitude
            ]
        )

        # Magnitude range and ratios
        mag_range = torch.max(mags, dim=-1)[0] - torch.min(mags, dim=-1)[0]
        features.append(mag_range)

        # Colors if multiple bands
        if len(self.bands) >= 2:
            try:
                colors_dict = self.compute_colors()
                for color_values in colors_dict.values():
                    if isinstance(color_values, torch.Tensor):
                        features.append(color_values)
                        break  # Just add first color for now
            except Exception:
                pass

        # Flux ratios between bands
        if len(self.bands) >= 2:
            # Convert to flux for ratios
            flux_data = 10 ** (-0.4 * mags)  # Simple flux conversion

            # Blue to red ratio (first vs last band)
            blue_red_ratio = flux_data[:, 0] / (flux_data[:, -1] + 1e-10)
            features.append(blue_red_ratio)

        # Error-based features if available
        if "errors" in self:
            errors = self["errors"]
            features.extend(
                [
                    torch.mean(errors, dim=-1),  # Mean error
                    torch.mean(mags / (errors + 1e-10), dim=-1),  # Mean S/N ratio
                ]
            )

        return (
            torch.stack(features, dim=-1) if features else torch.zeros(mags.shape[0], 0)
        )

    def validate(self) -> bool:
        """Validate photometric tensor data."""
        if not super().validate():
            return False

        return (
            "magnitudes" in self
            and self["magnitudes"].shape[-1] == len(self.bands)
            and len(self.bands) > 0
            and self.validate_finite_values("magnitudes")
        )
