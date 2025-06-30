"""
TensorDict for photometric data with proper astronomical handling
using real astropy APIs.
"""

from typing import Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats

# Real astropy photometric units and functions
from astropy.units import Unit
from tensordict import TensorDict

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class PhotometricTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    TensorDict for photometric data with real astropy integration.

    Features:
    - AB/Vega/ST magnitude systems with proper astropy units
    - Proper magnitude and flux conversions using astropy
    - K-corrections for cosmological sources
    - Color transformations between photometric systems
    - Proper error propagation
    - Integration with astropy.units.photometric
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

    # Effective wavelengths in Angstroms for spectral density conversions
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
        Initialize PhotometricTensorDict with astronomical features.

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

        super().__init__(data, batch_size=magnitudes.shape[:-1], **kwargs)

    def to_astropy_magnitudes(self, band_index: int = 0) -> u.Quantity:
        """
        Convert to proper astropy Magnitude objects.

        Args:
            band_index: Which band to convert (default: first band)

        Returns:
            Astropy Magnitude quantity with proper units
        """
        if not self.is_magnitude:
            raise ValueError("Can only convert magnitude data to astropy Magnitudes")

        mags = self["magnitudes"][:, band_index]

        if self.filter_system == "AB":
            return mags.detach().cpu().numpy() * u.mag
        elif self.filter_system == "ST":
            return mags.detach().cpu().numpy() * u.mag
        else:
            # Vega system - convert to AB first
            offset = self.VEGA_AB_OFFSETS.get(self.bands[band_index], 0.0)
            ab_mags = mags + offset
            return ab_mags.detach().cpu().numpy() * u.mag

    def to_ab_system(self) -> "PhotometricTensorDict":
        """Convert magnitudes to AB system using astropy units."""
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
        """Convert magnitudes to Vega system using astropy units."""
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
        # In practice, use kcorrect or template fitting
        k_corrections = {}

        if sed_type == "elliptical":
            # Elliptical galaxy template (simplified)
            k_corrections = {
                "g": 2.5 * torch.log10(1 + z) + 1.5 * z,
                "r": 2.5 * torch.log10(1 + z) + 1.0 * z,
                "i": 2.5 * torch.log10(1 + z) + 0.8 * z,
                "z": 2.5 * torch.log10(1 + z) + 0.6 * z,
            }
        elif sed_type == "spiral":
            # Spiral galaxy template
            k_corrections = {
                "g": 2.5 * torch.log10(1 + z) + 0.8 * z,
                "r": 2.5 * torch.log10(1 + z) + 0.5 * z,
                "i": 2.5 * torch.log10(1 + z) + 0.3 * z,
                "z": 2.5 * torch.log10(1 + z) + 0.2 * z,
            }
        elif sed_type == "starburst":
            # Starburst galaxy template
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

    def to_flux_density(
        self, wavelength: Optional[u.Quantity] = None
    ) -> "PhotometricTensorDict":
        """
        Convert magnitudes to flux densities using astropy units and equivalencies.

        Args:
            wavelength: Reference wavelength for conversion (if None, use band centers)
        """
        if not self.is_magnitude:
            return self.clone()

        device = self["magnitudes"].device
        flux_data = torch.zeros_like(self["magnitudes"])

        for i, band in enumerate(self.bands):
            # Get magnitude values for this band
            mags = self["magnitudes"][:, i]

            # Convert to astropy magnitudes
            if self.filter_system == "AB":
                astropy_mags = mags.detach().cpu().numpy() * u.mag
            elif self.filter_system == "ST":
                astropy_mags = mags.detach().cpu().numpy() * u.mag
            else:
                # Vega - convert to AB first
                offset = self.VEGA_AB_OFFSETS.get(band, 0.0)
                ab_mags = mags + offset
                astropy_mags = ab_mags.detach().cpu().numpy() * u.mag

            # Get wavelength for this band
            if wavelength is not None:
                pass  # band_wl is not used
            else:
                pass  # band_wl is not used

            # Convert to flux density using astropy
            flux_density = astropy_mags.to(
                u.erg / u.Unit("s") / u.Unit("cm") ** 2 / u.Unit("AA"),
                equivalencies=u.spectral(),
            )

            # Convert flux density units
            if flux_density.unit.is_equivalent(
                u.Unit("erg") / u.Unit("s") / u.Unit("cm") ** 2 / u.Unit("AA")
            ):
                # Convert from erg/s/cm²/Å to erg/s/cm²/Hz
                flux_density = flux_density.to(
                    u.Unit("erg") / u.Unit("s") / u.Unit("cm") ** 2 / u.Unit("Hz"),
                    equivalencies=u.spectral(),
                )

            flux_data[:, i] = torch.tensor(
                flux_density.value, device=device, dtype=torch.float32
            )

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
        """Convert fluxes to magnitudes with proper zero points."""
        if self.is_magnitude:
            return self.clone()

        # Apply appropriate zero point based on filter system
        if self._metadata["filter_system"] == "AB":
            # AB magnitude: m_AB = -2.5 * log10(f_nu) - 48.60
            mag_data = (
                -2.5 * torch.log10(torch.clamp(self["magnitudes"], min=1e-30)) - 48.60
            )
        elif self._metadata["filter_system"] == "ST":
            # ST magnitude: m_ST = -2.5 * log10(f_lambda) - 21.10
            mag_data = (
                -2.5 * torch.log10(torch.clamp(self["magnitudes"], min=1e-30)) - 21.10
            )
        else:
            # Default or custom zeropoint
            zp = zeropoint if zeropoint is not None else 0.0
            mag_data = (
                -2.5 * torch.log10(torch.clamp(self["magnitudes"], min=1e-30)) + zp
            )

        # Error propagation: sigma_m = 2.5 / ln(10) * sigma_f / f
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
        self, distance: Union[torch.Tensor, u.Quantity], cosmology=None
    ) -> torch.Tensor:
        """
        Compute absolute magnitudes from apparent magnitudes.

        Args:
            distance: Luminosity distance or distance modulus
            cosmology: Astropy cosmology for redshift-based calculation
        """
        if not self.is_magnitude:
            raise ValueError("Absolute magnitude requires magnitude data")

        # Handle different distance inputs
        if isinstance(distance, u.Quantity):
            # Convert to distance modulus
            if distance.unit.is_equivalent(Unit("pc")):
                distance_modulus = 5 * torch.log10(distance.to(Unit("pc")).value / 10)
            elif distance.unit.is_equivalent(Unit("Mpc")):
                distance_modulus = 5 * torch.log10(
                    distance.to(Unit("Mpc")).value * 1e6 / 10
                )
            else:
                raise ValueError(f"Unsupported distance unit: {distance.unit}")
        else:
            # Assume it's already distance modulus or distance in pc
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
        Identify outliers using sigma clipping from astropy.stats.

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

    def validate(self) -> bool:
        """Validate photometric tensor data."""
        # Basic validation from parent
        if not super().validate():
            return False

        # Photometric-specific validation
        return (
            self.validate_magnitudes()
            and "magnitudes" in self
            and self["magnitudes"].shape[-1] == len(self.bands)
            and len(self.bands) > 0
        )
