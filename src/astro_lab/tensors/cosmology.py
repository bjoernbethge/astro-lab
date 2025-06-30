"""
Cosmology TensorDict
===================

TensorDict for cosmological calculations with proper astropy integration.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from astropy import units as u
from astropy.constants import c as const_c
from astropy.coordinates import SkyCoord
from astropy.cosmology import FLRW, FlatLambdaCDM, Planck18

from .base import AstroTensorDict
from .mixins import NormalizationMixin, ValidationMixin

logger = logging.getLogger(__name__)


class CosmologyTensorDict(AstroTensorDict, NormalizationMixin, ValidationMixin):
    """
    TensorDict for cosmological calculations with proper astropy integration.

    Features:
    - Proper astropy.cosmology integration for distance calculations
    - Various cosmological models (ΛCDM, wCDM, etc.)
    - Luminosity distance, angular diameter distance, comoving volume
    - Age of universe, lookback time calculations
    - K-corrections and evolutionary corrections
    - Cosmological parameter inference support

    Best Practices:
    - Always use explicit cosmology objects instead of global defaults
    - For reproducible results, specify cosmology parameters explicitly
    - Use astropy.cosmology fitting utilities for parameter estimation
    - Validate input data before cosmological calculations
    """

    def __init__(
        self,
        redshifts: torch.Tensor,
        cosmology: Optional[Union[FLRW, Dict[str, float]]] = None,
        ra: Optional[torch.Tensor] = None,
        dec: Optional[torch.Tensor] = None,
        observed_magnitudes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Initialize CosmologyTensorDict with proper astronomical cosmology.

        Args:
            redshifts: [N] Source redshifts
            cosmology: Astropy cosmology object or parameter dict
            ra: Right ascension in degrees (optional)
            dec: Declination in degrees (optional)
            observed_magnitudes: Observed magnitudes for distance calculations

        Note:
            For reproducible results, always use explicit cosmology objects.
            See astropy.cosmology documentation for available models.
        """
        if redshifts.dim() != 1:
            raise ValueError(
                f"Redshifts must be 1D tensor, got shape {redshifts.shape}"
            )

        n_objects = redshifts.shape[0]

        # Handle cosmology - prefer explicit objects over defaults
        if cosmology is None:
            logger.warning(
                "No cosmology specified, using Planck18. "
                "For reproducible results, use explicit cosmology objects."
            )
            self.cosmology = Planck18
        elif isinstance(cosmology, dict):
            # Create cosmology from parameters
            logger.info("Creating cosmology from parameters")
            H0 = cosmology.get("H0", 70) * u.Unit("km") / u.Unit("s") / u.Unit("Mpc")
            Om0 = cosmology.get("Om0", 0.3)
            Tcmb0 = cosmology.get("Tcmb0", 2.725) * u.Unit("K")
            self.cosmology = FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0)
        else:
            # Use provided cosmology object
            if not isinstance(cosmology, FLRW):
                raise ValueError("Cosmology must be an astropy.cosmology.FLRW object")
            self.cosmology = cosmology
            logger.info(f"Using cosmology: {cosmology.name}")

        # Validate cosmology parameters
        if not hasattr(self.cosmology, "H0") or not hasattr(self.cosmology, "Om0"):
            raise ValueError("Cosmology must have H0 and Om0 parameters")

        data = {
            "redshifts": redshifts,
            "meta": {
                "n_objects": n_objects,
                "cosmology_name": self.cosmology.name,
                "cosmology_params": {
                    "H0": self.cosmology.H0.value,
                    "Om0": self.cosmology.Om0,
                    "Ode0": getattr(self.cosmology, "Ode0", None),
                },
                "calculated_quantities": [],
                "redshift_range": (redshifts.min().item(), redshifts.max().item()),
            },
        }

        # Add coordinates if provided
        if ra is not None and dec is not None:
            if ra.shape != redshifts.shape or dec.shape != redshifts.shape:
                raise ValueError("RA and Dec must have same shape as redshifts")

            coordinates = SkyCoord(
                ra=ra.detach().cpu().numpy() * u.Unit("deg"),
                dec=dec.detach().cpu().numpy() * u.Unit("deg"),
                frame="icrs",
            )
            data["coordinates"] = coordinates

        if observed_magnitudes is not None:
            data["observed_magnitudes"] = observed_magnitudes

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def redshifts(self) -> torch.Tensor:
        """Source redshifts."""
        return self["redshifts"]

    def compute_distances(self) -> Dict[str, torch.Tensor]:
        """
        Compute various cosmological distances using astropy.

        Returns:
            Dictionary with distance measures in Mpc
        """
        z = self.redshifts.detach().cpu().numpy()

        # Compute distances using astropy cosmology
        luminosity_distance = self.cosmology.luminosity_distance(z).to_value("Mpc")
        angular_diameter_distance = self.cosmology.angular_diameter_distance(
            z
        ).to_value("Mpc")
        comoving_distance = self.cosmology.comoving_distance(z).to_value("Mpc")

        # Distance modulus
        distance_modulus = 5 * np.log10(luminosity_distance * 1e5)  # Convert Mpc to pc

        distances = {
            "luminosity_distance": torch.tensor(
                luminosity_distance, dtype=torch.float32
            ),
            "angular_diameter_distance": torch.tensor(
                angular_diameter_distance, dtype=torch.float32
            ),
            "comoving_distance": torch.tensor(comoving_distance, dtype=torch.float32),
            "distance_modulus": torch.tensor(distance_modulus, dtype=torch.float32),
        }

        # Store in tensordict
        for key, value in distances.items():
            self[key] = value

        self._metadata["calculated_quantities"].extend(list(distances.keys()))
        self.add_history("compute_distances", cosmology=self.cosmology.name)

        return distances

    def compute_times(self) -> Dict[str, torch.Tensor]:
        """
        Compute cosmological time measures.

        Returns:
            Dictionary with time measures in Gyr
        """
        z = self.redshifts.detach().cpu().numpy()

        # Age of universe at redshift z
        age_at_z = self.cosmology.age(z).to_value("Gyr")

        # Lookback time
        lookback_time = self.cosmology.lookback_time(z).to_value("Gyr")

        # Age of universe today
        age_today = self.cosmology.age(0).to_value("Gyr")

        times = {
            "age_at_redshift": torch.tensor(age_at_z, dtype=torch.float32),
            "lookback_time": torch.tensor(lookback_time, dtype=torch.float32),
            "age_universe_today": torch.tensor(age_today, dtype=torch.float32),
        }

        # Store in tensordict
        for key, value in times.items():
            self[key] = value

        self._metadata["calculated_quantities"].extend(list(times.keys()))
        self.add_history("compute_times", cosmology=self.cosmology.name)

        return times

    def compute_volumes(self) -> Dict[str, torch.Tensor]:
        """
        Compute cosmological volumes.

        Returns:
            Dictionary with volume measures in Mpc³
        """
        z = self.redshifts.detach().cpu().numpy()

        # Comoving volume element
        dV_dz = self.cosmology.differential_comoving_volume(z).to_value("Mpc3/sr")

        # Comoving volume within redshift (for spherical survey)
        comoving_volume = self.cosmology.comoving_volume(z).to_value("Mpc3")

        volumes = {
            "differential_comoving_volume": torch.tensor(dV_dz, dtype=torch.float32),
            "comoving_volume": torch.tensor(comoving_volume, dtype=torch.float32),
        }

        # Store in tensordict
        for key, value in volumes.items():
            self[key] = value

        self._metadata["calculated_quantities"].extend(list(volumes.keys()))
        self.add_history("compute_volumes", cosmology=self.cosmology.name)

        return volumes

    def compute_absolute_magnitudes(
        self,
        observed_magnitudes: Optional[torch.Tensor] = None,
        k_corrections: Optional[torch.Tensor] = None,
        evolutionary_corrections: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute absolute magnitudes from observed magnitudes.

        Args:
            observed_magnitudes: Observed magnitudes (if not provided, use stored)
            k_corrections: K-corrections for each object
            evolutionary_corrections: Evolutionary corrections

        Returns:
            Absolute magnitudes
        """
        if observed_magnitudes is None:
            if "observed_magnitudes" not in self:
                raise ValueError("No observed magnitudes provided")
            observed_magnitudes = self["observed_magnitudes"]

        # Compute distance modulus if not already done
        if "distance_modulus" not in self:
            self.compute_distances()

        distance_modulus = self["distance_modulus"]

        # Basic calculation: M = m - μ
        absolute_magnitudes = observed_magnitudes - distance_modulus

        # Apply K-corrections if provided
        if k_corrections is not None:
            absolute_magnitudes -= k_corrections

        # Apply evolutionary corrections if provided
        if evolutionary_corrections is not None:
            absolute_magnitudes -= evolutionary_corrections

        self["absolute_magnitudes"] = absolute_magnitudes
        self._metadata["calculated_quantities"].append("absolute_magnitudes")
        self.add_history("compute_absolute_magnitudes")

        return absolute_magnitudes

    def compute_k_corrections(
        self, filter_observed: str, filter_rest: str, sed_type: str = "elliptical"
    ) -> torch.Tensor:
        """
        Compute K-corrections for magnitude conversion.

        Args:
            filter_observed: Observed filter name
            filter_rest: Rest-frame filter name
            sed_type: SED template type

        Returns:
            K-corrections
        """
        z = self.redshifts

        # Simplified K-correction templates
        # In practice, would use proper SED fitting
        if sed_type == "elliptical":
            # Early-type galaxy K-correction approximation
            k_corrections = 2.5 * torch.log10(1 + z) + 1.2 * z
        elif sed_type == "spiral":
            # Late-type galaxy K-correction
            k_corrections = 2.5 * torch.log10(1 + z) + 0.8 * z
        elif sed_type == "starburst":
            # Starburst galaxy K-correction
            k_corrections = 2.5 * torch.log10(1 + z) + 0.5 * z
        elif sed_type == "qso":
            # Quasar K-correction
            k_corrections = 2.5 * torch.log10(1 + z) + 0.3 * z
        else:
            raise ValueError(f"Unknown SED type: {sed_type}")

        self["k_corrections"] = k_corrections
        self._metadata["calculated_quantities"].append("k_corrections")
        self.add_history("compute_k_corrections", sed_type=sed_type)

        return k_corrections

    def hubble_parameter(self, redshift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Hubble parameter H(z) at given redshifts.

        Args:
            redshift: Redshifts (if None, use stored redshifts)

        Returns:
            Hubble parameter in km/s/Mpc
        """
        if redshift is None:
            redshift = self.redshifts

        z_array = redshift.detach().cpu().numpy()
        H_z = self.cosmology.H(z_array).to_value("km/s/Mpc")

        return torch.tensor(H_z, dtype=torch.float32)

    def critical_density(self, redshift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute critical density at given redshifts.

        Args:
            redshift: Redshifts (if None, use stored redshifts)

        Returns:
            Critical density in g/cm³
        """
        if redshift is None:
            redshift = self.redshifts

        z_array = redshift.detach().cpu().numpy()
        rho_crit = self.cosmology.critical_density(z_array).to(u.g / u.cm**3).value

        return torch.tensor(rho_crit, dtype=torch.float32)

    def scale_factor(self, redshift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scale factor a = 1/(1+z).

        Args:
            redshift: Redshifts (if None, use stored redshifts)

        Returns:
            Scale factor (dimensionless)
        """
        if redshift is None:
            redshift = self.redshifts

        scale_factor = 1.0 / (1.0 + redshift)
        return scale_factor

    def compute_survey_volume(
        self,
        area: u.Quantity,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> u.Quantity:
        """
        Compute survey volume for given sky area and redshift range.

        Args:
            area: Survey area on sky
            z_min: Minimum redshift (if None, use min of data)
            z_max: Maximum redshift (if None, use max of data)

        Returns:
            Survey volume in Mpc³
        """
        if z_min is None:
            z_min = self.redshifts.min().item()
        if z_max is None:
            z_max = self.redshifts.max().item()

        # Comoving volume between redshifts
        vol_max = self.cosmology.comoving_volume(z_max)
        vol_min = self.cosmology.comoving_volume(z_min)

        # Scale by survey area (assume full sphere is 4π steradians)
        area_steradians = area.to_value("steradian")
        full_sphere = 4 * np.pi * area.to_value("steradian")

        survey_volume = (vol_max - vol_min) * (area_steradians / full_sphere)

        return survey_volume.to(u.Unit("Mpc") ** 3)

    def estimate_cosmological_parameters(
        self,
        observed_data: Dict[str, torch.Tensor],
        parameter_priors: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Estimate cosmological parameters from observed data.

        Args:
            observed_data: Dictionary with observed quantities
            parameter_priors: Prior ranges for parameters

        Returns:
            Best-fit cosmological parameters

        Raises:
            NotImplementedError: If no real likelihood analysis is implemented.
        """
        raise NotImplementedError(
            "Cosmological parameter estimation requires real data and a likelihood function. "
            "Please implement a real fitting procedure."
        )

    def compare_cosmologies(
        self, cosmologies: List[FLRW], observed_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compare different cosmological models.

        Args:
            cosmologies: List of astropy cosmology objects
            observed_data: Observed data for comparison

        Returns:
            Dictionary with comparison metrics

        Raises:
            NotImplementedError: If no real comparison is implemented.
        """
        raise NotImplementedError(
            "Cosmology comparison requires real observed data and model predictions. "
            "Please implement a real comparison procedure."
        )

    def validate(self) -> bool:
        """Validate cosmology tensor data."""
        # Basic validation from parent
        if not super().validate():
            return False

        # Cosmology-specific validation
        return (
            "redshifts" in self
            and self.redshifts.dim() == 1
            and torch.all(self.redshifts >= 0)
            and torch.all(self.redshifts < 20)  # Reasonable redshift range
            and torch.all(torch.isfinite(self.redshifts))
        )

    # Calculate redshift using velocity
    c_kms = const_c.to_value("km/s")  # Speed of light in km/s


def create_cosmology_from_parameters(
    H0: float = 70.0,
    Om0: float = 0.3,
    Ode0: Optional[float] = None,
    Tcmb0: float = 2.725,
    name: str = "CustomCosmology",
) -> FLRW:
    """
    Create a cosmology object from parameters following Astropy best practices.

    Args:
        H0: Hubble constant in km/s/Mpc
        Om0: Matter density parameter
        Ode0: Dark energy density parameter (if None, assumes flat universe)
        Tcmb0: CMB temperature in K
        name: Name for the cosmology

    Returns:
        Astropy cosmology object
    """
    if Ode0 is None:
        # Flat universe: Ode0 = 1 - Om0
        Ode0 = 1.0 - Om0

    return FlatLambdaCDM(
        H0=H0 * u.Unit("km") / u.Unit("s") / u.Unit("Mpc"),
        Om0=Om0,
        Ode0=Ode0,
        Tcmb0=Tcmb0 * u.Unit("K"),
        name=name,
    )


def validate_cosmology_parameters(
    H0: float, Om0: float, Ode0: Optional[float] = None
) -> bool:
    """
    Validate cosmology parameters for physical consistency.

    Args:
        H0: Hubble constant in km/s/Mpc
        Om0: Matter density parameter
        Ode0: Dark energy density parameter

    Returns:
        True if parameters are physically reasonable

    Raises:
        ValueError: If parameters are unphysical
    """
    if H0 <= 0:
        raise ValueError("H0 must be positive")

    if Om0 < 0 or Om0 > 1:
        raise ValueError("Om0 must be between 0 and 1")

    if Ode0 is not None:
        if Ode0 < 0 or Ode0 > 1:
            raise ValueError("Ode0 must be between 0 and 1")

        # Check for flat universe consistency
        total = Om0 + Ode0
        if not (0.95 <= total <= 1.05):  # Allow small deviations from flatness
            logger.warning(f"Universe may not be flat: Om0 + Ode0 = {total}")

    return True
