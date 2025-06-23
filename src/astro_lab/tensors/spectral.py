"""
Spectral tensor for astronomical spectroscopy data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typing_extensions import Self
import numpy as np

from .base import AstroTensorBase

class SpectralTensor(AstroTensorBase):
    """
    Represents spectral data for celestial objects, including flux/intensity
    as a function of wavelength or frequency.
    """

    _metadata_fields = [
        "wavelengths",
        "redshift",
        "flux_units",
        "wavelength_units",
        "spectral_resolution",
        "instrument",
    ]

    def __init__(
        self,
        data: Union[torch.Tensor, List, Any],
        wavelengths: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        spectral_resolution: Optional[float] = None,
        **kwargs,
    ):
        """
        Initializes the SpectralTensor.

        Args:
            data: Spectral data, shape [..., n_wavelengths].
            wavelengths: A 1D tensor of wavelength values.
            uncertainties: Measurement uncertainties, same shape as data.
            spectral_resolution: The spectral resolution (R = lambda / delta_lambda).
            **kwargs: Additional metadata.
        """
        spectral_meta = {
            "wavelengths": wavelengths,
            "uncertainties": uncertainties,
            "spectral_resolution": spectral_resolution,
        }
        kwargs.update(spectral_meta)

        super().__init__(data=data, **kwargs)
        self._validate()

    def _validate(self) -> None:
        """Validates the structure of the spectral data."""
        if self.data.dim() < 1:
            raise ValueError("Spectral data must be at least 1D.")
        
        wavelengths = self.meta.get('wavelengths')
        if wavelengths is None:
            raise ValueError("SpectralTensor requires a 'wavelengths' tensor.")
            
        if not isinstance(wavelengths, torch.Tensor) or wavelengths.dim() != 1:
            raise ValueError("'wavelengths' must be a 1D torch.Tensor.")

        if self.data.shape[-1] != len(wavelengths):
            raise ValueError(
                f"Last dimension of data ({self.data.shape[-1]}) must match "
                f"the length of the wavelengths tensor ({len(wavelengths)})."
            )

    @property
    def wavelengths(self) -> torch.Tensor:
        return self.meta.get("wavelengths")

    @property
    def redshift(self) -> float:
        """Redshift value."""
        return self.meta.get("redshift", 0.0)

    @property
    def flux_units(self) -> str:
        """Flux units."""
        return self.meta.get("flux_units", "erg/s/cm2/A")

    @property
    def wavelength_units(self) -> str:
        """Wavelength units."""
        return self.meta.get("wavelength_units", "Angstrom")

    @property
    def spectral_resolution(self) -> Optional[float]:
        """Spectral resolution."""
        return self.meta.get("spectral_resolution")

    @property
    def instrument(self) -> Optional[str]:
        """Instrument name."""
        return self.meta.get("instrument")

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength bins."""
        return len(self.wavelengths)

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Wavelength range (min, max)."""
        wavelengths = self.wavelengths
        return (float(wavelengths.min()), float(wavelengths.max()))

    @property
    def delta_wavelength(self) -> torch.Tensor:
        """Wavelength differences between adjacent bins."""
        wavelengths = self.wavelengths
        return wavelengths[1:] - wavelengths[:-1]

    @property
    def rest_wavelengths(self) -> torch.Tensor:
        """Rest frame wavelengths (corrected for redshift)."""
        return self.wavelengths / (1 + self.redshift)

    @property
    def observed_wavelengths(self) -> torch.Tensor:
        """Observed wavelengths (original)."""
        return self.wavelengths

    def apply_redshift(self, z: float) -> "SpectralTensor":
        """
        Apply redshift to spectrum.

        Args:
            z: Redshift value

        Returns:
            New SpectralTensor with redshift applied
        """
        new_wavelengths = self.wavelengths * (1 + z)

        new_metadata = self.meta.copy()
        new_metadata["wavelengths"] = new_wavelengths
        new_metadata["redshift"] = self.redshift + z

        return SpectralTensor(self.data, **new_metadata)

    def deredshift(self) -> "SpectralTensor":
        """Remove redshift, returning to rest frame."""
        return self.apply_redshift(-self.redshift)

    def to_velocity_space(
        self, rest_wavelength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to velocity space around a rest wavelength.

        Args:
            rest_wavelength: Rest wavelength for velocity calculation

        Returns:
            Tuple of (velocities, flux_data)
        """
        c = 299792.458  # km/s
        wavelengths = self.wavelengths

        # Calculate velocities
        velocities = c * (wavelengths - rest_wavelength) / rest_wavelength

        return velocities, self.data

    def measure_line(
        self, wavelength: float, window: float = 50.0
    ) -> Dict[str, torch.Tensor]:
        """
        Measure spectral line properties.

        Args:
            wavelength: Central wavelength
            window: Window size around line (in wavelength units)

        Returns:
            Dictionary with line measurements
        """
        wavelengths = self.wavelengths
        flux = self.data

        # Find indices within window
        mask = torch.abs(wavelengths - wavelength) <= window / 2

        if not mask.any():
            raise ValueError(f"No data found within {window} of {wavelength}")

        line_wavelengths = wavelengths[mask]
        line_flux = flux[..., mask]

        # Basic measurements
        peak_idx = torch.argmax(line_flux, dim=-1)
        peak_wavelength = line_wavelengths[peak_idx]
        peak_flux = torch.max(line_flux, dim=-1)[0]

        continuum_flux = torch.mean(
            torch.cat([line_flux[..., :5], line_flux[..., -5:]], dim=-1), dim=-1
        )

        equivalent_width = torch.sum(
            1 - line_flux / continuum_flux.unsqueeze(-1), dim=-1
        ) * torch.mean(self.delta_wavelength)

        return {
            "line_flux": line_flux,
            "continuum_flux": continuum_flux,
            "equivalent_width": equivalent_width,
            "line_center": torch.tensor(wavelength),
        }

    def atmospheric_transmission_spectrum(
        self,
        planet_radius: float,
        stellar_radius: float,
        atmospheric_scale_height: float,
        molecular_species: Optional[List[str]] = None,
    ) -> "SpectralTensor":
        """
        Calculate atmospheric transmission spectrum for exoplanet transit.

        Args:
            planet_radius: Planet radius in Earth radii
            stellar_radius: Stellar radius in solar radii
            atmospheric_scale_height: Atmospheric scale height in km
            molecular_species: List of molecular species to include

        Returns:
            SpectralTensor with transmission spectrum
        """
        if molecular_species is None:
            molecular_species = ["H2O", "CO2", "O3", "CH4"]

        wavelengths = self.wavelengths
        earth_radius = 6371.0  # km
        solar_radius = 696000.0  # km

        # Convert to physical units
        rp_km = planet_radius * earth_radius
        rs_km = stellar_radius * solar_radius

        # Transit depth (geometric)
        transit_depth_base = (rp_km / rs_km) ** 2

        # Wavelength-dependent atmospheric effects
        # Simplified model: Rayleigh scattering + molecular absorption
        lambda_ref = 550.0  # nm reference wavelength

        # Convert wavelengths to nm for calculations
        wavelengths_nm = wavelengths / 10.0  # Assuming input in Angstroms

        # Rayleigh scattering (λ^-4 dependence)
        rayleigh_factor = (lambda_ref / wavelengths_nm) ** 4

        # Scale height effect on atmosphere thickness
        atmosphere_thickness = atmospheric_scale_height * torch.ones_like(wavelengths)

        # Enhanced transit depth due to atmosphere
        enhanced_depth = torch.zeros_like(wavelengths)

        for species in molecular_species:
            if species == "H2O":
                # Water vapor absorption features
                absorption_lines = [1400, 1900, 2700]  # nm (simplified)
                for line_center in absorption_lines:
                    line_strength = torch.exp(
                        -((wavelengths_nm - line_center) ** 2) / (50**2)
                    )
                    enhanced_depth += line_strength * 0.001  # Relative depth

            elif species == "CO2":
                # CO2 absorption features
                absorption_lines = [1400, 1600, 2000]  # nm
                for line_center in absorption_lines:
                    line_strength = torch.exp(
                        -((wavelengths_nm - line_center) ** 2) / (30**2)
                    )
                    enhanced_depth += line_strength * 0.0005

            elif species == "O3":
                # Ozone absorption (Chappuis band)
                line_strength = torch.exp(-((wavelengths_nm - 600) ** 2) / (100**2))
                enhanced_depth += line_strength * 0.0003

            elif species == "CH4":
                # Methane absorption
                absorption_lines = [890, 1200, 1650, 2200]  # nm
                for line_center in absorption_lines:
                    line_strength = torch.exp(
                        -((wavelengths_nm - line_center) ** 2) / (40**2)
                    )
                    enhanced_depth += line_strength * 0.0002

        # Add Rayleigh scattering contribution
        enhanced_depth += rayleigh_factor * 0.0001

        # Total transit depth
        total_depth = transit_depth_base + enhanced_depth

        # Create new spectral tensor
        metadata = self.meta.copy()
        return SpectralTensor(
            total_depth,
            wavelengths=wavelengths,
            redshift=self.redshift,
            **{
                k: v
                for k, v in metadata.items()
                if k not in ["wavelengths", "redshift"]
            },
        )

    def biosignature_detection(
        self,
        snr_threshold: float = 5.0,
        observation_time: float = 10.0,  # hours
    ) -> Dict[str, torch.Tensor]:
        """
        Assess biosignature detection potential in spectrum.

        Args:
            snr_threshold: Minimum SNR for detection
            observation_time: Total observation time in hours

        Returns:
            Dictionary with detection metrics for biosignatures
        """
        wavelengths = getattr(self, "wavelengths", torch.linspace(0.3, 30.0, len(self)))

        # Key biosignature wavelengths (μm)
        biosignatures = {
            "O2": 0.76,  # Oxygen A-band
            "O3": 9.6,  # Ozone
            "H2O": 1.4,  # Water vapor
            "CH4": 3.3,  # Methane
            "N2O": 4.5,  # Nitrous oxide
            "NH3": 10.5,  # Ammonia
            "PH3": 4.3,  # Phosphine (potential biosignature)
            "DMS": 9.2,  # Dimethyl sulfide
        }

        detection_results = {}

        for species, target_wavelength in biosignatures.items():
            # Find closest wavelength in spectrum
            wave_diff = torch.abs(wavelengths - target_wavelength)
            closest_idx = torch.argmin(wave_diff)

            # Get signal at this wavelength
            signal = self[closest_idx]

            # Estimate noise (simplified)
            # Real calculation would depend on instrument, stellar brightness, etc.
            noise_level = 0.001 * torch.sqrt(
                torch.tensor(1.0 / observation_time)
            )  # Simplified

            # Signal-to-noise ratio
            snr = torch.abs(signal) / noise_level

            # Detection probability
            detection_prob = torch.sigmoid((snr - snr_threshold) / 2.0)

            detection_results[species] = {
                "wavelength": target_wavelength,
                "signal_strength": signal,
                "snr": snr,
                "detection_probability": detection_prob,
                "detectable": snr > snr_threshold,
            }

        # Overall biosignature potential
        all_probs = torch.stack(
            [result["detection_probability"] for result in detection_results.values()]
        )
        overall_potential = 1.0 - torch.prod(1.0 - all_probs)  # At least one detection

        detection_results["overall_biosignature_potential"] = overall_potential

        return detection_results

    def interstellar_reddening_correction(
        self, distance_ly: float, av_per_kpc: float = 1.0
    ) -> "SpectralTensor":
        """
        Correct spectrum for interstellar reddening.

        Args:
            distance_ly: Distance to object in light-years
            av_per_kpc: Visual extinction per kiloparsec

        Returns:
            Dereddened SpectralTensor
        """
        wavelengths = getattr(self, "wavelengths", torch.linspace(0.3, 30.0, len(self)))

        # Convert distance to kpc
        distance_kpc = distance_ly / 3262.0

        # Total visual extinction
        av_total = av_per_kpc * distance_kpc

        # Cardelli, Clayton & Mathis (1989) extinction curve
        # Simplified version for UV/optical/NIR
        x = 1.0 / wavelengths  # 1/μm

        # Initialize extinction curve
        a = torch.zeros_like(x)
        b = torch.zeros_like(x)

        # Optical/NIR region (0.3 - 1.1 μm, x = 0.9 - 3.3)
        optical_mask = (x >= 0.9) & (x <= 3.3)
        y = x[optical_mask] - 1.82

        a[optical_mask] = (
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b[optical_mask] = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )

        # UV region (x > 3.3)
        uv_mask = x > 3.3
        if torch.any(uv_mask):
            x_uv = x[uv_mask]
            a[uv_mask] = 1.752 - 0.316 * x_uv - 0.104 / ((x_uv - 4.67) ** 2 + 0.341)
            b[uv_mask] = -3.090 + 1.825 * x_uv + 1.206 / ((x_uv - 4.62) ** 2 + 0.263)

        # IR region (x < 0.9)
        ir_mask = x < 0.9
        if torch.any(ir_mask):
            x_ir = x[ir_mask]
            a[ir_mask] = 0.574 * x_ir**1.61
            b[ir_mask] = -0.527 * x_ir**1.61

        # Extinction at each wavelength
        extinction = av_total * (a + b / 3.1)  # R_V = 3.1

        # Correct the spectrum
        corrected_flux = self * torch.pow(10, 0.4 * extinction)

        return SpectralTensor(
            corrected_flux,
            wavelengths=wavelengths,
            redshift=getattr(self, "redshift", 0.0),
            flux_units=getattr(self, "flux_units", "erg/s/cm2/A"),
            wavelength_units=getattr(self, "wavelength_units", "Angstrom"),
            spectral_resolution=getattr(self, "spectral_resolution", None),
            instrument=getattr(self, "instrument", None),
        )  # type: ignore[arg-type]

    def stellar_classification(self) -> Dict[str, Union[str, torch.Tensor, Dict]]:
        """
        Classify stellar spectrum and determine stellar parameters.

        Returns:
            Dictionary with stellar classification results
        """
        wavelengths = getattr(self, "wavelengths", torch.linspace(0.3, 30.0, len(self)))

        # Key spectral lines for classification (Angstroms)
        lines = {
            "H_alpha": 6563.0,
            "H_beta": 4861.0,
            "H_gamma": 4340.0,
            "Ca_II_K": 3934.0,
            "Ca_II_H": 3968.0,
            "Mg_I": 5175.0,
            "Na_I_D": 5893.0,
            "Fe_I": 5270.0,
        }

        # Convert wavelengths to Angstroms if needed
        if torch.max(wavelengths) < 100:  # Assume microns
            wavelengths_angstrom = wavelengths * 10000
        else:
            wavelengths_angstrom = wavelengths

        line_strengths = {}

        # Measure line strengths
        for line_name, line_wave in lines.items():
            # Find closest wavelength
            wave_diff = torch.abs(wavelengths_angstrom - line_wave)
            closest_idx = torch.argmin(wave_diff)

            # Simple line strength measurement (continuum - line)
            # In practice, would need proper continuum fitting
            if closest_idx > 5 and closest_idx < len(self) - 5:
                continuum = (self[closest_idx - 5] + self[closest_idx + 5]) / 2
                line_depth = continuum - self[closest_idx]
                line_strength = line_depth / continuum
            else:
                line_strength = torch.tensor(0.0)

            line_strengths[line_name] = line_strength

        # Simple spectral classification based on line ratios
        # This is very simplified - real classification uses many more features

        # Temperature indicators
        h_alpha_strength = line_strengths.get("H_alpha", torch.tensor(0.0))
        ca_ii_strength = (
            line_strengths.get("Ca_II_K", torch.tensor(0.0))
            + line_strengths.get("Ca_II_H", torch.tensor(0.0))
        ) / 2

        # Rough temperature estimate
        if h_alpha_strength > 0.1:
            if ca_ii_strength < 0.05:
                spectral_type = "A"
                temp_estimate = torch.tensor(8000.0)
            else:
                spectral_type = "F"
                temp_estimate = torch.tensor(6500.0)
        elif ca_ii_strength > 0.1:
            spectral_type = "G"
            temp_estimate = torch.tensor(5500.0)
        elif line_strengths.get("Fe_I", torch.tensor(0.0)) > 0.05:
            spectral_type = "K"
            temp_estimate = torch.tensor(4500.0)
        else:
            spectral_type = "M"
            temp_estimate = torch.tensor(3500.0)

        # Luminosity class (very simplified)
        # Real classification would use pressure-sensitive lines
        if h_alpha_strength > 0.2:
            luminosity_class = "V"  # Main sequence
        else:
            luminosity_class = "III"  # Giant

        return {
            "spectral_type": spectral_type,
            "luminosity_class": luminosity_class,
            "temperature_estimate": temp_estimate,
            "line_strengths": line_strengths,
            "full_classification": f"{spectral_type}{luminosity_class}",
        }

    def __repr__(self) -> str:
        """String representation of the tensor."""
        shape_str = f"shape={list(self.shape)}"
        z_str = f"z={self.meta.get('redshift', 0.0):.3f}"
        w_range = ""
        if "wavelengths" in self.meta and len(self.wavelengths) > 0:
            w_min = self.wavelengths.min().item()
            w_max = self.wavelengths.max().item()
            w_range = f"λ={w_min:.1f}-{w_max:.1f} {self.meta.get('wavelength_units', '')}"
        return f"SpectralTensor({shape_str}, {w_range}, {z_str})"

    def get_flux_at_wavelength(self, wavelength: float) -> torch.Tensor:
        """
        Get flux at a specific wavelength (requires interpolation).

        Args:
            wavelength: Wavelength to get flux at

        Returns:
            Flux at the specified wavelength
        """
        if self.flux_units is None or "erg" not in self.flux_units:
            raise ValueError("Flux unit must be specified for luminosity calculation")

        # Simplified luminosity calculation (assumes constant flux over wavelength)
        # Proper calculation requires integration over a bandpass.
        dl = self.meta.get("distance_lum")  # Luminosity distance
        if dl is None:
            raise ValueError("Luminosity distance ('distance_lum') required in metadata")

        # This is a very rough estimate
        return self.data * 4 * torch.pi * (dl**2)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        data_dict = super().to_dict()
        data_dict["flux"] = data_dict.pop("data", None)
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: Dict[str, any]) -> "SpectralTensor":
        """Create from dictionary."""
        flux = data_dict.pop("flux", None)
        return cls(data=flux, **data_dict)

    def normalize(self, wavelength: float) -> Self:
        """Normalizes the spectrum by the flux at a given wavelength."""
        # Find the index closest to the given wavelength
        idx = (torch.abs(self.wavelengths - wavelength)).argmin()
        
        # Get the flux value for normalization (avoiding division by zero)
        norm_flux = self.data[..., idx].unsqueeze(-1)
        safe_norm_flux = torch.where(norm_flux == 0, torch.tensor(1.0), norm_flux)
        
        normalized_data = self.data / safe_norm_flux
        return self._create_new_instance(new_data=normalized_data)

    def resample(self, new_wavelengths: torch.Tensor) -> Self:
        """Resamples the spectrum to a new set of wavelengths using interpolation."""
        original_wavelengths_np = self.wavelengths.cpu().numpy()
        new_wavelengths_np = new_wavelengths.cpu().numpy()
        
        # Ensure data is on CPU for numpy operations
        data_np = self.data.cpu().numpy()

        # Interpolate each spectrum in the batch
        resampled_data_list = []
        for i in range(data_np.shape[0]):
             resampled_data_list.append(
                torch.from_numpy(
                    np.interp(new_wavelengths_np, original_wavelengths_np, data_np[i])
                ).float()
            )
        
        resampled_data = torch.stack(resampled_data_list)
        
        new_meta = self.meta.copy()
        new_meta['wavelengths'] = new_wavelengths
        
        return self.__class__(data=resampled_data, **new_meta)
