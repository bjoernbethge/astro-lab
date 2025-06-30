"""
Spectral TensorDict for AstroLab
================================

TensorDict for spectroscopic data and spectral analysis with proper astropy integration.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from astropy import constants as const
from astropy import units as u
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.modeling.fitting import LevMarLSQFitter

# Spectral analysis tools
from astropy.modeling.models import Polynomial1D

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class SpectralTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    TensorDict for spectroscopic data with proper astronomical functionality.

    Features:
    - Proper wavelength/frequency handling with astropy units
    - Spectral line fitting and analysis
    - Redshift measurements and corrections
    - Continuum normalization and fitting
    - Equivalent width measurements
    - Radial velocity calculations
    - Spectral classification features
    """

    # Common spectral lines for analysis (air wavelengths in Angstroms)
    SPECTRAL_LINES = {
        # Hydrogen Balmer series
        "H_alpha": 6562.797,
        "H_beta": 4861.333,
        "H_gamma": 4340.462,
        "H_delta": 4101.742,
        # Calcium H&K lines
        "Ca_II_H": 3968.468,
        "Ca_II_K": 3933.664,
        # Metal lines
        "Na_D1": 5895.924,
        "Na_D2": 5889.951,
        "Mg_b": 5183.604,
        "Fe_5270": 5270.0,
        "Fe_5335": 5335.0,
        # Oxygen lines
        "O_II_3727": 3727.092,
        "O_III_4959": 4958.911,
        "O_III_5007": 5006.843,
        # Carbon lines
        "C_IV_1549": 1548.195,
        "C_III_1909": 1908.734,
    }

    def __init__(
        self,
        wavelengths: torch.Tensor,
        flux: torch.Tensor,
        flux_error: Optional[torch.Tensor] = None,
        wavelength_unit: Union[str, u.Unit] = u.Unit("AA"),
        flux_unit: Union[str, u.Unit] = u.erg / u.Unit("s") / u.cm**2 / u.Unit("AA"),
        instrument: str = "unknown",
        resolution: Optional[float] = None,
        redshift: Optional[Union[float, torch.Tensor]] = None,
        coordinates: Optional[SkyCoord] = None,
        **kwargs,
    ):
        """
        Initialize SpectralTensorDict with proper astronomical units.

        Args:
            wavelengths: [N, M] Wavelength grid
            flux: [N, M] Flux values
            flux_error: [N, M] Flux uncertainties (optional)
            wavelength_unit: Unit for wavelengths (Angstrom, nm, etc.)
            flux_unit: Unit for flux (erg/s/cm2/A, Jy, etc.)
            instrument: Instrument name/identifier
            resolution: Spectral resolution (R = λ/Δλ)
            redshift: Source redshift(s) for rest-frame corrections
            coordinates: Source coordinates
        """
        if wavelengths.shape != flux.shape:
            raise ValueError(
                f"Wavelengths and flux must have same shape, "
                f"got {wavelengths.shape} and {flux.shape}"
            )
        if flux_error is not None and flux_error.shape != flux.shape:
            raise ValueError(
                f"Flux error must have same shape as flux, got "
                f"{flux_error.shape} and {flux.shape}"
            )

        n_objects, n_wavelengths = flux.shape

        data = {
            "wavelengths": wavelengths,
            "flux": flux,
            "meta": {
                "instrument": instrument,
                "resolution": resolution,
                "n_objects": n_objects,
                "n_wavelengths": n_wavelengths,
                "wavelength_unit": str(wavelength_unit),
                "flux_unit": str(flux_unit),
                "wavelength_range": (
                    wavelengths.min().item(),
                    wavelengths.max().item(),
                ),
            },
        }

        if flux_error is not None:
            data["flux_error"] = flux_error

        if redshift is not None:
            if isinstance(redshift, (int, float)):
                redshift = torch.tensor(redshift, dtype=torch.float32)
            data["redshift"] = redshift

        if coordinates is not None:
            data["coordinates"] = coordinates

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def wavelengths(self) -> torch.Tensor:
        """Wavelength grid."""
        return self["wavelengths"]

    @property
    def flux(self) -> torch.Tensor:
        """Flux values."""
        return self["flux"]

    @property
    def flux_error(self) -> Optional[torch.Tensor]:
        """Flux uncertainties."""
        return self.get("flux_error", None)

    @property
    def instrument(self) -> str:
        """Instrument name."""
        return self._metadata["instrument"]

    @property
    def resolution(self) -> Optional[float]:
        """Spectral resolution."""
        return self._metadata["resolution"]

    @property
    def wavelength_unit(self) -> str:
        """Wavelength unit."""
        return self._metadata["wavelength_unit"]

    @property
    def flux_unit(self) -> str:
        """Flux unit."""
        return self._metadata["flux_unit"]

    def to_rest_frame(
        self, redshift: Optional[Union[float, torch.Tensor]] = None
    ) -> "SpectralTensorDict":
        """
        Convert spectra to rest frame using redshift correction.

        Args:
            redshift: Redshift value(s). If None, use stored redshift.
        """
        if redshift is None:
            if "redshift" not in self:
                raise ValueError("Redshift required for rest-frame conversion")
            z = self["redshift"]
        else:
            if isinstance(redshift, (int, float)):
                z = torch.tensor(redshift, dtype=torch.float32)
            else:
                z = redshift

        # Rest-frame wavelengths: λ_rest = λ_obs / (1 + z)
        if z.dim() == 0:  # Scalar redshift
            rest_wavelengths = self.wavelengths / (1 + z)
        else:  # Per-object redshift
            rest_wavelengths = self.wavelengths / (1 + z.unsqueeze(-1))

        result = SpectralTensorDict(
            rest_wavelengths,
            self.flux,
            self.flux_error,
            wavelength_unit=self.wavelength_unit,
            flux_unit=self.flux_unit,
            instrument=self.instrument,
            resolution=self.resolution,
            coordinates=self.get("coordinates", None),
        )
        result.add_history(
            "to_rest_frame", redshift=z.tolist() if hasattr(z, "tolist") else z
        )
        return result

    def normalize_continuum(
        self,
        method: str = "polynomial",
        order: int = 3,
        exclude_lines: bool = True,
        line_width: float = 10.0,
    ) -> "SpectralTensorDict":
        """
        Normalize spectra by fitting and dividing by continuum.

        Args:
            method: 'polynomial', 'spline', or 'median_filter'
            order: Polynomial order for fitting
            exclude_lines: Exclude known spectral lines from continuum fit
            line_width: Width around lines to exclude (in Angstroms)
        """
        normalized_flux = torch.zeros_like(self.flux)

        for i in range(self.n_objects):
            wavelengths = self.wavelengths[i].detach().cpu().numpy()
            flux = self.flux[i].detach().cpu().numpy()

            # Create mask excluding spectral lines
            mask = np.ones_like(wavelengths, dtype=bool)
            if exclude_lines:
                for line_name, line_wave in self.SPECTRAL_LINES.items():
                    line_mask = np.abs(wavelengths - line_wave) > line_width
                    mask &= line_mask

            # Fit continuum
            if method == "polynomial":
                # Use astropy polynomial fitting
                poly_model = Polynomial1D(degree=order)
                fitter = LevMarLSQFitter()

                # Fit only to continuum regions
                continuum_fit = fitter(poly_model, wavelengths[mask], flux[mask])
                continuum = continuum_fit(wavelengths)

            elif method == "median_filter":
                # median filter for continuum
                from scipy.ndimage import median_filter

                continuum = median_filter(flux, size=51)  # Adjust size as needed
            else:
                # Default: just use median value
                continuum = np.median(flux[mask])

            # Normalize
            normalized_flux[i] = torch.tensor(
                flux / (continuum + 1e-10), dtype=torch.float32
            )

        result = SpectralTensorDict(
            self.wavelengths,
            normalized_flux,
            self.flux_error,
            wavelength_unit=self.wavelength_unit,
            flux_unit="normalized",
            instrument=self.instrument,
            resolution=self.resolution,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history("normalize_continuum", method=method, order=order)
        return result

    def smooth_spectrum(
        self,
        method: str = "gaussian",
        kernel_size: Union[int, float] = 3.0,
        preserve_flux: bool = True,
    ) -> "SpectralTensorDict":
        """
        Smooth spectra using various kernels.

        Args:
            method: 'gaussian', 'boxcar', or 'savgol'
            kernel_size: Kernel size (pixels for boxcar, sigma for gaussian)
            preserve_flux: Whether to preserve total flux
        """
        smoothed_flux = torch.zeros_like(self.flux)

        for i in range(self.n_objects):
            flux = self.flux[i].detach().cpu().numpy()

            if method == "gaussian":
                # Use astropy Gaussian kernel
                kernel = Gaussian1DKernel(stddev=kernel_size)
                smoothed = convolve(flux, kernel, boundary="extend")
            elif method == "boxcar":
                # boxcar filter
                kernel_size_int = int(kernel_size)
                if kernel_size_int % 2 == 0:
                    kernel_size_int += 1
                kernel = np.ones(kernel_size_int) / kernel_size_int
                smoothed = np.convolve(flux, kernel, mode="same")
            elif method == "savgol":
                # Savitzky-Golay filter
                from scipy.signal import savgol_filter

                window_length = int(kernel_size)
                if window_length % 2 == 0:
                    window_length += 1
                smoothed = savgol_filter(flux, window_length, polyorder=3)
            else:
                raise ValueError(f"Unknown smoothing method: {method}")

            # Preserve flux if requested
            if preserve_flux:
                flux_ratio = np.sum(flux) / np.sum(smoothed)
                smoothed *= flux_ratio

            smoothed_flux[i] = torch.tensor(smoothed, dtype=torch.float32)

        result = SpectralTensorDict(
            self.wavelengths,
            smoothed_flux,
            self.flux_error,
            wavelength_unit=self.wavelength_unit,
            flux_unit=self.flux_unit,
            instrument=self.instrument,
            resolution=self.resolution,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history("smooth_spectrum", method=method, kernel_size=kernel_size)
        return result

    def measure_equivalent_widths(
        self,
        lines: Optional[Dict[str, float]] = None,
        continuum_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Measure equivalent widths of spectral lines.

        Args:
            lines: Dictionary of line names and rest wavelengths
            continuum_windows: Windows for continuum estimation

        Returns:
            Dictionary of equivalent widths for each line
        """
        if lines is None:
            lines = {
                "H_alpha": 6562.797,
                "H_beta": 4861.333,
                "Ca_II_K": 3933.664,
                "Mg_b": 5183.604,
            }

        equivalent_widths = {}

        for line_name, line_wave in lines.items():
            ews = torch.zeros(self.n_objects)

            for i in range(self.n_objects):
                wavelengths = self.wavelengths[i].detach().cpu().numpy()
                flux = self.flux[i].detach().cpu().numpy()

                # Find line region
                line_mask = np.abs(wavelengths - line_wave) < 20.0  # ±20 Å around line
                if not np.any(line_mask):
                    continue

                line_wavelengths = wavelengths[line_mask]
                line_flux = flux[line_mask]

                # Estimate continuum (simple linear interpolation from edges)
                if len(line_wavelengths) > 10:
                    edge_flux = np.mean([line_flux[:5].mean(), line_flux[-5:].mean()])
                    continuum = np.full_like(line_flux, edge_flux)

                    # Calculate equivalent width
                    # EW = ∫ (1 - F_line/F_continuum) dλ
                    ew = np.trapz((continuum - line_flux) / continuum, line_wavelengths)
                    ews[i] = ew

            equivalent_widths[line_name] = ews

        return equivalent_widths

    def measure_radial_velocities(
        self,
        template_lines: Optional[Dict[str, float]] = None,
        cross_correlation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Measure radial velocities from spectral line shifts.

        Args:
            template_lines: Reference line wavelengths
            cross_correlation: Use cross-correlation method

        Returns:
            Dictionary with radial velocities and uncertainties
        """
        if template_lines is None:
            template_lines = {"H_alpha": 6562.797, "Ca_II_K": 3933.664}

        results = {
            "radial_velocities": torch.zeros(self.n_objects),
            "rv_uncertainties": torch.zeros(self.n_objects),
            "line_shifts": {},
        }

        c_kms = const.c.to_value("km/s")  # Speed of light in km/s

        for line_name, rest_wave in template_lines.items():
            line_shifts = torch.zeros(self.n_objects)

            for i in range(self.n_objects):
                wavelengths = self.wavelengths[i].detach().cpu().numpy()
                flux = self.flux[i].detach().cpu().numpy()

                # Find line region
                line_mask = np.abs(wavelengths - rest_wave) < 50.0  # ±50 Å
                if not np.any(line_mask):
                    continue

                line_wavelengths = wavelengths[line_mask]
                line_flux = flux[line_mask]

                # Find minimum (absorption line) or maximum (emission line)
                if np.mean(line_flux) < np.median(flux):
                    # Absorption line
                    line_center_idx = np.argmin(line_flux)
                else:
                    # Emission line
                    line_center_idx = np.argmax(line_flux)

                observed_wave = line_wavelengths[line_center_idx]

                # Calculate velocity shift: v = c * (λ_obs - λ_rest) / λ_rest
                velocity_shift = c_kms * (observed_wave - rest_wave) / rest_wave
                line_shifts[i] = velocity_shift

            results["line_shifts"][line_name] = line_shifts

        # Average velocities across lines
        if results["line_shifts"]:
            all_velocities = torch.stack(list(results["line_shifts"].values()))
            results["radial_velocities"] = torch.mean(all_velocities, dim=0)
            results["rv_uncertainties"] = torch.std(all_velocities, dim=0)

        return results

    def extract_spectral_features(self) -> torch.Tensor:
        """
        Extract comprehensive spectral features for classification.

        Returns:
            [N, F] Feature tensor with spectral properties
        """
        features = []

        # Basic statistics
        features.extend(
            [
                torch.mean(self.flux, dim=-1),  # Mean flux
                torch.std(self.flux, dim=-1),  # Flux standard deviation
                torch.median(self.flux, dim=-1)[0],  # Median flux
                torch.min(self.flux, dim=-1)[0],  # Minimum flux
                torch.max(self.flux, dim=-1)[0],  # Maximum flux
            ]
        )

        # Spectral shape features
        flux_range = torch.max(self.flux, dim=-1)[0] - torch.min(self.flux, dim=-1)[0]
        features.append(flux_range)  # Dynamic range

        # Wavelength of peak flux
        peak_indices = torch.argmax(self.flux, dim=-1)
        peak_wavelengths = torch.gather(
            self.wavelengths, 1, peak_indices.unsqueeze(-1)
        ).squeeze(-1)
        features.append(peak_wavelengths)

        # Spectral slope (simple linear fit)
        # Using end points as approximation
        blue_flux = torch.mean(self.flux[:, :10], dim=-1)  # Blue end
        red_flux = torch.mean(self.flux[:, -10:], dim=-1)  # Red end
        blue_wave = torch.mean(self.wavelengths[:, :10], dim=-1)
        red_wave = torch.mean(self.wavelengths[:, -10:], dim=-1)

        spectral_slope = (red_flux - blue_flux) / (red_wave - blue_wave)
        features.append(spectral_slope)

        # Signal-to-noise ratio
        if self.flux_error is not None:
            snr = torch.mean(self.flux / (self.flux_error + 1e-10), dim=-1)
            features.append(snr)
        else:
            # Estimate SNR from flux statistics
            signal = torch.mean(self.flux, dim=-1)
            noise = torch.std(self.flux, dim=-1)
            estimated_snr = signal / (noise + 1e-10)
            features.append(estimated_snr)

        return torch.stack(features, dim=-1)

    def classify_spectra(self) -> Dict[str, torch.Tensor]:
        """
        Basic spectral classification based on features.

        Returns:
            Dictionary with classification results
        """
        features = self.extract_spectral_features()

        # heuristic classification
        peak_wavelength = features[:, 6]  # Peak wavelength feature
        spectral_slope = features[:, 7]

        # Initialize classification arrays
        stellar_class = torch.zeros(self.n_objects, dtype=torch.long)
        galaxy_type = torch.zeros(self.n_objects, dtype=torch.long)

        # Very basic classification rules (would use ML in practice)
        # Based on continuum shape and peak wavelength

        # Blue objects (hot stars, quasars)
        blue_mask = (peak_wavelength < 5000) & (spectral_slope > 0)
        stellar_class[blue_mask] = 1  # Hot stars

        # Red objects (cool stars, red galaxies)
        red_mask = (peak_wavelength > 6000) & (spectral_slope < 0)
        stellar_class[red_mask] = 2  # Cool stars

        # Flat spectrum (galaxies)
        flat_mask = torch.abs(spectral_slope) < 0.1
        galaxy_type[flat_mask] = 1  # Early-type galaxies

        return {
            "stellar_classification": stellar_class,
            "galaxy_type": galaxy_type,
            "features": features,
            "confidence": torch.abs(spectral_slope),  # Use slope as confidence proxy
        }

    def resample_wavelength_grid(
        self, new_wavelengths: torch.Tensor, method: str = "linear"
    ) -> "SpectralTensorDict":
        """
        Resample spectra to new wavelength grid.

        Args:
            new_wavelengths: Target wavelength grid
            method: Interpolation method ('linear', 'cubic')
        """

        resampled_flux = torch.zeros(self.n_objects, len(new_wavelengths))
        resampled_errors = None
        if self.flux_error is not None:
            resampled_errors = torch.zeros(self.n_objects, len(new_wavelengths))

        # Expand new_wavelengths to match batch size
        new_wave_grid = new_wavelengths.unsqueeze(0).expand(self.n_objects, -1)

        for i in range(self.n_objects):
            # Use torch interpolation
            old_flux = self.flux[i]

            # Use torch.nn.functional.interpolate for linear interpolation
            from torch.nn.functional import interpolate

            # Reshape for interpolate (expects [batch, channels, length])
            old_flux_reshaped = old_flux.unsqueeze(0).unsqueeze(0)  # [1, 1, old_length]

            # Interpolate
            resampled_flux_reshaped = interpolate(
                old_flux_reshaped,
                size=len(new_wavelengths),
                mode="linear",
                align_corners=False,
            )
            resampled_flux[i] = resampled_flux_reshaped.squeeze()

            if self.flux_error is not None:
                old_error = self.flux_error[i]
                old_error_reshaped = old_error.unsqueeze(0).unsqueeze(0)
                resampled_error_reshaped = interpolate(
                    old_error_reshaped,
                    size=len(new_wavelengths),
                    mode="linear",
                    align_corners=False,
                )
                resampled_errors[i] = resampled_error_reshaped.squeeze()

        result = SpectralTensorDict(
            new_wave_grid,
            resampled_flux,
            resampled_errors,
            wavelength_unit=self.wavelength_unit,
            flux_unit=self.flux_unit,
            instrument=self.instrument,
            resolution=self.resolution,
            redshift=self.get("redshift", None),
            coordinates=self.get("coordinates", None),
        )
        result.add_history(
            "resample_wavelength_grid", n_new_points=len(new_wavelengths)
        )
        return result

    def validate(self) -> bool:
        """Validate spectral tensor data."""
        # Basic validation from parent
        if not super().validate():
            return False

        # Spectral-specific validation
        return (
            "wavelengths" in self
            and "flux" in self
            and self.wavelengths.shape == self.flux.shape
            and self.wavelengths.shape[-1] > 10  # Minimum spectral points
            and torch.all(torch.isfinite(self.wavelengths))
            and torch.all(torch.isfinite(self.flux))
        )
