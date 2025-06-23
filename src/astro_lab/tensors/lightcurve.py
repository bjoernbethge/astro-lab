"""
Lightcurve Tensor for Time Series Astronomical Data
==================================================

Specialized tensor for handling astronomical time series data like:
- Variable star lightcurves
- Asteroid rotation lightcurves
- Supernova lightcurves
- Exoplanet transit lightcurves
- AGN variability
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import Field, field_validator
from typing_extensions import Self

from .base import AstroTensorBase

class LightcurveTensor(AstroTensorBase):
    """
    Represents time-series data for celestial objects, such as light curves.

    The `data` tensor is expected to have a shape of [N, D], where N is the
    number of observations and D >= 2. The columns are expected to be:
    - 0: Time
    - 1: Magnitude or Flux
    - 2: Error (optional)
    - 3+: Additional features (e.g., in different bands)
    """

    data: torch.Tensor
    bands: List[str] = Field(default_factory=list, description="Names of the bands or features.")
    time_format: str = Field("jd", description="Format of the time column (e.g., 'jd', 'mjd').")

    @field_validator("data")
    def validate_lightcurve_data(cls, v):
        if v.ndim != 2 or v.shape[1] < 2:
            raise ValueError(
                "Lightcurve data must be a 2D tensor with at least 2 columns (time, mag), "
                f"but got shape {v.shape}"
            )
        # Ensure time is monotonically increasing
        if not torch.all(v[1:, 0] >= v[:-1, 0]):
            raise ValueError("Time values must be monotonically increasing.")
        return v

    @property
    def times(self) -> torch.Tensor:
        """The time values of the light curve."""
        return self.data[:, 0]

    @property
    def magnitudes(self) -> torch.Tensor:
        """The magnitude or flux values of the light curve."""
        return self.data[:, 1]

    @property
    def errors(self) -> Optional[torch.Tensor]:
        """The errors, if they exist (3rd column)."""
        if self.data.shape[1] > 2:
            return self.data[:, 2]
        return None

    @property
    def sequence_length(self) -> int:
        """Number of points in the light curve."""
        return self.data.shape[0]

    def get_time_range(self) -> Tuple[float, float]:
        """Returns the minimum and maximum time."""
        return self.times.min().item(), self.times.max().item()

    def normalize_flux(self, method: str = "median") -> Self:
        """
        Normalizes the flux/magnitude of the light curve.
        This operation is performed on the magnitude column (index 1).
        """
        new_data = self.data.clone()
        magnitudes = new_data[:, 1]

        if method == "median":
            median_val = torch.median(magnitudes)
            new_data[:, 1] = magnitudes / median_val
        elif method == "z_score":
            mean = torch.mean(magnitudes)
            std = torch.std(magnitudes)
            new_data[:, 1] = (magnitudes - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Normalization method '{method}' not supported.")
        
        return self._create_new_instance(new_data=new_data).add_history_entry(
            "normalize_flux", method=method
        )

    def phase_fold(self, period: float, epoch: float = 0.0) -> "LightcurveTensor":
        """
        Computes a phase-folded light curve.
        The phase becomes the new time-like column.
        """
        phase = ((self.times - epoch) % period) / period
        
        new_data = self.data.clone()
        new_data[:, 0] = phase # Replace time with phase

        # Sort by phase
        sorted_indices = torch.argsort(phase)
        sorted_data = new_data[sorted_indices]

        return self._create_new_instance(
            new_data=sorted_data,
            time_format="phase"
        ).add_history_entry("phase_fold", period=period, epoch=epoch)

    def compute_statistics(self) -> Dict[str, float]:
        """Compute basic lightcurve statistics."""
        mags = self.magnitudes
        stats = {
            "mean": torch.mean(mags).item(),
            "std": torch.std(mags).item(),
            "median": torch.median(mags).item(),
            "min": torch.min(mags).item(),
            "max": torch.max(mags).item(),
            "time_span": self.time_span,
            "n_points": float(len(self.times)),
        }

        if len(self.times) > 1:
            time_diffs = self.times[1:] - self.times[:-1]
            stats["median_cadence"] = torch.median(time_diffs).item()
            stats["mean_cadence"] = torch.mean(time_diffs).item()

        return stats

    def time_bin(self, bin_size: Optional[float] = None, n_bins: int = 50) -> Self:
        """
        Bins the light curve in time, averaging points within each bin.
        """
        if bin_size is None:
            min_time, max_time = self.get_time_range()
            bin_size = (max_time - min_time) / n_bins

        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        binned_data = []
        min_time, max_time = self.get_time_range()
        
        for i in range(n_bins):
            bin_start = min_time + i * bin_size
            bin_end = bin_start + bin_size
            
            mask = (self.times >= bin_start) & (self.times < bin_end)
            if torch.any(mask):
                # Average all columns within the bin
                bin_mean = self.data[mask].mean(dim=0)
                # Set the bin's time to the center of the bin
                bin_mean[0] = bin_start + bin_size / 2.0
                binned_data.append(bin_mean)
        
        if not binned_data:
            return self._create_new_instance(new_data=torch.empty(0, self.data.shape[1]))

        return self._create_new_instance(new_data=torch.stack(binned_data)).add_history_entry(
            "time_bin", bin_size=bin_size, n_bins=n_bins
        )

    @property
    def object_ids(self) -> Optional[List[Any]]:
        """Object identifiers."""
        return self.meta.get("object_ids")

    @property
    def time_unit(self) -> str:
        """Unit of time measurements."""
        return self.meta.get("time_format", "jd")

    @property
    def magnitude_system(self) -> str:
        """Magnitude system."""
        return self.meta.get("magnitude_system", "AB")

    @property
    def periods(self) -> Optional[torch.Tensor]:
        """Periods for each object."""
        return self.meta.get("periods")

    @property
    def amplitudes(self) -> Optional[torch.Tensor]:
        """Amplitudes for each object."""
        return self.meta.get("amplitudes")

    @property
    def phases(self) -> Optional[torch.Tensor]:
        """Phases for each object."""
        return self.meta.get("phases")

    @property
    def variability_types(self) -> Optional[List[str]]:
        """Variability types for each object."""
        return self.meta.get("variability_types")

    def dim(self) -> int:
        """Number of dimensions."""
        return self.data.dim()

    def compute_period_folded(
        self, period: float, epoch: float = 0.0
    ) -> "LightcurveTensor":
        """
        Compute period-folded lightcurve.

        Args:
            period: Folding period
            epoch: Reference epoch

        Returns:
            Period-folded LightcurveTensor
        """
        phase = ((self.times - epoch) % period) / period

        # Sort by phase
        sorted_indices = torch.argsort(phase)

        # Create new metadata
        new_metadata = self.meta.copy()
        new_metadata["times"] = phase[sorted_indices]

        # Sort other arrays by phase
        folded_data = self.data[sorted_indices]

        if self.errors is not None:
            new_metadata["flux_errors"] = self.errors[sorted_indices]

        if self.object_ids is not None:
            new_metadata["object_ids"] = self.object_ids[sorted_indices]

        return LightcurveTensor(
            data=folded_data,
            times=new_metadata["times"],
            flux_errors=new_metadata["flux_errors"],
            object_ids=new_metadata["object_ids"],
            **{k: v for k, v in new_metadata.items() if k not in ["times", "flux_errors", "object_ids"]},
        )

    def phase_lightcurve(
        self, period: Optional[float] = None, epoch: float = 0.0
    ) -> torch.Tensor:
        """
        Calculate phase for each observation.

        Args:
            period: Period for phasing
            epoch: Reference epoch

        Returns:
            Phase values (0-1)
        """
        if period is None and self.periods is not None:
            period = float(self.periods[0])
        if period is None:
            raise ValueError("No period provided")

        times = self.times
        phases = ((times - epoch) % period) / period

        return phases

    def bin_lightcurve(
        self, n_bins: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bin lightcurve in phase.

        Args:
            n_bins: Number of phase bins

        Returns:
            Tuple of (bin_centers, binned_mags, bin_errors)
        """
        phases = self.phase_lightcurve()
        mags = self.magnitudes

        # Create phase bins
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Bin the data
        binned_mags = torch.zeros(n_bins)
        bin_counts = torch.zeros(n_bins)

        for i in range(n_bins):
            mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
            if mask.sum() > 0:
                binned_mags[i] = mags[mask].mean()
                bin_counts[i] = mask.sum()

        # Calculate bin errors
        bin_errors = torch.sqrt(1.0 / torch.clamp(bin_counts, min=1.0))

        return bin_centers, binned_mags, bin_errors

    @property
    def period(self) -> float:
        """Estimated period of the lightcurve in the same units as time."""
        if hasattr(self, '_period') and self._period is not None:
            return self._period
        
        # Simple period estimation using peak finding in frequency domain
        time_col = 0  # Assuming first column is time
        if self.data.shape[1] > 1:
            times = self.data[:, time_col].cpu().numpy()
            magnitudes = self.data[:, 1].cpu().numpy()  # Assuming second column is magnitude
            
            # Remove mean and compute FFT
            magnitudes = magnitudes - np.mean(magnitudes)
            if len(times) > 10:
                try:
                    # Simple frequency analysis
                    freqs = np.fft.fftfreq(len(times), d=np.median(np.diff(times)))
                    fft = np.abs(np.fft.fft(magnitudes))
                    
                    # Find peak frequency (excluding DC component)
                    peak_idx = np.argmax(fft[1:len(fft)//2]) + 1
                    if freqs[peak_idx] > 0:
                        return 1.0 / freqs[peak_idx]
                except:
                    pass
        
        # Fallback: return time span as rough period estimate
        if self.data.shape[0] > 1:
            return float(self.data[-1, 0] - self.data[0, 0])
        return 1.0

    @property
    def amplitude(self) -> float:
        """Amplitude of the lightcurve (peak-to-peak magnitude difference)."""
        if self.data.shape[1] > 1:
            magnitudes = self.data[:, 1]  # Assuming second column is magnitude
            return float(torch.max(magnitudes) - torch.min(magnitudes))
        return 0.0

    @property 
    def n_observations(self) -> int:
        """Number of observations in the lightcurve."""
        return self.data.shape[0]

    @property
    def time_span(self) -> float:
        """Return the time span of the lightcurve."""
        if self.data.shape[0] > 1:
            time_col = 0  # Assuming first column is time
            times = self.data[:, time_col]
            return float(torch.max(times) - torch.min(times))
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary for serialization."""
        return {
            "data": self.data.cpu().numpy().tolist(),
            "meta": self.meta,
            "tensor_type": "lightcurve",
            "period": self.period,
            "amplitude": self.amplitude,
            "n_observations": self.n_observations
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "LightcurveTensor":
        """Create from dictionary."""
        data = data_dict.pop("magnitudes", None) or data_dict.pop("data", None)
        return cls(magnitudes=data, **data_dict)

    def get_lightcurve_for_object(self, object_idx: int) -> "LightcurveTensor":
        """
        Get lightcurve for a specific object.

        Args:
            object_idx: Index of the object

        Returns:
            LightcurveTensor for the specified object
        """
        object_data = self._data[object_idx].unsqueeze(0)  # Keep dimensions

        new_metadata = self.meta.copy()
        if self.object_ids:
            new_metadata["object_ids"] = [self.object_ids[object_idx]]

        return LightcurveTensor(
            magnitudes=object_data,
            **new_metadata,
        )

    def sort_by_time(self) -> "LightcurveTensor":
        """
        Sort lightcurve data by time.

        Returns:
            Sorted LightcurveTensor
        """
        if "times" not in self.meta:
            raise ValueError("Cannot sort without 'times' in metadata")

        sorted_indices = torch.argsort(self.times)
        sorted_data = self._data[:, sorted_indices, :]
        sorted_times = self.times[sorted_indices]

        new_metadata = self.meta.copy()
        new_metadata["times"] = sorted_times

        return LightcurveTensor(magnitudes=sorted_data, **new_metadata)

    def filter_by_time(self, start_time: float, end_time: float) -> "LightcurveTensor":
        """
        Filter lightcurve data by time.

        Args:
            start_time: Start time for filtering
            end_time: End time for filtering

        Returns:
            Filtered LightcurveTensor
        """
        if "times" not in self.meta:
            raise ValueError("Cannot filter by time without 'times' in metadata")

        time_mask = (self.times >= start_time) & (self.times <= end_time)
        filtered_data = self._data[:, time_mask, :]
        filtered_times = self.times[time_mask]

        new_metadata = self.meta.copy()
        new_metadata["times"] = filtered_times

        return LightcurveTensor(magnitudes=filtered_data, **new_metadata)

    def normalize(self, method: str = "min-max") -> "LightcurveTensor":
        """
        Normalize lightcurve data.

        Args:
            method: Normalization method

        Returns:
            Normalized LightcurveTensor
        """
        if method == "min-max":
            min_val = self._data.min(dim=1, keepdim=True)[0]
            max_val = self._data.max(dim=1, keepdim=True)[0]
            normalized_data = (self._data - min_val) / (max_val - min_val + 1e-8)
        elif method == "z-score":
            mean = self._data.mean(dim=1, keepdim=True)
            std = self._data.std(dim=1, keepdim=True)
            normalized_data = (self._data - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return LightcurveTensor(magnitudes=normalized_data, **self.meta)

    def __repr__(self) -> str:
        """String representation."""
        n_points = self.shape[0] if self.dim() >= 1 else 1
        n_bands = len(self.bands) if self.bands else 1
        time_range = f"{float(self.times.min()):.2f}-{float(self.times.max()):.2f}"

        return (
            f"LightcurveTensor(n_points={n_points}, bands={n_bands}, "
            f"time_range={time_range} {self.time_unit}, "
            f"system={self.magnitude_system})"
        )

    @classmethod  # type: ignore
    def from_survey_data(
        cls,
        survey_data: Dict[str, torch.Tensor],
        time_col: str = "time",
        mag_col: str = "magnitude",
        error_col: Optional[str] = None,
        object_id_col: Optional[str] = None,
        **kwargs,
    ) -> "LightcurveTensor":
        """
        Create LightcurveTensor from survey data dictionary.

        Args:
            survey_data: Dictionary with survey measurements
            time_col: Name of time column
            mag_col: Name of magnitude column
            error_col: Name of error column
            object_id_col: Name of object ID column
            **kwargs: Additional arguments

        Returns:
            LightcurveTensor instance
        """
        times = survey_data[time_col]
        magnitudes = survey_data[mag_col]

        errors = survey_data.get(error_col) if error_col else None
        object_ids = survey_data.get(object_id_col) if object_id_col else None

        kwargs.update(
            {
                "errors": errors,
                "object_ids": object_ids,
            }
        )
        return cls(times, magnitudes, **kwargs)  # type: ignore[misc]
