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

from .base import AstroTensorBase


class LightcurveTensor(AstroTensorBase):
    """
    Tensor for astronomical time series and lightcurve data.

    Handles time-dependent photometric measurements with physical
    properties like periods, amplitudes, and variability characteristics.
    """

    _metadata_fields = [
        "object_ids",
        "bands",
        "time_unit",
        "magnitude_system",
        "periods",
        "amplitudes",
        "phases",
        "variability_types",
    ]

    def __init__(
        self,
        times: torch.Tensor,
        magnitudes: torch.Tensor,
        errors: Optional[torch.Tensor] = None,
        object_ids: Optional[torch.Tensor] = None,
        bands: Optional[List[str]] = None,
        time_unit: str = "days",
        magnitude_system: str = "AB",
        **kwargs,
    ):
        """
        Initialize lightcurve tensor.

        Args:
            times: Time measurements
            magnitudes: Magnitude/flux measurements
            errors: Measurement uncertainties
            object_ids: Object identifiers
            bands: Band names
            time_unit: Unit of time measurements
            magnitude_system: Magnitude system
        """
        # Store lightcurve-specific metadata
        metadata = {
            "times": times,
            "errors": errors,
            "object_ids": object_ids,
            "bands": bands,
            "time_unit": time_unit,
            "magnitude_system": magnitude_system,
        }
        metadata.update(kwargs)

        super().__init__(magnitudes, **metadata, tensor_type="lightcurve")
        self._validate()  # Call validation after initialization

    def _validate(self) -> None:
        """Validate lightcurve tensor data."""
        times = self._metadata.get("times")
        if times is None:
            raise ValueError("LightcurveTensor requires times")

        if not isinstance(times, torch.Tensor):
            raise ValueError("times must be a torch.Tensor")

        # Check time-magnitude compatibility
        if times.shape[0] != self._data.shape[0]:
            raise ValueError(
                f"times shape {times.shape} incompatible with data shape {self._data.shape}"
            )

        # Validate errors if provided
        errors = self._metadata.get("errors")
        if errors is not None:
            if not isinstance(errors, torch.Tensor):
                raise ValueError("errors must be a torch.Tensor")
            if errors.shape != self._data.shape:
                raise ValueError(
                    f"errors shape {errors.shape} doesn't match data shape {self._data.shape}"
                )

    @property
    def times(self) -> torch.Tensor:
        """Time measurements."""
        return self._metadata["times"]

    @property
    def magnitudes(self) -> torch.Tensor:
        """Magnitude/flux measurements (alias for data)."""
        return self._data

    @property
    def errors(self) -> Optional[torch.Tensor]:
        """Measurement uncertainties."""
        return self._metadata.get("errors")

    @property
    def object_ids(self) -> Optional[torch.Tensor]:
        """Object identifiers."""
        return self._metadata.get("object_ids")

    @property
    def bands(self) -> List[str]:
        """Band names."""
        return self._metadata.get("bands", [])

    @property
    def time_unit(self) -> str:
        """Unit of time measurements."""
        return self._metadata.get("time_unit", "days")

    @property
    def magnitude_system(self) -> str:
        """Magnitude system."""
        return self._metadata.get("magnitude_system", "AB")

    @property
    def periods(self) -> Optional[torch.Tensor]:
        """Periods for each object."""
        return self._metadata.get("periods")

    @property
    def amplitudes(self) -> Optional[torch.Tensor]:
        """Amplitudes for each object."""
        return self._metadata.get("amplitudes")

    @property
    def phases(self) -> Optional[torch.Tensor]:
        """Phases for each object."""
        return self._metadata.get("phases")

    @property
    def variability_types(self) -> Optional[List[str]]:
        """Variability types for each object."""
        return self._metadata.get("variability_types")

    def dim(self) -> int:
        """Number of dimensions."""
        return self._data.dim()

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
        new_metadata = self._metadata.copy()
        new_metadata["times"] = phase[sorted_indices]

        # Sort other arrays by phase
        folded_data = self._data[sorted_indices]

        if self.errors is not None:
            new_metadata["errors"] = self.errors[sorted_indices]

        if self.object_ids is not None:
            new_metadata["object_ids"] = self.object_ids[sorted_indices]

        return LightcurveTensor(
            times=new_metadata["times"],
            magnitudes=folded_data,
            **{k: v for k, v in new_metadata.items() if k not in ["times"]},
        )

    def phase_fold(self, period: float) -> "LightcurveTensor":
        """
        Phase fold the lightcurve with given period (Protocol compatibility).

        Args:
            period: Folding period

        Returns:
            Phase-folded LightcurveTensor
        """
        return self.compute_period_folded(period)

    def compute_statistics(self) -> Dict[str, torch.Tensor]:
        """Compute basic lightcurve statistics."""
        stats = {
            "mean": torch.mean(self._data, dim=0),
            "std": torch.std(self._data, dim=0),
            "median": torch.median(self._data, dim=0).values,
            "min": torch.min(self._data, dim=0).values,
            "max": torch.max(self._data, dim=0).values,
        }

        # Add time-based statistics
        stats["time_span"] = self.times.max() - self.times.min()
        stats["n_points"] = torch.tensor(len(self.times))

        if len(self.times) > 1:
            time_diffs = self.times[1:] - self.times[:-1]
            stats["median_cadence"] = torch.median(time_diffs)
            stats["mean_cadence"] = torch.mean(time_diffs)

        return stats

    def filter_by_band(self, band: str) -> "LightcurveTensor":
        """Filter lightcurve to specific band."""
        bands = self.bands
        if bands is None:
            raise ValueError("No band information available")

        if band not in bands:
            raise ValueError(f"Band '{band}' not found in {bands}")

        band_idx = bands.index(band)

        # Filter data for specific band
        if self._data.dim() == 1:
            # Single-band data
            filtered_data = self._data
        else:
            # Multi-band data
            filtered_data = self._data[..., band_idx]

        # Update metadata
        new_metadata = self._metadata.copy()
        new_metadata["bands"] = [band]

        if self.errors is not None:
            if self.errors.dim() == 1:
                new_metadata["errors"] = self.errors
            else:
                new_metadata["errors"] = self.errors[..., band_idx]

        return LightcurveTensor(
            times=self.times,
            magnitudes=filtered_data,
            **{k: v for k, v in new_metadata.items() if k not in ["times"]},
        )

    def time_bin(self, bin_size: float) -> "LightcurveTensor":
        """
        Bin lightcurve data in time.

        Args:
            bin_size: Size of time bins

        Returns:
            Binned LightcurveTensor
        """
        # Create time bins
        t_min = self.times.min()
        t_max = self.times.max()
        n_bins = int((t_max - t_min) / bin_size) + 1

        bin_centers = []
        binned_data = []
        binned_errors = []

        for i in range(n_bins):
            bin_start = t_min + i * bin_size
            bin_end = bin_start + bin_size

            mask = (self.times >= bin_start) & (self.times < bin_end)

            if mask.sum() > 0:
                bin_centers.append(bin_start + bin_size / 2)
                binned_data.append(torch.mean(self._data[mask], dim=0))

                if self.errors is not None:
                    # Combine errors in quadrature
                    n_points = mask.sum()
                    binned_errors.append(
                        torch.sqrt(torch.sum(self.errors[mask] ** 2, dim=0)) / n_points
                    )

        if not bin_centers:
            raise ValueError("No data points to bin")

        new_times = torch.stack(bin_centers)
        new_data = torch.stack(binned_data)

        new_metadata = self._metadata.copy()
        new_metadata["times"] = new_times

        if self.errors is not None and binned_errors:
            new_metadata["errors"] = torch.stack(binned_errors)
        else:
            new_metadata["errors"] = None

        return LightcurveTensor(
            times=new_times,
            magnitudes=new_data,
            **{k: v for k, v in new_metadata.items() if k not in ["times"]},
        )

    def get_period(self, object_idx: int = 0) -> Optional[float]:
        """Get period for specific object."""
        if self.periods is not None and len(self.periods) > object_idx:
            return float(self.periods[object_idx])
        return None

    def get_amplitude(self, object_idx: int = 0) -> Optional[float]:
        """Get amplitude for specific object."""
        if self.amplitudes is not None and len(self.amplitudes) > object_idx:
            return float(self.amplitudes[object_idx])
        return None

    def compute_variability_stats(self) -> Dict[str, torch.Tensor]:
        """
        Compute variability statistics for each lightcurve.

        Returns:
            Dictionary with variability metrics
        """
        mags = self.magnitudes

        # Basic statistics
        mean_mag = torch.mean(mags, dim=-1)
        std_mag = torch.std(mags, dim=-1)
        min_mag = torch.min(mags, dim=-1)[0]
        max_mag = torch.max(mags, dim=-1)[0]

        # Variability indices
        amplitude = max_mag - min_mag
        variability_index = std_mag / mean_mag

        return {
            "mean_magnitude": mean_mag,
            "std_magnitude": std_mag,
            "min_magnitude": min_mag,
            "max_magnitude": max_mag,
            "amplitude": amplitude,
            "variability_index": variability_index,
        }

    def fold_lightcurve(
        self, period: Optional[float] = None, object_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fold lightcurve with given period.

        Args:
            period: Period for folding (uses stored period if None)
            object_idx: Object index if multiple objects

        Returns:
            Tuple of (folded_times, magnitudes)
        """
        if period is None:
            period = self.get_period(object_idx)
            if period is None:
                raise ValueError("No period provided or stored")

        times = self.times
        mags = self.magnitudes

        # Handle multi-object case
        if times.dim() > 1:
            times = times[object_idx]
            mags = mags[object_idx]

        # Fold the lightcurve
        folded_times = (times % period) / period

        return folded_times, mags

    def detect_periods(
        self, min_period: float = 0.1, max_period: float = 100.0
    ) -> torch.Tensor:
        """
        Detect periods using Lomb-Scargle periodogram.

        Args:
            min_period: Minimum period to search
            max_period: Maximum period to search

        Returns:
            Detected periods for each object
        """
        # Simplified period detection (can be enhanced with scipy.signal)
        times = self.times
        mags = self.magnitudes

        # For now, return dummy periods
        # In practice, would use Lomb-Scargle or other methods
        n_objects = times.shape[0] if times.dim() > 1 else 1
        dummy_periods = torch.rand(n_objects) * (max_period - min_period) + min_period

        return dummy_periods

    def classify_variability(self) -> List[str]:
        """
        Classify variability type based on lightcurve properties.

        Returns:
            List of variability classifications
        """
        stats = self.compute_variability_stats()
        classifications = []

        # Simple classification based on amplitude and period
        amplitudes = stats["amplitude"]

        for i, amp in enumerate(amplitudes):
            period = self.get_period(i) if self.periods is not None else None

            if amp > 1.0:  # Large amplitude
                if period is not None and 0.2 < period < 1.0:
                    classifications.append("RR_Lyrae")
                elif period is not None and 1.0 < period < 100.0:
                    classifications.append("Cepheid")
                else:
                    classifications.append("Large_Amplitude_Variable")
            elif amp > 0.1:  # Medium amplitude
                classifications.append("Small_Amplitude_Variable")
            else:
                classifications.append("Stable")

        return classifications

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data_dict = {
            "times": self.times,
            "magnitudes": self.magnitudes,
            "time_unit": self.time_unit,
            "magnitude_system": self.magnitude_system,
            "bands": self.bands,
        }

        if self.errors is not None:
            data_dict["errors"] = self.errors

        if self.periods is not None:
            data_dict["periods"] = self.periods

        if self.amplitudes is not None:
            data_dict["amplitudes"] = self.amplitudes

        if self.phases is not None:
            data_dict["phases"] = self.phases

        if self.variability_types is not None:
            data_dict["variability_types"] = self.variability_types

        return data_dict

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
