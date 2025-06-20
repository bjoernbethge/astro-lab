"""
Statistics Tensor for Astronomical Statistics
============================================

Specialized tensor for astronomical statistical operations including
luminosity functions, correlation functions, and error estimation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import AstroTensorBase

# Optional dependencies
try:
    import scipy.integrate as integrate
    import scipy.interpolate as interpolate
    import scipy.stats as stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.utils import resample

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class StatisticsTensor(AstroTensorBase):
    """
    Tensor for astronomical statistical operations.

    Provides specialized statistical methods for:
    - Luminosity and mass functions
    - Color-magnitude diagrams
    - Two-point correlation functions
    - Bootstrap and jackknife error estimation
    - Completeness and selection function analysis
    - Astronomical hypothesis testing
    """

    _metadata_fields = [
        "statistical_methods",
        "computed_functions",
        "error_estimates",
        "bootstrap_samples",
        "correlation_functions",
        "selection_functions",
        "completeness_maps",
        "statistical_tests",
    ]

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        coordinates: Optional[Union[torch.Tensor, np.ndarray]] = None,
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        """
        Initialize statistics tensor.

        Args:
            data: Observable data [N, D] (magnitudes, colors, masses, etc.)
            coordinates: Spatial coordinates [N, 2] or [N, 3]
            weights: Statistical weights [N]
        """
        # Convert to tensor
        data_tensor = torch.as_tensor(data, dtype=torch.float32)
        if data_tensor.dim() == 1:
            data_tensor = data_tensor.unsqueeze(1)

        # Store coordinates and weights
        if coordinates is not None:
            coord_tensor = torch.as_tensor(coordinates, dtype=torch.float32)
            if coord_tensor.shape[0] != data_tensor.shape[0]:
                raise ValueError(
                    "Coordinates and data must have same number of objects"
                )
        else:
            coord_tensor = None

        if weights is not None:
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
            if weight_tensor.shape[0] != data_tensor.shape[0]:
                raise ValueError("Weights and data must have same number of objects")
        else:
            weight_tensor = torch.ones(data_tensor.shape[0])

        # Initialize metadata
        metadata = {
            "coordinates": coord_tensor,
            "weights": weight_tensor,
            "statistical_methods": {},
            "computed_functions": {},
            "error_estimates": {},
            "bootstrap_samples": {},
            "correlation_functions": {},
            "selection_functions": {},
            "completeness_maps": {},
            "statistical_tests": {},
            "tensor_type": "statistics",
        }
        metadata.update(kwargs)

        super().__init__(data_tensor, **metadata)

    @property
    def coordinates(self) -> Optional[torch.Tensor]:
        """Get spatial coordinates."""
        return self.get_metadata("coordinates")

    @property
    def weights(self) -> torch.Tensor:
        """Get statistical weights."""
        return self.get_metadata("weights", torch.ones(self.shape[0]))

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return self._data.shape[0]

    @property
    def n_observables(self) -> int:
        """Number of observables."""
        return self._data.shape[1]

    def luminosity_function(
        self,
        magnitude_column: int = 0,
        bins: Union[int, torch.Tensor] = 20,
        magnitude_range: Optional[Tuple[float, float]] = None,
        method: str = "1/Vmax",
        function_name: str = "luminosity_function",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute luminosity function.

        Args:
            magnitude_column: Column index for magnitudes
            bins: Number of bins or bin edges
            magnitude_range: Magnitude range (auto if None)
            method: Method ('1/Vmax', 'histogram', 'kde')
            function_name: Name for storing results

        Returns:
            Tuple of (bin_centers, phi) where phi is number density
        """
        magnitudes = self._data[:, magnitude_column]
        weights = self.weights

        # Handle magnitude range
        if magnitude_range is None:
            valid_mask = torch.isfinite(magnitudes)
            mag_min = magnitudes[valid_mask].min().item()
            mag_max = magnitudes[valid_mask].max().item()
            magnitude_range = (mag_min, mag_max)

        # Create bins
        if isinstance(bins, int):
            bin_edges = torch.linspace(magnitude_range[0], magnitude_range[1], bins + 1)
        else:
            bin_edges = torch.as_tensor(bins, dtype=torch.float32)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Compute histogram
        hist = torch.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            mask = (magnitudes >= bin_edges[i]) & (magnitudes < bin_edges[i + 1])
            if method == "1/Vmax":
                # Simple histogram for now, real 1/Vmax needs volume information
                hist[i] = weights[mask].sum()
            else:
                hist[i] = weights[mask].sum()

        # Convert to number density
        phi = hist / bin_widths

        # Store results
        self._store_function(
            function_name,
            {
                "bin_centers": bin_centers,
                "phi": phi,
                "bin_edges": bin_edges,
                "method": method,
                "magnitude_range": magnitude_range,
            },
        )

        return bin_centers, phi

    def color_magnitude_diagram(
        self,
        magnitude_column: int = 0,
        color_column: int = 1,
        bins: Union[int, Tuple[int, int]] = (50, 50),
        ranges: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        function_name: str = "cmd",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create color-magnitude diagram.

        Args:
            magnitude_column: Column for magnitude
            color_column: Column for color
            bins: Number of bins (mag_bins, color_bins)
            ranges: ((mag_min, mag_max), (color_min, color_max))
            function_name: Name for storing results

        Returns:
            Tuple of (mag_centers, color_centers, density)
        """
        magnitudes = self._data[:, magnitude_column]
        colors = self._data[:, color_column]
        weights = self.weights

        # Handle ranges
        if ranges is None:
            valid_mask = torch.isfinite(magnitudes) & torch.isfinite(colors)
            mag_range = (
                magnitudes[valid_mask].min().item(),
                magnitudes[valid_mask].max().item(),
            )
            color_range = (
                colors[valid_mask].min().item(),
                colors[valid_mask].max().item(),
            )
            ranges = (mag_range, color_range)

        # Create bins
        if isinstance(bins, int):
            mag_bins = color_bins = bins
        else:
            mag_bins, color_bins = bins

        mag_edges = torch.linspace(ranges[0][0], ranges[0][1], mag_bins + 1)
        color_edges = torch.linspace(ranges[1][0], ranges[1][1], color_bins + 1)

        mag_centers = (mag_edges[:-1] + mag_edges[1:]) / 2
        color_centers = (color_edges[:-1] + color_edges[1:]) / 2

        # Compute 2D histogram
        density = torch.zeros(mag_bins, color_bins)
        for i in range(mag_bins):
            for j in range(color_bins):
                mask = (
                    (magnitudes >= mag_edges[i])
                    & (magnitudes < mag_edges[i + 1])
                    & (colors >= color_edges[j])
                    & (colors < color_edges[j + 1])
                )
                density[i, j] = weights[mask].sum()

        # Store results
        self._store_function(
            function_name,
            {
                "mag_centers": mag_centers,
                "color_centers": color_centers,
                "density": density,
                "mag_edges": mag_edges,
                "color_edges": color_edges,
                "ranges": ranges,
            },
        )

        return mag_centers, color_centers, density

    def two_point_correlation(
        self,
        r_bins: Union[int, torch.Tensor] = 20,
        r_range: Tuple[float, float] = (0.1, 100.0),
        estimator: str = "landy_szalay",
        function_name: str = "xi_r",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute two-point correlation function.

        Args:
            r_bins: Number of radial bins or bin edges
            r_range: Radial range
            estimator: Estimator type ('landy_szalay', 'davis_peebles')
            function_name: Name for storing results

        Returns:
            Tuple of (r_centers, xi_r)
        """
        if self.coordinates is None:
            raise ValueError("Coordinates required for correlation function")

        coordinates = self.coordinates
        weights = self.weights

        # Create radial bins
        if isinstance(r_bins, int):
            r_edges = torch.logspace(
                np.log10(r_range[0]), np.log10(r_range[1]), r_bins + 1
            )
        else:
            r_edges = torch.as_tensor(r_bins, dtype=torch.float32)

        r_centers = torch.sqrt(r_edges[:-1] * r_edges[1:])  # Geometric mean

        # Compute pair counts
        dd_counts = self._count_pairs(
            coordinates, coordinates, r_edges, weights, weights
        )

        if estimator == "landy_szalay":
            # Generate random catalog (simplified)
            n_random = len(coordinates) * 10
            random_coords = self._generate_random_catalog(coordinates, n_random)
            random_weights = torch.ones(n_random)

            dr_counts = self._count_pairs(
                coordinates, random_coords, r_edges, weights, random_weights
            )
            rr_counts = self._count_pairs(
                random_coords, random_coords, r_edges, random_weights, random_weights
            )

            # Landy-Szalay estimator
            n_data = len(coordinates)
            n_rand = n_random
            dd_norm = dd_counts / (n_data * (n_data - 1) / 2)
            dr_norm = dr_counts / (n_data * n_rand)
            rr_norm = rr_counts / (n_rand * (n_rand - 1) / 2)

            xi_r = (dd_norm - 2 * dr_norm + rr_norm) / rr_norm

        else:  # Davis-Peebles estimator
            # Simplified version
            xi_r = dd_counts / dd_counts.mean() - 1

        # Store results
        self._store_function(
            function_name,
            {
                "r_centers": r_centers,
                "xi_r": xi_r,
                "r_edges": r_edges,
                "estimator": estimator,
                "dd_counts": dd_counts,
            },
        )

        return r_centers, xi_r

    def bootstrap_errors(
        self, function_name: str, n_bootstrap: int = 100, confidence_level: float = 0.68
    ) -> Dict[str, torch.Tensor]:
        """
        Compute bootstrap error estimates for a function.

        Args:
            function_name: Name of function to bootstrap
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with error estimates
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for bootstrap resampling")

        if function_name not in self.get_metadata("computed_functions", {}):
            raise ValueError(f"Function {function_name} not computed yet")

        # Get original function
        original_func = self.get_metadata("computed_functions")[function_name]

        # Perform bootstrap resampling
        bootstrap_results = []
        n_objects = self.n_objects

        for i in range(n_bootstrap):
            # Resample indices
            indices = torch.from_numpy(
                resample(np.arange(n_objects), n_samples=n_objects)
            )

            # Create bootstrap sample
            bootstrap_data = self._data[indices]
            bootstrap_weights = (
                self.weights[indices] if self.weights is not None else None
            )
            bootstrap_coords = (
                self.coordinates[indices] if self.coordinates is not None else None
            )

            # Create temporary tensor for bootstrap
            bootstrap_tensor = StatisticsTensor(
                bootstrap_data, coordinates=bootstrap_coords, weights=bootstrap_weights
            )

            # Recompute function
            if function_name == "luminosity_function":
                _, phi = bootstrap_tensor.luminosity_function()
                bootstrap_results.append(phi)
            elif function_name == "cmd":
                _, _, density = bootstrap_tensor.color_magnitude_diagram()
                bootstrap_results.append(density.flatten())
            elif function_name == "xi_r":
                _, xi_r = bootstrap_tensor.two_point_correlation()
                bootstrap_results.append(xi_r)

        # Stack results
        bootstrap_stack = torch.stack(bootstrap_results)

        # Compute statistics
        mean_values = bootstrap_stack.mean(dim=0)
        std_errors = bootstrap_stack.std(dim=0)

        # Confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_ci = torch.quantile(bootstrap_stack, lower_percentile / 100, dim=0)
        upper_ci = torch.quantile(bootstrap_stack, upper_percentile / 100, dim=0)

        error_results = {
            "mean": mean_values,
            "std_error": std_errors,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "bootstrap_samples": bootstrap_stack,
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
        }

        # Store results
        error_estimates = self.get_metadata("error_estimates", {})
        error_estimates[function_name] = error_results
        self.update_metadata(error_estimates=error_estimates)

        return error_results

    def jackknife_errors(
        self, function_name: str, n_jackknife: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute jackknife error estimates.

        Args:
            function_name: Name of function to jackknife
            n_jackknife: Number of jackknife samples (default: sqrt(N))

        Returns:
            Dictionary with error estimates
        """
        if function_name not in self.get_metadata("computed_functions", {}):
            raise ValueError(f"Function {function_name} not computed yet")

        n_objects = self.n_objects
        if n_jackknife is None:
            n_jackknife = int(np.sqrt(n_objects))

        # Divide data into jackknife regions
        indices = torch.randperm(n_objects)
        region_size = n_objects // n_jackknife

        jackknife_results = []

        for i in range(n_jackknife):
            # Remove one region
            start_idx = i * region_size
            end_idx = min((i + 1) * region_size, n_objects)

            mask = torch.ones(n_objects, dtype=torch.bool)
            mask[indices[start_idx:end_idx]] = False

            # Create jackknife sample
            jk_data = self._data[mask]
            jk_weights = self.weights[mask] if self.weights is not None else None
            jk_coords = self.coordinates[mask] if self.coordinates is not None else None

            # Create temporary tensor
            jk_tensor = StatisticsTensor(
                jk_data, coordinates=jk_coords, weights=jk_weights
            )

            # Recompute function
            if function_name == "luminosity_function":
                _, phi = jk_tensor.luminosity_function()
                jackknife_results.append(phi)
            elif function_name == "cmd":
                _, _, density = jk_tensor.color_magnitude_diagram()
                jackknife_results.append(density.flatten())
            elif function_name == "xi_r":
                _, xi_r = jk_tensor.two_point_correlation()
                jackknife_results.append(xi_r)

        # Stack results
        jackknife_stack = torch.stack(jackknife_results)

        # Compute jackknife error
        mean_values = jackknife_stack.mean(dim=0)
        jk_error = torch.sqrt(
            (n_jackknife - 1) * ((jackknife_stack - mean_values) ** 2).mean(dim=0)
        )

        error_results = {
            "mean": mean_values,
            "jackknife_error": jk_error,
            "jackknife_samples": jackknife_stack,
            "n_jackknife": n_jackknife,
        }

        # Store results
        error_estimates = self.get_metadata("error_estimates", {})
        error_estimates[f"{function_name}_jackknife"] = error_results
        self.update_metadata(error_estimates=error_estimates)

        return error_results

    def completeness_analysis(
        self,
        magnitude_column: int = 0,
        detection_limit: float = 25.0,
        function_name: str = "completeness",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze survey completeness as function of magnitude.

        Args:
            magnitude_column: Column for magnitude
            detection_limit: Nominal detection limit
            function_name: Name for storing results

        Returns:
            Tuple of (magnitudes, completeness_fraction)
        """
        magnitudes = self._data[:, magnitude_column]
        weights = self.weights

        # Create magnitude bins
        mag_min = magnitudes.min().item()
        mag_max = min(magnitudes.max().item(), detection_limit + 2.0)
        mag_bins = torch.linspace(mag_min, mag_max, 50)

        completeness = torch.zeros(len(mag_bins) - 1)

        for i in range(len(completeness)):
            # Objects in this magnitude bin
            mask = (magnitudes >= mag_bins[i]) & (magnitudes < mag_bins[i + 1])
            n_observed = weights[mask].sum()

            # Expected number (simplified model)
            # In reality, this would use external completeness simulations
            mag_center = (mag_bins[i] + mag_bins[i + 1]) / 2
            if mag_center < detection_limit:
                completeness[i] = 1.0
            else:
                # Simple exponential decay beyond limit
                completeness[i] = torch.exp(-(mag_center - detection_limit) / 0.5)

        mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

        # Store results
        self._store_function(
            function_name,
            {
                "magnitudes": mag_centers,
                "completeness": completeness,
                "detection_limit": detection_limit,
            },
        )

        return mag_centers, completeness

    # Helper methods
    def _count_pairs(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        r_edges: torch.Tensor,
        weights1: torch.Tensor,
        weights2: torch.Tensor,
    ) -> torch.Tensor:
        """Count pairs in radial bins."""
        counts = torch.zeros(len(r_edges) - 1)

        # Compute all pairwise distances (memory intensive for large datasets)
        if len(coords1) * len(coords2) > 1e6:
            # Use chunked computation for large datasets
            chunk_size = 1000
            for i in range(0, len(coords1), chunk_size):
                end_i = min(i + chunk_size, len(coords1))
                chunk1 = coords1[i:end_i]

                distances = torch.cdist(chunk1, coords2)

                for j in range(len(r_edges) - 1):
                    mask = (distances >= r_edges[j]) & (distances < r_edges[j + 1])
                    # Weight by product of weights
                    w1 = weights1[i:end_i].unsqueeze(1)
                    w2 = weights2.unsqueeze(0)
                    pair_weights = w1 * w2
                    counts[j] += (mask * pair_weights).sum()
        else:
            distances = torch.cdist(coords1, coords2)

            for j in range(len(r_edges) - 1):
                mask = (distances >= r_edges[j]) & (distances < r_edges[j + 1])
                w1 = weights1.unsqueeze(1)
                w2 = weights2.unsqueeze(0)
                pair_weights = w1 * w2
                counts[j] = (mask * pair_weights).sum()

        return counts

    def _generate_random_catalog(
        self, coordinates: torch.Tensor, n_random: int
    ) -> torch.Tensor:
        """Generate random catalog matching survey geometry."""
        # Simple uniform random in bounding box
        # In practice, this should match the survey footprint
        coord_min = coordinates.min(dim=0)[0]
        coord_max = coordinates.max(dim=0)[0]

        random_coords = torch.rand(n_random, coordinates.shape[1])
        random_coords = coord_min + random_coords * (coord_max - coord_min)

        return random_coords

    def _store_function(self, function_name: str, results: Dict[str, Any]):
        """Store computed function results."""
        computed_functions = self.get_metadata("computed_functions", {})
        computed_functions[function_name] = results
        self.update_metadata(computed_functions=computed_functions)

    def get_function(self, function_name: str) -> Dict[str, Any]:
        """Get stored function results."""
        computed_functions = self.get_metadata("computed_functions", {})
        if function_name not in computed_functions:
            raise ValueError(f"Function {function_name} not computed")
        return computed_functions[function_name]

    def list_functions(self) -> List[str]:
        """List all computed functions."""
        return list(self.get_metadata("computed_functions", {}).keys())

    def __repr__(self) -> str:
        n_functions = len(self.get_metadata("computed_functions", {}))
        return (
            f"StatisticsTensor(objects={self.n_objects}, "
            f"observables={self.n_observables}, functions={n_functions})"
        )
