"""
CrossMatch Tensor for Catalog Cross-Matching
===========================================

Specialized tensor for cross-matching astronomical catalogs with
proper handling of uncertainties and multi-survey matching.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance as distance
import torch
from sklearn.cluster import DBSCAN, KMeans

# Optional dependencies
from sklearn.neighbors import BallTree

from .base import AstroTensorBase


class CrossMatchTensor(AstroTensorBase):
    """
    Tensor for cross-matching astronomical catalogs.

    Provides specialized cross-matching algorithms for:
    - Sky coordinate matching with uncertainties
    - Multi-survey cross-matching
    - Proper motion matching for moving objects
    - Bayesian matching with source probabilities
    - Time-dependent matching for variable sources
    """

    _metadata_fields = [
        "catalog_info",
        "matching_results",
        "matching_parameters",
        "uncertainty_models",
        "match_statistics",
        "coordinate_systems",
        "proper_motion_data",
        "time_epochs",
    ]

    def __init__(
        self,
        catalog_a: Union[torch.Tensor, Dict[str, torch.Tensor]],
        catalog_b: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        catalog_names: Optional[Tuple[str, str]] = None,
        coordinate_columns: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ):
        """
        Initialize cross-match tensor.

        Args:
            catalog_a: First catalog data or dict with named columns
            catalog_b: Second catalog data (None for self-matching)
            catalog_names: Names of catalogs ("catalog_a", "catalog_b")
            coordinate_columns: Column indices for coordinates {"a": [ra_col, dec_col], "b": [...]}
        """
        # Handle catalog_a
        if isinstance(catalog_a, dict):
            # Convert numpy arrays to tensors if needed
            cat_a_tensors = []
            for key in sorted(catalog_a.keys()):
                data = catalog_a[key]
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).float()
                elif not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                cat_a_tensors.append(data)
            cat_a_data = torch.stack(cat_a_tensors, dim=1)
        else:
            # Direct tensor input
            if isinstance(catalog_a, np.ndarray):
                cat_a_data = torch.from_numpy(catalog_a).float()
            else:
                cat_a_data = (
                    catalog_a.float() if catalog_a.dtype != torch.float32 else catalog_a
                )

        # Handle catalog_b
        if catalog_b is not None:
            if isinstance(catalog_b, dict):
                # Convert numpy arrays to tensors if needed
                cat_b_tensors = []
                for key in sorted(catalog_b.keys()):
                    data = catalog_b[key]
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data).float()
                    elif not isinstance(data, torch.Tensor):
                        data = torch.tensor(data, dtype=torch.float32)
                    cat_b_tensors.append(data)
                cat_b_data = torch.stack(cat_b_tensors, dim=1)
            else:
                # Direct tensor input
                if isinstance(catalog_b, np.ndarray):
                    cat_b_data = torch.from_numpy(catalog_b).float()
                else:
                    cat_b_data = (
                        catalog_b.float()
                        if catalog_b.dtype != torch.float32
                        else catalog_b
                    )
        else:
            cat_b_data = None

        # Set default coordinate columns
        if coordinate_columns is None:
            coordinate_columns = {
                "a": [0, 1],
                "b": [0, 1] if cat_b_data is not None else [],
            }

        # Set default catalog names
        if catalog_names is None:
            catalog_names = (
                "catalog_a",
                "catalog_b" if cat_b_data is not None else "self",
            )

        # Initialize metadata
        metadata = {
            "catalog_info": {
                "catalog_a": {
                    "name": catalog_names[0],
                    "n_objects": len(cat_a_data),
                    "columns": [f"col_{i}" for i in range(cat_a_data.shape[1])],
                    "coordinate_columns": coordinate_columns["a"],
                },
                "catalog_b": {
                    "name": catalog_names[1],
                    "n_objects": len(cat_b_data) if cat_b_data is not None else 0,
                    "columns": [f"col_{i}" for i in range(cat_b_data.shape[1])],
                    "coordinate_columns": coordinate_columns.get("b", []),
                }
                if cat_b_data is not None
                else None,
            },
            "matching_results": {},
            "matching_parameters": {},
            "uncertainty_models": {},
            "match_statistics": {},
            "coordinate_systems": {"a": "icrs", "b": "icrs"},
            "proper_motion_data": {},
            "time_epochs": {},
            "tensor_type": "crossmatch",
        }
        metadata.update(kwargs)

        super().__init__(
            torch.cat([cat_a_data, cat_b_data])
            if cat_b_data is not None
            else cat_a_data,
            **metadata,
        )

    @property
    def catalog_a_data(self) -> torch.Tensor:
        """Get catalog A data."""
        if self.get_metadata("catalog_info")["catalog_b"] is not None:
            # Extract from combined data
            cat_ids = self._data[:, -1]
            return self._data[cat_ids == 0, :-1]
        else:
            return self._data

    @property
    def catalog_b_data(self) -> Optional[torch.Tensor]:
        """Get catalog B data."""
        if self.get_metadata("catalog_info")["catalog_b"] is not None:
            cat_ids = self._data[:, -1]
            return self._data[cat_ids == 1, :-1]
        return None

    def sky_coordinate_matching(
        self,
        tolerance_arcsec: float = 1.0,
        method: str = "nearest_neighbor",
        include_uncertainties: bool = False,
        match_name: str = "sky_match",
    ) -> Dict[str, torch.Tensor]:
        """
        Match catalogs based on sky coordinates.

        Args:
            tolerance_arcsec: Matching tolerance in arcseconds
            method: Matching method ('nearest_neighbor', 'all_pairs', 'bayesian')
            include_uncertainties: Include coordinate uncertainties
            match_name: Name for storing results

        Returns:
            Dictionary with match results
        """

        cat_a = self.catalog_a_data
        cat_b = self.catalog_b_data

        if cat_b is None:
            raise ValueError("Two catalogs required for cross-matching")

        # Get coordinate columns
        coord_info = self.get_metadata("catalog_info")
        ra_col_a, dec_col_a = coord_info["catalog_a"]["coordinate_columns"]
        ra_col_b, dec_col_b = coord_info["catalog_b"]["coordinate_columns"]

        # Extract coordinates
        ra_a, dec_a = cat_a[:, ra_col_a], cat_a[:, dec_col_a]
        ra_b, dec_b = cat_b[:, ra_col_b], cat_b[:, dec_col_b]

        # Convert to Cartesian coordinates for distance calculation
        coords_a = self._sky_to_cartesian(ra_a, dec_a)
        coords_b = self._sky_to_cartesian(ra_b, dec_b)

        # Convert tolerance to 3D distance
        tolerance_rad = np.radians(tolerance_arcsec / 3600.0)
        tolerance_3d = 2 * np.sin(tolerance_rad / 2)  # Chord distance

        if method == "nearest_neighbor":
            matches = self._nearest_neighbor_matching(coords_a, coords_b, tolerance_3d)
        elif method == "all_pairs":
            matches = self._all_pairs_matching(coords_a, coords_b, tolerance_3d)
        elif method == "bayesian":
            matches = self._bayesian_matching(
                coords_a, coords_b, tolerance_3d, include_uncertainties
            )
        else:
            raise ValueError(f"Unknown matching method: {method}")

        # Calculate match statistics
        stats = self._calculate_match_statistics(matches, len(cat_a), len(cat_b))

        # Store results
        results = {
            "matches": matches,
            "statistics": stats,
            "tolerance_arcsec": tolerance_arcsec,
            "method": method,
            "n_catalog_a": len(cat_a),
            "n_catalog_b": len(cat_b),
        }

        self._store_matching_results(match_name, results)
        return results

    def proper_motion_matching(
        self,
        spatial_tolerance_arcsec: float = 1.0,
        pm_tolerance_mas_yr: float = 10.0,
        epoch_a: float = 2000.0,
        epoch_b: float = 2000.0,
        match_name: str = "pm_match",
    ) -> Dict[str, torch.Tensor]:
        """
        Match catalogs including proper motion information.

        Args:
            spatial_tolerance_arcsec: Spatial tolerance in arcseconds
            pm_tolerance_mas_yr: Proper motion tolerance in mas/yr
            epoch_a: Epoch of catalog A
            epoch_b: Epoch of catalog B
            match_name: Name for storing results

        Returns:
            Dictionary with match results
        """
        cat_a = self.catalog_a_data
        cat_b = self.catalog_b_data

        if cat_b is None:
            raise ValueError("Two catalogs required for proper motion matching")

        # Get coordinate and proper motion columns
        coord_info = self.get_metadata("catalog_info")
        # Assume columns: [ra, dec, pmra, pmdec, ...]
        ra_col_a, dec_col_a = coord_info["catalog_a"]["coordinate_columns"]
        ra_col_b, dec_col_b = coord_info["catalog_b"]["coordinate_columns"]

        # Extract coordinates and proper motions
        ra_a, dec_a = cat_a[:, ra_col_a], cat_a[:, dec_col_a]
        ra_b, dec_b = cat_b[:, ra_col_b], cat_b[:, dec_col_b]

        # Try to find proper motion columns (assume they follow coordinates)
        if cat_a.shape[1] > ra_col_a + 2 and cat_b.shape[1] > ra_col_b + 2:
            pmra_a, pmdec_a = cat_a[:, ra_col_a + 2], cat_a[:, dec_col_a + 2]
            pmra_b, pmdec_b = cat_b[:, ra_col_b + 2], cat_b[:, dec_col_b + 2]
        else:
            # No proper motion data available
            pmra_a = pmdec_a = torch.zeros_like(ra_a)
            pmra_b = pmdec_b = torch.zeros_like(ra_b)

        # Propagate positions to common epoch
        common_epoch = max(epoch_a, epoch_b)
        dt_a = common_epoch - epoch_a
        dt_b = common_epoch - epoch_b

        # Propagate coordinates (simplified, ignoring spherical geometry)
        ra_a_prop = ra_a + pmra_a * dt_a / (3600.0 * 1000.0 * np.cos(np.radians(dec_a)))
        dec_a_prop = dec_a + pmdec_a * dt_a / (3600.0 * 1000.0)
        ra_b_prop = ra_b + pmra_b * dt_b / (3600.0 * 1000.0 * np.cos(np.radians(dec_b)))
        dec_b_prop = dec_b + pmdec_b * dt_b / (3600.0 * 1000.0)

        # Convert to Cartesian
        coords_a = self._sky_to_cartesian(ra_a_prop, dec_a_prop)
        coords_b = self._sky_to_cartesian(ra_b_prop, dec_b_prop)

        # Match on propagated positions
        tolerance_rad = np.radians(spatial_tolerance_arcsec / 3600.0)
        tolerance_3d = 2 * np.sin(tolerance_rad / 2)

        spatial_matches = self._nearest_neighbor_matching(
            coords_a, coords_b, tolerance_3d
        )

        # Filter by proper motion similarity
        pm_tolerance_deg = pm_tolerance_mas_yr / (3600.0 * 1000.0)
        filtered_matches = []

        for match in spatial_matches:
            if match["distance"] < tolerance_3d:
                idx_a, idx_b = match["index_a"], match["index_b"]

                # Compare proper motions
                dpm_ra = abs(pmra_a[idx_a] - pmra_b[idx_b])
                dpm_dec = abs(pmdec_a[idx_a] - pmdec_b[idx_b])

                if dpm_ra < pm_tolerance_mas_yr and dpm_dec < pm_tolerance_mas_yr:
                    match["pm_distance"] = float(torch.sqrt(dpm_ra**2 + dpm_dec**2))
                    filtered_matches.append(match)

        # Store results
        results = {
            "matches": filtered_matches,
            "spatial_tolerance_arcsec": spatial_tolerance_arcsec,
            "pm_tolerance_mas_yr": pm_tolerance_mas_yr,
            "epoch_a": epoch_a,
            "epoch_b": epoch_b,
            "n_spatial_matches": len(spatial_matches),
            "n_pm_matches": len(filtered_matches),
        }

        self._store_matching_results(match_name, results)
        return results

    def multi_survey_matching(
        self,
        surveys: List[str],
        tolerance_arcsec: float = 1.0,
        min_detections: int = 2,
        match_name: str = "multi_survey_match",
    ) -> Dict[str, Any]:
        """
        Match objects across multiple surveys.

        Args:
            surveys: List of survey names
            tolerance_arcsec: Matching tolerance
            min_detections: Minimum number of survey detections
            match_name: Name for storing results

        Returns:
            Dictionary with multi-survey matches
        """
        # This is a simplified implementation
        # In practice, would need multiple catalogs as input

        cat_a = self.catalog_a_data
        cat_b = self.catalog_b_data

        if cat_b is None:
            # Self-matching for duplicate detection
            matches = self._self_matching(cat_a, tolerance_arcsec)
        else:
            # Two-catalog matching
            basic_matches = self.sky_coordinate_matching(tolerance_arcsec)
            matches = basic_matches["matches"]

        # Group matches by object
        object_groups = self._group_matches_by_object(matches)

        # Filter by minimum detections
        filtered_groups = {
            obj_id: group
            for obj_id, group in object_groups.items()
            if len(group) >= min_detections
        }

        results = {
            "object_groups": filtered_groups,
            "surveys": surveys,
            "tolerance_arcsec": tolerance_arcsec,
            "min_detections": min_detections,
            "n_objects": len(filtered_groups),
            "total_detections": sum(len(group) for group in filtered_groups.values()),
        }

        self._store_matching_results(match_name, results)
        return results

    def bayesian_matching(
        self,
        prior_density: float = 1e-6,  # objects per square arcsecond
        tolerance_arcsec: float = 5.0,
        uncertainty_columns: Optional[Dict[str, List[int]]] = None,
        match_name: str = "bayesian_match",
    ) -> Dict[str, torch.Tensor]:
        """
        Bayesian cross-matching with source probabilities.

        Args:
            prior_density: Prior source density
            tolerance_arcsec: Maximum search radius
            uncertainty_columns: Columns with coordinate uncertainties
            match_name: Name for storing results

        Returns:
            Dictionary with probabilistic matches
        """
        cat_a = self.catalog_a_data
        cat_b = self.catalog_b_data

        if cat_b is None:
            raise ValueError("Two catalogs required for Bayesian matching")

        # Get coordinates
        coord_info = self.get_metadata("catalog_info")
        ra_col_a, dec_col_a = coord_info["catalog_a"]["coordinate_columns"]
        ra_col_b, dec_col_b = coord_info["catalog_b"]["coordinate_columns"]

        ra_a, dec_a = cat_a[:, ra_col_a], cat_a[:, dec_col_a]
        ra_b, dec_b = cat_b[:, ra_col_b], cat_b[:, dec_col_b]

        # Get uncertainties (simplified - assume constant if not provided)
        if uncertainty_columns is not None:
            err_ra_a, err_dec_a = cat_a[:, uncertainty_columns["a"]]
            err_ra_b, err_dec_b = cat_b[:, uncertainty_columns["b"]]
        else:
            # Default uncertainties in arcseconds
            err_ra_a = err_dec_a = torch.full_like(ra_a, 0.1)
            err_ra_b = err_dec_b = torch.full_like(ra_b, 0.1)

        # Find candidate matches within search radius
        coords_a = self._sky_to_cartesian(ra_a, dec_a)
        coords_b = self._sky_to_cartesian(ra_b, dec_b)

        tolerance_rad = np.radians(tolerance_arcsec / 3600.0)
        tolerance_3d = 2 * np.sin(tolerance_rad / 2)

        candidates = self._all_pairs_matching(coords_a, coords_b, tolerance_3d)

        # Calculate match probabilities
        probabilistic_matches = []

        for candidate in candidates:
            idx_a, idx_b = candidate["index_a"], candidate["index_b"]

            # Angular separation in arcseconds
            sep_arcsec = (
                self._angular_separation(
                    ra_a[idx_a], dec_a[idx_a], ra_b[idx_b], dec_b[idx_b]
                )
                * 3600.0
            )

            # Combined uncertainty
            err_total = torch.sqrt(
                (err_ra_a[idx_a] * np.cos(np.radians(dec_a[idx_a]))) ** 2
                + err_dec_a[idx_a] ** 2
                + (err_ra_b[idx_b] * np.cos(np.radians(dec_b[idx_b]))) ** 2
                + err_dec_b[idx_b] ** 2
            )

            # Likelihood (Gaussian)
            likelihood = torch.exp(-0.5 * (sep_arcsec / err_total) ** 2)
            likelihood /= 2 * np.pi * err_total**2

            # Prior for random match
            search_area = np.pi * tolerance_arcsec**2  # square arcseconds
            prior_random = prior_density * search_area

            # Posterior probability
            posterior = likelihood / (likelihood + prior_random)

            probabilistic_matches.append(
                {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "separation_arcsec": float(sep_arcsec),
                    "likelihood": float(likelihood),
                    "posterior_prob": float(posterior),
                    "uncertainty": float(err_total),
                }
            )

        # Sort by posterior probability
        probabilistic_matches.sort(key=lambda x: x["posterior_prob"], reverse=True)

        results = {
            "matches": probabilistic_matches,
            "prior_density": prior_density,
            "tolerance_arcsec": tolerance_arcsec,
            "n_candidates": len(candidates),
            "n_probable_matches": len(
                [m for m in probabilistic_matches if m["posterior_prob"] > 0.5]
            ),
        }

        self._store_matching_results(match_name, results)
        return results

    # Helper methods
    def _sky_to_cartesian(self, ra: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """Convert sky coordinates to unit sphere Cartesian."""
        ra_rad = torch.deg2rad(ra)
        dec_rad = torch.deg2rad(dec)

        x = torch.cos(dec_rad) * torch.cos(ra_rad)
        y = torch.cos(dec_rad) * torch.sin(ra_rad)
        z = torch.sin(dec_rad)

        return torch.stack([x, y, z], dim=1)

    def _angular_separation(
        self,
        ra1: torch.Tensor,
        dec1: torch.Tensor,
        ra2: torch.Tensor,
        dec2: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate angular separation using haversine formula."""
        ra1_rad, dec1_rad = torch.deg2rad(ra1), torch.deg2rad(dec1)
        ra2_rad, dec2_rad = torch.deg2rad(ra2), torch.deg2rad(dec2)

        dra = ra2_rad - ra1_rad
        ddec = dec2_rad - dec1_rad

        a = (
            torch.sin(ddec / 2) ** 2
            + torch.cos(dec1_rad) * torch.cos(dec2_rad) * torch.sin(dra / 2) ** 2
        )

        return 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))

    def _nearest_neighbor_matching(
        self, coords_a: torch.Tensor, coords_b: torch.Tensor, tolerance: float
    ) -> List[Dict[str, Any]]:
        """Find nearest neighbor matches."""

        tree = BallTree(coords_b.cpu().numpy(), metric="euclidean")
        distances, indices = tree.query(coords_a.cpu().numpy(), k=1)

        matches = []
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if dist <= tolerance:
                matches.append(
                    {
                        "index_a": i,
                        "index_b": int(idx),
                        "distance": float(dist),
                    }
                )

        return matches

    def _all_pairs_matching(
        self, coords_a: torch.Tensor, coords_b: torch.Tensor, tolerance: float
    ) -> List[Dict[str, Any]]:
        """Find all pairs within tolerance."""
        matches = []

        # Use chunked computation for large catalogs
        chunk_size = 1000
        for i in range(0, len(coords_a), chunk_size):
            end_i = min(i + chunk_size, len(coords_a))
            chunk_a = coords_a[i:end_i]

            distances = torch.cdist(chunk_a, coords_b)

            # Find all pairs within tolerance
            within_tolerance = distances <= tolerance
            idx_a, idx_b = torch.where(within_tolerance)

            for a_idx, b_idx in zip(idx_a, idx_b):
                matches.append(
                    {
                        "index_a": int(i + a_idx),
                        "index_b": int(b_idx),
                        "distance": float(distances[a_idx, b_idx]),
                    }
                )

        return matches

    def _bayesian_matching(
        self,
        coords_a: torch.Tensor,
        coords_b: torch.Tensor,
        tolerance: float,
        include_uncertainties: bool,
    ) -> List[Dict[str, Any]]:
        """Bayesian matching (simplified version)."""
        # For now, just return nearest neighbor matches
        # Full Bayesian matching would require uncertainty propagation
        return self._nearest_neighbor_matching(coords_a, coords_b, tolerance)

    def _self_matching(
        self, catalog: torch.Tensor, tolerance_arcsec: float
    ) -> List[Dict[str, Any]]:
        """Find duplicate sources within a catalog."""
        coord_info = self.get_metadata("catalog_info")
        ra_col, dec_col = coord_info["catalog_a"]["coordinate_columns"]

        ra, dec = catalog[:, ra_col], catalog[:, dec_col]
        coords = self._sky_to_cartesian(ra, dec)

        tolerance_rad = np.radians(tolerance_arcsec / 3600.0)
        tolerance_3d = 2 * np.sin(tolerance_rad / 2)

        matches = []

        tree = BallTree(coords.cpu().numpy(), metric="euclidean")

        for i in range(len(coords)):
            distances, indices = tree.query(
                [coords[i].cpu().numpy()], k=5
            )  # Find 5 nearest

            for dist, idx in zip(distances[0], indices[0]):
                if idx != i and dist <= tolerance_3d:  # Exclude self
                    matches.append(
                        {
                            "index_a": i,
                            "index_b": int(idx),
                            "distance": float(dist),
                        }
                    )

        return matches

    def _group_matches_by_object(
        self, matches: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group matches by object ID."""
        groups = {}
        for match in matches:
            obj_id = match["index_a"]  # Use catalog A as reference
            if obj_id not in groups:
                groups[obj_id] = []
            groups[obj_id].append(match)
        return groups

    def _calculate_match_statistics(
        self, matches: List[Dict[str, Any]], n_a: int, n_b: int
    ) -> Dict[str, float]:
        """Calculate matching statistics."""
        n_matches = len(matches)

        # Count unique matches
        matched_a = set(match["index_a"] for match in matches)
        matched_b = set(match["index_b"] for match in matches)

        return {
            "n_matches": n_matches,
            "n_matched_a": len(matched_a),
            "n_matched_b": len(matched_b),
            "match_rate_a": len(matched_a) / n_a if n_a > 0 else 0.0,
            "match_rate_b": len(matched_b) / n_b if n_b > 0 else 0.0,
            "completeness": len(matched_a) / min(n_a, n_b)
            if min(n_a, n_b) > 0
            else 0.0,
            "contamination": (n_matches - len(matched_a)) / n_matches
            if n_matches > 0
            else 0.0,
        }

    def _store_matching_results(self, match_name: str, results: Dict[str, Any]):
        """Store matching results in metadata."""
        matching_results = self.get_metadata("matching_results", {})
        matching_results[match_name] = results
        self.update_metadata(matching_results=matching_results)

    def get_matches(self, match_name: str) -> Dict[str, Any]:
        """Get stored matching results."""
        matching_results = self.get_metadata("matching_results", {})
        if match_name not in matching_results:
            raise ValueError(f"No matches found for: {match_name}")
        return matching_results[match_name]

    def list_matches(self) -> List[str]:
        """List all computed matches."""
        return list(self.get_metadata("matching_results", {}).keys())

    def __repr__(self) -> str:
        cat_info = self.get_metadata("catalog_info")
        n_a = cat_info["catalog_a"]["n_objects"]
        n_b = cat_info["catalog_b"]["n_objects"] if cat_info["catalog_b"] else 0
        n_matches = len(self.get_metadata("matching_results", {}))

        return (
            f"CrossMatchTensor(catalog_a={n_a}, catalog_b={n_b}, matches={n_matches})"
        )
