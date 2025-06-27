"""
Cosmic Web Analysis Module
=========================

Provides functionality for analyzing cosmic web structures across multiple scales,
from stellar neighborhoods to large-scale galactic structures.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from sklearn.cluster import DBSCAN

from ..tensors import SpatialTensorDict
from .config import data_config
from .loaders import load_catalog, load_survey_catalog

logger = logging.getLogger(__name__)

# Standard cosmology (Planck 2018)
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)


class CosmicWebAnalyzer:
    """Analyzer for cosmic web structures across different scales."""
    
    def __init__(self):
        """Initialize the cosmic web analyzer."""
        self.results_dir = data_config.results_dir / "cosmic_web"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_gaia_cosmic_web(
        self,
        catalog_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
        magnitude_limit: float = 12.0,
        clustering_scales: List[float] = [5.0, 10.0, 25.0, 50.0],
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze cosmic web structure in Gaia stellar data.
        
        Args:
            catalog_path: Path to Gaia catalog
            max_samples: Maximum number of stars to analyze
            magnitude_limit: Magnitude limit for star selection
            clustering_scales: List of clustering scales in parsecs
            min_samples: Minimum cluster size for DBSCAN
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("🌟 Starting Gaia cosmic web analysis")
        
        # Load Gaia data
        if catalog_path:
            df = load_catalog(catalog_path)
        else:
            df = load_survey_catalog("gaia", max_samples=max_samples)
            
        # Filter by magnitude
        if "phot_g_mean_mag" in df.columns:
            df = df.filter(pl.col("phot_g_mean_mag") <= magnitude_limit)
            
        # Convert to 3D coordinates
        spatial_tensor = self._gaia_to_spatial_tensor(df)
        
        # Perform multi-scale clustering
        results = {
            "n_stars": len(spatial_tensor),
            "clustering_results": {},
            "density_stats": {},
        }
        
        for scale_pc in clustering_scales:
            logger.info(f"  Clustering at {scale_pc} pc scale...")
            
            cluster_labels = self._cosmic_web_clustering(
                spatial_tensor, 
                eps_pc=scale_pc, 
                min_samples=min_samples
            )
            
            # Analyze clustering results
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = np.sum(cluster_labels == -1)
            
            results["clustering_results"][f"{scale_pc}_pc"] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_grouped": len(cluster_labels) - n_noise,
                "grouped_fraction": (len(cluster_labels) - n_noise) / len(cluster_labels),
                "cluster_labels": cluster_labels,
            }
            
            logger.info(f"    Found {n_clusters} clusters, {n_noise} isolated stars")
            
        # Save results
        self._save_results(results, "gaia_cosmic_web")
        
        return results
        
    def analyze_nsa_cosmic_web(
        self,
        catalog_path: Optional[Path] = None,
        redshift_limit: float = 0.15,
        clustering_scales: List[float] = [5.0, 10.0, 20.0, 50.0],
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze cosmic web structure in NSA galaxy data.
        
        Args:
            catalog_path: Path to NSA catalog
            redshift_limit: Maximum redshift
            clustering_scales: List of clustering scales in Mpc
            min_samples: Minimum cluster size for DBSCAN
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("🌌 Starting NSA cosmic web analysis")
        
        # Load NSA data
        if catalog_path:
            df = load_catalog(catalog_path)
        else:
            df = load_survey_catalog("nsa")
            
        # Filter by redshift
        if "z" in df.columns:
            df = df.filter(pl.col("z") <= redshift_limit)
            df = df.filter(pl.col("z") > 0)  # Remove invalid redshifts
            
        # Convert to 3D coordinates
        spatial_tensor = self._nsa_to_spatial_tensor(df)
        
        # Perform multi-scale clustering
        results = {
            "n_galaxies": len(spatial_tensor),
            "clustering_results": {},
            "density_stats": {},
        }
        
        for scale_mpc in clustering_scales:
            logger.info(f"  Clustering at {scale_mpc} Mpc scale...")
            
            cluster_labels = self._cosmic_web_clustering(
                spatial_tensor, 
                eps_pc=scale_mpc * 1e6,  # Convert Mpc to pc for consistency
                min_samples=min_samples
            )
            
            # Analyze clustering results
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = np.sum(cluster_labels == -1)
            
            results["clustering_results"][f"{scale_mpc}_mpc"] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_grouped": len(cluster_labels) - n_noise,
                "grouped_fraction": (len(cluster_labels) - n_noise) / len(cluster_labels),
                "cluster_labels": cluster_labels,
            }
            
            logger.info(f"    Found {n_clusters} clusters, {n_noise} isolated galaxies")
            
        # Save results
        self._save_results(results, "nsa_cosmic_web")
        
        return results
        
    def analyze_exoplanet_cosmic_web(
        self,
        catalog_path: Optional[Path] = None,
        clustering_scales: List[float] = [10.0, 25.0, 50.0, 100.0, 200.0],
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Analyze cosmic web structure in exoplanet host stars.
        
        Args:
            catalog_path: Path to exoplanet catalog
            clustering_scales: List of clustering scales in parsecs
            min_samples: Minimum cluster size for DBSCAN
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("🪐 Starting exoplanet host star cosmic web analysis")
        
        # Load exoplanet data
        if catalog_path:
            df = load_catalog(catalog_path)
        else:
            # Try to load from processed directory
            exo_path = data_config.processed_dir / "exoplanets" / "confirmed_exoplanets.parquet"
            if exo_path.exists():
                df = pl.read_parquet(exo_path)
            else:
                raise FileNotFoundError("Exoplanet catalog not found")
                
        # Convert to 3D coordinates
        spatial_tensor = self._exoplanet_to_spatial_tensor(df)
        
        # Perform multi-scale clustering
        results = {
            "n_systems": len(spatial_tensor),
            "clustering_results": {},
            "density_stats": {},
        }
        
        for scale_pc in clustering_scales:
            logger.info(f"  Clustering at {scale_pc} pc scale...")
            
            cluster_labels = self._cosmic_web_clustering(
                spatial_tensor, 
                eps_pc=scale_pc, 
                min_samples=min_samples
            )
            
            # Analyze clustering results
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = np.sum(cluster_labels == -1)
            
            results["clustering_results"][f"{scale_pc}_pc"] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_grouped": len(cluster_labels) - n_noise,
                "grouped_fraction": (len(cluster_labels) - n_noise) / len(cluster_labels),
                "cluster_labels": cluster_labels,
            }
            
            logger.info(f"    Found {n_clusters} clusters, {n_noise} isolated systems")
            
        # Save results
        self._save_results(results, "exoplanet_cosmic_web")
        
        return results
        
    def _gaia_to_spatial_tensor(self, df: pl.DataFrame) -> SpatialTensorDict:
        """Convert Gaia catalog to SpatialTensorDict."""
        # Extract coordinates and parallax
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        parallax = df["parallax"].to_numpy()
        
        # Convert parallax to distance (parsecs)
        # Handle negative/zero parallaxes
        parallax_safe = np.where(parallax > 0, parallax, 0.1)  # 10 kpc for bad parallaxes
        distance_pc = 1000.0 / parallax_safe
        
        # Create SkyCoord
        coords = SkyCoord(
            ra=ra * u.degree,
            dec=dec * u.degree,
            distance=distance_pc * u.pc,
            frame='icrs'
        )
        
        # Convert to cartesian
        cartesian = coords.cartesian
        x = cartesian.x.value
        y = cartesian.y.value
        z = cartesian.z.value
        
        # Stack coordinates
        coords_3d = np.column_stack([x, y, z])
        
        # Create tensor
        return SpatialTensorDict(
            coordinates=torch.tensor(coords_3d, dtype=torch.float32),
            coordinate_system="icrs",
            unit="parsec"
        )
        
    def _nsa_to_spatial_tensor(self, df: pl.DataFrame) -> SpatialTensorDict:
        """Convert NSA catalog to SpatialTensorDict."""
        # Check for column names (case-insensitive)
        column_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == "ra":
                column_map["ra"] = col
            elif col_lower == "dec":
                column_map["dec"] = col
            elif col_lower == "z" or col_lower == "redshift":
                column_map["z"] = col
                
        if "ra" not in column_map or "dec" not in column_map or "z" not in column_map:
            raise ValueError(f"Required columns not found. Available columns: {df.columns}")
            
        # Extract coordinates and redshift
        ra = df[column_map["ra"]].to_numpy()
        dec = df[column_map["dec"]].to_numpy()
        z = df[column_map["z"]].to_numpy()
        
        # Convert to radians
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        # Simple Hubble flow approximation for nearby galaxies
        c_km_s = 299792.458  # km/s
        H0 = 67.4  # km/s/Mpc
        distance_mpc = (c_km_s * z) / H0
        
        # Convert to comoving coordinates
        x = distance_mpc * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance_mpc * np.cos(dec_rad) * np.sin(ra_rad)
        z_coord = distance_mpc * np.sin(dec_rad)
        
        # Stack coordinates
        coords_3d = np.column_stack([x, y, z_coord])
        
        # Create tensor (convert Mpc to pc for consistency)
        return SpatialTensorDict(
            coordinates=torch.tensor(coords_3d * 1e6, dtype=torch.float32),
            coordinate_system="icrs",
            unit="parsec"
        )
        
    def _exoplanet_to_spatial_tensor(self, df: pl.DataFrame) -> SpatialTensorDict:
        """Convert exoplanet catalog to SpatialTensorDict."""
        # Extract coordinates
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        
        # Distance handling - check various possible column names
        distance_columns = ["sy_dist", "st_dist", "distance", "dist"]
        distance = None
        
        for col in distance_columns:
            if col in df.columns:
                distance = df[col].to_numpy()
                break
                
        if distance is None:
            # If no distance, try to compute from parallax
            if "sy_plx" in df.columns:
                parallax = df["sy_plx"].to_numpy()
                distance = 1000.0 / np.where(parallax > 0, parallax, 1.0)
            else:
                raise ValueError("No distance or parallax information found")
                
        # Create SkyCoord
        coords = SkyCoord(
            ra=ra * u.degree,
            dec=dec * u.degree,
            distance=distance * u.pc,
            frame='icrs'
        )
        
        # Convert to cartesian
        cartesian = coords.cartesian
        x = cartesian.x.value
        y = cartesian.y.value
        z = cartesian.z.value
        
        # Stack coordinates
        coords_3d = np.column_stack([x, y, z])
        
        # Create tensor
        return SpatialTensorDict(
            coordinates=torch.tensor(coords_3d, dtype=torch.float32),
            coordinate_system="icrs",
            unit="parsec"
        )
        
    def _cosmic_web_clustering(
        self,
        spatial_tensor: SpatialTensorDict,
        eps_pc: float,
        min_samples: int = 5,
        algorithm: str = "dbscan"
    ) -> np.ndarray:
        """Perform cosmic web clustering analysis."""
        # Get coordinates
        coords = spatial_tensor["coordinates"].numpy()
        
        if algorithm == "dbscan":
            # DBSCAN clustering
            clustering = DBSCAN(eps=eps_pc, min_samples=min_samples, n_jobs=-1)
            labels = clustering.fit_predict(coords)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        return labels
        
    def _save_results(self, results: Dict[str, Any], prefix: str) -> None:
        """Save analysis results."""
        # Save summary
        summary_file = self.results_dir / f"{prefix}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Cosmic Web Analysis Results: {prefix}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total objects: {results.get('n_stars', results.get('n_galaxies', results.get('n_systems', 0)))}\n\n")
            
            f.write("Clustering Results:\n")
            for scale, stats in results["clustering_results"].items():
                f.write(f"\n{scale}:\n")
                f.write(f"  Clusters: {stats['n_clusters']}\n")
                f.write(f"  Grouped: {stats['n_grouped']} ({stats['grouped_fraction']:.1%})\n")
                f.write(f"  Isolated: {stats['n_noise']}\n")
                
        # Save cluster labels
        for scale, stats in results["clustering_results"].items():
            labels_file = self.results_dir / f"{prefix}_labels_{scale}.pt"
            torch.save(torch.tensor(stats["cluster_labels"]), labels_file)
            
        logger.info(f"Results saved to {self.results_dir}")
    
    def detect_filaments(
        self,
        spatial_tensor: SpatialTensorDict,
        method: str = "mst",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect filamentary structures in cosmic web.
        
        Args:
            spatial_tensor: Spatial coordinates tensor
            method: Detection method ("mst", "morse_theory", "hessian")
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with detected filaments
        """
        coords = spatial_tensor["coordinates"].numpy()
        
        if method == "mst":
            return self._detect_filaments_mst(coords, **kwargs)
        elif method == "morse_theory":
            return self._detect_filaments_morse(coords, **kwargs)
        elif method == "hessian":
            return self._detect_filaments_hessian(coords, **kwargs)
        else:
            raise ValueError(f"Unknown filament detection method: {method}")
    
    def _detect_filaments_mst(
        self,
        coords: np.ndarray,
        n_neighbors: int = 20,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect filaments using Minimal Spanning Tree approach.
        
        Args:
            coords: Coordinate array [N, D]
            n_neighbors: Number of nearest neighbors to consider
            distance_threshold: Maximum distance for connections
            
        Returns:
            Dictionary with filament information
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        from sklearn.neighbors import NearestNeighbors
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Create sparse distance matrix
        n_points = coords.shape[0]
        row_ind = np.repeat(np.arange(n_points), n_neighbors)
        col_ind = indices.flatten()
        data = distances.flatten()
        
        # Apply distance threshold if specified
        if distance_threshold is not None:
            mask = data <= distance_threshold
            row_ind = row_ind[mask]
            col_ind = col_ind[mask]
            data = data[mask]
        
        # Create sparse matrix
        sparse_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_points, n_points))
        
        # Compute MST
        mst = minimum_spanning_tree(sparse_matrix)
        
        # Extract filament segments
        mst_coo = mst.tocoo()
        filament_edges = list(zip(mst_coo.row, mst_coo.col))
        
        # Find filament chains (paths with degree 2)
        degrees = np.zeros(n_points)
        for i, j in filament_edges:
            degrees[i] += 1
            degrees[j] += 1
        
        # Identify nodes with degree 2 (part of filaments)
        filament_nodes = np.where(degrees == 2)[0]
        
        return {
            "method": "mst",
            "filament_edges": filament_edges,
            "filament_nodes": filament_nodes.tolist(),
            "n_filament_segments": len(filament_edges),
            "mean_segment_length": mst.data.mean() if len(mst.data) > 0 else 0,
            "total_filament_length": mst.data.sum(),
        }
    
    def _detect_filaments_morse(
        self,
        coords: np.ndarray,
        smoothing_scale: float = 10.0,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Detect filaments using simplified Morse theory approach.
        
        Args:
            coords: Coordinate array [N, D]
            smoothing_scale: Scale for density estimation
            threshold: Threshold for filament detection
            
        Returns:
            Dictionary with filament information
        """
        from scipy.ndimage import gaussian_filter
        from sklearn.neighbors import KernelDensity
        
        # Estimate density field
        kde = KernelDensity(kernel='gaussian', bandwidth=smoothing_scale)
        kde.fit(coords)
        
        # Create grid for evaluation
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        grid_points = 50
        
        if coords.shape[1] == 3:
            # 3D case
            x = np.linspace(mins[0], maxs[0], grid_points)
            y = np.linspace(mins[1], maxs[1], grid_points)
            z = np.linspace(mins[2], maxs[2], grid_points)
            xx, yy, zz = np.meshgrid(x, y, z)
            grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        else:
            # 2D case
            x = np.linspace(mins[0], maxs[0], grid_points)
            y = np.linspace(mins[1], maxs[1], grid_points)
            xx, yy = np.meshgrid(x, y)
            grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        
        # Compute density on grid
        log_density = kde.score_samples(grid)
        density = np.exp(log_density).reshape([grid_points] * coords.shape[1])
        
        # Smooth density field
        density_smooth = gaussian_filter(density, sigma=2)
        
        # Find ridges (simplified - just high density regions)
        filament_mask = density_smooth > threshold * density_smooth.max()
        filament_points = grid[filament_mask.ravel()]
        
        return {
            "method": "morse_theory",
            "filament_points": filament_points,
            "n_filament_points": len(filament_points),
            "density_threshold": threshold * density_smooth.max(),
            "smoothing_scale": smoothing_scale,
        }
    
    def _detect_filaments_hessian(
        self,
        coords: np.ndarray,
        scale: float = 10.0,
        eigenvalue_threshold: float = -0.1,
    ) -> Dict[str, Any]:
        """
        Detect filaments using Hessian eigenvalue analysis.
        
        Args:
            coords: Coordinate array [N, D]
            scale: Scale for density estimation
            eigenvalue_threshold: Threshold for ridge detection
            
        Returns:
            Dictionary with filament information
        """
        from scipy.ndimage import gaussian_filter, sobel
        from sklearn.neighbors import KernelDensity
        
        # Estimate density field on a grid
        kde = KernelDensity(kernel='gaussian', bandwidth=scale)
        kde.fit(coords)
        
        # Create evaluation grid
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        grid_points = 40
        
        if coords.shape[1] == 3:
            x = np.linspace(mins[0], maxs[0], grid_points)
            y = np.linspace(mins[1], maxs[1], grid_points)
            z = np.linspace(mins[2], maxs[2], grid_points)
            xx, yy, zz = np.meshgrid(x, y, z)
            grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
            shape = (grid_points, grid_points, grid_points)
        else:
            x = np.linspace(mins[0], maxs[0], grid_points)
            y = np.linspace(mins[1], maxs[1], grid_points)
            xx, yy = np.meshgrid(x, y)
            grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
            shape = (grid_points, grid_points)
        
        # Compute density
        log_density = kde.score_samples(grid)
        density = np.exp(log_density).reshape(shape)
        
        # Compute Hessian (simplified using finite differences)
        if coords.shape[1] == 3:
            # 3D Hessian
            dx = sobel(density, axis=0)
            dy = sobel(density, axis=1)
            dz = sobel(density, axis=2)
            
            dxx = sobel(dx, axis=0)
            dyy = sobel(dy, axis=1)
            dzz = sobel(dz, axis=2)
            
            # Find ridges where two eigenvalues are negative
            ridge_measure = dxx + dyy + dzz
            filament_mask = ridge_measure < eigenvalue_threshold
        else:
            # 2D Hessian
            dx = sobel(density, axis=0)
            dy = sobel(density, axis=1)
            
            dxx = sobel(dx, axis=0)
            dyy = sobel(dy, axis=1)
            
            # Find ridges where one eigenvalue is negative
            ridge_measure = dxx + dyy
            filament_mask = ridge_measure < eigenvalue_threshold
        
        filament_points = grid[filament_mask.ravel()]
        
        return {
            "method": "hessian",
            "filament_points": filament_points,
            "n_filament_points": len(filament_points),
            "scale": scale,
            "eigenvalue_threshold": eigenvalue_threshold,
        }


# Convenience functions
def analyze_gaia_cosmic_web(**kwargs) -> Dict[str, Any]:
    """Quick Gaia cosmic web analysis."""
    analyzer = CosmicWebAnalyzer()
    return analyzer.analyze_gaia_cosmic_web(**kwargs)


def analyze_nsa_cosmic_web(**kwargs) -> Dict[str, Any]:
    """Quick NSA cosmic web analysis."""
    analyzer = CosmicWebAnalyzer()
    return analyzer.analyze_nsa_cosmic_web(**kwargs)


def analyze_exoplanet_cosmic_web(**kwargs) -> Dict[str, Any]:
    """Quick exoplanet cosmic web analysis."""
    analyzer = CosmicWebAnalyzer()
    return analyzer.analyze_exoplanet_cosmic_web(**kwargs)
