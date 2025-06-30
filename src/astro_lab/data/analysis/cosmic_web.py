"""
Cosmic Web Analysis
===================

Clean, efficient cosmic web analysis using modern PyTorch Geometric.
Optimized for 50M+ astronomical objects with TensorDict integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
import torch
from astropy.time import Time
from tensordict import TensorDict, MemoryMappedTensor

from astro_lab.data.converters import create_spatial_tensor_from_survey
from astro_lab.data.preprocessors import get_preprocessor
from astro_lab.models.core.astro_cosmic_web_gnn import AstroCosmicWebGNN
from astro_lab.models.core.astro_pointnet import create_scalable_astro_pointnet
from astro_lab.tensors.spatial import SpatialTensorDict

logger = logging.getLogger(__name__)


class ScalableCosmicWebAnalyzer:
    """
    Scalable cosmic web analyzer for 50M+ objects.
    
    Features:
    - TensorDict-based data management
    - Memory-mapped tensors for large datasets
    - PyTorch Geometric native operations
    - Multi-scale hierarchical analysis
    - GPU-accelerated processing
    """
    
    def __init__(
        self, 
        config: Optional[Dict] = None,
        use_memory_mapping: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        results_dir: Optional[Path] = None,
    ):
        """Initialize scalable cosmic web analyzer."""
        self.config = config or {}
        self.use_memory_mapping = use_memory_mapping
        self.device = device
        self.results_dir = results_dir or Path("./results/cosmic_web")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        logger.info(f"🌌 ScalableCosmicWebAnalyzer initialized on {self.device}")
        logger.info(f"   Memory mapping: {use_memory_mapping}")
        
    def _init_models(self):
        """Initialize neural network models for analysis."""
        # Cosmic web structure classifier
        self.cosmic_web_model = AstroCosmicWebGNN(
            num_features=10,  # Basic features
            num_classes=4,    # void, sheet, filament, node
            hidden_dim=128,
            use_multi_scale=True,
            cosmic_web_scales=[5.0, 10.0, 25.0, 50.0],
        ).to(self.device)
        
        # Large-scale point cloud processor
        self.pointnet_model = create_scalable_astro_pointnet(
            num_features=10,
            num_classes=4,
            task="cosmic_web_classification",
            num_objects=50_000_000,
        ).to(self.device)
        
    def analyze_survey(
        self,
        survey: str,
        max_samples: Optional[int] = None,
        clustering_scales: Optional[List[float]] = None,
        min_samples: int = 5,
        chunk_size: int = 1_000_000,
        **filter_kwargs,
    ) -> TensorDict:
        """
        Analyze cosmic web structure for a survey using TensorDict.
        
        Args:
            survey: Survey name ('gaia', 'nsa', 'sdss', etc.)
            max_samples: Maximum objects to analyze
            clustering_scales: Scales in survey-appropriate units
            min_samples: Minimum cluster size
            chunk_size: Processing chunk size for large data
            **filter_kwargs: Survey-specific filters
            
        Returns:
            TensorDict with comprehensive analysis results
        """
        logger.info(f"🌌 Analyzing {survey.upper()} cosmic web structure")
        
        # Set default scales
        if clustering_scales is None:
            clustering_scales = self._get_default_scales(survey)
            
        # Load and prepare data as SpatialTensorDict
        spatial_tensor = self._load_and_prepare_spatial_tensor(
            survey, max_samples, chunk_size, **filter_kwargs
        )
        
        n_objects = spatial_tensor["meta"]["n_objects"]
        logger.info(f"📊 Processing {n_objects:,} objects")
        
        # Create analysis TensorDict
        analysis_td = TensorDict({
            "survey": survey,
            "n_objects": n_objects,
            "device": str(self.device),
            "clustering_scales": torch.tensor(clustering_scales),
            "timestamp": Time.now().iso,
        }, batch_size=[])
        
        # Multi-scale clustering analysis
        logger.info("🔍 Performing multi-scale clustering...")
        clustering_results = self._multi_scale_clustering(
            spatial_tensor, clustering_scales, min_samples
        )
        analysis_td["clustering_results"] = clustering_results
        
        # Neural network structure classification
        if n_objects <= 10_000_000:
            logger.info("🧠 Running neural network analysis...")
            nn_results = self._neural_network_analysis(spatial_tensor)
            analysis_td["nn_results"] = nn_results
        else:
            logger.info("⚡ Using hierarchical analysis for 50M+ objects...")
            nn_results = self._hierarchical_nn_analysis(spatial_tensor, chunk_size)
            analysis_td["nn_results"] = nn_results
            
        # Cosmic web statistics
        stats = self._compute_cosmic_web_statistics(
            spatial_tensor, clustering_results, nn_results
        )
        analysis_td["statistics"] = stats
        
        # Save results
        self._save_tensordict_results(analysis_td, f"{survey}_analysis")
        
        return analysis_td
        
    def _load_and_prepare_spatial_tensor(
        self,
        survey: str,
        max_samples: Optional[int],
        chunk_size: int,
        **filter_kwargs
    ) -> SpatialTensorDict:
        """Load data and create SpatialTensorDict with optional memory mapping."""
        
        # Use preprocessor to load data
        preprocessor = get_preprocessor(survey)
        df = preprocessor.load_data(max_samples=max_samples)
        logger.info(f"📂 Loaded {len(df)} {survey.upper()} objects")
        
        # Apply filters
        df = self._apply_filters(df, survey, **filter_kwargs)
        logger.info(f"🔍 Filtered to {len(df)} objects")
        
        # Convert to SpatialTensorDict
        spatial_tensor = create_spatial_tensor_from_survey(df, survey)
        
        # Enable memory mapping for large datasets
        if self.use_memory_mapping and len(df) > 10_000_000:
            logger.info("💾 Converting to memory-mapped tensors...")
            spatial_tensor = self._convert_to_memory_mapped(spatial_tensor, chunk_size)
            
        return spatial_tensor
        
    def _convert_to_memory_mapped(
        self,
        spatial_tensor: SpatialTensorDict,
        chunk_size: int
    ) -> SpatialTensorDict:
        """Convert large SpatialTensorDict to use memory-mapped tensors."""
        coords = spatial_tensor.coordinates
        n_objects = len(coords)
        
        # Create memory-mapped tensor
        mmap_coords = MemoryMappedTensor.empty(
            (n_objects, 3),
            dtype=torch.float32,
            filename=f"coords_{n_objects}.memmap"
        )
        
        # Copy in chunks
        for i in range(0, n_objects, chunk_size):
            end_idx = min(i + chunk_size, n_objects)
            mmap_coords[i:end_idx] = coords[i:end_idx]
            
        # Create new SpatialTensorDict with memory-mapped data
        return SpatialTensorDict(
            mmap_coords,
            coordinate_system=spatial_tensor["meta"]["coordinate_system"],
            unit=spatial_tensor["meta"]["unit"],
            use_memory_mapping=True,
            chunk_size=chunk_size,
        )
        
    def _multi_scale_clustering(
        self,
        spatial_tensor: SpatialTensorDict,
        scales: List[float],
        min_samples: int
    ) -> TensorDict:
        """Perform multi-scale clustering analysis."""
        clustering_td = TensorDict({}, batch_size=[])
        
        for scale in scales:
            logger.info(f"  Scale {scale}: ", end="")
            
            # Perform clustering
            labels = spatial_tensor.cosmic_web_clustering(
                method="fof",
                linking_length=scale,
                min_group_size=min_samples,
                batch_size=1_000_000,  # Process in batches
            )
            
            # Compute statistics
            unique_labels = torch.unique(labels[labels >= 0])
            n_clusters = len(unique_labels)
            n_grouped = (labels >= 0).sum().item()
            grouped_fraction = n_grouped / len(labels)
            
            logger.info(f"{n_clusters} clusters ({grouped_fraction:.1%} grouped)")
            
            # Store results
            scale_key = f"{scale}pc"
            clustering_td[scale_key] = TensorDict({
                "labels": labels,
                "n_clusters": n_clusters,
                "n_grouped": n_grouped,
                "grouped_fraction": grouped_fraction,
            }, batch_size=[])
            
        return clustering_td
        
    def _neural_network_analysis(
        self,
        spatial_tensor: SpatialTensorDict
    ) -> TensorDict:
        """Run neural network cosmic web classification."""
        
        # Build PyG data
        data = spatial_tensor.build_pyg_data(
            method="knn",
            k=20,
            use_spatial_partitioning=True,
        )
        
        # Add features if not present
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 10), device=self.device)
            
        # Move to device
        data = data.to(self.device)
        
        # Run model
        self.cosmic_web_model.eval()
        with torch.no_grad():
            predictions = self.cosmic_web_model(data)
            
        # Get structure statistics
        stats = self.cosmic_web_model.analyze_cosmic_web_statistics(data)
        
        return TensorDict({
            "predictions": predictions.cpu(),
            "statistics": stats,
        }, batch_size=[])
        
    def _hierarchical_nn_analysis(
        self,
        spatial_tensor: SpatialTensorDict,
        chunk_size: int
    ) -> TensorDict:
        """Hierarchical neural network analysis for 50M+ objects."""
        
        # Use ScalableAstroPointNet for very large data
        coords = spatial_tensor.coordinates
        n_objects = len(coords)
        
        # Process in chunks
        all_predictions = []
        
        for i in range(0, n_objects, chunk_size):
            end_idx = min(i + chunk_size, n_objects)
            
            # Create chunk data
            chunk_coords = coords[i:end_idx]
            chunk_td = TensorDict({
                "pos": chunk_coords.to(self.device),
                "features": torch.ones((len(chunk_coords), 10), device=self.device),
            }, batch_size=[])
            
            # Run model
            self.pointnet_model.eval()
            with torch.no_grad():
                chunk_pred = self.pointnet_model(chunk_td)
                all_predictions.append(chunk_pred.cpu())
                
            if i % (chunk_size * 10) == 0:
                logger.info(f"  Processed {i:,} / {n_objects:,} objects")
                
        # Combine predictions
        predictions = torch.cat(all_predictions, dim=0)
        
        # Compute statistics
        structure_names = ["void", "sheet", "filament", "node"]
        pred_labels = predictions.argmax(dim=1)
        
        stats = {
            "structure_counts": {
                name: (pred_labels == i).sum().item()
                for i, name in enumerate(structure_names)
            },
            "structure_fractions": {
                name: count / n_objects
                for name, count in stats["structure_counts"].items()
            }
        }
        
        return TensorDict({
            "predictions": predictions,
            "statistics": stats,
        }, batch_size=[])
        
    def _compute_cosmic_web_statistics(
        self,
        spatial_tensor: SpatialTensorDict,
        clustering_results: TensorDict,
        nn_results: TensorDict
    ) -> Dict:
        """Compute comprehensive cosmic web statistics."""
        
        stats = {
            "coordinate_system": spatial_tensor["meta"]["coordinate_system"],
            "total_objects": spatial_tensor["meta"]["n_objects"],
            "volume_sampled": self._estimate_volume(spatial_tensor),
            "clustering_summary": {},
            "nn_summary": nn_results.get("statistics", {}),
        }
        
        # Clustering summary across scales
        for scale_key, scale_data in clustering_results.items():
            stats["clustering_summary"][scale_key] = {
                "n_clusters": scale_data["n_clusters"].item(),
                "grouped_fraction": scale_data["grouped_fraction"].item(),
            }
            
        return stats
        
    def _estimate_volume(self, spatial_tensor: SpatialTensorDict) -> float:
        """Estimate the volume of the sampled region."""
        coords = spatial_tensor.coordinates
        
        # Get bounding box
        min_coords = coords.min(dim=0)[0]
        max_coords = coords.max(dim=0)[0]
        
        # Volume in cubic units
        volume = torch.prod(max_coords - min_coords).item()
        
        return volume
        
    def _save_tensordict_results(self, results: TensorDict, prefix: str) -> None:
        """Save TensorDict results efficiently."""
        
        # Save as PyTorch checkpoint
        checkpoint_file = self.results_dir / f"{prefix}.pt"
        torch.save(results.to_dict(), checkpoint_file)
        
        # Save summary
        summary_file = self.results_dir / f"{prefix}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Cosmic Web Analysis: {results['survey'].upper()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Objects analyzed: {results['n_objects']:,}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            # Clustering summary
            if "clustering_results" in results:
                f.write("Multi-Scale Clustering:\n")
                f.write("-" * 25 + "\n")
                for scale_key, scale_data in results["clustering_results"].items():
                    f.write(f"{scale_key}: {scale_data['n_clusters']} clusters ")
                    f.write(f"({scale_data['grouped_fraction']:.1%} grouped)\n")
                    
        logger.info(f"💾 Results saved to {self.results_dir}")
        
    def _apply_filters(self, df: pl.DataFrame, survey: str, **kwargs) -> pl.DataFrame:
        """Apply survey-specific astronomical filters."""
        
        if survey in ["gaia", "exoplanet"]:
            # Stellar surveys - quality cuts
            if "phot_g_mean_mag" in df.columns:
                mag_limit = kwargs.get("magnitude_limit", 15.0)
                df = df.filter(pl.col("phot_g_mean_mag") <= mag_limit)
                
            if "parallax" in df.columns and "parallax_error" in df.columns:
                snr_limit = kwargs.get("parallax_snr", 3.0)
                df = df.filter(
                    (pl.col("parallax") / pl.col("parallax_error")) >= snr_limit
                )
                
        elif survey in ["nsa", "sdss"]:
            # Galaxy surveys - redshift cuts
            if "z" in df.columns:
                z_min = kwargs.get("redshift_min", 0.001)
                z_max = kwargs.get("redshift_limit", 0.15)
                df = df.filter(pl.col("z").is_between(z_min, z_max))
                
        return df
        
    def _get_default_scales(self, survey: str) -> List[float]:
        """Get appropriate default scales for survey."""
        scales = {
            # Stellar surveys (parsec scales)
            "gaia": [5.0, 10.0, 25.0, 50.0],
            "exoplanet": [10.0, 25.0, 50.0, 100.0],
            # Galaxy surveys (Megaparsec scales)
            "nsa": [2.0, 5.0, 10.0, 20.0],
            "sdss": [3.0, 6.0, 12.0, 25.0],
        }
        return scales.get(survey, [5.0, 10.0, 25.0, 50.0])


def analyze_cosmic_web_50m(
    survey: str = "gaia",
    n_objects: int = 50_000_000,
    **kwargs
) -> TensorDict:
    """
    Quick function to analyze 50M objects.
    
    Args:
        survey: Survey name
        n_objects: Number of objects to analyze
        **kwargs: Additional parameters
        
    Returns:
        TensorDict with analysis results
    """
    analyzer = ScalableCosmicWebAnalyzer(
        use_memory_mapping=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return analyzer.analyze_survey(
        survey=survey,
        max_samples=n_objects,
        chunk_size=1_000_000,
        **kwargs
    )
