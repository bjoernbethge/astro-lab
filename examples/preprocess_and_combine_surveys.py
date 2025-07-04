"""
Example: Preprocessing and Combining Multiple Surveys
=====================================================

Shows how to:
1. Preprocess individual surveys (Gaia, SDSS, 2MASS)
2. Cross-match between surveys
3. Build a combined multi-wavelength graph
"""

import logging
from pathlib import Path

import polars as pl
import torch
from torch_geometric.data import Data

from astro_lab.data.cross_match import SurveyCrossMatcher
from astro_lab.data.preprocessors import get_preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate multi-survey preprocessing and cross-matching."""

    # 1. Preprocess individual surveys
    logger.info("=== Step 1: Preprocessing Individual Surveys ===")

    surveys = {}
    graphs = {}

    # Process Gaia (astrometry + optical)
    logger.info("\nProcessing Gaia DR3...")
    gaia_proc = get_preprocessor("gaia")
    gaia_df, gaia_graph = gaia_proc.preprocess(max_samples=100000)
    surveys["gaia"] = gaia_df
    graphs["gaia"] = gaia_graph

    # Process SDSS (deep optical photometry)
    logger.info("\nProcessing SDSS...")
    sdss_proc = get_preprocessor("sdss")
    sdss_df, sdss_graph = sdss_proc.preprocess(max_samples=50000)
    surveys["sdss"] = sdss_df
    graphs["sdss"] = sdss_graph

    # Process 2MASS (near-infrared)
    logger.info("\nProcessing 2MASS...")
    twomass_proc = get_preprocessor("twomass")
    twomass_df, twomass_graph = twomass_proc.preprocess(max_samples=50000)
    surveys["twomass"] = twomass_df
    graphs["twomass"] = twomass_graph

    # 2. Cross-match surveys
    logger.info("\n=== Step 2: Cross-Matching Surveys ===")

    matcher = SurveyCrossMatcher(max_separation=1.0)  # 1 arcsec

    # Match all to Gaia as reference
    combined_df = matcher.multi_survey_match(surveys=surveys, reference_survey="gaia")

    logger.info(f"\nCombined catalog has {len(combined_df):,} sources")
    logger.info(f"Columns: {len(combined_df.columns)}")

    # 3. Create multi-wavelength features
    logger.info("\n=== Step 3: Creating Multi-Wavelength Features ===")

    # Build feature vector combining all surveys
    feature_cols = []

    # Gaia features
    gaia_features = [
        "gaia_phot_g_mean_mag",
        "gaia_phot_bp_mean_mag",
        "gaia_phot_rp_mean_mag",
        "gaia_bp_rp",
        "gaia_parallax",
        "gaia_pmra",
        "gaia_pmdec",
    ]
    feature_cols.extend([f for f in gaia_features if f in combined_df.columns])

    # SDSS features
    sdss_features = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    feature_cols.extend([f for f in sdss_features if f in combined_df.columns])

    # 2MASS features
    twomass_features = ["twomass_j_m", "twomass_h_m", "twomass_k_m"]
    feature_cols.extend([f for f in twomass_features if f in combined_df.columns])

    logger.info(f"Total features: {len(feature_cols)}")

    # 4. Build combined graph
    logger.info("\n=== Step 4: Building Combined Graph ===")

    # Use Gaia positions (most accurate)
    pos_cols = ["gaia_x", "gaia_y", "gaia_z"]

    # Filter to objects with positions and features
    mask = (
        combined_df.select(pos_cols + feature_cols).null_count().sum_horizontal() == 0
    )
    filtered_df = combined_df.filter(mask)

    logger.info(f"Filtered to {len(filtered_df):,} complete sources")

    # Extract data
    positions = torch.tensor(
        filtered_df.select(pos_cols).to_numpy(), dtype=torch.float32
    )

    features = torch.tensor(
        filtered_df.select(feature_cols).to_numpy(), dtype=torch.float32
    )

    # Build graph using Gaia preprocessor's method
    edge_index = gaia_proc._build_graph_structure(positions)

    # Create combined graph
    combined_graph = Data(x=features, pos=positions, edge_index=edge_index)

    # Add metadata
    combined_graph.feature_names = feature_cols
    combined_graph.surveys = ["gaia", "sdss", "twomass"]
    combined_graph.n_sources = len(filtered_df)

    # Add source IDs
    if "gaia_source_id" in filtered_df.columns:
        combined_graph.source_id = torch.tensor(
            filtered_df["gaia_source_id"].to_numpy(), dtype=torch.long
        )

    # 5. Save results
    logger.info("\n=== Step 5: Saving Results ===")

    output_dir = Path("data/processed/combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined catalog
    catalog_path = output_dir / "gaia_sdss_twomass.parquet"
    filtered_df.write_parquet(catalog_path)
    logger.info(f"Saved catalog to {catalog_path}")

    # Save combined graph
    graph_path = output_dir / "gaia_sdss_twomass.pt"
    torch.save(combined_graph, graph_path)
    logger.info(f"Saved graph to {graph_path}")

    # 6. Print statistics
    logger.info("\n=== Final Statistics ===")
    logger.info(f"Nodes: {combined_graph.num_nodes}")
    logger.info(f"Edges: {combined_graph.num_edges}")
    logger.info(f"Features: {combined_graph.x.shape[1]}")
    logger.info(
        f"Average degree: {combined_graph.num_edges / combined_graph.num_nodes:.1f}"
    )

    # Color statistics (optical to NIR)
    if all(f in feature_cols for f in ["gaia_phot_g_mean_mag", "twomass_k_m"]):
        g_idx = feature_cols.index("gaia_phot_g_mean_mag")
        k_idx = feature_cols.index("twomass_k_m")
        g_k_color = features[:, g_idx] - features[:, k_idx]
        logger.info(f"G-K color range: [{g_k_color.min():.2f}, {g_k_color.max():.2f}]")

    return combined_graph


if __name__ == "__main__":
    combined_graph = main()

    # Example: Select red giants using multi-wavelength data
    print("\n=== Example: Finding Red Giants ===")

    # Red giants typically have:
    # - Red optical colors (BP-RP > 1.0)
    # - Bright in infrared (K < 10)
    # - Low parallax (distant)

    if hasattr(combined_graph, "feature_names"):
        feature_names = combined_graph.feature_names

        # Find relevant indices
        bp_rp_idx = (
            feature_names.index("gaia_bp_rp") if "gaia_bp_rp" in feature_names else None
        )
        k_idx = (
            feature_names.index("twomass_k_m")
            if "twomass_k_m" in feature_names
            else None
        )
        parallax_idx = (
            feature_names.index("gaia_parallax")
            if "gaia_parallax" in feature_names
            else None
        )

        if all(idx is not None for idx in [bp_rp_idx, k_idx, parallax_idx]):
            # Select red giants
            red_giants_mask = (
                (combined_graph.x[:, bp_rp_idx] > 1.0)  # Red color
                & (combined_graph.x[:, k_idx] < 10.0)  # Bright in K
                & (combined_graph.x[:, parallax_idx] < 1.0)  # Distant (> 1 kpc)
            )

            n_red_giants = red_giants_mask.sum().item()
            print(f"Found {n_red_giants:,} potential red giants")
            print(f"({n_red_giants / combined_graph.num_nodes * 100:.1f}% of sample)")
