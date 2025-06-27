#!/usr/bin/env python3
"""Test cosmic web functionality with AstroLab Widget visualization."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create results directory
results_dir = Path("results/cosmic_web_test")
results_dir.mkdir(parents=True, exist_ok=True)

try:
    # Test direct import first
    from astro_lab.data.cosmic_web import CosmicWebAnalyzer
    from astro_lab.data.loaders import list_available_catalogs, load_survey_catalog
    from astro_lab.tensors.tensordict_astro import SpatialTensorDict
    from astro_lab.widgets import AstroLabWidget
    from astro_lab.widgets.tensor_bridge import visualize_cosmic_web

    logger.info("✅ Imports successful!")

    # Create analyzer and widget
    analyzer = CosmicWebAnalyzer()
    widget = AstroLabWidget()

    # Check available catalogs
    logger.info("📊 Checking available catalogs...")
    catalogs = list_available_catalogs(survey="nsa")

    if len(catalogs) > 0:
        logger.info(f"📁 Found {len(catalogs)} NSA catalogs")
        print(catalogs)

        # Try to analyze with small dataset
        logger.info("🔬 Running cosmic web analysis...")
        results = analyzer.analyze_nsa_cosmic_web(
            clustering_scales=[5.0, 10.0, 25.0],
            min_samples=3,
            redshift_limit=0.1,
        )

        logger.info("✅ Analysis complete!")
        logger.info(f"  🌌 Galaxies analyzed: {results['n_galaxies']}")

        # Extract spatial data for visualization
        spatial_tensor = results.get("spatial_tensor")

        if spatial_tensor is not None:
            logger.info("🎨 Creating visualizations with AstroLab Widget...")

            # 1. Basic scatter plot
            try:
                logger.info("  📍 Creating basic galaxy distribution plot...")
                basic_viz = widget.plot(
                    spatial_tensor,
                    plot_type="scatter_3d",
                    title="NSA Galaxy Distribution",
                    point_size=2,
                    color_by="density",
                    max_points=10000,
                )

                # Save basic plot
                if hasattr(basic_viz, "write_html"):
                    basic_viz.write_html(
                        str(results_dir / "nsa_galaxy_distribution.html")
                    )
                    logger.info(
                        f"  💾 Saved: {results_dir / 'nsa_galaxy_distribution.html'}"
                    )

            except Exception as e:
                logger.warning(f"  ⚠️ Basic plot failed: {e}")

            # 2. Clustered visualizations for each scale
            for scale, stats in results["clustering_results"].items():
                logger.info(f"  🎯 Creating cluster visualization for {scale}...")

                try:
                    cluster_labels = stats.get("labels")
                    if cluster_labels is not None:
                        # Convert to numpy if tensor
                        if torch.is_tensor(cluster_labels):
                            cluster_labels = cluster_labels.numpy()

                        # Create clustered visualization
                        cluster_viz = widget.plot(
                            spatial_tensor,
                            plot_type="scatter_3d",
                            cluster_labels=cluster_labels,
                            title=f"Cosmic Web Clustering - {scale}",
                            point_size=3,
                            show_clusters=True,
                            max_points=10000,
                        )

                        # Save cluster plot
                        if hasattr(cluster_viz, "write_html"):
                            filename = (
                                f"cosmic_web_clusters_{scale.replace('.', '_')}.html"
                            )
                            cluster_viz.write_html(str(results_dir / filename))
                            logger.info(f"  💾 Saved: {results_dir / filename}")

                        logger.info(
                            f"    📊 {scale}: {stats['n_clusters']} clusters, {stats.get('n_grouped', 0)} grouped galaxies"
                        )

                except Exception as e:
                    logger.warning(f"  ⚠️ Cluster plot for {scale} failed: {e}")

            # 3. Open3D visualization (external window)
            try:
                logger.info("  🎯 Creating Open3D visualization...")

                # Use first clustering result for Open3D
                first_scale = list(results["clustering_results"].keys())[0]
                first_labels = results["clustering_results"][first_scale].get("labels")

                if first_labels is not None:
                    if torch.is_tensor(first_labels):
                        first_labels = first_labels.numpy()

                    # Create Open3D visualization
                    open3d_viz = visualize_cosmic_web(
                        spatial_tensor,
                        cluster_labels=first_labels,
                        backend="open3d",
                        show=True,  # Opens external window
                        window_name=f"NSA Cosmic Web - {first_scale}",
                        point_size=2.0,
                    )

                    logger.info("  ✅ Open3D visualization opened in external window!")

            except Exception as e:
                logger.warning(f"  ⚠️ Open3D visualization failed: {e}")

            # 4. Save analysis summary
            try:
                logger.info("📝 Saving analysis summary...")

                summary_path = results_dir / "cosmic_web_analysis_summary.txt"
                with open(summary_path, "w") as f:
                    f.write("NSA Cosmic Web Analysis Results\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Total Galaxies Analyzed: {results['n_galaxies']}\n")
                    f.write(
                        f"Redshift Limit: {results.get('redshift_limit', 'Unknown')}\n"
                    )
                    f.write(
                        f"Analysis Date: {results.get('analysis_date', 'Unknown')}\n\n"
                    )

                    f.write("Clustering Results by Scale:\n")
                    f.write("-" * 30 + "\n")

                    for scale, stats in results["clustering_results"].items():
                        f.write(f"\n{scale}:\n")
                        f.write(f"  Clusters Found: {stats['n_clusters']}\n")
                        f.write(f"  Grouped Galaxies: {stats.get('n_grouped', 0)}\n")
                        f.write(f"  Noise Points: {stats.get('n_noise', 0)}\n")
                        f.write(
                            f"  Grouping Fraction: {stats.get('grouped_fraction', 0):.1%}\n"
                        )

                    f.write("\nVisualization Files:\n")
                    f.write("-" * 20 + "\n")
                    for html_file in results_dir.glob("*.html"):
                        f.write(f"  {html_file.name}\n")

                logger.info(f"💾 Summary saved: {summary_path}")

            except Exception as e:
                logger.warning(f"⚠️ Summary save failed: {e}")

        else:
            logger.warning("⚠️ No spatial tensor available for visualization")

        # Final summary
        logger.info("\n" + "=" * 50)
        logger.info("🎉 COSMIC WEB TEST COMPLETED!")
        logger.info(f"📁 Results saved in: {results_dir}")
        logger.info("🔍 Check the following files:")
        for result_file in results_dir.iterdir():
            if result_file.is_file():
                logger.info(f"  📄 {result_file.name}")
        logger.info("=" * 50)

    else:
        logger.warning("❌ No NSA catalogs found")
        logger.info(
            "💡 Try downloading data first with: python -m astro_lab.cli download nsa --max-samples 10000"
        )

except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error(
        "💡 Make sure to restart the Python kernel or use 'python -m astro_lab.cli cosmic-web nsa'"
    )
except Exception as e:
    logger.error(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
finally:
    logger.info("🏁 Test script finished.")
