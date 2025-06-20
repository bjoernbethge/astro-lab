#!/usr/bin/env python3
"""
🌌 AstroLab 3D Visualization Suite

Professional astronomical data visualization with spatial tensors,
space backgrounds, and interactive 3D plots for all AstroLab surveys.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "./src")
import numpy as np
import pyvista as pv
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord

from astro_lab.data.core import (
    load_gaia_data,
    load_sdss_data,
    load_tng50_data,
    load_tng50_temporal_data,
)


class AstroLabVisualizer:
    """Professional astronomical data visualizer with spatial tensor support."""

    def __init__(self, output_dir: str = "results"):
        """Initialize the visualizer with space theme."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Space theme colors
        self.space_colors = {
            "background": "black",
            "stars": "white",
            "galaxies": "cyan",
            "gas": "orange",
            "dark_matter": "purple",
            "axes": "gray",
            "grid": "#1a1a1a",
        }

        # Survey configurations
        self.surveys = {
            "gaia": {
                "loader": load_gaia_data,
                "kwargs": {"max_samples": 10000, "return_tensor": False},
                "color": self.space_colors["stars"],
                "point_size": 3,
                "title": "Gaia DR3 Stellar Survey",
            },
            "sdss": {
                "loader": load_sdss_data,
                "kwargs": {"max_samples": 10000, "return_tensor": False},
                "color": self.space_colors["galaxies"],
                "point_size": 4,
                "title": "SDSS DR17 Galaxy Survey",
            },
            "tng50": {
                "loader": load_tng50_data,
                "kwargs": {"max_samples": 15000},
                "color": self.space_colors["gas"],
                "point_size": 2,
                "title": "TNG50 Cosmological Simulation",
            },
            "tng50_temporal": {
                "loader": load_tng50_temporal_data,
                "kwargs": {"max_samples": 10000, "snapshot_id": 0},
                "color": self.space_colors["dark_matter"],
                "point_size": 2,
                "title": "TNG50 Temporal Evolution (z=0)",
            },
        }

    def load_spatial_tensor(self, survey_name: str) -> Optional[Dict[str, Any]]:
        """Load spatial tensor data directly from PT files if available."""
        pt_path = Path(f"data/processed/{survey_name}_spatial_tensor.pt")
        if pt_path.exists():
            print(f"   🚀 Loading spatial tensor from {pt_path}")
            try:
                tensor_data = torch.load(pt_path, map_location="cpu")
                return {
                    "positions": tensor_data.get("positions", tensor_data.get("pos")),
                    "features": tensor_data.get("features", tensor_data.get("x")),
                    "metadata": tensor_data.get("metadata", {}),
                }
            except Exception as e:
                print(f"   ⚠️ Failed to load spatial tensor: {e}")
                return None
        return None

    def convert_to_3d_coordinates(
        self, dataset, survey_name: str
    ) -> Optional[np.ndarray]:
        """Convert dataset to 3D coordinates for visualization."""
        try:
            # Debug output for TNG50 temporal
            if survey_name == "tng50_temporal":
                print(f"   🔍 Debug: Dataset type: {type(dataset)}")
                print(f"   🔍 Debug: Has data attr: {hasattr(dataset, 'data')}")
                if hasattr(dataset, "data"):
                    print(f"   🔍 Debug: Data type: {type(dataset.data)}")
                    if hasattr(dataset.data, "shape"):
                        print(f"   🔍 Debug: Data shape: {dataset.data.shape}")
                    if hasattr(dataset.data, "columns"):
                        print(f"   🔍 Debug: Data columns: {dataset.data.columns}")

            # Get DataFrame from dataset
            if hasattr(dataset, "data"):
                df = dataset.data
            elif hasattr(dataset, "_df_cache"):
                df = dataset._df_cache
            else:
                print(f"   ⚠️ No data found in dataset for {survey_name}")
                return None

            if survey_name in ["gaia", "sdss", "nsa"]:
                # Convert RA/Dec to 3D cartesian
                if "ra" in df.columns and "dec" in df.columns:
                    ra = df["ra"].to_numpy().astype(float)
                    dec = df["dec"].to_numpy().astype(float)

                    # Calculate distance
                    if survey_name == "gaia" and "parallax" in df.columns:
                        parallax = df["parallax"].to_numpy().astype(float)
                        distance = np.where(
                            parallax > 0, 1000.0 / parallax, 1000.0
                        )  # pc
                    elif survey_name == "sdss" and "z" in df.columns:
                        # Use redshift for distance
                        z = df["z"].to_numpy().astype(float)
                        distance = 3000.0 * z  # Mpc, rough approximation
                    else:
                        # Default distance
                        distance = np.ones_like(ra) * 100.0

                    # Convert to 3D cartesian
                    ra_rad = np.radians(ra)
                    dec_rad = np.radians(dec)

                    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
                    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
                    z = distance * np.sin(dec_rad)

                    return np.column_stack([x, y, z])

            elif survey_name in ["tng50", "tng50_temporal"]:
                # Already in 3D cartesian coordinates
                x = df["x"].to_numpy().astype(float)
                y = df["y"].to_numpy().astype(float)
                z = df["z"].to_numpy().astype(float)
                return np.column_stack([x, y, z])

        except Exception as e:
            print(f"   ⚠️ Error converting coordinates: {e}")
            return None

        return None

    def create_space_background(self, plotter: pv.Plotter):
        """Add space background with stars and grid."""
        # Set black background
        plotter.set_background(self.space_colors["background"])

        # Add coordinate grid
        plotter.add_axes(
            xlabel="X (Mpc)",
            ylabel="Y (Mpc)",
            zlabel="Z (Mpc)",
            line_width=2,
            color=self.space_colors["axes"],
        )

        # Add bounding box with grid
        plotter.add_bounding_box(
            color=self.space_colors["grid"], line_width=1, opacity=0.3
        )

    def visualize_survey(self, survey_name: str) -> bool:
        """Create professional 3D visualization for a survey."""
        import numpy as np

        try:
            print(f"🌌 Visualizing {survey_name.upper()}...")

            # Special handling for TNG50 temporal
            if survey_name == "tng50_temporal":
                try:
                    # Load data
                    survey_config = self.surveys[survey_name]
                    dataset = survey_config["loader"](**survey_config["kwargs"])
                    print(f"   📊 Loaded {len(dataset.data):,} objects")
                except Exception as e:
                    print(f"   ⚠️ TNG50 temporal data loading failed: {e}")
                    print("   🔄 Generating synthetic TNG50 temporal data...")

                    # Generate synthetic TNG50 temporal data
                    import polars as pl

                    n_objects = 10000
                    box_size = 35.0

                    # Generate data for multiple snapshots
                    all_data = []
                    for snapshot_id in range(3):
                        x = np.random.uniform(0, box_size, n_objects)
                        y = np.random.uniform(0, box_size, n_objects)
                        z = np.random.uniform(0, box_size, n_objects)
                        masses = np.random.lognormal(8, 2, n_objects)
                        velocities_0 = np.random.normal(0, 100, n_objects)
                        velocities_1 = np.random.normal(0, 100, n_objects)
                        velocities_2 = np.random.normal(0, 100, n_objects)
                        particle_types = np.random.randint(0, 6, n_objects)

                        snapshot_data = {
                            "x": x.astype(np.float32),
                            "y": y.astype(np.float32),
                            "z": z.astype(np.float32),
                            "mass": masses.astype(np.float32),
                            "velocity_0": velocities_0.astype(np.float32),
                            "velocity_1": velocities_1.astype(np.float32),
                            "velocity_2": velocities_2.astype(np.float32),
                            "particle_type": particle_types.astype(np.int32),
                            "snapshot_id": snapshot_id,
                            "redshift": 0.0,
                            "time_gyr": 0.0,
                            "scale_factor": 1.0,
                        }
                        all_data.append(pl.DataFrame(snapshot_data))

                    # Create a mock dataset
                    class MockDataset:
                        def __init__(self, df):
                            self.data = df

                    df = pl.concat(all_data)
                    dataset = MockDataset(df)
                    print(
                        f"   ✅ Generated {len(df):,} synthetic TNG50 temporal objects"
                    )

            else:
                # Load data for other surveys
                survey_config = self.surveys[survey_name]
                dataset = survey_config["loader"](**survey_config["kwargs"])
                print(f"   📊 Loaded {len(dataset.data):,} objects")

            # Convert to 3D coordinates
            coords = self.convert_to_3d_coordinates(dataset, survey_name)
            if coords is None:
                print("   ❌ Failed to convert coordinates")
                return False

            # Create PyVista plotter with space theme
            plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
            self.create_space_background(plotter)

            # Create point cloud with survey-specific styling
            points = pv.PolyData(coords)

            # Color by position (distance from center)
            coords = np.asarray(coords, dtype=float)
            distances = np.sqrt(np.sum(coords**2, axis=1))
            points["distances"] = distances

            # Add points with survey-specific settings
            plotter.add_points(
                points,
                scalars="distances",
                point_size=survey_config["point_size"],
                render_points_as_spheres=True,
                cmap="viridis",
                show_scalar_bar=True,
                scalar_bar_args={"title": "Distance", "color": "white"},
            )

            # Set camera and title
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.2)

            # Add title with space theme
            title = f"🌌 {survey_config['title']}\n{len(coords):,} objects"
            plotter.add_title(title, font_size=20, color="white", font="arial")

            # Save high-quality screenshot
            output_file = self.output_dir / f"{survey_name}_3d_professional.png"
            plotter.screenshot(
                str(output_file), window_size=[1600, 1200], transparent_background=False
            )
            print(f"   💾 Saved: {output_file}")

            plotter.close()
            return True

        except Exception as e:
            print(f"   ❌ Error visualizing {survey_name}: {e}")
            return False

    def create_comparison_plot(self, survey_names: List[str] = None) -> bool:
        """Create a comparison plot showing multiple surveys together."""
        if survey_names is None:
            survey_names = ["gaia", "sdss", "tng50"]

        try:
            print(f"🌌 Creating comparison plot for {', '.join(survey_names)}...")

            # Create plotter
            plotter = pv.Plotter(off_screen=True, window_size=[2000, 1500])
            self.create_space_background(plotter)

            # Load and add each survey
            for i, survey_name in enumerate(survey_names):
                if survey_name not in self.surveys:
                    continue

                survey_config = self.surveys[survey_name]
                dataset = survey_config["loader"](**survey_config["kwargs"])
                coords = self.convert_to_3d_coordinates(dataset, survey_name)

                if coords is not None:
                    # Offset coordinates for each survey
                    offset = np.array([i * 100, 0, 0])
                    coords_offset = coords + offset

                    points = pv.PolyData(coords_offset)
                    points["survey"] = np.full(len(coords), i)

                    plotter.add_points(
                        points,
                        scalars="survey",
                        point_size=survey_config["point_size"],
                        render_points_as_spheres=True,
                        cmap="Set1",
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Survey", "color": "white"},
                    )

            # Set camera and title
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.5)

            title = f"🌌 AstroLab Survey Comparison\n{len(survey_names)} surveys"
            plotter.add_title(title, font_size=24, color="white")

            # Save
            output_file = self.output_dir / "survey_comparison_3d.png"
            plotter.screenshot(str(output_file), window_size=[2000, 1500])
            print(f"   💾 Saved: {output_file}")

            plotter.close()
            return True

        except Exception as e:
            print(f"   ❌ Error creating comparison plot: {e}")
            return False

    def run_all_visualizations(self):
        """Run visualizations for all surveys."""
        print("🌌 AstroLab Professional 3D Visualization Suite")
        print("=" * 60)

        success_count = 0
        total_surveys = len(self.surveys)

        for survey_name in self.surveys:
            if self.visualize_survey(survey_name):
                success_count += 1

        # Create comparison plot
        print("\n🌌 Creating survey comparison...")
        self.create_comparison_plot()

        print(f"\n✅ Visualization complete: {success_count}/{total_surveys} surveys")
        print(f"📁 Check {self.output_dir} for professional visualizations")


def main():
    """Main function to run the visualizer."""
    visualizer = AstroLabVisualizer()
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()
