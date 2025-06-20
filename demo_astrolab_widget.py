#!/usr/bin/env python3
"""
🌌 AstroLab 3D Visualization Suite

Professional astronomical data visualization with spatial tensors,
space backgrounds, and interactive 3D plots for all AstroLab surveys.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the current directory is in the path to find the src module
sys.path.insert(0, os.getcwd())

import astropy.units as u

# We need to import bpy at the top level for the script to use it.
import bpy
import imageio
import polars as pl
from astropy.coordinates import SkyCoord

from src.astro_lab.data.core import (
    load_gaia_data,
    load_sdss_data,
    load_tng50_data,
    load_tng50_temporal_data,
)
from src.astro_lab.utils.blender.core import (
    AstroPlotter,
    create_camera,
    create_cosmic_grid,
    create_light,
    create_material,
    create_text_legend,
    render_scene,
    reset_scene,
    setup_render_settings,
)

# Particle type mapping and colors for TNG50
TNG50_PARTICLE_TYPES = {0: "Gas", 1: "Dark Matter", 4: "Stars", 5: "Black Holes"}
TNG50_COLORS = {
    "Gas": [0.1, 0.2, 0.8, 1.0],
    "Dark Matter": [0.5, 0.1, 0.5, 1.0],
    "Stars": [1.0, 1.0, 0.2, 1.0],
    "Black Holes": [1.0, 0.0, 0.0, 1.0],
}


class AstroLabBlenderVisualizer:
    """Professional astronomical data visualizer using Blender."""

    def __init__(self, output_dir: str = "results"):
        if bpy:
            self.output_dir = Path(bpy.path.abspath(f"//{output_dir}"))
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(output_dir)

        self.plotter = AstroPlotter(scene_name="AstroLabScene")
        self.surveys = {
            "gaia": {
                "loader": load_gaia_data,
                "kwargs": {"max_samples": 10000},
                "title": "Gaia DR3",
            },
            "sdss": {
                "loader": load_sdss_data,
                "kwargs": {"max_samples": 10000},
                "title": "SDSS DR17",
            },
            "tng50": {
                "loader": load_tng50_data,
                "kwargs": {"max_samples": 15000},
                "title": "TNG50",
            },
            "tng50_temporal": {
                "loader": load_tng50_temporal_data,
                "kwargs": {},
                "title": "TNG50 Evolution",
            },
        }

    def _setup_blender_scene(self, title: str):
        reset_scene()
        self.plotter.setup_scene()
        create_camera(position=[15, -15, 10], target=[0, 0, 0], fov=35)
        create_light(light_type="SUN", position=[10, -10, 10], power=1.5, angle=0.2)
        create_cosmic_grid(size=50, divisions=10)

    def visualize_survey(self, survey_name: str) -> bool:
        try:
            print(f"🌌 Visualizing {survey_name.upper()} with Blender...")
            survey_config = self.surveys[survey_name]
            dataset = survey_config["loader"](**survey_config["kwargs"])
            df = dataset.data if hasattr(dataset, "data") else dataset._df_cache

            self._setup_blender_scene(survey_config["title"])

            if "tng50" in survey_name and "particle_type" in df.columns:
                particle_colors = {}
                for p_type, p_name in TNG50_PARTICLE_TYPES.items():
                    df_subset = df.filter(pl.col("particle_type") == p_type)
                    if not df_subset.is_empty():
                        color = TNG50_COLORS[p_name]
                        particle_colors[p_name] = color
                        mat = create_material(
                            name=f"{p_name}_Mat",
                            material_type="emission",
                            base_color=color[:3],
                            emission_strength=2.5,
                        )
                        self.plotter.plot_data(
                            df_subset,
                            position_cols=["x", "y", "z"],
                            material=mat,
                            scale=0.02,
                        )
                create_text_legend(particle_colors, position=(-12, 8, 0), font_size=0.6)
            else:
                mat = create_material(
                    name="SurveyMat",
                    material_type="emission",
                    base_color=[0.9, 0.7, 0.3],
                    emission_strength=2.0,
                )
                # Convert sky coords to 3D for Gaia/SDSS
                if "ra" in df.columns and "dec" in df.columns:
                    distance = (
                        df["z_phot"].to_numpy() * 3000
                        if "z_phot" in df.columns
                        else 1 / df["parallax"].to_numpy() * 1000
                    )
                    coords = SkyCoord(
                        ra=df["ra"].to_numpy() * u.degree,
                        dec=df["dec"].to_numpy() * u.degree,
                        distance=distance * u.Mpc,
                    )
                    df = df.with_columns(
                        [
                            pl.Series("x", coords.cartesian.x.value),
                            pl.Series("y", coords.cartesian.y.value),
                            pl.Series("z", coords.cartesian.z.value),
                        ]
                    )

                self.plotter.plot_data(
                    df, position_cols=["x", "y", "z"], material=mat, scale=0.05
                )

            output_file = self.output_dir / f"{survey_name}_blender.png"
            setup_render_settings(output_path=str(output_file))
            render_scene()
            print(f"   💾 Saved: {output_file}")
            return True
        except Exception as e:
            print(f"   ❌ Error visualizing {survey_name}: {e}")
            return False

    def animate_tng50_temporal(self) -> bool:
        try:
            print("🎬 Creating TNG50 temporal animation...")
            dataset = load_tng50_temporal_data()
            df = dataset.data if hasattr(dataset, "data") else dataset._df_cache
            snapshot_ids = sorted(df["snapshot_id"].unique())
            frame_files = []

            for i, snapshot_id in enumerate(snapshot_ids):
                print(f"   🔄 Rendering frame {i + 1}/{len(snapshot_ids)}...")
                self._setup_blender_scene(f"TNG50 Snap: {snapshot_id}")
                df_snapshot = df.filter(pl.col("snapshot_id") == snapshot_id)

                particle_colors = {}
                for p_type, p_name in TNG50_PARTICLE_TYPES.items():
                    df_subset = df_snapshot.filter(pl.col("particle_type") == p_type)
                    if not df_subset.is_empty():
                        color = TNG50_COLORS[p_name]
                        particle_colors[p_name] = color
                        mat = create_material(
                            name=f"{p_name}_Mat",
                            material_type="emission",
                            base_color=color[:3],
                            emission_strength=2.5,
                        )
                        self.plotter.plot_data(
                            df_subset,
                            position_cols=["x", "y", "z"],
                            material=mat,
                            scale=0.02,
                        )

                create_text_legend(particle_colors, position=(-12, 8, 0), font_size=0.6)

                frame_file = self.output_dir / f"frame_{i:03d}.png"
                setup_render_settings(output_path=str(frame_file))
                render_scene()
                frame_files.append(frame_file)

            # Create GIF
            images = [imageio.imread(f) for f in frame_files]
            gif_path = self.output_dir / "tng50_evolution.gif"
            imageio.mimsave(gif_path, images, fps=2)
            print(f"   ✅ Animation saved: {gif_path}")

            # Clean up frames
            for f in frame_files:
                os.remove(f)
            return True
        except Exception as e:
            print(f"   ❌ Error creating animation: {e}")
            return False

    def run_all_visualizations(self):
        print("🌌 AstroLab Professional Blender Visualization Suite")
        print("=" * 60)
        for survey_name in self.surveys:
            if survey_name != "tng50_temporal":
                self.visualize_survey(survey_name)
        self.animate_tng50_temporal()
        print("\n✅ All visualizations complete.")


def main():
    if bpy:
        visualizer = AstroLabBlenderVisualizer()
        visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()
