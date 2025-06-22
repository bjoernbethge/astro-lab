"""
AstroLab Widget - Main Interactive Widget for Astronomical Data Analysis
======================================================================

Provides a unified interface for astronomical data visualization, analysis,
and interactive exploration using various backends (PyVista, Blender, etc.).
"""

import logging
import numpy as np
import torch
import astropy.units as u
from astropy.coordinates import SkyCoord
import polars as pl
import pyvista as pv
from typing import Any, Dict, List, Optional, Union, Tuple

from ..data.core import create_cosmic_web_loader
from ..tensors.spatial_3d import Spatial3DTensor
from ..utils.viz.bidirectional_bridge import BidirectionalPyVistaBlenderBridge
from ..utils.blender import bpy, AstroLabApi

from pathlib import Path
from typing import Optional, Union, Any, Callable

# Configure logging
logger = logging.getLogger(__name__)

# Centralized Blender/Astropy availability checks
try:
    from astropy.cosmology import Planck18 as cosmo

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Import bidirectional bridge
try:
    from ..utils.viz.bidirectional_bridge import (
        SyncConfig,
        quick_convert_pyvista_to_blender,
        quick_convert_blender_to_pyvista
    )
    BIDIRECTIONAL_BRIDGE_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_BRIDGE_AVAILABLE = False

from ..data.core import load_gaia_data, load_sdss_data, load_tng50_data


class AstroPipeline:
    """
    🚀 State-of-the-Art Astronomy Pipeline: Polars, Astropy & PyVista

    A complete pipeline combining Polars for data processing,
    Astropy for scientific accuracy, and PyVista for
    interactive 3D visualization.
    """

    def __init__(self, num_galaxies: int = 5000):
        """Initializes the pipeline with simulated data."""
        self.num_galaxies = num_galaxies
        self.galaxy_df = self._generate_survey_data()

    def _generate_survey_data(self) -> pl.DataFrame:
        """Generates realistic astronomical survey data using Polars."""
        logger.info(f"📊 Generating {self.num_galaxies} galaxies with Polars...")
        np.random.seed(42)

        # Generate base data
        ra = np.random.uniform(130, 230, self.num_galaxies)  # SDSS Stripe 82
        dec = np.random.uniform(-1.25, 1.25, self.num_galaxies)
        redshift_raw = np.random.gamma(2, 0.05, self.num_galaxies)
        redshift = np.array(np.maximum(0.01, np.minimum(0.5, redshift_raw)))

        g_mag = 18 + 5 * redshift + np.random.normal(0, 0.5, self.num_galaxies)
        r_mag = g_mag - np.random.normal(0.5, 0.3, self.num_galaxies)
        i_mag = r_mag - np.random.normal(0.2, 0.2, self.num_galaxies)

        # Polars DataFrame with feature engineering
        df = (
            pl.DataFrame(
                {
                    "ra": ra,
                    "dec": dec,
                    "redshift": redshift,
                    "g_mag": g_mag,
                    "r_mag": r_mag,
                    "i_mag": i_mag,
                }
            )
            .with_columns(
                [
                    (pl.col("g_mag") - pl.col("r_mag")).alias("g_r_color"),
                    (pl.col("r_mag") - pl.col("i_mag")).alias("r_i_color"),
                    (
                        10.5
                        - 0.4
                        * (pl.col("r_mag") - 5 * pl.col("redshift").log10() - 42.36)
                        + 1.1 * (pl.col("g_mag") - pl.col("r_mag"))
                    ).alias("log_stellar_mass"),
                ]
            )
            .with_columns(
                pl.when(pl.col("g_r_color") > 0.7)
                .then(pl.lit("elliptical"))
                .otherwise(pl.lit("spiral"))
                .alias("galaxy_type")
            )
        )

        logger.info("✅ Polars DataFrame created.")
        return df

    def get_3d_coordinates(self):
        """Converts 2D + Redshift to 3D coordinates using Astropy."""
        if not ASTROPY_AVAILABLE:
            logger.warning("⚠️ Astropy not available - using simplified coordinates.")
            # Fallback without Astropy
            ra_array = np.array(self.galaxy_df["ra"].to_numpy())
            dec_array = np.array(self.galaxy_df["dec"].to_numpy())
            redshift_array = np.array(self.galaxy_df["redshift"].to_numpy()) * 1000  # Scaling
            coords = np.vstack([ra_array, dec_array, redshift_array]).T
            return coords, None

        logger.info("🔭 Converting to 3D coordinates with Astropy...")

        # Convert redshift to distance
        redshift_values = self.galaxy_df["redshift"].to_numpy()
        distance = cosmo.comoving_distance(redshift_values)  # type: ignore

        # Create SkyCoord object
        sky_coords = SkyCoord(
            ra=self.galaxy_df["ra"].to_numpy() * u.deg,
            dec=self.galaxy_df["dec"].to_numpy() * u.deg,
            distance=distance,
            frame="icrs",
        )

        # Convert to cartesian coordinates
        coords_xyz = sky_coords.cartesian.xyz.to(u.Mpc).value.T

        logger.info("✅ 3D coordinates calculated.")
        return coords_xyz, sky_coords

    def create_visualization(self):
        """Creates an interactive 3D visualization with PyVista."""
        logger.info("🎨 Creating 3D visualization...")

        coords_xyz, sky_coords = self.get_3d_coordinates()

        # Create PyVista PolyData
        point_cloud = pv.PolyData(coords_xyz)

        # Add scalar data
        point_cloud["redshift"] = self.galaxy_df["redshift"].to_numpy()
        point_cloud["log_stellar_mass"] = self.galaxy_df["log_stellar_mass"].to_numpy()
        point_cloud["g_r_color"] = self.galaxy_df["g_r_color"].to_numpy()

        # Create glyphs
        glyphs = point_cloud.glyph(
            orient=False,
            scale="log_stellar_mass",
            factor=0.05,
            geom=pv.Sphere(theta_resolution=8, phi_resolution=8),
        )

        # Setup plotter
        plotter = pv.Plotter(window_size=[1200, 900])
        plotter.set_background("black")

        # Add mesh
        plotter.add_mesh(
            glyphs,
            scalars="redshift",
            cmap="viridis",
            scalar_bar_args={"title": "Redshift (z)"},
        )

        plotter.add_axes()
        plotter.add_bounding_box(color="grey")
        plotter.camera_position = "xy"
        plotter.camera.zoom(1.5)

        return plotter


class AstroLabWidget(AstroPipeline):
    """
    🎛️ Interactive AstroLab Widget

    Extends AstroPipeline with interactive features and a powerful
    bridge to Blender for advanced scientific visualization.

    The Blender API is directly available via:
    - widget.al: The full AstroLab API for advanced functions.
    - widget.ops: Blender Operations (bpy.ops)
    - widget.data: Blender Data (bpy.data)
    - widget.context: Blender Context (bpy.context)
    """

    def __init__(self, data_source: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize the widget.

        Parameters:
        -----------
        data_source : str, Path, optional
            Path to data file or None for simulated data
        **kwargs : dict
            Additional parameters for AstroPipeline
        """
        if data_source:
            self.load_real_data(data_source)
        else:
            super().__init__(**kwargs)

        self.plotter = None
        self.widgets = {}
        
        # Setup Blender API with proper error handling
        self._setup_blender_api()
        
        # Setup bidirectional bridge
        self._setup_bidirectional_bridge()

    def _setup_blender_api(self):
        """
        Sets up the direct API to Blender.

        If Blender is available, this method initializes the AstroLab API
        and provides direct access to bpy's core components.
        """
        if bpy is None:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.warning("⚠️ Blender not available - API access disabled.")
            return

        try:
            self.al = AstroLabApi()
            self.ops = bpy.ops
            self.data = bpy.data
            self.context = bpy.context
            self.scene = bpy.context.scene
            logger.info("✅ Blender API connected. Access via `widget.al`.")
        except Exception as e:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.error(f"❌ Failed to connect Blender API: {e}")
            
    def _setup_bidirectional_bridge(self):
        """Initializes the PyVista-Blender bidirectional bridge."""
        if not self.blender_available() or not BIDIRECTIONAL_BRIDGE_AVAILABLE:
            self.bridge = None
            return
        
        self.bridge = BidirectionalPyVistaBlenderBridge()

    def bidirectional_bridge_available(self) -> bool:
        """Check if bidirectional bridge is available."""
        return BIDIRECTIONAL_BRIDGE_AVAILABLE and self.bridge is not None

    def pyvista_to_blender(self, mesh: "pv.PolyData", name: str = "converted_mesh") -> Optional[Any]:
        """
        Convert PyVista mesh to Blender object using bidirectional bridge.
        
        Args:
            mesh: PyVista PolyData mesh
            name: Name for the Blender object
            
        Returns:
            Blender object or None if conversion failed
        """
        if not self.bidirectional_bridge_available():
            logger.error("❌ Bidirectional bridge not available")
            return None
            
        try:
            return self.bridge.pyvista_to_blender(mesh, name)
        except Exception as e:
            logger.error(f"❌ PyVista to Blender conversion failed: {e}")
            return None

    def blender_to_pyvista(self, obj: Any) -> Optional["pv.PolyData"]:
        """
        Convert Blender object to PyVista mesh using bidirectional bridge.
        
        Args:
            obj: Blender object
            
        Returns:
            PyVista PolyData mesh or None if conversion failed
        """
        if not self.bidirectional_bridge_available():
            logger.error("❌ Bidirectional bridge not available")
            return None
            
        try:
            return self.bridge.blender_to_pyvista(obj)
        except Exception as e:
            logger.error(f"❌ Blender to PyVista conversion failed: {e}")
            return None

    def sync_mesh(self, source: Union["pv.PolyData", Any], target: Union["pv.PolyData", Any]):
        """
        Synchronize mesh data between PyVista and Blender objects.
        
        Args:
            source: Source mesh (PyVista or Blender)
            target: Target mesh (PyVista or Blender)
        """
        if not self.bidirectional_bridge_available():
            logger.error("❌ Bidirectional bridge not available")
            return
            
        try:
            self.bridge.sync_mesh(source, target)
            logger.info("✅ Mesh synchronized")
        except Exception as e:
            logger.error(f"❌ Mesh synchronization failed: {e}")

    def start_live_sync(self, pyvista_plotter: "pv.Plotter", sync_interval: float = 0.1) -> bool:
        """
        Start live synchronization between PyVista and Blender.
        
        Args:
            pyvista_plotter: PyVista plotter
            sync_interval: Sync interval in seconds
            
        Returns:
            True if live sync started successfully
        """
        if not self.bidirectional_bridge_available():
            logger.error("❌ Bidirectional bridge not available")
            return False
            
        if not self.blender_available():
            logger.error("❌ Blender API not available")
            return False
            
        try:
            success = self.bridge.create_live_sync(pyvista_plotter, self.scene, sync_interval)
            if success:
                logger.info(f"✅ Live sync started (interval: {sync_interval}s)")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to start live sync: {e}")
            return False

    def stop_live_sync(self):
        """Stop live synchronization."""
        if self.bidirectional_bridge_available():
            self.bridge.stop_live_sync()
            logger.info("✅ Live sync stopped")

    def add_sync_callback(self, callback: Callable):
        """Add a callback function to be executed after each sync."""
        if self.bridge:
            self.bridge.add_callback(callback)

    def blender_available(self) -> bool:
        """Check if Blender API is available."""
        return bpy is not None and self.al is not None

    def create_blender_scene(self, clear_existing: bool = True):
        """
        Creates a basic Blender scene with camera and lighting.

        Parameters:
        -----------
        clear_existing : bool
            If True, clears the existing Blender scene.
        """
        if not self.blender_available():
            logger.warning("Blender not available. Cannot create scene.")
            return

        logger.info("🎬 Creating Blender scene...")
        if clear_existing:
            self.al.core['reset_scene']()

        self.al.core['create_camera'](location=(0, -30, 10), target=(0, 0, 0))
        self.al.core['create_light'](light_type='SUN', location=(10, 20, 10), energy=2.5)
        self.al.core['create_cosmic_grid'](scale=(10, 10, 10))
        logger.info("✅ Blender scene created.")

    def add_astronomical_data_to_blender(
        self, point_size: float = 0.1, use_colors: bool = True
    ):
        """
        Adds the astronomical data from the pipeline to the Blender scene.

        Parameters:
        -----------
        point_size : float
            The size of the points representing stars/galaxies.
        use_colors : bool
            Whether to color points based on redshift.
        """
        if not self.blender_available():
            logger.warning("Blender not available. Cannot add data.")
            return

        logger.info("🌠 Adding astronomical data to Blender...")
        coords, _ = self.get_3d_coordinates()

        if coords is None or len(coords) == 0:
            logger.warning("No 3D coordinate data to add.")
            return

        # Use the advanced geometry node visualizer from the API
        self.al.advanced.create_point_cloud_visualizer(
            points=coords,
            obj_name="AstronomicalData",
            point_size=point_size,
            use_colors=use_colors,
            redshift_data=self.galaxy_df["redshift"].to_numpy() if use_colors else None
        )

        logger.info("✅ Data added to Blender scene.")

    def _redshift_to_color(self, redshift: float) -> tuple:
        """Convert redshift to RGB color."""
        # Simple color mapping: Blue -> Green -> Red
        if redshift < 0.1:
            return (0, 0, 1)  # Blue
        elif redshift < 0.3:
            return (0, 1, 0)  # Green
        else:
            return (1, 0, 0)  # Red

    def export_blender_scene(self, filename: str = "astronomical_scene.blend"):
        """
        Export the current Blender scene.
        
        Parameters:
        -----------
        filename : str
            Filename for export
        """
        if not self.blender_available():
            logger.warning("Blender not available. Cannot export scene.")
            return

        logger.info(f"💾 Exporting Blender scene to {filename}...")
        try:
            self.ops.wm.save_as_mainfile(filepath=str(Path(filename).resolve()))
            logger.info("✅ Scene exported successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to export Blender scene: {e}")

    def load_real_data(self, data_source: Union[str, Path]):
        """Loads real astronomical data using existing loaders. For TNG50, all particle types are combined."""
        data_path = Path(data_source)

        logger.info(f"📂 Loading data: {data_path}")

        # Use existing AstroLab loaders
        if "gaia" in str(data_path).lower():
            tensor = load_gaia_data(max_samples=10000)
            self.galaxy_df = self._tensor_to_polars(tensor, "gaia")
        elif "sdss" in str(data_path).lower():
            tensor = load_sdss_data(max_samples=10000)
            self.galaxy_df = self._tensor_to_polars(tensor, "sdss")
        elif "tng50" in str(data_path).lower():
            # Load and combine all particle types
            particle_types = ["PartType0", "PartType1", "PartType4", "PartType5"]
            dfs = []
            for ptype in particle_types:
                try:
                    tensor = load_tng50_data(max_samples=10000, particle_type=ptype)
                    df = self._tensor_to_polars(tensor, "tng50")
                    df = df.with_columns(pl.lit(ptype).alias("particle_type"))
                    dfs.append(df)
                    logger.info(f"   ✅ {ptype}: {len(df):,} particles loaded")
                except Exception as e:
                    logger.warning(f"   ⚠️  {ptype} could not be loaded: {e}")
            if dfs:
                self.galaxy_df = pl.concat(dfs, how="vertical")
            else:
                raise RuntimeError("No TNG50 particle types could be loaded!")
        else:
            # Fallback: load directly with Polars
            if data_path.suffix == ".parquet":
                self.galaxy_df = pl.read_parquet(data_path)
            elif data_path.suffix == ".csv":
                self.galaxy_df = pl.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported format: {data_path.suffix}")

        logger.info(f"✅ {len(self.galaxy_df):,} objects loaded")

    def _tensor_to_polars(self, tensor, survey_type: str) -> pl.DataFrame:
        """Converts AstroLab tensor to Polars DataFrame."""
        if hasattr(tensor, "data"):
            data = tensor.data.numpy()
        elif hasattr(tensor, "positions"):
            data = tensor.positions.numpy()
        else:
            data = tensor.numpy()

        # Base schema depending on survey type
        if survey_type == "gaia":
            schema = {
                "ra": float,
                "dec": float,
                "parallax": float,
                "pmra": float,
                "pmdec": float,
                "phot_g_mean_mag": float,
                "bp_rp": float,
                "teff_val": float,
            }
        elif survey_type == "sdss":
            schema = {
                "ra": float,
                "dec": float,
                "u": float,
                "g": float,
                "r": float,
                "i": float,
                "z": float,
                "redshift": float,
            }
        else:
            schema = {"ra": float, "dec": float, "magnitude": float}

        # Feature engineering for astronomical data
        if survey_type == "gaia":
            # For Gaia: Estimate redshift from parallax
            if "parallax" in data and data["parallax"] > 0:
                distance_pc = 1000 / data["parallax"]  # Distance in parsecs
                redshift = distance_pc * 0.0000001  # Rough estimate
            else:
                redshift = 0.0

            # Estimate stellar mass
            if "teff_val" in data and data["teff_val"] > 0:
                # Simple mass estimation from temperature
                mass = (data["teff_val"] / 5777) ** 0.5  # Solar mass units
            else:
                mass = 1.0

        df = pl.DataFrame(data, schema=schema)

        # Feature engineering for astronomical data
        if survey_type in ["gaia", "sdss"] and "ra" in schema and "dec" in schema:
            if "redshift" not in schema:
                # For Gaia: Estimate redshift from parallax
                if "parallax" in schema:
                    df = df.with_columns(
                        (1.0 / (pl.col("parallax") * 1000)).alias("distance_pc")
                    ).with_columns(
                        (pl.col("distance_pc") / 1e6 * 0.1).alias(
                            "redshift"
                        )  # Rough estimate
                    )
                else:
                    df = df.with_columns(pl.lit(0.1).alias("redshift"))

            # Estimate stellar mass
            if "phot_g_mean_mag" in schema and "bp_rp" in schema:
                df = df.with_columns(
                    [
                        (pl.col("phot_g_mean_mag") - pl.col("bp_rp")).alias("g_r_color"),
                        (10.0 + np.random.normal(0, 1, len(df))).alias(
                            "log_stellar_mass"
                        ),
                    ]
                )

        return df

    def add_interactive_controls(self):
        """Adds interactive slider widgets."""
        if not self.plotter:
            logger.warning("⚠️  Create visualization first")
            return self

        logger.info("🎛️  Adding interactive controls...")

        # Point Size Slider
        def update_point_size(value):
            # Update all mesh objects
            for actor in self.plotter.renderer.actors.values():
                if hasattr(actor, "GetProperty"):
                    actor.GetProperty().SetPointSize(value)
            self.plotter.render()

        self.widgets["point_size"] = self.plotter.add_slider_widget(
            callback=update_point_size,
            rng=[0.5, 5.0],
            value=2.0,
            title="Point Size",
            pointa=(0.02, 0.9),
            pointb=(0.3, 0.9),
            style="modern",
        )

        # Opacity Slider
        def update_opacity(value):
            for actor in self.plotter.renderer.actors.values():
                if hasattr(actor, "GetProperty"):
                    actor.GetProperty().SetOpacity(value)
            self.plotter.render()

        self.widgets["opacity"] = self.plotter.add_slider_widget(
            callback=update_opacity,
            rng=[0.1, 1.0],
            value=0.8,
            title="Opacity",
            pointa=(0.02, 0.8),
            pointb=(0.3, 0.8),
            style="modern",
        )

        return self

    def show(self, add_controls: bool = True):
        """Shows the interactive visualization."""
        self.plotter = self.create_visualization()

        if add_controls:
            self.add_interactive_controls()

        logger.info("🎭 Starting interactive visualization...")
        self.plotter.show()

        return self

    def analyze_data(self):
        """Performs data analysis with Polars."""
        logger.info("📈 Data analysis with Polars...")

        # Basic statistics
        numeric_cols = [
            col
            for col, dtype in self.galaxy_df.schema.items()
            if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        if numeric_cols:
            stats = self.galaxy_df.select(
                [
                    pl.col(numeric_cols).mean().name.suffix("_mean"),
                    pl.col(numeric_cols).std().name.suffix("_std"),
                    pl.col(numeric_cols).count().name.suffix("_count"),
                ]
            )
            logger.info("Statistics:")
            logger.info(stats)

        # Specific astronomical analyses
        if "redshift" in self.galaxy_df.columns:
            z_stats = self.galaxy_df.select(
                [
                    pl.col("redshift").min().alias("z_min"),
                    pl.col("redshift").max().alias("z_max"),
                    pl.col("redshift").mean().alias("z_mean"),
                    pl.col("redshift").count().alias("n_galaxies"),
                ]
            )
            logger.info("\nRedshift distribution:")
            logger.info(z_stats)

        return self


# 🎯 Convenience Functions
def quick_demo():
    """Creates a quick demo visualization."""
    widget = AstroLabWidget(num_galaxies=5000)
    widget.show()
    return widget


def load_and_visualize(data_path: str):
    """Loads real data and visualizes it."""
    widget = AstroLabWidget(data_source=data_path)
    widget.analyze_data()
    widget.show()
    return widget


def compare_surveys():
    """Compares GAIA and SDSS surveys side-by-side."""
    logger.info("--- GAIA Survey ---")
    gaia_widget = AstroLabWidget(data_source='gaia')
    gaia_plotter = gaia_widget.show(add_controls=False)
    if gaia_plotter:
        gaia_plotter.subplot(0, 0)

    logger.info("\n--- SDSS Survey ---")
    sdss_widget = AstroLabWidget(data_source='sdss')
    sdss_plotter = sdss_widget.show(add_controls=False)
    if sdss_plotter:
        sdss_plotter.subplot(0, 1)
        
    if gaia_plotter:
        gaia_plotter.show()


# Main execution for tests
if __name__ == "__main__":
    logger.info("🌌 AstroLab Widget Test")

    # Demo with simulated data
    logger.info("\n1. Demo with simulated data:")
    demo_widget = quick_demo()

    logger.info("\n✨ Test completed!")
