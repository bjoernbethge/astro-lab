"""
🌌 AstroLab Widget: Simple Interactive Astronomical Visualization

Combines Polars, Astropy, and PyVista for high-performance astronomical
data analysis and 3D visualization, with a direct Blender API bridge.
"""

from pathlib import Path
from typing import Optional, Union, Any, Callable

import numpy as np
import polars as pl
import pyvista as pv

# Centralized Blender/Astropy availability checks
from ..utils.blender import BLENDER_AVAILABLE, AstroLabApi, bpy
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18 as cosmo

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Import bidirectional bridge
try:
    from ..utils.viz.bidirectional_bridge import (
        BidirectionalPyVistaBlenderBridge,
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
        print(f"📊 Generating {self.num_galaxies} galaxies with Polars...")
        np.random.seed(42)

        # Generate base data
        ra = np.random.uniform(130, 230, self.num_galaxies)  # SDSS Stripe 82
        dec = np.random.uniform(-1.25, 1.25, self.num_galaxies)
        redshift = np.clip(np.random.gamma(2, 0.05, self.num_galaxies), 0.01, 0.5)

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

        print("✅ Polars DataFrame created.")
        return df

    def get_3d_coordinates(self):
        """Converts 2D + Redshift to 3D coordinates using Astropy."""
        if not ASTROPY_AVAILABLE:
            print("⚠️ Astropy not available - using simplified coordinates.")
            # Fallback without Astropy
            coords = np.column_stack(
                [
                    self.galaxy_df["ra"].to_numpy(),
                    self.galaxy_df["dec"].to_numpy(),
                    self.galaxy_df["redshift"].to_numpy() * 1000,  # Scaling
                ]
            )
            return coords, None

        print("🔭 Converting to 3D coordinates with Astropy...")

        # Convert redshift to distance
        distance = cosmo.comoving_distance(self.galaxy_df["redshift"].to_numpy())

        # Create SkyCoord object
        sky_coords = SkyCoord(
            ra=self.galaxy_df["ra"].to_numpy() * u.deg,
            dec=self.galaxy_df["dec"].to_numpy() * u.deg,
            distance=distance,
            frame="icrs",
        )

        # Convert to cartesian coordinates
        coords_xyz = sky_coords.cartesian.xyz.to(u.Mpc).value.T

        print("✅ 3D coordinates calculated.")
        return coords_xyz, sky_coords

    def create_visualization(self):
        """Creates an interactive 3D visualization with PyVista."""
        print("🎨 Creating 3D visualization...")

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
        if not BLENDER_AVAILABLE:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            print("⚠️ Blender not available - API access disabled.")
            return

        try:
            self.al = AstroLabApi()
            self.ops = bpy.ops
            self.data = bpy.data
            self.context = bpy.context
            self.scene = bpy.context.scene
            print("✅ Blender API connected. Access via `widget.al`.")
        except Exception as e:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            print(f"❌ Failed to connect Blender API: {e}")
            
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
            print("❌ Bidirectional bridge not available")
            return None
            
        try:
            return self.bridge.pyvista_to_blender(mesh, name)
        except Exception as e:
            print(f"❌ PyVista to Blender conversion failed: {e}")
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
            print("❌ Bidirectional bridge not available")
            return None
            
        try:
            return self.bridge.blender_to_pyvista(obj)
        except Exception as e:
            print(f"❌ Blender to PyVista conversion failed: {e}")
            return None

    def sync_mesh(self, source: Union["pv.PolyData", Any], target: Union["pv.PolyData", Any]):
        """
        Synchronize mesh data between PyVista and Blender objects.
        
        Args:
            source: Source mesh (PyVista or Blender)
            target: Target mesh (PyVista or Blender)
        """
        if not self.bidirectional_bridge_available():
            print("❌ Bidirectional bridge not available")
            return
            
        try:
            self.bridge.sync_mesh(source, target)
            print("✅ Mesh synchronized")
        except Exception as e:
            print(f"❌ Mesh synchronization failed: {e}")

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
            print("❌ Bidirectional bridge not available")
            return False
            
        if not self.blender_available():
            print("❌ Blender API not available")
            return False
            
        try:
            success = self.bridge.create_live_sync(pyvista_plotter, self.scene, sync_interval)
            if success:
                print(f"✅ Live sync started (interval: {sync_interval}s)")
            return success
        except Exception as e:
            print(f"❌ Failed to start live sync: {e}")
            return False

    def stop_live_sync(self):
        """Stop live synchronization."""
        if self.bidirectional_bridge_available():
            self.bridge.stop_live_sync()
            print("✅ Live sync stopped")

    def add_sync_callback(self, callback: Callable):
        """Add a callback function to be executed after each sync."""
        if self.bridge:
            self.bridge.add_callback(callback)

    def blender_available(self) -> bool:
        """Check if the Blender API is available."""
        return self.al is not None

    def create_blender_scene(self, clear_existing: bool = True):
        """
        Creates a basic Blender scene with camera and lighting.

        Parameters:
        -----------
        clear_existing : bool
            If True, clears the existing Blender scene.
        """
        if not self.blender_available():
            print("Blender not available. Cannot create scene.")
            return

        print("🎬 Creating Blender scene...")
        if clear_existing:
            self.al.core['reset_scene']()

        self.al.core['create_camera'](location=(0, -30, 10), target=(0, 0, 0))
        self.al.core['create_light'](light_type='SUN', location=(10, 20, 10), energy=2.5)
        self.al.core['create_cosmic_grid'](scale=(10, 10, 10))
        print("✅ Blender scene created.")

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
            print("Blender not available. Cannot add data.")
            return

        print("🌠 Adding astronomical data to Blender...")
        coords, _ = self.get_3d_coordinates()

        if coords is None or len(coords) == 0:
            print("No 3D coordinate data to add.")
            return

        # Use the advanced geometry node visualizer from the API
        self.al.advanced.create_point_cloud_visualizer(
            points=coords,
            obj_name="AstronomicalData",
            point_size=point_size,
            use_colors=use_colors,
            redshift_data=self.galaxy_df["redshift"].to_numpy() if use_colors else None
        )

        print("✅ Data added to Blender scene.")

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
            print("Blender not available. Cannot export scene.")
            return

        print(f"💾 Exporting Blender scene to {filename}...")
        try:
            self.ops.wm.save_as_mainfile(filepath=str(Path(filename).resolve()))
            print("✅ Scene exported successfully.")
        except Exception as e:
            print(f"❌ Failed to export Blender scene: {e}")

    def load_real_data(self, data_source: Union[str, Path]):
        """Lädt echte astronomische Daten mit bestehenden Loadern. Für TNG50 werden alle Partikeltypen kombiniert."""
        data_path = Path(data_source)

        print(f"📂 Lade Daten: {data_path}")

        # Verwende bestehende AstroLab-Loader
        if "gaia" in str(data_path).lower():
            tensor = load_gaia_data(max_samples=10000)
            self.galaxy_df = self._tensor_to_polars(tensor, "gaia")
        elif "sdss" in str(data_path).lower():
            tensor = load_sdss_data(max_samples=10000)
            self.galaxy_df = self._tensor_to_polars(tensor, "sdss")
        elif "tng50" in str(data_path).lower():
            # Alle Partikeltypen laden und kombinieren
            particle_types = ["PartType0", "PartType1", "PartType4", "PartType5"]
            dfs = []
            for ptype in particle_types:
                try:
                    tensor = load_tng50_data(max_samples=10000, particle_type=ptype)
                    df = self._tensor_to_polars(tensor, "tng50")
                    df = df.with_columns(pl.lit(ptype).alias("particle_type"))
                    dfs.append(df)
                    print(f"   ✅ {ptype}: {len(df):,} Partikel geladen")
                except Exception as e:
                    print(f"   ⚠️  {ptype} konnte nicht geladen werden: {e}")
            if dfs:
                self.galaxy_df = pl.concat(dfs, how="vertical")
            else:
                raise RuntimeError("Keine TNG50-Partikeltypen konnten geladen werden!")
        else:
            # Fallback: direkt mit Polars laden
            if data_path.suffix == ".parquet":
                self.galaxy_df = pl.read_parquet(data_path)
            elif data_path.suffix == ".csv":
                self.galaxy_df = pl.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported format: {data_path.suffix}")

        print(f"✅ {len(self.galaxy_df):,} Objekte geladen")

    def _tensor_to_polars(self, tensor, survey_type: str) -> pl.DataFrame:
        """Konvertiert AstroLab-Tensor zu Polars DataFrame."""
        if hasattr(tensor, "data"):
            data = tensor.data.numpy()
        elif hasattr(tensor, "positions"):
            data = tensor.positions.numpy()
        else:
            data = tensor.numpy()

        # Basis-Schema abhängig vom Survey-Typ
        if survey_type == "gaia":
            columns = ["ra", "dec", "parallax", "pmra", "pmdec", "g_mag"]
        elif survey_type == "sdss":
            columns = ["ra", "dec", "redshift", "g_mag", "r_mag", "i_mag"]
        elif survey_type == "tng50":
            columns = ["x", "y", "z", "vx", "vy", "vz"]
        else:
            columns = [f"col_{i}" for i in range(data.shape[1])]

        # Spaltenanzahl anpassen
        columns = columns[: data.shape[1]]

        df = pl.DataFrame(data, schema=columns)

        # Feature-Engineering für astronomische Daten
        if survey_type in ["gaia", "sdss"] and "ra" in columns and "dec" in columns:
            if "redshift" not in columns:
                # Für Gaia: Redshift aus Parallax schätzen
                if "parallax" in columns:
                    df = df.with_columns(
                        (1.0 / (pl.col("parallax") * 1000)).alias("distance_pc")
                    ).with_columns(
                        (pl.col("distance_pc") / 1e6 * 0.1).alias(
                            "redshift"
                        )  # Grobe Schätzung
                    )
                else:
                    df = df.with_columns(pl.lit(0.1).alias("redshift"))

            # Stellarmasse schätzen
            if "g_mag" in columns and "r_mag" in columns:
                df = df.with_columns(
                    [
                        (pl.col("g_mag") - pl.col("r_mag")).alias("g_r_color"),
                        (10.0 + np.random.normal(0, 1, len(df))).alias(
                            "log_stellar_mass"
                        ),
                    ]
                )

        return df

    def add_interactive_controls(self):
        """Fügt interaktive Slider-Widgets hinzu."""
        if not self.plotter:
            print("⚠️  Erstelle zuerst eine Visualisierung")
            return self

        print("🎛️  Füge interaktive Controls hinzu...")

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
        """Zeigt die interaktive Visualisierung an."""
        self.plotter = self.create_visualization()

        if add_controls:
            self.add_interactive_controls()

        print("🎭 Starte interaktive Visualisierung...")
        self.plotter.show()

        return self

    def analyze_data(self):
        """Führt Datenanalyse mit Polars durch."""
        print("📈 Datenanalyse mit Polars...")

        # Basis-Statistiken
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
            print("Statistiken:")
            print(stats)

        # Spezielle astronomische Analysen
        if "redshift" in self.galaxy_df.columns:
            z_stats = self.galaxy_df.select(
                [
                    pl.col("redshift").min().alias("z_min"),
                    pl.col("redshift").max().alias("z_max"),
                    pl.col("redshift").mean().alias("z_mean"),
                    pl.col("redshift").count().alias("n_galaxies"),
                ]
            )
            print("\nRedshift-Verteilung:")
            print(z_stats)

        return self


# 🎯 Convenience Functions
def quick_demo():
    """Erstellt schnell eine Demo-Visualisierung."""
    widget = AstroLabWidget(num_galaxies=5000)
    widget.show()
    return widget


def load_and_visualize(data_path: str):
    """Lädt echte Daten und visualisiert sie."""
    widget = AstroLabWidget(data_source=data_path)
    widget.analyze_data()
    widget.show()
    return widget


def compare_surveys():
    """Compares GAIA and SDSS surveys side-by-side."""
    print("--- GAIA Survey ---")
    gaia_widget = AstroLabWidget(data_source='gaia')
    gaia_plotter = gaia_widget.show(add_controls=False)
    if gaia_plotter:
        gaia_plotter.subplot(0, 0)

    print("\n--- SDSS Survey ---")
    sdss_widget = AstroLabWidget(data_source='sdss')
    sdss_plotter = sdss_widget.show(add_controls=False)
    if sdss_plotter:
        sdss_plotter.subplot(0, 1)
        
    if gaia_plotter:
        gaia_plotter.show()


# Hauptausführung für Tests
if __name__ == "__main__":
    print("🌌 AstroLab Widget Test")

    # Demo mit simulierten Daten
    print("\n1. Demo mit simulierten Daten:")
    demo_widget = quick_demo()

    print("\n✨ Test abgeschlossen!")
