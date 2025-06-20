"""
🌌 AstroLab Widget: Einfache interaktive astronomische Visualisierung

Kombiniert Polars, Astropy und PyVista für hochperformante astronomische
Datenanalyse und 3D-Visualisierung mit Vererbung - ohne das Rad neu zu erfinden.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import pyvista as pv

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18 as cosmo

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

from ..data.core import load_gaia_data, load_sdss_data, load_tng50_data


class AstroPipeline:
    """
    🚀 State-of-the-Art Astronomie Pipeline: Polars, Astropy & PyVista

    Eine vollständige Pipeline, die Polars für Datenverarbeitung,
    Astropy für wissenschaftliche Genauigkeit und PyVista für
    interaktive 3D-Visualisierung kombiniert.
    """

    def __init__(self, num_galaxies: int = 5000):
        """Initialisiert die Pipeline mit simulierten Daten."""
        self.num_galaxies = num_galaxies
        self.galaxy_df = self._generate_survey_data()

    def _generate_survey_data(self) -> pl.DataFrame:
        """Generiert realistische astronomische Survey-Daten mit Polars."""
        print(f"📊 Generiere {self.num_galaxies} Galaxien mit Polars...")
        np.random.seed(42)

        # Basis-Daten generieren
        ra = np.random.uniform(130, 230, self.num_galaxies)  # SDSS Stripe 82
        dec = np.random.uniform(-1.25, 1.25, self.num_galaxies)
        redshift = np.clip(np.random.gamma(2, 0.05, self.num_galaxies), 0.01, 0.5)

        g_mag = 18 + 5 * redshift + np.random.normal(0, 0.5, self.num_galaxies)
        r_mag = g_mag - np.random.normal(0.5, 0.3, self.num_galaxies)
        i_mag = r_mag - np.random.normal(0.2, 0.2, self.num_galaxies)

        # Polars DataFrame mit Feature-Engineering
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

        print("✅ Polars DataFrame erstellt.")
        return df

    def get_3d_coordinates(self):
        """Konvertiert 2D + Redshift in 3D-Koordinaten mit Astropy."""
        if not ASTROPY_AVAILABLE:
            print("⚠️  Astropy nicht verfügbar - verwende vereinfachte Koordinaten")
            # Fallback ohne Astropy
            coords = np.column_stack(
                [
                    self.galaxy_df["ra"].to_numpy(),
                    self.galaxy_df["dec"].to_numpy(),
                    self.galaxy_df["redshift"].to_numpy() * 1000,  # Skalierung
                ]
            )
            return coords, None

        print("🔭 Konvertiere zu 3D-Koordinaten mit Astropy...")

        # Redshift in Distanz umrechnen
        distance = cosmo.comoving_distance(self.galaxy_df["redshift"].to_numpy())

        # SkyCoord-Objekt erstellen
        sky_coords = SkyCoord(
            ra=self.galaxy_df["ra"].to_numpy() * u.deg,
            dec=self.galaxy_df["dec"].to_numpy() * u.deg,
            distance=distance,
            frame="icrs",
        )

        # In kartesische Koordinaten umwandeln
        coords_xyz = sky_coords.cartesian.xyz.to(u.Mpc).value.T

        print("✅ 3D-Koordinaten berechnet.")
        return coords_xyz, sky_coords

    def create_visualization(self):
        """Erstellt interaktive 3D-Visualisierung mit PyVista."""
        print("🎨 Erstelle 3D-Visualisierung...")

        coords_xyz, sky_coords = self.get_3d_coordinates()

        # PyVista PolyData erstellen
        point_cloud = pv.PolyData(coords_xyz)

        # Skalare Daten hinzufügen
        point_cloud["redshift"] = self.galaxy_df["redshift"].to_numpy()
        point_cloud["log_stellar_mass"] = self.galaxy_df["log_stellar_mass"].to_numpy()
        point_cloud["g_r_color"] = self.galaxy_df["g_r_color"].to_numpy()

        # Glyphen erstellen
        glyphs = point_cloud.glyph(
            orient=False,
            scale="log_stellar_mass",
            factor=0.05,
            geom=pv.Sphere(theta_resolution=8, phi_resolution=8),
        )

        # Plotter setup
        plotter = pv.Plotter(window_size=[1200, 900])
        plotter.set_background("black")

        # Mesh hinzufügen
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
    🎛️ Interaktives AstroLab Widget

    Erweitert AstroPipeline um interaktive Widgets und
    Integration mit bestehenden AstroLab-Datenloadern.
    """

    def __init__(self, data_source: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialisiert das Widget.

        Parameters:
        -----------
        data_source : str, Path, optional
            Pfad zu Datendatei oder None für simulierte Daten
        **kwargs : dict
            Weitere Parameter für AstroPipeline
        """
        if data_source:
            self.load_real_data(data_source)
        else:
            super().__init__(**kwargs)

        self.plotter = None
        self.widgets = {}

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

    def export_to_blender(self, filename: str = "astrolab_scene.blend"):
        """Exportiert Szene zu Blender (falls verfügbar)."""
        if not BLENDER_AVAILABLE:
            print("⚠️  Blender nicht verfügbar")
            return self

        print(f"📦 Exportiere zu Blender: {filename}")

        try:
            # Bestehende Objekte löschen
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)

            # Sample für Performance
            sample_df = self.galaxy_df.sample(min(1000, len(self.galaxy_df)))
            coords, _ = self.get_3d_coordinates()

            # Objekte erstellen
            for i, row in enumerate(sample_df.iter_rows(named=True)):
                if i < len(coords):
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=0.01, location=coords[i]
                    )

            bpy.ops.wm.save_as_mainfile(filepath=filename)
            print(f"✅ Blender-Datei gespeichert: {filename}")

        except Exception as e:
            print(f"❌ Blender-Export Fehler: {e}")

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
    """Vergleicht verschiedene Surveys."""
    print("🔬 Vergleiche verschiedene astronomische Surveys...")

    # TNG50 Simulation
    print("\n--- TNG50 Simulation ---")
    tng_widget = AstroLabWidget()
    tng_widget.galaxy_df = tng_widget._tensor_to_polars(
        load_tng50_data(max_samples=2000), "tng50"
    )

    # Gaia Daten
    print("\n--- Gaia Survey ---")
    gaia_widget = AstroLabWidget()
    gaia_widget.galaxy_df = gaia_widget._tensor_to_polars(
        load_gaia_data(max_samples=2000), "gaia"
    )

    return tng_widget, gaia_widget


# Hauptausführung für Tests
if __name__ == "__main__":
    print("🌌 AstroLab Widget Test")

    # Demo mit simulierten Daten
    print("\n1. Demo mit simulierten Daten:")
    demo_widget = quick_demo()

    print("\n✨ Test abgeschlossen!")
