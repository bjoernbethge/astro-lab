"""
Survey-spezifisches Preprocessing für AstroGNN

Nutzt die bestehende Preprocessing-Pipeline aus astro_lab.data
und passt sie für Punktwolken-Verarbeitung an.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

# Import bestehende Preprocessing-Funktionen
from astro_lab.utils.config.surveys import get_survey_config

logger = logging.getLogger(__name__)


# Survey-spezifische Feature-Definitionen für Punktwolken
SURVEY_POINTCLOUD_FEATURES = {
    "gaia": {
        "position_cols": ["ra", "dec", "parallax"],  # 3D Position durch Parallaxe
        "feature_cols": ["phot_g_mean_mag", "bp_rp_color", "pmra", "pmdec"],
        "classes": {
            "stellar_type": [
                "O",
                "B",
                "A",
                "F",
                "G",
                "K",
                "M",
            ],  # Basierend auf BP-RP Farbe
            "evolution_stage": ["main_sequence", "giant", "white_dwarf", "other"],
        },
    },
    "sdss": {
        "position_cols": ["ra", "dec", "z"],  # z als Rotverschiebung → Distanz
        "feature_cols": [
            "modelMag_r",
            "modelMag_g",
            "modelMag_i",
            "petroRad_r",
            "fracDeV_r",
        ],
        "classes": {
            "galaxy_type": ["elliptical", "spiral", "irregular"],
            "morphology": ["early", "late", "peculiar"],
        },
    },
    "nsa": {
        "position_cols": ["ra", "dec", "z"],
        "feature_cols": ["mag_r", "mag_g", "mag_i", "mass", "sersic_n"],
        "classes": {"galaxy_type": ["elliptical", "spiral", "irregular"]},
    },
    "tng50": {
        "position_cols": ["x", "y", "z"],  # Direkte 3D Koordinaten
        "feature_cols": [
            "masses",
            "density",
            "velocities_0",
            "velocities_1",
            "velocities_2",
        ],
        "classes": {"particle_type": ["gas", "dark_matter", "stars", "black_holes"]},
    },
}


def preprocess_survey_for_pointcloud(
    survey: str,
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    k_neighbors: int = 8,
    **kwargs,
) -> Dict[str, Path]:
    """
    Preprocess survey data specifically for point cloud GNN analysis.

    This creates sparse point cloud data suitable for 3D graph neural networks.
    Different from the main preprocessing which focuses on tabular features.
    """
    from astro_lab.data.loaders import load_survey_catalog

    logger.info(f"🌌 Preprocessing {survey} for point cloud analysis...")

    # Load data using existing loader
    df = load_survey_catalog(survey, max_samples=max_samples)

    # Apply point cloud specific preprocessing
    lf = df.lazy()  # Convert to LazyFrame for processing

    # Collect DataFrame
    df = lf.collect()
    logger.info(f"📊 Collected {len(df)} Objekte")

    # Survey-spezifische Features
    config = SURVEY_POINTCLOUD_FEATURES.get(survey, {})
    position_cols = config.get("position_cols", ["ra", "dec"])
    feature_cols = config.get("feature_cols", [])

    # Extrahiere Positionen
    positions = extract_3d_positions(df, survey, position_cols)

    # Normalisiere wenn gewünscht
    if kwargs.get("normalize_positions", True):
        positions = normalize_to_unit_sphere(positions)

    # Extrahiere Features
    features = extract_features(
        df, feature_cols, kwargs.get("add_velocity_features", True)
    )

    # Erstelle Labels (falls verfügbar)
    labels = create_astronomical_labels(df, survey)

    # Konvertiere zu Tensoren
    position_tensor = torch.tensor(positions, dtype=torch.float32)
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = (
        torch.tensor(labels, dtype=torch.long) if labels is not None else None
    )

    result = {
        "dataframe": df,
        "positions": position_tensor,
        "features": feature_tensor,
        "labels": label_tensor,
        "survey_info": {
            "name": survey,
            "num_objects": len(df),
            "position_dim": position_tensor.shape[1],
            "feature_dim": feature_tensor.shape[1],
            "position_cols": position_cols,
            "feature_cols": feature_cols,
        },
    }

    # Speichere wenn Output-Dir angegeben
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Speichere als Parquet
        df.write_parquet(output_dir / f"{survey}_pointcloud.parquet")

        # Speichere Tensoren
        torch.save(
            {
                "positions": position_tensor,
                "features": feature_tensor,
                "labels": label_tensor,
                "survey_info": result["survey_info"],
            },
            output_dir / f"{survey}_pointcloud_tensors.pt",
        )

        logger.info(f"💾 Gespeichert in {output_dir}")

    return result


def extract_3d_positions(
    df: pl.DataFrame, survey: str, position_cols: List[str]
) -> np.ndarray:
    """
    Extrahiert 3D-Positionen aus Survey-Daten.

    Verschiedene Surveys haben unterschiedliche Koordinatensysteme:
    - Gaia: RA, Dec, Parallaxe → 3D kartesisch
    - SDSS/NSA: RA, Dec, Rotverschiebung → 3D kosmologisch
    - TNG50: Direkte kartesische Koordinaten
    """
    if survey == "gaia":
        # Konvertiere RA/Dec/Parallaxe zu 3D
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()

        # Parallaxe in Millibogensekunden → Distanz in Parsec
        if "parallax" in df.columns:
            parallax = df["parallax"].to_numpy()
            # Bereinige negative/null Parallaxen
            parallax = np.where(parallax > 0, parallax, 0.1)  # Min 0.1 mas
            distance = 1000.0 / parallax  # Distanz in Parsec
        else:
            # Fallback: Verwende feste Distanz
            distance = np.full(len(df), 100.0)  # 100 pc

        # Konvertiere zu kartesischen Koordinaten
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)

        positions = np.stack([x, y, z], axis=1)

    elif survey in ["sdss", "nsa"]:
        # Konvertiere RA/Dec/z zu 3D kosmologisch
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()

        # Rotverschiebung → Distanz (vereinfacht, ohne kosmologische Korrekturen)
        if "z" in df.columns:
            redshift = df["z"].to_numpy()
            # Hubble-Konstante: 70 km/s/Mpc
            c = 299792.458  # km/s
            H0 = 70.0  # km/s/Mpc
            distance = (c * redshift) / H0  # Mpc
        else:
            distance = np.full(len(df), 100.0)  # 100 Mpc

        # Konvertiere zu kartesischen Koordinaten
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)

        positions = np.stack([x, y, z], axis=1)

    elif survey == "tng50":
        # Direkte kartesische Koordinaten
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()
        positions = np.stack([x, y, z], axis=1)

    else:
        # Fallback: 2D → 3D mit z=0
        coords = df.select(position_cols).to_numpy()
        if coords.shape[1] == 2:
            # Füge z=0 hinzu
            z_zeros = np.zeros((coords.shape[0], 1))
            positions = np.hstack([coords, z_zeros])
        else:
            positions = coords

    return positions


def normalize_to_unit_sphere(positions: np.ndarray) -> np.ndarray:
    """
    Normalisiert Positionen auf Einheitssphäre.
    Zentriert um Schwerpunkt und skaliert auf Radius 1.
    """
    # Zentriere um Schwerpunkt
    center = np.mean(positions, axis=0)
    positions_centered = positions - center

    # Skaliere auf maximalen Radius 1
    max_radius = np.max(np.linalg.norm(positions_centered, axis=1))
    if max_radius > 0:
        positions_normalized = positions_centered / max_radius
    else:
        positions_normalized = positions_centered

    return positions_normalized


def extract_features(
    df: pl.DataFrame, feature_cols: List[str], add_velocity_features: bool = True
) -> np.ndarray:
    """
    Extrahiert Features aus DataFrame.

    Fügt optional Geschwindigkeitsfeatures hinzu (für Gaia: Eigenbewegung).
    """
    # Basis-Features
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        # Fallback: Verwende alle numerischen Spalten
        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
        available_features = numeric_cols[:10]  # Maximal 10 Features

    features = df.select(available_features).to_numpy()

    # Füge Geschwindigkeitsfeatures hinzu wenn verfügbar
    if add_velocity_features:
        velocity_features = []

        # Gaia: Eigenbewegung
        if "pmra" in df.columns and "pmdec" in df.columns:
            pmra = df["pmra"].to_numpy()
            pmdec = df["pmdec"].to_numpy()
            # Berechne Gesamtgeschwindigkeit
            pm_total = np.sqrt(pmra**2 + pmdec**2)
            velocity_features.append(pm_total)

        # TNG50: Geschwindigkeitskomponenten
        vel_cols = [col for col in df.columns if col.startswith("velocities_")]
        if vel_cols:
            for col in vel_cols:
                velocity_features.append(df[col].to_numpy())

        # Füge Geschwindigkeitsfeatures hinzu
        if velocity_features:
            velocity_array = np.column_stack(velocity_features)
            features = np.hstack([features, velocity_array])

    # Bereinige NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

    return features


def create_astronomical_labels(df: pl.DataFrame, survey: str) -> Optional[np.ndarray]:
    """
    Erstellt astronomische Labels basierend auf Survey-Typ.

    - Gaia: Sterntypen basierend auf Farbe
    - SDSS/NSA: Galaxientypen basierend auf Morphologie
    - TNG50: Partikeltypen
    """
    if survey == "gaia":
        # Klassifiziere Sterne basierend auf BP-RP Farbe
        if "bp_rp_color" in df.columns:
            bp_rp = df["bp_rp_color"].to_numpy()
            # Bins für Sterntypen (O, B, A, F, G, K, M)
            bins = [-np.inf, -0.3, 0.0, 0.3, 0.6, 0.9, 1.5, np.inf]
            labels = np.digitize(bp_rp, bins) - 1
            labels = np.clip(labels, 0, 6)  # 7 Klassen
            return labels

    elif survey in ["sdss", "nsa"]:
        # Vereinfachte Galaxienklassifikation
        # TODO: Implementiere basierend auf Sersic-Index oder anderen Parametern
        n_objects = len(df)
        # Dummy: Zufällige Klassen für Demo
        labels = np.random.randint(0, 3, size=n_objects)  # 3 Galaxientypen
        return labels

    elif survey == "tng50":
        # Partikeltypen (falls verfügbar)
        if "particle_type" in df.columns:
            return df["particle_type"].to_numpy()
        else:
            # Dummy für Demo
            n_objects = len(df)
            labels = np.random.randint(0, 4, size=n_objects)  # 4 Partikeltypen
            return labels

    return None


def create_survey_pointcloud_graph(
    positions: torch.Tensor,
    features: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    k_neighbors: int = 16,
    survey_name: str = "unknown",
) -> Data:
    """
    Erstellt einen PyG Graph aus Punktwolken-Daten.

    Nutzt GPU-beschleunigte k-NN Graph-Erstellung.
    """
    from torch_geometric.nn import knn_graph

    # Erstelle k-NN Graph
    edge_index = knn_graph(positions, k=k_neighbors, batch=None, loop=False)

    # Erstelle PyG Data Objekt
    data = Data(x=features, pos=positions, edge_index=edge_index, y=labels)

    # Füge Metadaten hinzu
    data.survey_name = survey_name
    data.num_nodes = len(positions)

    return data
