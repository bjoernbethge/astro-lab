"""
TensorDict-Integrations-Tests
============================

Tests für die Integration zwischen verschiedenen TensorDict-Klassen
und komplexe Arbeitsabläufe mit mehreren Tensor-Typen.
"""

import pytest
import torch
import numpy as np
from astro_lab.tensors import (
    SpatialTensorDict, PhotometricTensorDict, OrbitTensorDict,
    FeatureTensorDict, ClusteringTensorDict, SurveyTensorDict,
    CrossMatchTensorDict, AstroTensorDict
)


class TestTensorDictIntegration:
    """Tests für die Integration verschiedener TensorDict-Klassen."""

    def test_survey_data_workflow(self):
        """Teste kompletten Survey-Daten-Workflow."""
        n_objects = 1000

        # 1. Räumliche Koordinaten
        ra = torch.rand(n_objects) * 360
        dec = (torch.rand(n_objects) - 0.5) * 180
        distance = torch.rand(n_objects) * 1000 + 10
        coordinates = torch.stack([ra, dec, distance], dim=-1)

        spatial = SpatialTensorDict(
            coordinates,
            coordinate_system="equatorial",
            frame="icrs",
            units=["deg", "deg", "pc"]
        )

        # 2. Photometrische Daten
        bands = ["u", "g", "r", "i", "z"]
        magnitudes = torch.randn(n_objects, len(bands)) + 20
        errors = torch.rand(n_objects, len(bands)) * 0.1

        photometric = PhotometricTensorDict(
            magnitudes,
            bands=bands,
            errors=errors,
            system="AB"
        )

        # 3. Survey zusammenfügen
        survey = SurveyTensorDict(
            spatial=spatial,
            photometric=photometric,
            survey_name="TestSurvey",
            survey_area=100.0  # deg²
        )

        assert survey.n_objects == n_objects
        assert survey.survey_name == "TestSurvey"
        assert survey["spatial"]["coordinates"].shape == (n_objects, 3)
        assert survey["photometric"]["magnitudes"].shape == (n_objects, 5)

        # 4. Farben berechnen
        colors = survey.compute_colors([("g", "r"), ("r", "i")])
        assert "g_r" in colors
        assert "r_i" in colors

        # 5. Räumliche Statistiken
        spatial_stats = survey.compute_spatial_statistics()
        assert "density" in spatial_stats
        assert "clustering_strength" in spatial_stats

    def test_stellar_classification_workflow(self):
        """Teste Stellar-Klassifikations-Workflow."""
        n_stars = 5000

        # 1. Stellar photometry (simulate main sequence, giants, white dwarfs)
        bands = ["B", "V", "R", "I", "J", "H", "K"]
        n_bands = len(bands)

        # Main sequence stars
        ms_mags = torch.randn(3000, n_bands) + torch.tensor([12, 11, 10.5, 10, 9.5, 9, 8.8])
        # Giants
        giant_mags = torch.randn(1500, n_bands) + torch.tensor([10, 9, 8.5, 8, 7.5, 7, 6.8])
        # White dwarfs
        wd_mags = torch.randn(500, n_bands) + torch.tensor([15, 14.5, 14, 13.8, 13.5, 13.3, 13.2])

        all_mags = torch.cat([ms_mags, giant_mags, wd_mags], dim=0)
        true_labels = torch.cat([
            torch.zeros(3000),     # Main sequence
            torch.ones(1500),      # Giants  
            torch.full((500,), 2)  # White dwarfs
        ])

        photometric = PhotometricTensorDict(
            all_mags,
            bands=bands,
            system="Johnson"
        )

        # 2. Farben als Features
        colors = photometric.compute_colors([
            ("B", "V"), ("V", "R"), ("R", "I"),
            ("I", "J"), ("J", "H"), ("H", "K")
        ])

        color_data = torch.stack([
            colors["B_V"], colors["V_R"], colors["R_I"],
            colors["I_J"], colors["J_H"], colors["H_K"]
        ], dim=-1)

        features = FeatureTensorDict(
            color_data,
            feature_names=["B_V", "V_R", "R_I", "I_J", "J_H", "H_K"]
        )

        # 3. Normalisierung
        normalized_features = features.normalize(method="standard")

        # 4. Clustering
        clustering = ClusteringTensorDict(
            normalized_features["features"],
            n_clusters=3,
            method="kmeans"
        )
        clustering.fit()

        predicted_labels = clustering["labels"]

        # 5. Validierung (vereinfacht)
        assert len(torch.unique(predicted_labels)) <= 3
        assert predicted_labels.shape == (n_stars,)

    def test_galaxy_morphology_analysis(self):
        """Teste Galaxy-Morphologie-Analyse-Workflow."""
        n_galaxies = 2000

        # 1. Morphologische Features (simuliert)
        # Konzentration, Asymmetrie, Smoothness, Gini, M20, etc.
        morphology_features = torch.randn(n_galaxies, 8)
        morphology_names = [
            "concentration", "asymmetry", "smoothness", "gini",
            "m20", "ellipticity", "petrosian_radius", "sersic_index"
        ]

        features = FeatureTensorDict(
            morphology_features,
            feature_names=morphology_names
        )

        # 2. Photometrische Daten
        bands = ["u", "g", "r", "i", "z"]
        magnitudes = torch.randn(n_galaxies, len(bands)) + 22

        photometric = PhotometricTensorDict(
            magnitudes,
            bands=bands,
            system="AB"
        )

        # 3. Redshift-Information
        redshifts = torch.rand(n_galaxies) * 0.5  # z < 0.5

        # 4. Feature-Engineering
        colors = photometric.compute_colors([("u", "g"), ("g", "r"), ("r", "i")])

        # Kombiniere morphologische und photometrische Features
        combined_features = torch.cat([
            features["features"],
            torch.stack([colors["u_g"], colors["g_r"], colors["r_i"]], dim=-1),
            redshifts.unsqueeze(-1)
        ], dim=-1)

        all_feature_names = morphology_names + ["u_g", "g_r", "r_i", "redshift"]

        combined_feature_tensor = FeatureTensorDict(
            combined_features,
            feature_names=all_feature_names
        )

        # 5. Dimensionsreduktion
        reduced_features = combined_feature_tensor.reduce_dimensions(
            method="pca",
            n_components=5
        )

        assert reduced_features["features"].shape == (n_galaxies, 5)

        # 6. Morphologie-Clustering
        morphology_clustering = ClusteringTensorDict(
            reduced_features["features"],
            n_clusters=4,  # E, S0, Sa/Sb, Sc/Irr
            method="gaussian_mixture"
        )
        morphology_clustering.fit()

        assert morphology_clustering["labels"].shape == (n_galaxies,)
        assert len(torch.unique(morphology_clustering["labels"])) <= 4

    def test_crossmatch_workflow(self):
        """Teste Cross-Match-Workflow zwischen Katalogen."""
        # Katalog 1: Optische Daten
        n_optical = 10000
        ra_opt = torch.rand(n_optical) * 360
        dec_opt = (torch.rand(n_optical) - 0.5) * 180
        optical_coords = torch.stack([ra_opt, dec_opt], dim=-1)

        opt_mags = torch.randn(n_optical, 5) + 20
        optical_bands = ["u", "g", "r", "i", "z"]

        optical_spatial = SpatialTensorDict(
            torch.cat([optical_coords, torch.ones(n_optical, 1) * 1000], dim=-1),
            coordinate_system="equatorial"
        )
        optical_phot = PhotometricTensorDict(opt_mags, bands=optical_bands)

        # Katalog 2: Infrarot-Daten (nur Teilmenge überlappt)
        n_ir = 3000
        # 70% Überlappung mit optischem Katalog
        n_overlap = int(0.7 * n_ir)

        # Überlappende Objekte (mit kleinem Positionsfehler)
        overlap_indices = torch.randperm(n_optical)[:n_overlap]
        ra_ir_overlap = ra_opt[overlap_indices] + torch.randn(n_overlap) * 0.001  # 3.6" scatter
        dec_ir_overlap = dec_opt[overlap_indices] + torch.randn(n_overlap) * 0.001

        # Neue IR-Objekte
        ra_ir_new = torch.rand(n_ir - n_overlap) * 360
        dec_ir_new = (torch.rand(n_ir - n_overlap) - 0.5) * 180

        ra_ir = torch.cat([ra_ir_overlap, ra_ir_new])
        dec_ir = torch.cat([dec_ir_overlap, dec_ir_new])
        ir_coords = torch.stack([ra_ir, dec_ir], dim=-1)

        ir_mags = torch.randn(n_ir, 3) + 18
        ir_bands = ["J", "H", "K"]

        ir_spatial = SpatialTensorDict(
            torch.cat([ir_coords, torch.ones(n_ir, 1) * 1000], dim=-1),
            coordinate_system="equatorial"
        )
        ir_phot = PhotometricTensorDict(ir_mags, bands=ir_bands)

        # Cross-Match durchführen
        crossmatch = CrossMatchTensorDict(
            catalog1_spatial=optical_spatial,
            catalog2_spatial=ir_spatial,
            match_radius=5.0  # arcsec
        )

        matches = crossmatch.find_matches()

        assert "matches" in matches
        assert "distances" in matches
        assert matches["matches"].shape[1] == 2  # index1, index2

        # Sollte etwa n_overlap Matches finden
        n_matches = matches["matches"].shape[0]
        assert n_matches > 0.5 * n_overlap  # Mindestens 50% der erwarteten Matches
        assert n_matches < 1.5 * n_overlap  # Nicht zu viele falsche Matches

    def test_orbit_analysis_workflow(self):
        """Teste Orbit-Analyse-Workflow für Asteroiden/Kometen."""
        n_objects = 500

        # 1. Orbital-Elemente für verschiedene Populations
        # Near-Earth Asteroids
        nea_elements = torch.rand(200, 6)
        nea_elements[:, 0] = 0.5 + torch.rand(200) * 1.5  # a ∈ [0.5, 2.0] AU
        nea_elements[:, 1] = torch.rand(200) * 0.8         # e ∈ [0, 0.8]

        # Main Belt Asteroids  
        mba_elements = torch.rand(250, 6)
        mba_elements[:, 0] = 2.1 + torch.rand(250) * 1.4  # a ∈ [2.1, 3.5] AU
        mba_elements[:, 1] = torch.rand(250) * 0.3         # e ∈ [0, 0.3]

        # Jupiter Trojans
        trojan_elements = torch.rand(50, 6)
        trojan_elements[:, 0] = 5.2 + torch.randn(50) * 0.1  # a ≈ 5.2 AU (Jupiter)
        trojan_elements[:, 1] = torch.rand(50) * 0.2          # e ∈ [0, 0.2]

        all_elements = torch.cat([nea_elements, mba_elements, trojan_elements])
        true_populations = torch.cat([
            torch.zeros(200),       # NEA
            torch.ones(250),        # MBA
            torch.full((50,), 2)    # Trojans
        ])

        orbits = OrbitTensorDict(
            all_elements,
            central_body="sun",
            element_type="keplerian"
        )

        # 2. Orbital-Parameter berechnen
        periods = orbits.compute_period()
        eccentricities = orbits["elements"][:, 1]
        inclinations = orbits["elements"][:, 2]

        # 3. Tisserand-Parameter (für Komet-Klassifikation)
        tisserand = orbits.compute_tisserand_parameter(planet="jupiter")

        # 4. Feature-Vektor für Klassifikation
        orbital_features = torch.stack([
            orbits["elements"][:, 0],  # semi-major axis
            eccentricities,
            inclinations,
            periods / 365.25,         # Period in years
            tisserand
        ], dim=-1)

        features = FeatureTensorDict(
            orbital_features,
            feature_names=["a", "e", "i", "period_years", "tisserand"]
        )

        # 5. Normalisierung und Clustering
        normalized = features.normalize(method="minmax")

        clustering = ClusteringTensorDict(
            normalized["features"],
            n_clusters=3,
            method="kmeans"
        )
        clustering.fit()

        predicted_populations = clustering["labels"]

        # 6. Validierung der Klassifikation
        # NEAs sollten kleine a, hohe e haben
        nea_mask = predicted_populations == 0
        if torch.sum(nea_mask) > 0:
            nea_a = orbital_features[nea_mask, 0]
            assert torch.mean(nea_a) < 2.5  # Typisch für NEAs

    def test_performance_with_large_datasets(self):
        """Teste Performance mit großen Datensätzen."""
        n_objects = 100000

        # Große räumliche Daten
        coordinates = torch.randn(n_objects, 3) * 1000
        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        # Große photometrische Daten
        magnitudes = torch.randn(n_objects, 10) + 20
        bands = [f"band_{i}" for i in range(10)]
        photometric = PhotometricTensorDict(magnitudes, bands=bands)

        # Performance-Test: Batch-Operationen
        import time

        start_time = time.time()

        # Räumliche Statistiken
        spatial_stats = spatial.compute_spatial_statistics()

        # Photometrische Farben
        color_pairs = [(f"band_{i}", f"band_{i+1}") for i in range(9)]
        colors = photometric.compute_colors(color_pairs)

        end_time = time.time()
        computation_time = end_time - start_time

        # Sollte für 100k Objekte unter 10 Sekunden dauern
        assert computation_time < 10.0

        # Ergebnisse sollten korrekt sein
        assert spatial_stats["center_of_mass"].shape == (3,)
        assert len(colors) == 9

    def test_gpu_acceleration_workflow(self):
        """Teste GPU-beschleunigte Workflows falls verfügbar."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        n_objects = 50000
        device = "cuda"

        # GPU-Daten erstellen
        coordinates = torch.randn(n_objects, 3, device=device) * 1000
        magnitudes = torch.randn(n_objects, 5, device=device) + 20

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")
        photometric = PhotometricTensorDict(
            magnitudes, 
            bands=["u", "g", "r", "i", "z"]
        )

        # Alle Operationen sollten auf GPU bleiben
        assert spatial["coordinates"].device.type == "cuda"
        assert photometric["magnitudes"].device.type == "cuda"

        # Komplexe Operationen
        colors = photometric.compute_colors([("g", "r"), ("r", "i")])
        spatial_stats = spatial.compute_spatial_statistics()

        assert colors["g_r"].device.type == "cuda"
        assert spatial_stats["center_of_mass"].device.type == "cuda"

    def test_interoperability_with_external_libraries(self):
        """Teste Interoperabilität mit externen Bibliotheken."""
        n_objects = 1000

        # TensorDict-Daten
        coordinates = torch.randn(n_objects, 3)
        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        # Export zu NumPy
        numpy_coords = spatial.to_numpy()
        assert isinstance(numpy_coords["coordinates"], np.ndarray)
        assert numpy_coords["coordinates"].shape == (n_objects, 3)

        # Export zu pandas (falls verfügbar)
        try:
            import pandas as pd
            df = spatial.to_pandas()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == n_objects
        except ImportError:
            pass  # pandas nicht verfügbar

        # Export zu Astropy (falls verfügbar)
        try:
            import astropy.coordinates as coord
            skycoord = spatial.to_astropy_skycoord()
            assert len(skycoord) == n_objects
        except ImportError:
            pass  # astropy nicht verfügbar

    def test_serialization_of_complex_workflows(self):
        """Teste Serialisierung komplexer Multi-Tensor-Workflows."""
        # Erstelle komplexen Workflow-State
        n_objects = 1000

        coordinates = torch.randn(n_objects, 3)
        magnitudes = torch.randn(n_objects, 5) + 20

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")
        photometric = PhotometricTensorDict(
            magnitudes, 
            bands=["u", "g", "r", "i", "z"]
        )

        # Survey zusammenstellen
        survey = SurveyTensorDict(
            spatial=spatial,
            photometric=photometric,
            survey_name="ComplexWorkflowTest"
        )

        # Verarbeitung
        colors = survey.compute_colors([("g", "r"), ("r", "i")])
        feature_data = torch.stack([colors["g_r"], colors["r_i"]], dim=-1)

        features = FeatureTensorDict(feature_data, ["g_r", "r_i"])
        clustering = ClusteringTensorDict(feature_data, n_clusters=3)
        clustering.fit()

        # Workflow-Zustand serialisieren
        workflow_state = {
            "survey": survey.to_dict(),
            "features": features.to_dict(),
            "clustering": clustering.to_dict(),
            "metadata": {
                "n_objects": n_objects,
                "processing_steps": ["color_computation", "feature_extraction", "clustering"],
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }

        # Validiere Serialisierung
        assert "survey" in workflow_state
        assert "features" in workflow_state
        assert "clustering" in workflow_state
        assert workflow_state["metadata"]["n_objects"] == n_objects

        # Deserialisierung testen
        restored_survey = SurveyTensorDict.from_dict(workflow_state["survey"])
        restored_features = FeatureTensorDict.from_dict(workflow_state["features"])

        assert restored_survey.n_objects == n_objects
        assert restored_features.n_features == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
