"""
Tests für die vollständige TensorDict-Integration
==============================================

Testet alle neuen TensorDict-Klassen und ihre Integration.
"""

import pytest
import torch
import math
from typing import List, Tuple

from .orbital_tensordict import OrbitTensorDict, ManeuverTensorDict
from .satellite_tensordict import EarthSatelliteTensorDict
from .simulation_tensordict import SimulationTensorDict, CosmologyTensorDict
from .feature_tensordict import FeatureTensorDict, StatisticsTensorDict, ClusteringTensorDict
from .crossmatch_tensordict import CrossMatchTensorDict
from .tensordict_astro import SpatialTensorDict, PhotometricTensorDict, SurveyTensorDict
from .factories import (
    create_kepler_orbits, create_asteroid_population, create_hohmann_transfer,
    create_nbody_simulation, create_cosmology_sample, create_crossmatch_example,
    merge_surveys
)


class TestOrbitTensorDict:
    """Tests für OrbitTensorDict."""

    def test_initialization(self):
        """Test der Orbit-Initialisierung."""
        elements = torch.tensor([
            [1.0, 0.1, 10.0, 0.0, 0.0, 0.0],  # Erde-ähnlicher Orbit
            [5.2, 0.05, 1.3, 0.0, 0.0, 0.0]   # Jupiter-ähnlicher Orbit
        ])

        orbit = OrbitTensorDict(elements)

        assert orbit.n_objects == 2
        assert torch.allclose(orbit.semi_major_axis, torch.tensor([1.0, 5.2]))
        assert torch.allclose(orbit.eccentricity, torch.tensor([0.1, 0.05]))

    def test_period_calculation(self):
        """Test der Periodenberechnung."""
        elements = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # 1 AU kreisförmiger Orbit
        orbit = OrbitTensorDict(elements)

        period = orbit.compute_period()
        assert torch.allclose(period, torch.tensor([1.0]), atol=1e-6)  # 1 Jahr

    def test_cartesian_conversion(self):
        """Test der Konvertierung zu kartesischen Koordinaten."""
        elements = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        orbit = OrbitTensorDict(elements)

        cartesian = orbit.to_cartesian()
        assert cartesian.shape == (1, 6)  # x, y, z, vx, vy, vz

    def test_orbit_propagation(self):
        """Test der Orbit-Propagation."""
        elements = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        orbit = OrbitTensorDict(elements)

        # Propagiere um 1 Jahr
        propagated = orbit.propagate(torch.tensor([365.25]))

        assert propagated.n_objects == 1
        assert propagated["meta", "central_body"] == "Sun"


class TestManeuverTensorDict:
    """Tests für ManeuverTensorDict."""

    def test_initialization(self):
        """Test der Manöver-Initialisierung."""
        delta_v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        time = torch.tensor([0.0, 100.0])

        maneuver = ManeuverTensorDict(delta_v, time)

        assert maneuver.n_objects == 2
        assert torch.allclose(maneuver.delta_v_magnitude, torch.tensor([1.0, 1.0]))

    def test_apply_to_orbit(self):
        """Test der Manöver-Anwendung auf Orbits."""
        # Erstelle Test-Orbit
        elements = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        orbit = OrbitTensorDict(elements)

        # Erstelle Test-Manöver
        delta_v = torch.tensor([[0.1, 0.0, 0.0]])
        time = torch.tensor([0.0])
        maneuver = ManeuverTensorDict(delta_v, time)

        # Wende Manöver an
        modified_orbit = maneuver.apply_to_orbit(orbit)

        assert modified_orbit.n_objects == 1
        assert modified_orbit["meta", "central_body"] == "Sun"


class TestEarthSatelliteTensorDict:
    """Tests für EarthSatelliteTensorDict."""

    def test_initialization(self):
        """Test der Satelliten-Initialisierung."""
        # Vereinfachte TLE-Daten: [i, RAAN, e, arg_perigee, M, n, bstar, epoch]
        tle_data = torch.tensor([
            [98.0, 0.0, 0.001, 0.0, 0.0, 14.0, 0.0, 2024.0],  # Polarer Orbit
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2024.0]       # Äquatorialer Orbit
        ])

        satellites = EarthSatelliteTensorDict(tle_data)

        assert satellites.n_objects == 2
        assert len(satellites.satellite_names) == 2
        assert satellites["meta", "reference_epoch"] == 2000.0

    def test_sgp4_propagation(self):
        """Test der SGP4-Propagation."""
        tle_data = torch.tensor([[98.0, 0.0, 0.001, 0.0, 0.0, 14.0, 0.0, 2024.0]])
        satellites = EarthSatelliteTensorDict(tle_data)

        # Propagiere um 1 Stunde
        propagated = satellites.propagate_sgp4(torch.tensor([60.0]))

        assert propagated.n_objects == 1
        assert propagated["tle_data"].shape == (1, 8)

    def test_ground_track(self):
        """Test der Bodenspur-Berechnung."""
        tle_data = torch.tensor([[98.0, 0.0, 0.001, 0.0, 0.0, 14.0, 0.0, 2024.0]])
        satellites = EarthSatelliteTensorDict(tle_data)

        longitudes, latitudes = satellites.compute_ground_track(n_points=10)

        assert longitudes.shape == (1, 10)
        assert latitudes.shape == (1, 10)
        # Prüfe ob Koordinaten im gültigen Bereich
        assert torch.all(torch.abs(longitudes) <= 180)
        assert torch.all(torch.abs(latitudes) <= 90)


class TestSimulationTensorDict:
    """Tests für SimulationTensorDict."""

    def test_initialization(self):
        """Test der Simulations-Initialisierung."""
        positions = torch.randn(10, 3)
        velocities = torch.randn(10, 3)
        masses = torch.ones(10)

        sim = SimulationTensorDict(positions, velocities, masses)

        assert sim.n_objects == 10
        assert sim["meta", "simulation_type"] == "nbody"
        assert torch.allclose(sim.masses, masses)

    def test_force_calculation(self):
        """Test der Kraft-Berechnung."""
        # Zwei-Körper-System
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = torch.zeros(2, 3)
        masses = torch.ones(2)

        sim = SimulationTensorDict(positions, velocities, masses)
        forces = sim.compute_gravitational_forces()

        assert forces.shape == (2, 3)
        # Kräfte sollten entgegengesetzt und gleich groß sein
        assert torch.allclose(forces[0], -forces[1], atol=1e-6)

    def test_energy_conservation(self):
        """Test der Energie-Erhaltung."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        masses = torch.ones(2)

        sim = SimulationTensorDict(positions, velocities, masses)

        initial_energy = sim.compute_total_energy()

        # Ein Zeitschritt
        sim_next = sim.leapfrog_step()
        final_energy = sim_next.compute_total_energy()

        # Energie sollte ungefähr erhalten bleiben
        assert torch.allclose(initial_energy, final_energy, rtol=1e-2)

    def test_center_of_mass(self):
        """Test der Schwerpunkt-Berechnung."""
        positions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        velocities = torch.zeros(2, 3)
        masses = torch.ones(2)

        sim = SimulationTensorDict(positions, velocities, masses)
        com = sim.compute_center_of_mass()

        # Schwerpunkt sollte bei Ursprung liegen
        assert torch.allclose(com, torch.zeros(3), atol=1e-6)


class TestFeatureTensorDict:
    """Tests für FeatureTensorDict."""

    def test_initialization(self):
        """Test der Feature-Initialisierung."""
        features = torch.randn(100, 5)
        feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]

        feat = FeatureTensorDict(features, feature_names)

        assert feat.n_objects == 100
        assert feat.n_features == 5
        assert feat.feature_names == feature_names

    def test_feature_access(self):
        """Test des Feature-Zugriffs."""
        features = torch.randn(100, 3)
        feature_names = ["x", "y", "z"]

        feat = FeatureTensorDict(features, feature_names)

        x_feature = feat.get_feature("x")
        assert torch.allclose(x_feature, features[:, 0])

    def test_normalization(self):
        """Test der Feature-Normalisierung."""
        features = torch.randn(100, 3) * 10 + 5  # Nicht-normalisierte Daten
        feature_names = ["a", "b", "c"]

        feat = FeatureTensorDict(features, feature_names)
        normalized = feat.normalize(method="standard")

        # Nach Standard-Normalisierung sollte Mittelwert ~0 und Std ~1 sein
        norm_features = normalized["features"]
        assert torch.allclose(torch.mean(norm_features, dim=0), torch.zeros(3), atol=1e-5)
        assert torch.allclose(torch.std(norm_features, dim=0), torch.ones(3), atol=1e-5)

    def test_feature_selection(self):
        """Test der Feature-Auswahl."""
        features = torch.randn(100, 5)
        feature_names = ["a", "b", "c", "d", "e"]

        feat = FeatureTensorDict(features, feature_names)
        selected = feat.select_features(["a", "c", "e"])

        assert selected.n_features == 3
        assert selected.feature_names == ["a", "c", "e"]


class TestStatisticsTensorDict:
    """Tests für StatisticsTensorDict."""

    def test_basic_statistics(self):
        """Test der grundlegenden Statistiken."""
        data = torch.randn(100, 3)
        stats = StatisticsTensorDict(data)

        stats.compute_basic_stats()

        assert "mean" in stats["statistics"]
        assert "std" in stats["statistics"]
        assert "min" in stats["statistics"]
        assert "max" in stats["statistics"]

    def test_percentiles(self):
        """Test der Perzentil-Berechnung."""
        data = torch.randn(1000, 2)
        stats = StatisticsTensorDict(data)

        stats.compute_percentiles([25, 50, 75])

        assert "p25" in stats["statistics"]
        assert "p50" in stats["statistics"]
        assert "p75" in stats["statistics"]


class TestClusteringTensorDict:
    """Tests für ClusteringTensorDict."""

    def test_kmeans_clustering(self):
        """Test des K-Means-Clustering."""
        # Erstelle drei klar getrennte Cluster
        cluster1 = torch.randn(30, 2) + torch.tensor([5.0, 5.0])
        cluster2 = torch.randn(30, 2) + torch.tensor([-5.0, 5.0])
        cluster3 = torch.randn(30, 2) + torch.tensor([0.0, -5.0])

        data = torch.cat([cluster1, cluster2, cluster3], dim=0)

        clustering = ClusteringTensorDict(data, n_clusters=3)
        clustering.fit_kmeans(max_iters=50)

        assert clustering["meta", "fitted"] == True
        assert clustering["labels"].shape == (90,)
        assert clustering["centroids"].shape == (3, 2)

    def test_inertia_calculation(self):
        """Test der Inertia-Berechnung."""
        data = torch.randn(100, 2)
        clustering = ClusteringTensorDict(data, n_clusters=3)
        clustering.fit_kmeans()

        inertia = clustering.compute_inertia()
        assert inertia >= 0  # Inertia muss positiv sein

    def test_prediction(self):
        """Test der Vorhersage für neue Daten."""
        data = torch.randn(100, 2)
        clustering = ClusteringTensorDict(data, n_clusters=3)
        clustering.fit_kmeans()

        new_data = torch.randn(10, 2)
        predictions = clustering.predict(new_data)

        assert predictions.shape == (10,)
        assert torch.all(predictions >= 0)
        assert torch.all(predictions < 3)


class TestCrossMatchTensorDict:
    """Tests für CrossMatchTensorDict."""

    def test_initialization(self):
        """Test der CrossMatch-Initialisierung."""
        # Erstelle zwei einfache Kataloge
        coords1 = torch.randn(100, 3)
        coords2 = torch.randn(80, 3)

        spatial1 = SpatialTensorDict(coords1)
        spatial2 = SpatialTensorDict(coords2)

        cat1 = SurveyTensorDict(spatial=spatial1, survey_name="Cat1")
        cat2 = SurveyTensorDict(spatial=spatial2, survey_name="Cat2")

        crossmatch = CrossMatchTensorDict(cat1, cat2)

        assert crossmatch["catalog1"].n_objects == 100
        assert crossmatch["catalog2"].n_objects == 80
        assert crossmatch["meta", "match_radius"] == 1.0

    def test_nearest_neighbor_match(self):
        """Test des Nearest-Neighbor-Matching."""
        # Erstelle teilweise überlappende Kataloge
        coords1 = torch.randn(50, 3)
        coords2 = coords1[:30] + torch.randn(30, 3) * 0.01  # 30 überlappende mit kleinem Fehler
        coords2 = torch.cat([coords2, torch.randn(20, 3)], dim=0)  # 20 zusätzliche

        spatial1 = SpatialTensorDict(coords1)
        spatial2 = SpatialTensorDict(coords2)

        cat1 = SurveyTensorDict(spatial=spatial1, survey_name="Cat1")
        cat2 = SurveyTensorDict(spatial=spatial2, survey_name="Cat2")

        crossmatch = CrossMatchTensorDict(cat1, cat2, match_radius=5.0)
        crossmatch.perform_crossmatch()

        # Sollte Matches finden
        assert crossmatch["meta", "n_matches"] > 0
        assert len(crossmatch["matches"]) > 0

    def test_match_statistics(self):
        """Test der Match-Statistiken."""
        coords1 = torch.randn(100, 3)
        coords2 = coords1[:50] + torch.randn(50, 3) * 0.01

        spatial1 = SpatialTensorDict(coords1)
        spatial2 = SpatialTensorDict(coords2)

        cat1 = SurveyTensorDict(spatial=spatial1, survey_name="Cat1")
        cat2 = SurveyTensorDict(spatial=spatial2, survey_name="Cat2")

        crossmatch = CrossMatchTensorDict(cat1, cat2, match_radius=10.0)
        crossmatch.perform_crossmatch()

        stats = crossmatch["meta", "match_statistics"]

        assert "completeness" in stats
        assert "mean_distance" in stats
        assert stats["completeness"] >= 0
        assert stats["completeness"] <= 1


class TestFactoryFunctions:
    """Tests für Factory-Funktionen."""

    def test_create_kepler_orbits(self):
        """Test der Kepler-Orbit-Erstellung."""
        a = torch.tensor([1.0, 5.2, 9.5])  # AU
        e = torch.tensor([0.0, 0.05, 0.1])
        i = torch.tensor([0.0, 1.3, 2.5])  # Grad

        orbits = create_kepler_orbits(a, e, i)

        assert orbits.n_objects == 3
        assert torch.allclose(orbits.semi_major_axis, a)
        assert torch.allclose(orbits.eccentricity, e)
        assert torch.allclose(orbits.inclination, i)

    def test_create_asteroid_population(self):
        """Test der Asteroiden-Population."""
        asteroids = create_asteroid_population(n_asteroids=100, belt_type="main")

        assert asteroids.n_objects == 100
        # Hauptgürtel: 2.1 - 3.3 AU
        assert torch.all(asteroids.semi_major_axis >= 2.0)
        assert torch.all(asteroids.semi_major_axis <= 3.5)

    def test_create_nbody_simulation(self):
        """Test der N-Body-Simulation."""
        sim = create_nbody_simulation(n_particles=50, system_type="cluster")

        assert sim.n_objects == 50
        assert sim["meta", "simulation_type"] == "cluster"
        assert sim.positions.shape == (50, 3)
        assert sim.velocities.shape == (50, 3)
        assert sim.masses.shape == (50,)

    def test_create_cosmology_sample(self):
        """Test der kosmologischen Probe."""
        cosmo = create_cosmology_sample(z_min=0.1, z_max=2.0, n_objects=1000)

        assert cosmo.n_objects == 1000
        assert torch.all(cosmo["redshifts"] >= 0.1)
        assert torch.all(cosmo["redshifts"] <= 2.0)

    def test_merge_surveys(self):
        """Test des Survey-Merging."""
        # Erstelle zwei Survey-TensorDicts
        coords1 = torch.randn(50, 3)
        coords2 = torch.randn(30, 3)

        spatial1 = SpatialTensorDict(coords1)
        spatial2 = SpatialTensorDict(coords2)

        survey1 = SurveyTensorDict(spatial=spatial1, survey_name="Survey1")
        survey2 = SurveyTensorDict(spatial=spatial2, survey_name="Survey2")

        merged = merge_surveys(survey1, survey2)

        assert merged.n_objects == 80
        assert "Survey1_Survey2" in merged.survey_name


class TestIntegration:
    """Integrationstests für das gesamte System."""

    def test_full_workflow(self):
        """Test eines kompletten Workflows."""
        # 1. Erstelle Gaia-ähnliche Daten
        n_objects = 1000
        coordinates = torch.randn(n_objects, 3) * 10
        g_mag = torch.randn(n_objects) + 15
        bp_mag = g_mag + torch.randn(n_objects) * 0.1 + 0.5
        rp_mag = g_mag - torch.randn(n_objects) * 0.1 + 0.3

        spatial = SpatialTensorDict(coordinates)
        magnitudes = torch.stack([g_mag, bp_mag, rp_mag], dim=-1)
        photometric = PhotometricTensorDict(magnitudes, bands=["G", "BP", "RP"])

        gaia_survey = SurveyTensorDict(
            spatial=spatial,
            photometric=photometric,
            survey_name="Gaia_Mock"
        )

        # 2. Extrahiere Features
        colors = gaia_survey.compute_colors([("BP", "RP"), ("G", "BP")])
        feature_data = torch.stack([colors["BP_RP"], colors["G_BP"]], dim=-1)
        features = FeatureTensorDict(
            feature_data, 
            feature_names=["BP_RP", "G_BP"],
            raw_data=gaia_survey
        )

        # 3. Führe Clustering durch
        clustering = ClusteringTensorDict(feature_data, n_clusters=5)
        clustering.fit_kmeans()

        # 4. Berechne Statistiken
        stats = StatisticsTensorDict(feature_data)
        stats.compute_basic_stats()
        stats.compute_percentiles()

        # Prüfe ob alles funktioniert hat
        assert gaia_survey.n_objects == n_objects
        assert features.n_features == 2
        assert clustering["meta", "fitted"] == True
        assert len(stats["statistics"]) > 0

    def test_gpu_compatibility(self):
        """Test der GPU-Kompatibilität."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Erstelle Daten auf CPU
        coordinates = torch.randn(100, 3)
        spatial = SpatialTensorDict(coordinates)

        # Übertrage zu GPU
        spatial_gpu = spatial.cuda()

        assert spatial_gpu.device.type == "cuda"
        assert spatial_gpu["coordinates"].device.type == "cuda"

        # Führe Operationen auf GPU durch
        ra, dec, distance = spatial_gpu.to_spherical()

        assert ra.device.type == "cuda"
        assert dec.device.type == "cuda"
        assert distance.device.type == "cuda"

    def test_memory_efficiency(self):
        """Test der Speicher-Effizienz."""
        # Erstelle große Datenstrukturen
        n_objects = 10000
        coordinates = torch.randn(n_objects, 3)

        spatial = SpatialTensorDict(coordinates)

        # Prüfe Speicher-Informationen
        mem_info = spatial.memory_info()

        assert mem_info["n_tensors"] >= 1
        assert mem_info["total_bytes"] > 0
        assert mem_info["total_mb"] > 0

        # Speicher sollte angemessen sein (nicht zu viel Overhead)
        expected_bytes = coordinates.numel() * coordinates.element_size()
        assert mem_info["total_bytes"] >= expected_bytes


if __name__ == "__main__":
    pytest.main([__file__])
