"""
Tests für SpatialTensorDict - Moderne TensorDict-basierte Spatial-Tensor-Implementierung
==================================================================================
"""

import pytest
import torch
import numpy as np
from astro_lab.tensors import SpatialTensorDict


class TestSpatialTensorDict:
    """Tests für die SpatialTensorDict-Klasse."""

    def test_spatial_creation(self):
        """Teste die Erstellung von SpatialTensorDict-Objekten."""
        n_objects = 100

        # 3D Kartesische Koordinaten
        coordinates = torch.randn(n_objects, 3) * 1000  # kpc

        spatial = SpatialTensorDict(
            coordinates,
            coordinate_system="cartesian",
            units="kpc",
            frame="galactic"
        )

        assert spatial.n_objects == n_objects
        assert spatial["coordinates"].shape == (n_objects, 3)
        assert spatial["meta", "coordinate_system"] == "cartesian"
        assert spatial["meta", "units"] == "kpc"
        assert spatial["meta", "frame"] == "galactic"

    def test_coordinate_system_validation(self):
        """Teste die Validierung verschiedener Koordinatensysteme."""
        n_objects = 50

        # Kartesische Koordinaten (x, y, z)
        cartesian = torch.randn(n_objects, 3)
        spatial_cart = SpatialTensorDict(cartesian, coordinate_system="cartesian")
        assert spatial_cart["coordinates"].shape == (n_objects, 3)

        # Sphärische Koordinaten (r, θ, φ)
        spherical = torch.rand(n_objects, 3)
        spherical[:, 1] = spherical[:, 1] * np.pi        # θ ∈ [0, π]
        spherical[:, 2] = spherical[:, 2] * 2 * np.pi    # φ ∈ [0, 2π]
        spatial_sph = SpatialTensorDict(spherical, coordinate_system="spherical")
        assert spatial_sph["coordinates"].shape == (n_objects, 3)

        # Zylindrische Koordinaten (ρ, φ, z)
        cylindrical = torch.rand(n_objects, 3)
        cylindrical[:, 1] = cylindrical[:, 1] * 2 * np.pi  # φ ∈ [0, 2π]
        spatial_cyl = SpatialTensorDict(cylindrical, coordinate_system="cylindrical")
        assert spatial_cyl["coordinates"].shape == (n_objects, 3)

    def test_equatorial_coordinates(self):
        """Teste äquatoriale Koordinaten (RA, Dec, Distance)."""
        n_objects = 200

        # RA ∈ [0, 360°], Dec ∈ [-90°, 90°], Distance > 0
        ra = torch.rand(n_objects) * 360
        dec = (torch.rand(n_objects) - 0.5) * 180
        distance = torch.rand(n_objects) * 1000 + 10  # 10-1010 pc

        equatorial_coords = torch.stack([ra, dec, distance], dim=-1)

        spatial = SpatialTensorDict(
            equatorial_coords,
            coordinate_system="equatorial",
            units=["deg", "deg", "pc"],
            frame="icrs"
        )

        assert spatial["coordinates"].shape == (n_objects, 3)
        assert spatial["meta", "frame"] == "icrs"

        # Validiere Bereichsgrenzen
        assert torch.all(spatial["coordinates"][:, 0] >= 0)     # RA >= 0
        assert torch.all(spatial["coordinates"][:, 0] <= 360)   # RA <= 360
        assert torch.all(spatial["coordinates"][:, 1] >= -90)   # Dec >= -90
        assert torch.all(spatial["coordinates"][:, 1] <= 90)    # Dec <= 90
        assert torch.all(spatial["coordinates"][:, 2] > 0)      # Distance > 0

    def test_galactic_coordinates(self):
        """Teste galaktische Koordinaten (l, b, Distance)."""
        n_objects = 150

        # l ∈ [0, 360°], b ∈ [-90°, 90°], Distance > 0
        l = torch.rand(n_objects) * 360
        b = (torch.rand(n_objects) - 0.5) * 180
        distance = torch.rand(n_objects) * 5000  # pc

        galactic_coords = torch.stack([l, b, distance], dim=-1)

        spatial = SpatialTensorDict(
            galactic_coords,
            coordinate_system="galactic",
            units=["deg", "deg", "pc"],
            frame="galactic"
        )

        assert spatial["coordinates"].shape == (n_objects, 3)
        assert spatial["meta", "coordinate_system"] == "galactic"

    def test_coordinate_transformations(self):
        """Teste Koordinatentransformationen zwischen Systemen."""
        # Einfache Kartesische Koordinaten
        cartesian = torch.tensor([
            [1.0, 0.0, 0.0],  # x-Achse
            [0.0, 1.0, 0.0],  # y-Achse
            [0.0, 0.0, 1.0]   # z-Achse
        ])

        spatial = SpatialTensorDict(cartesian, coordinate_system="cartesian")

        # Transformiere zu sphärischen Koordinaten
        spherical = spatial.to_spherical()

        assert "coordinates" in spherical
        assert spherical["coordinates"].shape == (3, 3)

        # Prüfe bekannte Transformationen
        # (1,0,0) → (r=1, θ=π/2, φ=0)
        torch.testing.assert_close(spherical["coordinates"][0, 0], torch.tensor(1.0))  # r
        torch.testing.assert_close(spherical["coordinates"][0, 1], torch.tensor(np.pi/2))  # θ
        torch.testing.assert_close(spherical["coordinates"][0, 2], torch.tensor(0.0))  # φ

        # Rücktransformation sollte Original ergeben
        cartesian_back = spatial.from_spherical(spherical["coordinates"])
        torch.testing.assert_close(
            cartesian_back["coordinates"], cartesian, atol=1e-6
        )

    def test_distance_calculations(self):
        """Teste Entfernungsberechnungen zwischen Objekten."""
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        # Paarweise Entfernungen
        distances = spatial.compute_pairwise_distances()

        assert distances.shape == (4, 4)

        # Diagonale sollte Null sein
        torch.testing.assert_close(
            torch.diag(distances), torch.zeros(4), atol=1e-6
        )

        # Bekannte Entfernungen prüfen
        torch.testing.assert_close(distances[0, 1], torch.tensor(1.0))  # (0,0,0) zu (1,0,0)
        torch.testing.assert_close(distances[0, 2], torch.tensor(1.0))  # (0,0,0) zu (0,1,0)
        torch.testing.assert_close(distances[0, 3], torch.tensor(np.sqrt(2)))  # (0,0,0) zu (1,1,0)

    def test_nearest_neighbors(self):
        """Teste die Suche nach nächsten Nachbarn."""
        # Erstelle Gitter von Punkten
        x = torch.linspace(-5, 5, 11)
        y = torch.linspace(-5, 5, 11)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        coordinates = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.zeros(xx.numel())
        ], dim=-1)

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        # Finde k nächste Nachbarn für jeden Punkt
        k = 5
        neighbors = spatial.find_nearest_neighbors(k=k)

        assert "indices" in neighbors
        assert "distances" in neighbors
        assert neighbors["indices"].shape == (coordinates.shape[0], k)
        assert neighbors["distances"].shape == (coordinates.shape[0], k)

        # Erster Nachbar sollte der Punkt selbst sein (Entfernung 0)
        torch.testing.assert_close(
            neighbors["distances"][:, 0], torch.zeros(coordinates.shape[0]), atol=1e-6
        )

    def test_spatial_clustering(self):
        """Teste räumliches Clustering."""
        # Erstelle drei Cluster
        cluster1 = torch.randn(50, 3) + torch.tensor([0.0, 0.0, 0.0])
        cluster2 = torch.randn(50, 3) + torch.tensor([10.0, 0.0, 0.0])
        cluster3 = torch.randn(50, 3) + torch.tensor([0.0, 10.0, 0.0])

        coordinates = torch.cat([cluster1, cluster2, cluster3], dim=0)

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        # DBSCAN-Clustering
        clustering = spatial.cluster_dbscan(eps=2.0, min_samples=5)

        assert "labels" in clustering
        assert clustering["labels"].shape == (150,)

        # Sollte mindestens 3 Cluster finden (Labels 0, 1, 2)
        unique_labels = torch.unique(clustering["labels"])
        n_clusters = len(unique_labels[unique_labels >= 0])  # -1 ist Rauschen
        assert n_clusters >= 3

    def test_spatial_statistics(self):
        """Teste räumliche Statistiken."""
        coordinates = torch.randn(1000, 3) * 10

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        stats = spatial.compute_spatial_statistics()

        assert "center_of_mass" in stats
        assert "bounding_box" in stats
        assert "extent" in stats
        assert "density" in stats

        # Schwerpunkt sollte nahe Null sein (zufällige Verteilung)
        torch.testing.assert_close(
            stats["center_of_mass"], torch.zeros(3), atol=2.0
        )

        # Bounding Box sollte alle Punkte umfassen
        bbox = stats["bounding_box"]
        assert bbox["min"].shape == (3,)
        assert bbox["max"].shape == (3,)
        assert torch.all(bbox["min"] <= torch.min(coordinates, dim=0).values)
        assert torch.all(bbox["max"] >= torch.max(coordinates, dim=0).values)

    def test_convex_hull(self):
        """Teste die Berechnung der konvexen Hülle."""
        # Einfaches Tetraeder
        coordinates = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0]
        ])

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        hull = spatial.compute_convex_hull()

        assert "vertices" in hull
        assert "faces" in hull
        assert "volume" in hull

        # Alle Original-Punkte sollten Eckpunkte der konvexen Hülle sein
        assert hull["vertices"].shape[0] == 4

    def test_voronoi_tessellation(self):
        """Teste Voronoi-Tessellation."""
        # Regelmäßiges Gitter von Punkten
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ])

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        voronoi = spatial.compute_voronoi_tessellation()

        assert "regions" in voronoi
        assert "ridge_points" in voronoi
        assert "volumes" in voronoi

        # Sollte für jeden Punkt eine Region haben
        assert len(voronoi["regions"]) == coordinates.shape[0]

    def test_gravitational_potential(self):
        """Teste die Berechnung des Gravitationspotentials."""
        # Punktmassen
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        masses = torch.tensor([1.0, 0.5, 0.3])  # Einheiten von M_sun

        spatial = SpatialTensorDict(
            coordinates, 
            coordinate_system="cartesian",
            masses=masses
        )

        # Testpunkte für Potentialberechnung
        test_points = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [2.0, 2.0, 0.0]
        ])

        potential = spatial.compute_gravitational_potential(test_points)

        assert potential.shape == (3,)

        # Potential sollte negativer sein (stärker gebunden) näher zu den Massen
        assert potential[0] < potential[2]  # Näher zu Masse 1
        assert potential[1] < potential[2]  # Näher zu Masse 2

    def test_tidal_forces(self):
        """Teste die Berechnung von Gezeitenkräften."""
        # Zentralmasse und Testpartikel
        central_mass = torch.tensor([0.0, 0.0, 0.0])
        test_particles = torch.tensor([
            [1.0, 0.0, 0.0],   # Radial
            [0.0, 1.0, 0.0],   # Tangential
            [0.0, 0.0, 1.0]    # Vertikal
        ])

        coordinates = torch.cat([central_mass.unsqueeze(0), test_particles], dim=0)
        masses = torch.tensor([1.0, 0.001, 0.001, 0.001])  # Zentrale Masse viel größer

        spatial = SpatialTensorDict(
            coordinates,
            coordinate_system="cartesian",
            masses=masses
        )

        # Berechne Gezeitenkräfte auf Testpartikel
        tidal_forces = spatial.compute_tidal_forces(central_body_index=0)

        assert tidal_forces.shape == (3, 3)

        # Radiale Kraft sollte anziehend sein
        assert tidal_forces[0, 0] < 0  # Kraft in -x Richtung

        # Tangentiale Kräfte sollten schwächer sein
        assert abs(tidal_forces[1, 0]) < abs(tidal_forces[0, 0])

    def test_coordinate_frame_transformations(self):
        """Teste Transformationen zwischen Referenzsystemen."""
        # ICRS-Koordinaten (äquatorial)
        ra = torch.tensor([0.0, 90.0, 180.0])  # deg
        dec = torch.tensor([0.0, 45.0, -30.0])  # deg
        distance = torch.tensor([100.0, 200.0, 150.0])  # pc

        icrs_coords = torch.stack([ra, dec, distance], dim=-1)

        spatial_icrs = SpatialTensorDict(
            icrs_coords,
            coordinate_system="equatorial",
            frame="icrs",
            units=["deg", "deg", "pc"]
        )

        # Transformiere zu galaktischen Koordinaten
        galactic = spatial_icrs.transform_frame("galactic")

        assert galactic["coordinates"].shape == (3, 3)
        assert galactic["meta", "frame"] == "galactic"

        # Galaktische Breite sollte in [-90°, 90°] sein
        assert torch.all(galactic["coordinates"][:, 1] >= -90)
        assert torch.all(galactic["coordinates"][:, 1] <= 90)

    def test_proper_motion_integration(self):
        """Teste Integration von Eigenbewegungen."""
        # Startpositionen
        positions = torch.tensor([
            [10.0, 20.0, 100.0],   # RA, Dec, Distance
            [15.0, -10.0, 200.0]
        ])

        # Eigenbewegungen (mas/yr, mas/yr, km/s)
        proper_motions = torch.tensor([
            [5.0, -2.0, 10.0],     # μ_α, μ_δ, v_r
            [-3.0, 8.0, -5.0]
        ])

        spatial = SpatialTensorDict(
            positions,
            coordinate_system="equatorial",
            proper_motions=proper_motions,
            units=["deg", "deg", "pc"]
        )

        # Propagiere über 1000 Jahre
        time_years = 1000.0
        future_positions = spatial.propagate_proper_motion(time_years)

        assert future_positions["coordinates"].shape == (2, 3)

        # Positionen sollten sich geändert haben
        assert not torch.allclose(
            future_positions["coordinates"], positions, atol=1e-6
        )

    def test_batch_operations(self):
        """Teste Batch-Operationen mit vielen Objekten."""
        n_objects = 50000
        coordinates = torch.randn(n_objects, 3) * 1000  # kpc

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        # Alle Operationen sollten effizient funktionieren
        stats = spatial.compute_spatial_statistics()

        assert stats["center_of_mass"].shape == (3,)
        assert stats["extent"] > 0

        # Transformationen sollten vektorisiert funktionieren
        spherical = spatial.to_spherical()
        assert spherical["coordinates"].shape == (n_objects, 3)

    def test_gpu_compatibility(self):
        """Teste GPU-Kompatibilität falls verfügbar."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        coordinates = torch.randn(1000, 3, device="cuda")

        spatial = SpatialTensorDict(coordinates, coordinate_system="cartesian")

        assert spatial["coordinates"].device.type == "cuda"

        # Operationen sollten auf GPU funktionieren
        spherical = spatial.to_spherical()
        stats = spatial.compute_spatial_statistics()

        assert spherical["coordinates"].device.type == "cuda"
        assert stats["center_of_mass"].device.type == "cuda"

    def test_serialization(self):
        """Teste Serialisierung und Deserialisierung."""
        coordinates = torch.randn(100, 3)
        masses = torch.rand(100)

        original_spatial = SpatialTensorDict(
            coordinates,
            coordinate_system="cartesian",
            units="kpc",
            frame="galactic",
            masses=masses
        )

        # Serialisiere zu Dictionary
        spatial_dict = original_spatial.to_dict()

        # Deserialisiere zurück
        restored_spatial = SpatialTensorDict.from_dict(spatial_dict)

        assert restored_spatial.n_objects == original_spatial.n_objects
        assert restored_spatial["meta", "coordinate_system"] == "cartesian"
        assert restored_spatial["meta", "units"] == "kpc"
        assert restored_spatial["meta", "frame"] == "galactic"

        torch.testing.assert_close(
            restored_spatial["coordinates"],
            original_spatial["coordinates"]
        )
        torch.testing.assert_close(
            restored_spatial["masses"],
            original_spatial["masses"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
