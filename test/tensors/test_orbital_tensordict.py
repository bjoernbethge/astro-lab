"""
Tests für OrbitTensorDict - Moderne TensorDict-basierte Orbital-Tensor-Implementierung
================================================================================
"""

import pytest
import torch
import numpy as np
from astro_lab.tensors import OrbitTensorDict


class TestOrbitTensorDict:
    """Tests für die OrbitTensorDict-Klasse."""

    def test_orbital_creation(self):
        """Teste die Erstellung von OrbitTensorDict-Objekten."""
        # Orbital elements: [a, e, i, Ω, ω, ν] für N Objekte
        n_objects = 10
        orbital_elements = torch.rand(n_objects, 6)
        orbital_elements[:, 1] = torch.clamp(orbital_elements[:, 1], 0, 0.9)  # e < 1

        orbit = OrbitTensorDict(
            orbital_elements,
            element_type="keplerian",
            central_body="earth"
        )

        assert orbit.n_objects == n_objects
        assert orbit["meta", "element_type"] == "keplerian"
        assert orbit["meta", "central_body"] == "earth"
        assert orbit["elements"].shape == (n_objects, 6)
        assert orbit["constants", "mu"].item() > 0  # Earth's gravitational parameter

    def test_orbital_validation(self):
        """Teste die Validierung von Orbital-Elementen."""
        # Falsche Anzahl von Elementen
        with pytest.raises(ValueError):
            OrbitTensorDict(torch.randn(10, 5))  # Sollte 6 Elemente sein

        # Negative Halbachse
        elements = torch.randn(5, 6)
        elements[:, 0] = -1.0  # Negative semi-major axis
        with pytest.raises(ValueError):
            OrbitTensorDict(elements)

        # Exzentrizität >= 1 (hyperbolische Bahnen)
        elements = torch.rand(5, 6)
        elements[:, 1] = 1.5  # e > 1
        orbit = OrbitTensorDict(elements, allow_hyperbolic=True)
        assert orbit.n_objects == 5  # Sollte mit allow_hyperbolic funktionieren

    def test_orbital_period_computation(self):
        """Teste die Berechnung von Orbital-Perioden."""
        # Erdnahe Kreisbahnen
        elements = torch.tensor([
            [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],  # 7000 km Radius
            [42164e3, 0.0, 0.0, 0.0, 0.0, 0.0],  # GEO-Bahn
        ])

        orbit = OrbitTensorDict(elements, central_body="earth")
        periods = orbit.compute_period()

        assert periods.shape == (2,)
        assert periods[0] < periods[1]  # Niedrigere Bahn = kürzere Periode

        # Prüfe gegen Kepler's 3. Gesetz: T² ∝ a³
        a = elements[:, 0]
        mu = orbit["constants", "mu"]
        expected_periods = 2 * torch.pi * torch.sqrt(a**3 / mu)

        torch.testing.assert_close(periods, expected_periods, rtol=1e-6)

    def test_cartesian_conversion(self):
        """Teste die Konvertierung zu kartesischen Koordinaten."""
        # Einfache Kreisbahn in der xy-Ebene
        elements = torch.tensor([[
            7000e3,  # a = 7000 km
            0.0,     # e = 0 (circular)
            0.0,     # i = 0 (equatorial)
            0.0,     # Ω = 0
            0.0,     # ω = 0
            0.0      # ν = 0 (at periapsis)
        ]])

        orbit = OrbitTensorDict(elements, central_body="earth")
        cartesian = orbit.to_cartesian()

        assert "position" in cartesian
        assert "velocity" in cartesian
        assert cartesian["position"].shape == (1, 3)
        assert cartesian["velocity"].shape == (1, 3)

        # Für Kreisbahn bei ν=0 sollte Position bei (a, 0, 0) sein
        expected_pos = torch.tensor([[7000e3, 0.0, 0.0]])
        torch.testing.assert_close(
            cartesian["position"], expected_pos, atol=1e-3
        )

    def test_orbit_propagation(self):
        """Teste die Orbit-Propagation über Zeit."""
        elements = torch.tensor([[
            7000e3, 0.1, 10.0, 0.0, 0.0, 0.0
        ]])

        orbit = OrbitTensorDict(elements, central_body="earth")

        # Propagiere für verschiedene Zeiten
        times = torch.tensor([0.0, 3600.0, 7200.0])  # 0, 1, 2 Stunden
        propagated = orbit.propagate(times)

        assert propagated["elements"].shape == (3, 1, 6)
        assert propagated["times"].shape == (3,)

        # Halbachse und Exzentrizität sollten konstant bleiben
        a_propagated = propagated["elements"][:, 0, 0]
        e_propagated = propagated["elements"][:, 0, 1]

        torch.testing.assert_close(
            a_propagated, torch.full_like(a_propagated, 7000e3), rtol=1e-6
        )
        torch.testing.assert_close(
            e_propagated, torch.full_like(e_propagated, 0.1), rtol=1e-6
        )

    def test_mean_motion_computation(self):
        """Teste die Berechnung der mittleren Bewegung."""
        elements = torch.tensor([
            [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [42164e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        orbit = OrbitTensorDict(elements, central_body="earth")
        mean_motion = orbit.compute_mean_motion()

        assert mean_motion.shape == (2,)
        assert mean_motion[0] > mean_motion[1]  # Niedrigere Bahn = höhere Bewegung

        # Prüfe gegen n = √(μ/a³)
        a = elements[:, 0]
        mu = orbit["constants", "mu"]
        expected_n = torch.sqrt(mu / a**3)

        torch.testing.assert_close(mean_motion, expected_n, rtol=1e-6)

    def test_apoapsis_periapsis(self):
        """Teste die Berechnung von Apoapsis und Periapsis."""
        elements = torch.tensor([
            [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],   # Kreisbahn
            [7000e3, 0.5, 0.0, 0.0, 0.0, 0.0],   # Elliptische Bahn
        ])

        orbit = OrbitTensorDict(elements, central_body="earth")

        rp = orbit.compute_periapsis()
        ra = orbit.compute_apoapsis()

        assert rp.shape == (2,)
        assert ra.shape == (2,)

        # Für Kreisbahn: rp = ra = a
        torch.testing.assert_close(rp[0], torch.tensor(7000e3), rtol=1e-6)
        torch.testing.assert_close(ra[0], torch.tensor(7000e3), rtol=1e-6)

        # Für elliptische Bahn: rp = a(1-e), ra = a(1+e)
        a, e = elements[1, 0], elements[1, 1]
        expected_rp = a * (1 - e)
        expected_ra = a * (1 + e)

        torch.testing.assert_close(rp[1], expected_rp, rtol=1e-6)
        torch.testing.assert_close(ra[1], expected_ra, rtol=1e-6)

    def test_energy_computation(self):
        """Teste die Berechnung der spezifischen Energie."""
        elements = torch.tensor([
            [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [14000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        orbit = OrbitTensorDict(elements, central_body="earth")
        energy = orbit.compute_specific_energy()

        assert energy.shape == (2,)
        assert energy[0] < 0  # Gebundene Bahn
        assert energy[1] < 0  # Gebundene Bahn
        assert abs(energy[0]) > abs(energy[1])  # Niedrigere Bahn = höhere Energie

        # Prüfe gegen E = -μ/(2a)
        a = elements[:, 0]
        mu = orbit["constants", "mu"]
        expected_energy = -mu / (2 * a)

        torch.testing.assert_close(energy, expected_energy, rtol=1e-6)

    def test_batch_operations(self):
        """Teste Batch-Operationen mit vielen Orbits."""
        n_orbits = 1000
        elements = torch.rand(n_orbits, 6)
        elements[:, 0] = 7000e3 + torch.rand(n_orbits) * 35164e3  # 7000-42164 km
        elements[:, 1] = torch.rand(n_orbits) * 0.9  # e < 0.9
        elements[:, 2] = torch.rand(n_orbits) * 180  # i in [0, 180]

        orbit = OrbitTensorDict(elements, central_body="earth")

        # Alle Berechnungen sollten vektorisiert funktionieren
        periods = orbit.compute_period()
        energies = orbit.compute_specific_energy()
        cartesian = orbit.to_cartesian()

        assert periods.shape == (n_orbits,)
        assert energies.shape == (n_orbits,)
        assert cartesian["position"].shape == (n_orbits, 3)
        assert cartesian["velocity"].shape == (n_orbits, 3)

    def test_different_central_bodies(self):
        """Teste verschiedene Zentralkörper."""
        elements = torch.tensor([[7000e3, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # Erde
        earth_orbit = OrbitTensorDict(elements, central_body="earth")
        earth_period = earth_orbit.compute_period()

        # Mars (kleinere Masse)
        mars_orbit = OrbitTensorDict(elements, central_body="mars")
        mars_period = mars_orbit.compute_period()

        # Sonne (viel größere Masse)
        sun_orbit = OrbitTensorDict(elements, central_body="sun")
        sun_period = sun_orbit.compute_period()

        # Mars sollte längere Periode haben als Erde (kleinere μ)
        assert mars_period > earth_period

        # Sonne sollte kürzere Periode haben als Erde (größere μ)
        assert sun_period < earth_period

    def test_gpu_compatibility(self):
        """Teste GPU-Kompatibilität falls verfügbar."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        elements = torch.rand(100, 6, device="cuda")
        elements[:, 1] = torch.clamp(elements[:, 1], 0, 0.9)

        orbit = OrbitTensorDict(elements, central_body="earth")

        assert orbit["elements"].device.type == "cuda"
        assert orbit["constants", "mu"].device.type == "cuda"

        # Alle Operationen sollten auf GPU funktionieren
        periods = orbit.compute_period()
        cartesian = orbit.to_cartesian()

        assert periods.device.type == "cuda"
        assert cartesian["position"].device.type == "cuda"

    def test_serialization(self):
        """Teste Serialisierung und Deserialisierung."""
        elements = torch.rand(10, 6)
        elements[:, 1] = torch.clamp(elements[:, 1], 0, 0.9)

        original_orbit = OrbitTensorDict(
            elements, 
            central_body="earth",
            element_type="keplerian"
        )

        # Serialisiere zu Dictionary
        orbit_dict = original_orbit.to_dict()

        # Deserialisiere zurück
        restored_orbit = OrbitTensorDict.from_dict(orbit_dict)

        assert restored_orbit.n_objects == original_orbit.n_objects
        assert restored_orbit["meta", "central_body"] == "earth"
        assert restored_orbit["meta", "element_type"] == "keplerian"

        torch.testing.assert_close(
            restored_orbit["elements"], 
            original_orbit["elements"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
