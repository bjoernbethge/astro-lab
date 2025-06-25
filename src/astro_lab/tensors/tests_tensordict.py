"""
Tests für die TensorDict-basierte astro_lab.tensors Implementierung
================================================================

Vollständige Testsuite für alle TensorDict-Klassen und Funktionen.
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from .tensordict_astro import (
    AstroTensorDict,
    SpatialTensorDict,
    PhotometricTensorDict,
    SpectralTensorDict,
    LightcurveTensorDict,
    SurveyTensorDict,
)
from .factories import (
    create_gaia_survey,
    create_sdss_survey,
    create_kepler_lightcurves,
    create_generic_survey,
)


class TestAstroTensorDict:
    """Tests für die Basis AstroTensorDict Klasse."""

    def test_initialization(self):
        """Test der Basis-Initialisierung."""
        data = {
            "tensor1": torch.randn(10, 3),
            "tensor2": torch.randn(10, 5),
        }

        astro_td = AstroTensorDict(data)

        assert astro_td.n_objects == 10
        assert astro_td.batch_size == (10,)
        assert "meta" in astro_td
        assert astro_td["meta", "tensor_type"] == "AstroTensorDict"

    def test_inconsistent_batch_sizes(self):
        """Test für inkonsistente Batch-Größen."""
        data = {
            "tensor1": torch.randn(10, 3),
            "tensor2": torch.randn(15, 5),  # Verschiedene Batch-Größe
        }

        with pytest.raises(ValueError, match="Inconsistent batch sizes"):
            AstroTensorDict(data)

    def test_history_tracking(self):
        """Test der Operationshistorie."""
        data = {"tensor": torch.randn(5, 3)}
        astro_td = AstroTensorDict(data)

        astro_td.add_history("test_operation", param1="value1")

        history = astro_td["meta", "history"]
        assert len(history) == 1
        assert history[0]["operation"] == "test_operation"
        assert history[0]["details"]["param1"] == "value1"

    def test_memory_info(self):
        """Test der Speicher-Informationen."""
        data = {
            "tensor1": torch.randn(10, 3),
            "tensor2": torch.randn(10, 5),
        }
        astro_td = AstroTensorDict(data)

        mem_info = astro_td.memory_info()

        assert "total_bytes" in mem_info
        assert "n_tensors" in mem_info
        assert "total_mb" in mem_info
        assert mem_info["n_tensors"] >= 2  # mindestens unsere 2 Tensoren


class TestSpatialTensorDict:
    """Tests für SpatialTensorDict."""

    def test_initialization(self):
        """Test der räumlichen Tensor-Initialisierung."""
        coords = torch.randn(100, 3)
        spatial = SpatialTensorDict(coords)

        assert spatial.n_objects == 100
        assert spatial.coordinate_system == "icrs"
        assert torch.allclose(spatial.x, coords[:, 0])
        assert torch.allclose(spatial.y, coords[:, 1])
        assert torch.allclose(spatial.z, coords[:, 2])

    def test_invalid_coordinates(self):
        """Test für ungültige Koordinaten-Shape."""
        coords = torch.randn(100, 2)  # Falsche Anzahl Dimensionen

        with pytest.raises(ValueError, match="must have shape"):
            SpatialTensorDict(coords)

    def test_to_spherical(self):
        """Test der Konvertierung zu sphärischen Koordinaten."""
        # Teste mit bekannten Koordinaten
        coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        spatial = SpatialTensorDict(coords)

        ra, dec, distance = spatial.to_spherical()

        # Erwartete Werte
        expected_ra = torch.tensor([0.0, 90.0, 0.0])
        expected_dec = torch.tensor([0.0, 0.0, 90.0])
        expected_distance = torch.ones(3)

        assert torch.allclose(ra, expected_ra, atol=1e-5)
        assert torch.allclose(dec, expected_dec, atol=1e-5)
        assert torch.allclose(distance, expected_distance, atol=1e-5)

    def test_angular_separation(self):
        """Test der Winkeltrennungs-Berechnung."""
        coords1 = torch.tensor([[1.0, 0.0, 0.0]])
        coords2 = torch.tensor([[0.0, 1.0, 0.0]])

        spatial1 = SpatialTensorDict(coords1)
        spatial2 = SpatialTensorDict(coords2)

        separation = spatial1.angular_separation(spatial2)

        # Erwartete Trennung: 90 Grad
        assert torch.allclose(separation, torch.tensor([90.0]), atol=1e-5)

    def test_cone_search(self):
        """Test der Kegel-Suche."""
        # Erstelle Punkte um Ursprung
        coords = torch.randn(1000, 3)
        spatial = SpatialTensorDict(coords)

        center = torch.zeros(3)
        matches = spatial.cone_search(center, radius_deg=45.0)

        # Mindestens einige Matches sollten gefunden werden
        assert len(matches) > 0
        assert len(matches) < spatial.n_objects


class TestPhotometricTensorDict:
    """Tests für PhotometricTensorDict."""

    def test_initialization(self):
        """Test der photometrischen Tensor-Initialisierung."""
        mags = torch.randn(100, 3)
        bands = ["g", "r", "i"]

        phot = PhotometricTensorDict(mags, bands)

        assert phot.n_objects == 100
        assert phot.n_bands == 3
        assert phot.bands == bands
        assert phot.is_magnitude == True

    def test_band_mismatch(self):
        """Test für falsche Anzahl Bänder."""
        mags = torch.randn(100, 3)
        bands = ["g", "r"]  # Nur 2 Bänder für 3 Spalten

        with pytest.raises(ValueError, match="Number of bands"):
            PhotometricTensorDict(mags, bands)

    def test_get_band(self):
        """Test des Band-Zugriffs."""
        mags = torch.randn(100, 3)
        bands = ["g", "r", "i"]
        phot = PhotometricTensorDict(mags, bands)

        g_band = phot.get_band("g")
        assert torch.allclose(g_band, mags[:, 0])

        with pytest.raises(ValueError, match="not found"):
            phot.get_band("z")

    def test_compute_colors(self):
        """Test der Farbindex-Berechnung."""
        mags = torch.randn(100, 3) + torch.tensor([15.0, 16.0, 17.0])
        bands = ["g", "r", "i"]
        phot = PhotometricTensorDict(mags, bands)

        colors = phot.compute_colors([("g", "r"), ("r", "i")])

        assert "g_r" in colors
        assert "r_i" in colors
        assert torch.allclose(colors["g_r"], mags[:, 0] - mags[:, 1])
        assert torch.allclose(colors["r_i"], mags[:, 1] - mags[:, 2])

    def test_magnitude_flux_conversion(self):
        """Test der Magnituden/Fluss-Konvertierung."""
        mags = torch.tensor([[15.0, 16.0, 17.0]])
        bands = ["g", "r", "i"]
        phot = PhotometricTensorDict(mags, bands)

        # Zu Flüssen konvertieren
        flux_phot = phot.to_flux()
        assert flux_phot.is_magnitude == False

        # Zurück zu Magnituden
        mag_phot = flux_phot.to_magnitude()
        assert mag_phot.is_magnitude == True

        # Teste Rundungsfehler
        assert torch.allclose(mags, mag_phot["magnitudes"], atol=1e-5)

    def test_flux_color_error(self):
        """Test dass Farbberechnung mit Flüssen fehlschlägt."""
        flux = torch.randn(100, 3)
        bands = ["g", "r", "i"]
        phot = PhotometricTensorDict(flux, bands, is_magnitude=False)

        with pytest.raises(ValueError, match="Color computation requires magnitude"):
            phot.compute_colors([("g", "r")])


class TestSpectralTensorDict:
    """Tests für SpectralTensorDict."""

    def test_initialization(self):
        """Test der spektralen Tensor-Initialisierung."""
        flux = torch.randn(50, 1000)
        wavelengths = torch.linspace(4000, 7000, 1000)

        spec = SpectralTensorDict(flux, wavelengths)

        assert spec.n_objects == 50
        assert spec["meta", "n_wavelengths"] == 1000
        assert spec.redshift == 0.0

    def test_wavelength_mismatch(self):
        """Test für inkompatible Wellenlängen."""
        flux = torch.randn(50, 1000)
        wavelengths = torch.linspace(4000, 7000, 500)  # Falsche Länge

        with pytest.raises(ValueError, match="incompatible"):
            SpectralTensorDict(flux, wavelengths)

    def test_rest_wavelengths(self):
        """Test der Rest-frame Wellenlängen."""
        flux = torch.randn(50, 1000)
        wavelengths = torch.linspace(4000, 7000, 1000)
        redshift = 0.1

        spec = SpectralTensorDict(flux, wavelengths, redshift=redshift)

        expected_rest = wavelengths / (1 + redshift)
        assert torch.allclose(spec.rest_wavelengths, expected_rest)

    def test_apply_redshift(self):
        """Test der Rotverschiebungs-Anwendung."""
        flux = torch.randn(50, 1000)
        wavelengths = torch.linspace(4000, 7000, 1000)

        spec = SpectralTensorDict(flux, wavelengths, redshift=0.1)
        redshifted = spec.apply_redshift(0.05)

        assert redshifted.redshift == 0.15
        expected_wavelengths = wavelengths * (1 + 0.05)
        assert torch.allclose(redshifted["wavelengths"], expected_wavelengths)

    def test_normalize(self):
        """Test der Spektren-Normalisierung."""
        flux = torch.ones(10, 1000) * 2.0  # Konstanter Fluss
        wavelengths = torch.linspace(4000, 7000, 1000)

        spec = SpectralTensorDict(flux, wavelengths)
        normalized = spec.normalize(5500.0)

        # Nach Normalisierung sollte Fluss bei 5500Å gleich 1 sein
        idx = torch.argmin(torch.abs(wavelengths - 5500.0))
        assert torch.allclose(normalized["flux"][:, idx], torch.ones(10), atol=1e-5)


class TestLightcurveTensorDict:
    """Tests für LightcurveTensorDict."""

    def test_initialization(self):
        """Test der Lichtkurven-Initialisierung."""
        times = torch.linspace(0, 100, 200).unsqueeze(0).expand(10, -1)
        magnitudes = torch.randn(10, 200, 1)  # 1 Band
        bands = ["V"]

        lc = LightcurveTensorDict(times, magnitudes, bands)

        assert lc.n_objects == 10
        assert lc["meta", "n_times"] == 200
        assert lc["meta", "bands"] == bands

    def test_shape_mismatch(self):
        """Test für inkompatible Shapes."""
        times = torch.linspace(0, 100, 200).unsqueeze(0).expand(10, -1)
        magnitudes = torch.randn(10, 150, 1)  # Falsche Zeit-Dimension
        bands = ["V"]

        with pytest.raises(ValueError, match="incompatible"):
            LightcurveTensorDict(times, magnitudes, bands)

    def test_time_span(self):
        """Test der Zeitspannen-Berechnung."""
        times = torch.tensor([[0.0, 50.0, 100.0], [10.0, 60.0, 110.0]])
        magnitudes = torch.randn(2, 3, 1)
        bands = ["V"]

        lc = LightcurveTensorDict(times, magnitudes, bands)
        spans = lc.time_span

        expected = torch.tensor([100.0, 100.0])
        assert torch.allclose(spans, expected)

    def test_phase_fold(self):
        """Test der Phasen-Faltung."""
        n_times = 100
        times = torch.linspace(0, 200, n_times).unsqueeze(0).expand(5, -1)

        # Erstelle sinusförmige Lichtkurven mit bekannten Perioden
        periods = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        magnitudes = torch.zeros(5, n_times, 1)

        for i, period in enumerate(periods):
            magnitudes[i, :, 0] = 15.0 + torch.sin(2 * torch.pi * times[i] / period)

        lc = LightcurveTensorDict(times, magnitudes, ["V"])
        folded = lc.phase_fold(periods)

        # Nach Faltung sollten Phasen zwischen 0 und 1 liegen
        assert torch.all(folded["times"] >= 0)
        assert torch.all(folded["times"] <= 1)


class TestSurveyTensorDict:
    """Tests für SurveyTensorDict."""

    def test_initialization(self):
        """Test der Survey-Initialisierung."""
        coords = torch.randn(100, 3)
        mags = torch.randn(100, 3)

        spatial = SpatialTensorDict(coords)
        photometric = PhotometricTensorDict(mags, ["g", "r", "i"])

        survey = SurveyTensorDict(spatial, photometric, "TestSurvey")

        assert survey.survey_name == "TestSurvey"
        assert survey.n_objects == 100
        assert "spatial" in survey
        assert "photometric" in survey

    def test_batch_size_mismatch(self):
        """Test für inkompatible Batch-Größen."""
        coords = torch.randn(100, 3)
        mags = torch.randn(80, 3)  # Verschiedene Batch-Größe

        spatial = SpatialTensorDict(coords)
        photometric = PhotometricTensorDict(mags, ["g", "r", "i"])

        with pytest.raises(ValueError, match="same batch size"):
            SurveyTensorDict(spatial, photometric, "TestSurvey")

    def test_delegated_operations(self):
        """Test der delegierten Operationen."""
        coords = torch.randn(100, 3)
        mags = torch.randn(100, 3) + torch.tensor([15.0, 16.0, 17.0])

        spatial = SpatialTensorDict(coords)
        photometric = PhotometricTensorDict(mags, ["g", "r", "i"])
        survey = SurveyTensorDict(spatial, photometric, "TestSurvey")

        # Test Farbberechnung
        colors = survey.compute_colors([("g", "r")])
        assert "g_r" in colors

        # Test Cone Search
        center = torch.zeros(3)
        matches = survey.cone_search(center, 45.0)
        assert isinstance(matches, torch.Tensor)

    def test_query_region(self):
        """Test der Regions-Abfrage."""
        # Erstelle Daten in bekanntem Bereich
        n_in_region = 50
        n_outside = 50

        coords_in = torch.rand(n_in_region, 3) * 10  # 0-10 Grad
        coords_out = torch.rand(n_outside, 3) * 10 + 20  # 20-30 Grad

        all_coords = torch.cat([coords_in, coords_out], dim=0)
        mags = torch.randn(100, 3)

        spatial = SpatialTensorDict(all_coords)
        photometric = PhotometricTensorDict(mags, ["g", "r", "i"])
        survey = SurveyTensorDict(spatial, photometric, "TestSurvey")

        # Abfrage Region, die nur coords_in enthalten sollte
        subset = survey.query_region(ra_range=(0, 15), dec_range=(0, 15))

        # Sollte ungefähr n_in_region Objekte enthalten
        assert subset.n_objects <= n_in_region + 10  # Etwas Toleranz


class TestFactory:
    """Tests für Factory-Funktionen."""

    def test_create_gaia_survey(self):
        """Test der Gaia Survey-Erstellung."""
        n_objects = 100
        coords = torch.randn(n_objects, 2)
        g_mag = torch.randn(n_objects) + 15
        bp_mag = g_mag + torch.randn(n_objects) * 0.1
        rp_mag = g_mag - torch.randn(n_objects) * 0.1
        parallax = torch.abs(torch.randn(n_objects)) + 1.0

        survey = create_gaia_survey(coords, g_mag, bp_mag, rp_mag, parallax)

        assert survey.survey_name == "Gaia"
        assert survey.n_objects == n_objects
        assert survey["photometric"].bands == ["G", "BP", "RP"]
        assert "parallax" in survey

    def test_create_sdss_survey(self):
        """Test der SDSS Survey-Erstellung."""
        n_objects = 100
        coords = torch.randn(n_objects, 2)
        u_mag = torch.randn(n_objects) + 18
        g_mag = torch.randn(n_objects) + 16
        r_mag = torch.randn(n_objects) + 15
        i_mag = torch.randn(n_objects) + 14
        z_mag = torch.randn(n_objects) + 14

        survey = create_sdss_survey(coords, u_mag, g_mag, r_mag, i_mag, z_mag)

        assert survey.survey_name == "SDSS"
        assert survey.n_objects == n_objects
        assert survey["photometric"].bands == ["u", "g", "r", "i", "z"]

    def test_create_kepler_lightcurves(self):
        """Test der Kepler Lichtkurven-Erstellung."""
        n_objects = 50
        n_times = 200
        coords = torch.randn(n_objects, 2)
        times = torch.linspace(0, 100, n_times).unsqueeze(0).expand(n_objects, -1)
        magnitudes = torch.randn(n_objects, n_times)

        survey = create_kepler_lightcurves(coords, times, magnitudes)

        assert survey.survey_name == "Kepler"
        assert survey.n_objects == n_objects
        assert "lightcurves" in survey
        assert survey["lightcurves"]["meta", "bands"] == ["Kepler"]

    def test_create_generic_survey(self):
        """Test der generischen Survey-Erstellung."""
        coords = torch.randn(100, 2)
        mags = torch.randn(100, 4)
        bands = ["B", "V", "R", "I"]

        survey = create_generic_survey(
            coords, mags, bands, "MySurvey", filter_system="Johnson"
        )

        assert survey.survey_name == "MySurvey"
        assert survey["photometric"].bands == bands
        assert survey["photometric"]["meta", "filter_system"] == "Johnson"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUSupport:
    """Tests für GPU-Unterstützung."""

    def test_gpu_transfer(self):
        """Test der GPU-Übertragung."""
        coords = torch.randn(100, 3)
        mags = torch.randn(100, 3)

        survey = create_gaia_survey(coords[:, :2], mags[:, 0], mags[:, 1], mags[:, 2])

        # Transfer to GPU
        survey_gpu = survey.cuda()

        assert survey_gpu.device.type == "cuda"
        assert survey_gpu["spatial"]["coordinates"].device.type == "cuda"
        assert survey_gpu["photometric"]["magnitudes"].device.type == "cuda"

    def test_gpu_operations(self):
        """Test der GPU-Operationen."""
        coords = torch.randn(1000, 3)
        mags = torch.randn(1000, 3)

        survey = create_gaia_survey(coords[:, :2], mags[:, 0], mags[:, 1], mags[:, 2])
        survey_gpu = survey.cuda()

        # Test operations on GPU
        colors = survey_gpu.compute_colors([("BP", "RP")])
        assert colors.device.type == "cuda"

        center = torch.zeros(3).cuda()
        matches = survey_gpu.cone_search(center, 45.0)
        assert matches.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
