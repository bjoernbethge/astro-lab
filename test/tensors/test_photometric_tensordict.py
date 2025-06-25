"""
Tests für PhotometricTensorDict - Moderne TensorDict-basierte Photometric-Tensor-Implementierung
============================================================================================
"""

import pytest
import torch
import numpy as np
from astro_lab.tensors import PhotometricTensorDict


class TestPhotometricTensorDict:
    """Tests für die PhotometricTensorDict-Klasse."""

    def test_photometric_creation(self):
        """Teste die Erstellung von PhotometricTensorDict-Objekten."""
        n_objects = 20
        bands = ["u", "g", "r", "i", "z"]
        magnitudes = torch.randn(n_objects, len(bands)) + 15
        errors = torch.rand(n_objects, len(bands)) * 0.1

        phot = PhotometricTensorDict(
            magnitudes,
            bands=bands,
            errors=errors,
            system="AB",
            survey="SDSS"
        )

        assert phot.n_objects == n_objects
        assert phot.n_bands == len(bands)
        assert phot.bands == bands
        assert phot["meta", "system"] == "AB"
        assert phot["meta", "survey"] == "SDSS"
        assert phot["magnitudes"].shape == (n_objects, len(bands))
        assert phot["errors"].shape == (n_objects, len(bands))

    def test_band_validation(self):
        """Teste die Validierung von Bändern."""
        magnitudes = torch.randn(10, 5)

        # Teste mit korrekter Anzahl von Bändern
        bands = ["g", "r", "i", "z", "y"]
        phot = PhotometricTensorDict(magnitudes, bands=bands)
        assert phot.n_bands == 5
        assert phot.bands == bands

        # Teste mit falscher Anzahl von Bändern
        wrong_bands = ["g", "r", "i"]  # 3 Bänder für 5-dimensionale Daten
        with pytest.raises(ValueError):
            PhotometricTensorDict(magnitudes, bands=wrong_bands)

        # Teste automatische Band-Erstellung
        phot_auto = PhotometricTensorDict(magnitudes)
        assert phot_auto.n_bands == 5
        assert len(phot_auto.bands) == 5

    def test_error_validation(self):
        """Teste die Validierung von Messfehlern."""
        magnitudes = torch.randn(10, 3)
        bands = ["g", "r", "i"]

        # Korrekte Fehler
        correct_errors = torch.rand(10, 3) * 0.1
        phot = PhotometricTensorDict(magnitudes, bands=bands, errors=correct_errors)
        torch.testing.assert_close(phot["errors"], correct_errors)

        # Falsche Form der Fehler
        wrong_errors = torch.rand(10, 2)
        with pytest.raises(ValueError):
            PhotometricTensorDict(magnitudes, bands=bands, errors=wrong_errors)

    def test_magnitude_to_flux_conversion(self):
        """Teste die Konvertierung von Magnituden zu Flüssen."""
        magnitudes = torch.tensor([
            [15.0, 16.0, 17.0],
            [20.0, 21.0, 22.0]
        ])
        bands = ["g", "r", "i"]

        phot = PhotometricTensorDict(magnitudes, bands=bands, system="AB")
        fluxes = phot.to_flux()

        assert "fluxes" in fluxes
        assert fluxes["fluxes"].shape == magnitudes.shape

        # Prüfe AB-System: f = 10^(-0.4*(m - zp))
        # Für AB-System ist zp = 8.90 (typisch)
        zp = phot["constants", "zeropoints"]["g"]  # Sollte ~8.90 sein
        expected_flux_g = 10**(-0.4 * (magnitudes[:, 0] - zp))

        torch.testing.assert_close(
            fluxes["fluxes"][:, 0], expected_flux_g, rtol=1e-5
        )

    def test_flux_to_magnitude_conversion(self):
        """Teste die Konvertierung von Flüssen zu Magnituden."""
        fluxes = torch.tensor([
            [1e-6, 5e-7, 2e-7],
            [1e-8, 5e-9, 2e-9]
        ])
        bands = ["g", "r", "i"]

        phot = PhotometricTensorDict(fluxes, bands=bands, data_type="flux", system="AB")
        magnitudes = phot.to_magnitude()

        assert "magnitudes" in magnitudes
        assert magnitudes["magnitudes"].shape == fluxes.shape

        # Hellere Objekte (größere Flüsse) sollten kleinere Magnituden haben
        assert torch.all(magnitudes["magnitudes"][0] < magnitudes["magnitudes"][1])

    def test_color_computation(self):
        """Teste die Berechnung von Farben."""
        magnitudes = torch.tensor([
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 20.5, 21.0, 21.5, 22.0]
        ])
        bands = ["u", "g", "r", "i", "z"]

        phot = PhotometricTensorDict(magnitudes, bands=bands)

        # Berechne spezifische Farben
        colors = phot.compute_colors([("g", "r"), ("r", "i"), ("i", "z")])

        assert "g_r" in colors
        assert "r_i" in colors
        assert "i_z" in colors

        # Prüfe Farb-Berechnungen
        expected_g_r = magnitudes[:, 1] - magnitudes[:, 2]  # g - r
        expected_r_i = magnitudes[:, 2] - magnitudes[:, 3]  # r - i

        torch.testing.assert_close(colors["g_r"], expected_g_r)
        torch.testing.assert_close(colors["r_i"], expected_r_i)

    def test_color_excess_computation(self):
        """Teste die Berechnung von Farbexzessen (Rötung)."""
        # Simuliere gerötete Objekte
        intrinsic_mags = torch.tensor([
            [15.0, 15.5, 16.0, 16.3, 16.5],  # Blauer Stern
            [18.0, 18.2, 18.4, 18.5, 18.6]   # Roter Stern
        ])

        # Füge Rötung hinzu (mehr in blauen Bändern)
        extinction = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1]])  # A_λ
        observed_mags = intrinsic_mags + extinction

        bands = ["u", "g", "r", "i", "z"]
        phot = PhotometricTensorDict(observed_mags, bands=bands)

        # Berechne Farbexzess E(B-V) ~ E(g-r)
        colors = phot.compute_colors([("g", "r")])
        intrinsic_colors = phot.compute_colors([("g", "r")], magnitudes=intrinsic_mags)

        color_excess = colors["g_r"] - intrinsic_colors["g_r"]
        expected_excess = extinction[0, 1] - extinction[0, 2]  # A_g - A_r

        torch.testing.assert_close(
            color_excess, 
            torch.full_like(color_excess, expected_excess),
            rtol=1e-6
        )

    def test_extinction_correction(self):
        """Teste die Extinktionskorrektur."""
        observed_mags = torch.tensor([
            [16.0, 16.5, 17.0, 17.3, 17.5],
            [19.0, 19.2, 19.4, 19.5, 19.6]
        ])
        bands = ["u", "g", "r", "i", "z"]

        phot = PhotometricTensorDict(observed_mags, bands=bands)

        # Galaktische Extinktion (E(B-V) = 0.1)
        ebv = torch.tensor([0.1, 0.05])
        corrected = phot.correct_extinction(ebv)

        assert "magnitudes" in corrected
        assert corrected["magnitudes"].shape == observed_mags.shape

        # Korrigierte Magnituden sollten heller sein (kleinere Werte)
        assert torch.all(corrected["magnitudes"] < observed_mags)

    def test_k_correction(self):
        """Teste K-Korrekturen für kosmologische Distanzen."""
        observed_mags = torch.tensor([
            [20.0, 20.5, 21.0, 21.3, 21.5],
            [22.0, 22.2, 22.4, 22.5, 22.6]
        ])
        bands = ["u", "g", "r", "i", "z"]
        redshifts = torch.tensor([0.1, 0.5])

        phot = PhotometricTensorDict(observed_mags, bands=bands)

        # Einfache K-Korrektur (vereinfacht)
        k_corrected = phot.apply_k_correction(redshifts)

        assert "magnitudes" in k_corrected
        assert k_corrected["magnitudes"].shape == observed_mags.shape

        # K-Korrektur sollte für höhere Redshifts stärker sein
        k_corr_z01 = k_corrected["magnitudes"][0] - observed_mags[0]
        k_corr_z05 = k_corrected["magnitudes"][1] - observed_mags[1]

        # Höhere Redshift sollte stärkere Korrektur haben
        assert torch.mean(torch.abs(k_corr_z05)) > torch.mean(torch.abs(k_corr_z01))

    def test_luminosity_distance_modulus(self):
        """Teste die Berechnung des Entfernungsmoduls."""
        apparent_mags = torch.tensor([
            [20.0, 20.5, 21.0],
            [25.0, 25.2, 25.4]
        ])
        absolute_mags = torch.tensor([
            [-5.0, -4.5, -4.0],
            [-2.0, -1.8, -1.6]
        ])
        bands = ["g", "r", "i"]

        phot = PhotometricTensorDict(apparent_mags, bands=bands)

        distance_moduli = phot.compute_distance_modulus(absolute_mags)

        assert distance_moduli.shape == (2, 3)

        # μ = m - M
        expected_dm = apparent_mags - absolute_mags
        torch.testing.assert_close(distance_moduli, expected_dm)

        # Konvertiere zu Entfernungen in Parsec
        distances = phot.distance_modulus_to_distance(distance_moduli)
        expected_distances = 10**(distance_moduli / 5 + 1)  # d = 10^(μ/5 + 1) pc

        torch.testing.assert_close(distances, expected_distances)

    def test_spectral_energy_distribution(self):
        """Teste die Erstellung von spektralen Energieverteilungen."""
        magnitudes = torch.tensor([
            [15.0, 15.5, 16.0, 16.3, 16.5],  # Blauer Stern
            [18.0, 17.8, 17.6, 17.5, 17.4]   # Roter Stern  
        ])
        bands = ["u", "g", "r", "i", "z"]

        phot = PhotometricTensorDict(magnitudes, bands=bands, system="AB")

        # Erstelle SED
        sed = phot.compute_sed()

        assert "wavelengths" in sed
        assert "fluxes" in sed
        assert sed["wavelengths"].shape[0] == len(bands)
        assert sed["fluxes"].shape == (2, len(bands))

        # Blauer Stern sollte bei kürzeren Wellenlängen heller sein
        blue_star_sed = sed["fluxes"][0]
        red_star_sed = sed["fluxes"][1]

        # Bei u-Band (kürzeste Wellenlänge) sollte blauer Stern heller sein
        assert blue_star_sed[0] > red_star_sed[0]

        # Bei z-Band (längste Wellenlänge) sollte roter Stern heller sein
        assert red_star_sed[-1] > blue_star_sed[-1]

    def test_photometric_redshift_estimation(self):
        """Teste die Schätzung photometrischer Redshifts."""
        # Simuliere Galaxien bei verschiedenen Redshifts
        z_true = torch.tensor([0.1, 0.3, 0.5, 0.8])
        n_galaxies = len(z_true)

        # Vereinfachte Template-basierte Magnituden
        bands = ["u", "g", "r", "i", "z"]
        base_template = torch.tensor([20.0, 19.5, 19.0, 18.8, 18.7])

        # K-Korrektur und Entfernungsmodul simulieren
        magnitudes = base_template.unsqueeze(0).repeat(n_galaxies, 1)
        magnitudes += torch.outer(z_true, torch.tensor([2.0, 1.5, 1.0, 0.5, 0.2]))

        phot = PhotometricTensorDict(magnitudes, bands=bands)

        # Template-Fitting für Photo-z
        z_phot = phot.estimate_photometric_redshift(template_type="elliptical")

        assert z_phot.shape == (n_galaxies,)

        # Photo-z sollte grob mit spektroskopischem z übereinstimmen
        # (bei vereinfachtem Template wird Genauigkeit nicht perfekt sein)
        assert torch.all(z_phot > 0)
        assert torch.all(z_phot < 2.0)

    def test_batch_operations(self):
        """Teste Batch-Operationen mit vielen Objekten."""
        n_objects = 10000
        n_bands = 5
        magnitudes = torch.randn(n_objects, n_bands) + 20
        errors = torch.rand(n_objects, n_bands) * 0.1

        bands = ["u", "g", "r", "i", "z"]
        phot = PhotometricTensorDict(magnitudes, bands=bands, errors=errors)

        # Alle Operationen sollten vektorisiert funktionieren
        fluxes = phot.to_flux()
        colors = phot.compute_colors([("g", "r"), ("r", "i")])

        assert fluxes["fluxes"].shape == (n_objects, n_bands)
        assert colors["g_r"].shape == (n_objects,)
        assert colors["r_i"].shape == (n_objects,)

    def test_different_photometric_systems(self):
        """Teste verschiedene photometrische Systeme."""
        magnitudes = torch.tensor([[15.0, 16.0, 17.0]])
        bands = ["B", "V", "R"]

        # AB-System
        phot_ab = PhotometricTensorDict(magnitudes, bands=bands, system="AB")
        flux_ab = phot_ab.to_flux()

        # Vega-System
        phot_vega = PhotometricTensorDict(magnitudes, bands=bands, system="Vega")
        flux_vega = phot_vega.to_flux()

        # Flüsse sollten unterschiedlich sein (verschiedene Nullpunkte)
        assert not torch.allclose(flux_ab["fluxes"], flux_vega["fluxes"])

        # ST-System
        phot_st = PhotometricTensorDict(magnitudes, bands=bands, system="ST")
        flux_st = phot_st.to_flux()

        assert not torch.allclose(flux_ab["fluxes"], flux_st["fluxes"])

    def test_gpu_compatibility(self):
        """Teste GPU-Kompatibilität falls verfügbar."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        magnitudes = torch.randn(100, 5, device="cuda") + 20
        bands = ["u", "g", "r", "i", "z"]

        phot = PhotometricTensorDict(magnitudes, bands=bands)

        assert phot["magnitudes"].device.type == "cuda"

        # Alle Operationen sollten auf GPU funktionieren
        fluxes = phot.to_flux()
        colors = phot.compute_colors([("g", "r")])

        assert fluxes["fluxes"].device.type == "cuda"
        assert colors["g_r"].device.type == "cuda"

    def test_serialization(self):
        """Teste Serialisierung und Deserialisierung."""
        magnitudes = torch.randn(10, 5) + 20
        errors = torch.rand(10, 5) * 0.1
        bands = ["u", "g", "r", "i", "z"]

        original_phot = PhotometricTensorDict(
            magnitudes, 
            bands=bands,
            errors=errors,
            system="AB",
            survey="SDSS"
        )

        # Serialisiere zu Dictionary
        phot_dict = original_phot.to_dict()

        # Deserialisiere zurück
        restored_phot = PhotometricTensorDict.from_dict(phot_dict)

        assert restored_phot.n_objects == original_phot.n_objects
        assert restored_phot.n_bands == original_phot.n_bands
        assert restored_phot.bands == original_phot.bands
        assert restored_phot["meta", "system"] == "AB"
        assert restored_phot["meta", "survey"] == "SDSS"

        torch.testing.assert_close(
            restored_phot["magnitudes"],
            original_phot["magnitudes"]
        )
        torch.testing.assert_close(
            restored_phot["errors"],
            original_phot["errors"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
