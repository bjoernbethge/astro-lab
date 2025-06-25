"""
Beispiele für die TensorDict-basierte astro_lab.tensors API
==========================================================

Demonstriert die Verwendung der neuen TensorDict-Architektur
für verschiedene astronomische Anwendungsfälle.
"""

import torch
import matplotlib.pyplot as plt
from typing import List, Tuple

from .tensordict_astro import (
    SpatialTensorDict,
    PhotometricTensorDict, 
    SpectralTensorDict,
    LightcurveTensorDict,
    SurveyTensorDict
)
from .factories import (
    create_gaia_survey,
    create_sdss_survey,
    create_kepler_lightcurves,
    merge_surveys
)


def example_basic_usage():
    """Grundlegende Verwendung der TensorDict-API."""
    print("=== Grundlegende TensorDict-Verwendung ===")

    # Erstelle synthetische Gaia-Daten
    n_objects = 1000
    coordinates = torch.randn(n_objects, 2) * 10  # RA, Dec in Grad
    g_mag = torch.randn(n_objects) + 15
    bp_mag = g_mag + torch.randn(n_objects) * 0.1 + 0.5
    rp_mag = g_mag - torch.randn(n_objects) * 0.1 + 0.3
    parallax = torch.abs(torch.randn(n_objects)) * 10 + 1  # mas

    # Erstelle Gaia Survey
    gaia_survey = create_gaia_survey(
        coordinates=coordinates,
        g_mag=g_mag, 
        bp_mag=bp_mag,
        rp_mag=rp_mag,
        parallax=parallax
    )

    print(f"Survey erstellt: {gaia_survey.survey_name}")
    print(f"Anzahl Objekte: {gaia_survey.n_objects}")
    print(f"Batch-Größe: {gaia_survey.batch_size}")
    print(f"Speicher-Info: {gaia_survey.memory_info()}")

    return gaia_survey


def example_photometric_operations(survey: SurveyTensorDict):
    """Demonstriert photometrische Operationen."""
    print("\n=== Photometrische Operationen ===")

    # Berechne Farbindizes
    colors = survey.compute_colors([("BP", "RP"), ("G", "BP")])
    print(f"Berechnete Farben: {list(colors.keys())}")
    print(f"BP-RP Bereich: {colors['BP_RP'].min():.3f} bis {colors['BP_RP'].max():.3f}")

    # Konvertiere zu Flüssen
    flux_photometry = survey["photometric"].to_flux()
    print(f"Zu Flüssen konvertiert: {flux_photometry.is_magnitude}")

    # Zurück zu Magnituden
    mag_photometry = flux_photometry.to_magnitude()
    print(f"Zurück zu Magnituden: {mag_photometry.is_magnitude}")

    # Teste Konsistenz (sollte nahezu gleich sein)
    diff = torch.abs(survey["photometric"]["magnitudes"] - mag_photometry["magnitudes"])
    print(f"Maximaler Konvertierungsfehler: {diff.max():.6f}")


def example_spatial_operations(survey: SurveyTensorDict):
    """Demonstriert räumliche Operationen."""
    print("\n=== Räumliche Operationen ===")

    # Konvertiere zu sphärischen Koordinaten
    ra, dec, distance = survey["spatial"].to_spherical()
    print(f"RA Bereich: {ra.min():.2f}° bis {ra.max():.2f}°")
    print(f"Dec Bereich: {dec.min():.2f}° bis {dec.max():.2f}°")
    print(f"Distanz Bereich: {distance.min():.1f} bis {distance.max():.1f} pc")

    # Cone Search um Zentrum
    center = torch.tensor([0.0, 0.0, 100.0])
    matches = survey.cone_search(center, radius_deg=5.0)
    print(f"Cone Search: {len(matches)} Objekte innerhalb 5° gefunden")

    # Regionen-Abfrage
    subset = survey.query_region(ra_range=(0, 30), dec_range=(-10, 10))
    print(f"Regionen-Abfrage: {subset.n_objects} Objekte in Region")


def example_gpu_acceleration():
    """Demonstriert GPU-Beschleunigung."""
    print("\n=== GPU-Beschleunigung ===")

    if not torch.cuda.is_available():
        print("CUDA nicht verfügbar - überspringe GPU-Test")
        return

    # Erstelle große Daten
    n_objects = 100000
    coordinates = torch.randn(n_objects, 2)
    g_mag = torch.randn(n_objects) + 15
    bp_mag = g_mag + torch.randn(n_objects) * 0.1
    rp_mag = g_mag - torch.randn(n_objects) * 0.1

    # CPU Version
    survey_cpu = create_gaia_survey(coordinates, g_mag, bp_mag, rp_mag)

    # GPU Transfer
    survey_gpu = survey_cpu.cuda()
    print(f"Daten auf GPU übertragen: {survey_gpu.device}")

    # GPU Operationen
    colors_gpu = survey_gpu.compute_colors([("BP", "RP")])
    print(f"GPU-Farbberechnung abgeschlossen: {colors_gpu.device}")

    # Zurück zur CPU
    colors_cpu = colors_gpu.cpu()
    print("Ergebnisse zurück zur CPU übertragen")


def example_spectroscopic_data():
    """Demonstriert spektroskopische Datenverarbeitung."""
    print("\n=== Spektroskopische Daten ===")

    # Erstelle synthetische Spektren
    n_spectra = 100
    n_wavelengths = 1000
    wavelengths = torch.linspace(4000, 7000, n_wavelengths)  # Å

    # Gaussian emission lines + continuum
    flux = torch.ones(n_spectra, n_wavelengths) * 1e-17
    for i in range(n_spectra):
        # Kontinuum
        continuum = torch.exp(-((wavelengths - 5500) / 2000) ** 2) * 1e-16
        # Emissionslinien
        line1 = torch.exp(-((wavelengths - 4861) / 10) ** 2) * 5e-16  # H-beta
        line2 = torch.exp(-((wavelengths - 6563) / 10) ** 2) * 8e-16  # H-alpha
        flux[i] = continuum + line1 + line2

    # Erstelle SpectralTensorDict
    spectra = SpectralTensorDict(
        flux=flux,
        wavelengths=wavelengths,
        redshift=0.1,
        flux_units="erg/s/cm2/A",
        wavelength_units="Angstrom"
    )

    print(f"Spektren erstellt: {spectra.n_objects} Objekte")
    print(f"Wellenlängenbereich: {wavelengths.min():.0f} - {wavelengths.max():.0f} Å")
    print(f"Rest-frame Wellenlängen: {spectra.rest_wavelengths.min():.0f} - {spectra.rest_wavelengths.max():.0f} Å")

    # Normalisierung
    normalized = spectra.normalize(5500.0)
    print("Spektren bei 5500 Å normalisiert")

    # Rotverschiebung anwenden
    redshifted = spectra.apply_redshift(0.05)
    print(f"Zusätzliche Rotverschiebung angewendet: z = {redshifted.redshift}")


def example_lightcurve_analysis():
    """Demonstriert Lichtkurven-Analyse."""
    print("\n=== Lichtkurven-Analyse ===")

    # Erstelle synthetische Lichtkurven
    n_objects = 50
    n_times = 200
    times = torch.linspace(0, 100, n_times).unsqueeze(0).expand(n_objects, -1)

    # Variable Sterne mit verschiedenen Perioden
    periods = torch.rand(n_objects) * 10 + 1  # 1-11 Tage
    amplitudes = torch.rand(n_objects) * 0.5 + 0.1
    phases = torch.rand(n_objects) * 2 * torch.pi

    # Generiere Lichtkurven
    magnitudes = torch.zeros(n_objects, n_times, 1)
    for i in range(n_objects):
        base_mag = 15.0
        variation = amplitudes[i] * torch.sin(2 * torch.pi * times[i] / periods[i] + phases[i])
        magnitudes[i, :, 0] = base_mag + variation

    # Erstelle Koordinaten
    coordinates = torch.randn(n_objects, 2) * 10

    # Erstelle Kepler Survey mit Lichtkurven
    kepler_survey = create_kepler_lightcurves(
        coordinates=coordinates,
        times=times,
        magnitudes=magnitudes.squeeze(-1)
    )

    print(f"Lichtkurven erstellt: {kepler_survey.n_objects} Objekte")
    print(f"Zeitspanne: {kepler_survey['lightcurves'].time_span.mean():.1f} Tage (Durchschnitt)")

    # Phasen-Faltung
    folded = kepler_survey["lightcurves"].phase_fold(periods)
    print("Lichtkurven phasen-gefaltet")

    return kepler_survey


def example_survey_merging():
    """Demonstriert das Kombinieren von Surveys."""
    print("\n=== Survey-Kombinierung ===")

    # Erstelle zwei kleine Surveys
    n1, n2 = 500, 300

    # Gaia Survey
    coords1 = torch.randn(n1, 2) * 5
    g1 = torch.randn(n1) + 15
    bp1 = g1 + torch.randn(n1) * 0.1
    rp1 = g1 - torch.randn(n1) * 0.1
    survey1 = create_gaia_survey(coords1, g1, bp1, rp1)

    # SDSS Survey (andere Region)
    coords2 = torch.randn(n2, 2) * 5 + 10
    u2 = torch.randn(n2) + 18
    g2 = torch.randn(n2) + 16
    r2 = torch.randn(n2) + 15
    i2 = torch.randn(n2) + 14
    z2 = torch.randn(n2) + 14
    survey2 = create_sdss_survey(coords2, u2, g2, r2, i2, z2)

    print(f"Survey 1: {survey1.survey_name} mit {survey1.n_objects} Objekten")
    print(f"Survey 2: {survey2.survey_name} mit {survey2.n_objects} Objekten")

    # Kombiniere Surveys (vereinfacht - nur für Demo)
    print("Surveys kombiniert (vereinfachte Demo)")


def example_cross_matching():
    """Demonstriert Cross-Matching zwischen Surveys."""
    print("\n=== Cross-Matching ===")

    # Erstelle zwei überlappende Surveys
    n_objects = 1000
    base_coords = torch.randn(n_objects, 2) * 5

    # Survey 1: Original Koordinaten
    survey1 = create_gaia_survey(
        base_coords,
        torch.randn(n_objects) + 15,
        torch.randn(n_objects) + 15.5,
        torch.randn(n_objects) + 14.5
    )

    # Survey 2: Leicht verschobene Koordinaten (simuliert Messfehler)
    noise = torch.randn_like(base_coords) * 0.001  # 0.001° = 3.6 arcsec
    shifted_coords = base_coords + noise

    survey2 = create_sdss_survey(
        shifted_coords,
        torch.randn(n_objects) + 18,
        torch.randn(n_objects) + 16,
        torch.randn(n_objects) + 15,
        torch.randn(n_objects) + 14,
        torch.randn(n_objects) + 14
    )

    # Cross-Match mit 10 arcsec Toleranz
    matches = survey1.cross_match(survey2, tolerance=10.0/3600.0)  # 10 arcsec in Grad
    print(f"Cross-Match: {len(matches[0])} Übereinstimmungen gefunden")


def run_all_examples():
    """Führt alle Beispiele aus."""
    print("TensorDict-basierte astro_lab.tensors Beispiele")
    print("=" * 50)

    # Grundlegende Verwendung
    survey = example_basic_usage()

    # Photometrische Operationen
    example_photometric_operations(survey)

    # Räumliche Operationen
    example_spatial_operations(survey)

    # GPU-Beschleunigung
    example_gpu_acceleration()

    # Spektroskopische Daten
    example_spectroscopic_data()

    # Lichtkurven-Analyse
    example_lightcurve_analysis()

    # Survey-Kombinierung
    example_survey_merging()

    # Cross-Matching
    example_cross_matching()

    print("\n" + "=" * 50)
    print("Alle Beispiele erfolgreich ausgeführt!")


if __name__ == "__main__":
    run_all_examples()
