"""
Factory-Funktionen für die Erstellung von Survey TensorDicts
===========================================================

Vereinfacht die Erstellung von TensorDict-Strukturen für gängige
astronomische Survey-Formate wie Gaia, SDSS, 2MASS, etc.
"""

import math
from typing import Any, List, Optional

import torch

from .crossmatch_tensordict import CrossMatchTensorDict
from .feature_tensordict import (
    ClusteringTensorDict,
    FeatureTensorDict,
    StatisticsTensorDict,
)
from .orbital_tensordict import ManeuverTensorDict, OrbitTensorDict
from .satellite_tensordict import EarthSatelliteTensorDict
from .simulation_tensordict import CosmologyTensorDict, SimulationTensorDict
from .tensordict_astro import (
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
    SurveyTensorDict,
)


def create_gaia_survey(
    coordinates: torch.Tensor,
    g_mag: torch.Tensor,
    bp_mag: torch.Tensor,
    rp_mag: torch.Tensor,
    parallax: Optional[torch.Tensor] = None,
    proper_motions: Optional[torch.Tensor] = None,
    **kwargs,
) -> SurveyTensorDict:
    """
    Creates SurveyTensorDict with Gaia data.

    Args:
        coordinates: [N, 2] oder [N, 3] - RA, Dec, (Distance)
        g_mag: [N] G-Band Magnituden
        bp_mag: [N] BP-Band Magnituden
        rp_mag: [N] RP-Band Magnituden
        parallax: [N] Parallaxen in mas (optional)
        proper_motions: [N, 2] Eigenbewegungen (optional)

    Returns:
        SurveyTensorDict with Gaia data
    """
    # Spatial component with parallax -> distance conversion
    if parallax is not None:
        distance = 1000.0 / (torch.abs(parallax) + 1e-6)  # mas to parsec
        if coordinates.shape[-1] == 2:
            # RA, Dec -> RA, Dec, Distance
            coords = torch.cat([coordinates, distance.unsqueeze(-1)], dim=-1)
        else:
            coords = coordinates
    else:
        coords = coordinates

    spatial = SpatialTensorDict(
        coords, coordinate_system="icrs", unit="parsec", epoch=2016.0
    )

    # Photometric component
    magnitudes = torch.stack([g_mag, bp_mag, rp_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["G", "BP", "RP"], filter_system="Gaia"
    )

    survey = SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="Gaia",
        data_release="DR3",
        **kwargs,
    )

    # Add additional Gaia-specific data
    if proper_motions is not None:
        survey["proper_motions"] = proper_motions
    if parallax is not None:
        survey["parallax"] = parallax

    return survey


def create_sdss_survey(
    coordinates: torch.Tensor,
    u_mag: torch.Tensor,
    g_mag: torch.Tensor,
    r_mag: torch.Tensor,
    i_mag: torch.Tensor,
    z_mag: torch.Tensor,
    spectra: Optional[SpectralTensorDict] = None,
    **kwargs,
) -> SurveyTensorDict:
    """
    Creates SDSS survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in degrees
        u_mag, g_mag, r_mag, i_mag, z_mag: [N] SDSS band magnitudes
        spectra: Optional spectroscopic data

    Returns:
        SurveyTensorDict with SDSS data
    """
    spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="degree")

    magnitudes = torch.stack([u_mag, g_mag, r_mag, i_mag, z_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["u", "g", "r", "i", "z"], filter_system="SDSS"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="SDSS",
        data_release="DR17",
        spectral=spectra,
        **kwargs,
    )


def create_2mass_survey(
    coordinates: torch.Tensor,
    j_mag: torch.Tensor,
    h_mag: torch.Tensor,
    k_mag: torch.Tensor,
    **kwargs,
) -> SurveyTensorDict:
    """
    Erstellt 2MASS-Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        j_mag, h_mag, k_mag: [N] 2MASS-Band Magnituden

    Returns:
        SurveyTensorDict mit 2MASS-Daten
    """
    spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="degree")

    magnitudes = torch.stack([j_mag, h_mag, k_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["J", "H", "K"], filter_system="2MASS"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="2MASS",
        data_release="All-Sky",
        **kwargs,
    )


def create_pan_starrs_survey(
    coordinates: torch.Tensor,
    g_mag: torch.Tensor,
    r_mag: torch.Tensor,
    i_mag: torch.Tensor,
    z_mag: torch.Tensor,
    y_mag: torch.Tensor,
    **kwargs,
) -> SurveyTensorDict:
    """
    Erstellt Pan-STARRS Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        g_mag, r_mag, i_mag, z_mag, y_mag: [N] Pan-STARRS Band Magnituden

    Returns:
        SurveyTensorDict mit Pan-STARRS Daten
    """
    spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="degree")

    magnitudes = torch.stack([g_mag, r_mag, i_mag, z_mag, y_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["g", "r", "i", "z", "y"], filter_system="Pan-STARRS"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="Pan-STARRS",
        data_release="DR2",
        **kwargs,
    )


def create_wise_survey(
    coordinates: torch.Tensor,
    w1_mag: torch.Tensor,
    w2_mag: torch.Tensor,
    w3_mag: torch.Tensor,
    w4_mag: torch.Tensor,
    **kwargs,
) -> SurveyTensorDict:
    """
    Erstellt WISE Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        w1_mag, w2_mag, w3_mag, w4_mag: [N] WISE Band Magnituden

    Returns:
        SurveyTensorDict mit WISE Daten
    """
    spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="degree")

    magnitudes = torch.stack([w1_mag, w2_mag, w3_mag, w4_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["W1", "W2", "W3", "W4"], filter_system="WISE"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="WISE",
        data_release="AllWISE",
        **kwargs,
    )


def create_kepler_orbits(
    semi_major_axes: torch.Tensor,
    eccentricities: torch.Tensor,
    inclinations: torch.Tensor,
    **kwargs,
) -> OrbitTensorDict:
    """
    Erstellt Kepler-Orbits aus grundlegenden Parametern.

    Args:
        semi_major_axes: [N] Große Halbachsen in AU
        eccentricities: [N] Exzentrizitäten
        inclinations: [N] Inklinationen in Grad

    Returns:
        OrbitTensorDict mit Kepler-Elementen
    """
    n_objects = semi_major_axes.shape[0]

    # Erstelle vollständige Orbital-Elemente
    elements = torch.stack(
        [
            semi_major_axes,
            eccentricities,
            inclinations,
            torch.zeros(n_objects),  # Omega (RAAN)
            torch.zeros(n_objects),  # omega (Argument of Periapsis)
            torch.rand(n_objects) * 360,  # M (zufällige mittlere Anomalie)
        ],
        dim=-1,
    )

    return OrbitTensorDict(elements, **kwargs)


def create_asteroid_population(
    n_asteroids: int = 1000, belt_type: str = "main"
) -> OrbitTensorDict:
    """
    Erstellt synthetische Asteroidenpopulation.

    Args:
        n_asteroids: Anzahl der Asteroiden
        belt_type: Art des Gürtels ("main", "trojan", "neo")

    Returns:
        OrbitTensorDict mit Asteroiden-Orbits
    """
    if belt_type == "main":
        # Hauptgürtel: 2.1 - 3.3 AU
        a = torch.rand(n_asteroids) * (3.3 - 2.1) + 2.1
        e = torch.exponential(torch.ones(n_asteroids)) * 0.1
        e = torch.clamp(e, 0, 0.9)
        i = torch.abs(torch.randn(n_asteroids) * 5)
    elif belt_type == "trojan":
        # Jupiter Trojaner bei ~5.2 AU
        a = torch.randn(n_asteroids) * 0.1 + 5.2
        e = torch.exponential(torch.ones(n_asteroids)) * 0.05
        e = torch.clamp(e, 0, 0.3)
        i = torch.abs(torch.randn(n_asteroids) * 10)
    elif belt_type == "neo":
        # Near-Earth Objects: a < 1.3 AU
        a = torch.rand(n_asteroids) * (1.3 - 0.8) + 0.8
        e = torch.rand(n_asteroids) * 0.8
        i = torch.abs(torch.randn(n_asteroids) * 15)
    else:
        raise ValueError(f"Unknown belt type: {belt_type}")

    return create_kepler_orbits(a, e, i, central_body="Sun")


def create_hohmann_transfer(
    orbit1: OrbitTensorDict, orbit2: OrbitTensorDict
) -> ManeuverTensorDict:
    """
    Berechnet Hohmann-Transfer zwischen zwei Orbits.

    Args:
        orbit1: Start-Orbit
        orbit2: Ziel-Orbit

    Returns:
        ManeuverTensorDict mit Transfer-Manövern
    """
    # Vereinfachte Hohmann-Transfer-Berechnung
    r1 = orbit1.semi_major_axis
    r2 = orbit2.semi_major_axis

    # Delta-V Berechnungen
    mu = 1.327e11  # Gravitationsparameter Sonne (km³/s²)

    v1 = torch.sqrt(mu / (r1 * 1.496e8))  # Convert AU to km
    v_transfer_1 = torch.sqrt(mu * (2 / (r1 * 1.496e8) - 2 / ((r1 + r2) * 1.496e8 / 2)))
    delta_v1 = v_transfer_1 - v1

    v2 = torch.sqrt(mu / (r2 * 1.496e8))
    v_transfer_2 = torch.sqrt(mu * (2 / (r2 * 1.496e8) - 2 / ((r1 + r2) * 1.496e8 / 2)))
    delta_v2 = v2 - v_transfer_2

    # Erstelle Manöver-Sequenz
    n_orbits = r1.shape[0]
    delta_v_tensor = torch.zeros(n_orbits * 2, 3)
    delta_v_tensor[::2, 0] = delta_v1  # Erste Manöver in x-Richtung
    delta_v_tensor[1::2, 0] = delta_v2  # Zweite Manöver in x-Richtung

    # Transfer-Zeit berechnen
    transfer_time = (
        math.pi * torch.sqrt(((r1 + r2) * 1.496e8 / 2) ** 3 / mu) / 86400
    )  # Tage
    times = torch.zeros(n_orbits * 2)
    times[1::2] = transfer_time.repeat(n_orbits)

    return ManeuverTensorDict(
        delta_v=delta_v_tensor, time=times, maneuver_type="hohmann_transfer"
    )


def create_tle_satellites(tle_lines: List[str]) -> EarthSatelliteTensorDict:
    """
    Erstellt EarthSatelliteTensorDict aus TLE-Daten.

    Args:
        tle_lines: Liste von TLE-Zeilen (je 3 Zeilen pro Satellit)

    Returns:
        EarthSatelliteTensorDict
    """
    if len(tle_lines) % 3 != 0:
        raise ValueError("TLE data must be in groups of 3 lines")

    n_satellites = len(tle_lines) // 3
    tle_data = torch.zeros(n_satellites, 8)
    satellite_names = []
    catalog_numbers = []

    for i in range(n_satellites):
        name_line = tle_lines[i * 3].strip()
        line1 = tle_lines[i * 3 + 1].strip()
        line2 = tle_lines[i * 3 + 2].strip()

        satellite_names.append(name_line)

        # Parse TLE (vereinfacht)
        catalog_num = int(line1[2:7])
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33])
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        epoch = float(line1[18:32])

        catalog_numbers.append(catalog_num)
        tle_data[i] = torch.tensor(
            [
                inclination,
                raan,
                eccentricity,
                arg_perigee,
                mean_anomaly,
                mean_motion,
                0.0,
                epoch,
            ]
        )

    return EarthSatelliteTensorDict(
        tle_data=tle_data,
        satellite_names=satellite_names,
        catalog_numbers=catalog_numbers,
    )


def create_nbody_simulation(
    n_particles: int = 100, system_type: str = "cluster"
) -> SimulationTensorDict:
    """
    Erstellt N-Body-Simulation.

    Args:
        n_particles: Anzahl der Teilchen
        system_type: Art des Systems ("cluster", "galaxy", "solar_system")

    Returns:
        SimulationTensorDict
    """
    if system_type == "cluster":
        # Kugelhaufen
        positions = torch.randn(n_particles, 3) * 10  # kpc
        velocities = torch.randn(n_particles, 3) * 10  # km/s
        masses = torch.ones(n_particles)  # Solar masses

    elif system_type == "galaxy":
        # Scheibengalaxie
        r = torch.exponential(torch.ones(n_particles)) * 5  # kpc
        theta = torch.uniform(0, 2 * math.pi, (n_particles,))
        z = torch.normal(0, 0.5, (n_particles,))

        positions = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)

        # Rotationsgeschwindigkeit
        v_rot = torch.sqrt(200 * r / (r + 1))  # km/s
        velocities = torch.stack(
            [
                -v_rot * torch.sin(theta),
                v_rot * torch.cos(theta),
                torch.zeros(n_particles),
            ],
            dim=-1,
        )

        masses = torch.ones(n_particles)

    elif system_type == "solar_system":
        # Vereinfachtes Sonnensystem
        if n_particles != 9:
            n_particles = 9  # Sonne + 8 Planeten

        # Planetenabstände in AU
        distances = torch.tensor([0, 0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30.1])
        masses = torch.tensor(
            [1.0, 0.055, 0.815, 1.0, 0.107, 317.8, 95.2, 14.5, 17.1]
        )  # Erdmassen

        positions = torch.zeros(n_particles, 3)
        velocities = torch.zeros(n_particles, 3)

        for i in range(1, n_particles):
            positions[i, 0] = distances[i] * 1.496e8  # Convert to km
            velocities[i, 1] = torch.sqrt(
                1.327e20 / (distances[i] * 1.496e8)
            )  # Orbital velocity

    else:
        raise ValueError(f"Unknown system type: {system_type}")

    return SimulationTensorDict(
        positions=positions,
        velocities=velocities,
        masses=masses,
        simulation_type=system_type,
    )


def create_cosmology_sample(
    z_min: float = 0.0, z_max: float = 5.0, n_objects: int = 1000
) -> CosmologyTensorDict:
    """
    Erstellt kosmologische Probe.

    Args:
        z_min: Minimale Rotverschiebung
        z_max: Maximale Rotverschiebung
        n_objects: Anzahl der Objekte

    Returns:
        CosmologyTensorDict
    """
    # Gleichmäßige Verteilung in Rotverschiebung
    redshifts = torch.uniform(z_min, z_max, (n_objects,))

    return CosmologyTensorDict(redshifts)


def create_crossmatch_example(
    n_objects1: int = 1000, n_objects2: int = 800, overlap_fraction: float = 0.7
) -> CrossMatchTensorDict:
    """
    Erstellt Beispiel für Cross-Matching.

    Args:
        n_objects1: Anzahl Objekte in Katalog 1
        n_objects2: Anzahl Objekte in Katalog 2
        overlap_fraction: Anteil der überlappenden Objekte

    Returns:
        CrossMatchTensorDict mit synthetischen Katalogen
    """
    # Erstelle ersten Katalog
    coords1 = torch.randn(n_objects1, 3) * 10
    mags1 = torch.randn(n_objects1, 3) + 15

    spatial1 = SpatialTensorDict(coords1)
    phot1 = PhotometricTensorDict(mags1, bands=["g", "r", "i"])
    cat1 = SurveyTensorDict(spatial=spatial1, photometric=phot1, survey_name="Survey1")

    # Erstelle zweiten Katalog mit teilweiser Überlappung
    n_overlap = int(n_objects2 * overlap_fraction)
    n_unique = n_objects2 - n_overlap

    # Überlappende Objekte (mit kleinen Positionsfehlern)
    coords2_overlap = coords1[:n_overlap] + torch.randn(n_overlap, 3) * 0.1
    mags2_overlap = mags1[:n_overlap, :2] + torch.randn(n_overlap, 2) * 0.1

    # Einzigartige Objekte
    coords2_unique = torch.randn(n_unique, 3) * 10
    mags2_unique = torch.randn(n_unique, 2) + 16

    coords2 = torch.cat([coords2_overlap, coords2_unique], dim=0)
    mags2 = torch.cat([mags2_overlap, mags2_unique], dim=0)

    spatial2 = SpatialTensorDict(coords2)
    phot2 = PhotometricTensorDict(mags2, bands=["u", "g"])
    cat2 = SurveyTensorDict(spatial=spatial2, photometric=phot2, survey_name="Survey2")

    return CrossMatchTensorDict(cat1, cat2, match_radius=2.0)


def create_kepler_lightcurves(
    coordinates: torch.Tensor,
    times: torch.Tensor,
    magnitudes: torch.Tensor,
    errors: Optional[torch.Tensor] = None,
    **kwargs,
) -> SurveyTensorDict:
    """
    Erstellt Kepler Lichtkurven TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec
        times: [N, T] Zeitpunkte in BJD
        magnitudes: [N, T] Kepler Magnituden
        errors: [N, T] Optionale Fehler

    Returns:
        SurveyTensorDict mit Kepler Lichtkurven
    """
    spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="degree")

    # Dummy photometric für Konsistenz
    avg_mag = magnitudes.mean(dim=-1)
    photometric = PhotometricTensorDict(
        avg_mag.unsqueeze(-1), bands=["Kepler"], filter_system="Kepler"
    )

    # Reshape magnitudes für Lightcurve format [N, T, 1]
    lc_magnitudes = magnitudes.unsqueeze(-1)
    lc_errors = errors.unsqueeze(-1) if errors is not None else None

    lightcurves = LightcurveTensorDict(
        times=times,
        magnitudes=lc_magnitudes,
        bands=["Kepler"],
        errors=lc_errors,
        time_format="bjd",
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        lightcurves=lightcurves,
        survey_name="Kepler",
        data_release="DR25",
        **kwargs,
    )


def create_generic_survey(
    coordinates: torch.Tensor,
    magnitudes: torch.Tensor,
    bands: List[str],
    survey_name: str,
    filter_system: str = "Generic",
    errors: Optional[torch.Tensor] = None,
    **kwargs,
) -> SurveyTensorDict:
    """
    Erstellt einen generischen Survey TensorDict.

    Args:
        coordinates: [N, 2] oder [N, 3] Koordinaten
        magnitudes: [N, B] Magnituden
        bands: Liste der Bandnamen
        survey_name: Name des Surveys
        filter_system: Filtersystem
        errors: Optionale Fehler

    Returns:
        SurveyTensorDict mit generischen Survey-Daten
    """
    spatial = SpatialTensorDict(
        coordinates,
        coordinate_system="icrs",
        unit="degree" if coordinates.shape[-1] == 2 else "parsec",
    )

    photometric = PhotometricTensorDict(
        magnitudes, bands=bands, errors=errors, filter_system=filter_system
    )

    return SurveyTensorDict(
        spatial=spatial, photometric=photometric, survey_name=survey_name, **kwargs
    )


def merge_surveys(
    *surveys: SurveyTensorDict, new_name: str = "Merged"
) -> SurveyTensorDict:
    """
    Kombiniert mehrere Survey TensorDicts.

    Args:
        *surveys: Survey TensorDicts zum Kombinieren
        new_name: Name des kombinierten Surveys

    Returns:
        Kombinierter SurveyTensorDict
    """
    if not surveys:
        raise ValueError("At least one survey required")

    # Sammle alle Koordinaten und photometrischen Daten
    all_coordinates = []
    all_magnitudes = []
    all_bands = []

    for survey in surveys:
        all_coordinates.append(survey["spatial"]["coordinates"])
        all_magnitudes.append(survey["photometric"]["magnitudes"])
        all_bands.extend(survey["photometric"].bands)

    # Kombiniere Daten
    combined_coords = torch.cat(all_coordinates, dim=0)
    combined_mags = torch.cat(all_magnitudes, dim=0)

    # Erstelle kombinierten Survey
    return create_generic_survey(
        coordinates=combined_coords,
        magnitudes=combined_mags,
        bands=list(set(all_bands)),  # Remove duplicates
        survey_name=new_name,
    )
