"""
Factory-Funktionen für die Erstellung von Survey TensorDicts
===========================================================

Vereinfacht die Erstellung von TensorDict-Strukturen für gängige
astronomische Survey-Formate wie Gaia, SDSS, 2MASS, etc.
"""

from typing import Optional, List
import torch
from .tensordict_astro import (
    SpatialTensorDict, 
    PhotometricTensorDict, 
    SurveyTensorDict,
    SpectralTensorDict,
    LightcurveTensorDict
)


def create_gaia_survey(coordinates: torch.Tensor, g_mag: torch.Tensor, 
                       bp_mag: torch.Tensor, rp_mag: torch.Tensor,
                       parallax: Optional[torch.Tensor] = None,
                       proper_motions: Optional[torch.Tensor] = None,
                       **kwargs) -> SurveyTensorDict:
    """
    Erstellt Gaia-Survey TensorDict.

    Args:
        coordinates: [N, 2] oder [N, 3] - RA, Dec, (Distance)
        g_mag: [N] G-Band Magnituden
        bp_mag: [N] BP-Band Magnituden  
        rp_mag: [N] RP-Band Magnituden
        parallax: [N] Parallaxen in mas (optional)
        proper_motions: [N, 2] Eigenbewegungen (optional)

    Returns:
        SurveyTensorDict mit Gaia-Daten
    """
    # Spatial component mit parallax -> distance conversion
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
        **kwargs
    )

    # Füge zusätzliche Gaia-spezifische Daten hinzu
    if proper_motions is not None:
        survey["proper_motions"] = proper_motions
    if parallax is not None:
        survey["parallax"] = parallax

    return survey


def create_sdss_survey(coordinates: torch.Tensor, u_mag: torch.Tensor,
                       g_mag: torch.Tensor, r_mag: torch.Tensor,
                       i_mag: torch.Tensor, z_mag: torch.Tensor,
                       spectra: Optional[SpectralTensorDict] = None,
                       **kwargs) -> SurveyTensorDict:
    """
    Erstellt SDSS-Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        u_mag, g_mag, r_mag, i_mag, z_mag: [N] SDSS-Band Magnituden
        spectra: Optionale spektroskopische Daten

    Returns:
        SurveyTensorDict mit SDSS-Daten
    """
    spatial = SpatialTensorDict(
        coordinates, coordinate_system="icrs", unit="degree"
    )

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
        **kwargs
    )


def create_2mass_survey(coordinates: torch.Tensor, j_mag: torch.Tensor,
                        h_mag: torch.Tensor, k_mag: torch.Tensor,
                        **kwargs) -> SurveyTensorDict:
    """
    Erstellt 2MASS-Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        j_mag, h_mag, k_mag: [N] 2MASS-Band Magnituden

    Returns:
        SurveyTensorDict mit 2MASS-Daten
    """
    spatial = SpatialTensorDict(
        coordinates, coordinate_system="icrs", unit="degree"
    )

    magnitudes = torch.stack([j_mag, h_mag, k_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["J", "H", "K"], filter_system="2MASS"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="2MASS",
        data_release="All-Sky",
        **kwargs
    )


def create_pan_starrs_survey(coordinates: torch.Tensor, g_mag: torch.Tensor,
                             r_mag: torch.Tensor, i_mag: torch.Tensor,
                             z_mag: torch.Tensor, y_mag: torch.Tensor,
                             **kwargs) -> SurveyTensorDict:
    """
    Erstellt Pan-STARRS Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        g_mag, r_mag, i_mag, z_mag, y_mag: [N] Pan-STARRS Band Magnituden

    Returns:
        SurveyTensorDict mit Pan-STARRS Daten
    """
    spatial = SpatialTensorDict(
        coordinates, coordinate_system="icrs", unit="degree"
    )

    magnitudes = torch.stack([g_mag, r_mag, i_mag, z_mag, y_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["g", "r", "i", "z", "y"], filter_system="Pan-STARRS"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="Pan-STARRS",
        data_release="DR2",
        **kwargs
    )


def create_wise_survey(coordinates: torch.Tensor, w1_mag: torch.Tensor,
                       w2_mag: torch.Tensor, w3_mag: torch.Tensor,
                       w4_mag: torch.Tensor, **kwargs) -> SurveyTensorDict:
    """
    Erstellt WISE Survey TensorDict.

    Args:
        coordinates: [N, 2] RA, Dec in Grad
        w1_mag, w2_mag, w3_mag, w4_mag: [N] WISE Band Magnituden

    Returns:
        SurveyTensorDict mit WISE Daten
    """
    spatial = SpatialTensorDict(
        coordinates, coordinate_system="icrs", unit="degree"
    )

    magnitudes = torch.stack([w1_mag, w2_mag, w3_mag, w4_mag], dim=-1)
    photometric = PhotometricTensorDict(
        magnitudes, bands=["W1", "W2", "W3", "W4"], filter_system="WISE"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="WISE",
        data_release="AllWISE",
        **kwargs
    )


def create_kepler_lightcurves(coordinates: torch.Tensor, times: torch.Tensor,
                              magnitudes: torch.Tensor, 
                              errors: Optional[torch.Tensor] = None,
                              **kwargs) -> SurveyTensorDict:
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
    spatial = SpatialTensorDict(
        coordinates, coordinate_system="icrs", unit="degree"
    )

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
        time_format="bjd"
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        lightcurves=lightcurves,
        survey_name="Kepler",
        data_release="DR25",
        **kwargs
    )


def create_generic_survey(coordinates: torch.Tensor, magnitudes: torch.Tensor,
                          bands: List[str], survey_name: str,
                          filter_system: str = "Generic",
                          errors: Optional[torch.Tensor] = None,
                          **kwargs) -> SurveyTensorDict:
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
        coordinates, coordinate_system="icrs", 
        unit="degree" if coordinates.shape[-1] == 2 else "parsec"
    )

    photometric = PhotometricTensorDict(
        magnitudes, bands=bands, errors=errors, filter_system=filter_system
    )

    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name=survey_name,
        **kwargs
    )


def merge_surveys(*surveys: SurveyTensorDict, 
                  new_name: str = "Merged") -> SurveyTensorDict:
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
        survey_name=new_name
    )
