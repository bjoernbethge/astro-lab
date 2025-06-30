import numpy as np
import torch

from .base import BaseSurveyProcessor


class RRLyraePreprocessor(BaseSurveyProcessor):
    def __init__(self, survey_name="rrlyrae", data_config=None):
        super().__init__(survey_name, data_config)

    def get_coordinate_columns(self):
        # Use RAJ2000/DEJ2000 as coordinates, Dist as distance if available
        return ["RAJ2000", "DEJ2000", "Dist"]

    def extract_coordinates(self, df):
        ra = df["RAJ2000"].to_numpy()
        dec = df["DEJ2000"].to_numpy()
        if "Dist" in df.columns:
            distance = df["Dist"].to_numpy()
        else:
            distance = np.ones_like(ra)
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(
            ra=ra * u.deg, dec=dec * u.deg, distance=distance * u.kpc, frame="icrs"
        )
        cart = coords.cartesian
        x = cart.x.value
        y = cart.y.value
        z = cart.z.value
        return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

    def extract_features(self, df):
        # Use all numeric columns except coordinates
        coord_cols = set(self.get_coordinate_columns())
        feature_cols = [
            col
            for col in df.columns
            if col not in coord_cols
            and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]
        ]
        features = [
            torch.tensor(df[col].to_numpy(), dtype=torch.float32)
            for col in feature_cols
        ]
        return torch.stack(features, dim=1) if features else None
