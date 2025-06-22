from typing import Optional
import torch

class SurveyTensor(Spatial3DTensorProtocol, PhotometricTensorProtocol):
    def __init__(
        self,
        spatial: Spatial3DTensorProtocol,
        photometric: PhotometricTensorProtocol,
        features: Optional[torch.Tensor] = None,
        survey_name: str = "unknown",
        **kwargs,
    ):
        """
        Initializes the SurveyTensor from specialized sub-tensors.
        """
        super().__init__(data=spatial.data, survey_name=survey_name, **kwargs)
        self.spatial = spatial
        self.photometric = photometric
        self.features = features
        self._metadata["survey_name"] = survey_name
        if hasattr(photometric, 'filter_system'):
            self._metadata["filter_system"] = photometric.filter_system
        if hasattr(spatial, 'coordinate_system'):
            self._metadata["coordinate_system"] = spatial.coordinate_system
        self._validate() 