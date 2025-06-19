"""
Advanced tensor processing for astronomical data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProcessingConfig(BaseModel):
    """Type-safe configuration for tensor processing."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )

    device: str = Field(default="auto", description="Device for tensor operations")
    batch_size: int = Field(
        default=32, ge=1, le=10000, description="Batch size for processing"
    )
    max_samples: Optional[Dict[str, int]] = Field(
        default=None, description="Maximum samples per survey"
    )
    surveys: Optional[List[str]] = Field(
        default=None, description="List of surveys to process"
    )

    @field_validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device string."""
        if v not in ["auto", "cpu", "cuda", "mps"]:
            raise ValueError(
                f"device must be one of ['auto', 'cpu', 'cuda', 'mps'], got {v}"
            )
        return v

    def __init__(self, **data: Any) -> None:
        """Initialize with default values."""
        # Set defaults for max_samples and surveys if not provided
        if "max_samples" not in data or data["max_samples"] is None:
            data["max_samples"] = {
                "gaia": 10000,
                "sdss_spectral": 1000,
                "linear_lightcurve": 500,
                "rrlyrae": 300,
                "nsa": 5000,
                "exoplanet": 2000,
                "satellite": 100,
                "tng50": 5000,
                "astrophot": 1000,
            }

        if "surveys" not in data or data["surveys"] is None:
            data["surveys"] = [
                "gaia",
                "sdss_spectral",
                "linear_lightcurve",
                "rrlyrae",
                "nsa",
                "exoplanet",
                "satellite",
                "tng50",
                "astrophot",
            ]

        super().__init__(**data)


class AstroTensorProcessor:
    """
    Simplified processor for astronomical tensors.

    Focuses on core tensor operations without complex batch processing.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the tensor processor.

        Parameters
        ----------
        config : ProcessingConfig, optional
            Processing configuration
        """
        self.config = config or ProcessingConfig()

        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        print(f"🚀 AstroTensorProcessor initialized on {self.device}")

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to processing device."""
        return tensor.to(self.device)

    def create_spatial_data(
        self,
        coordinates: torch.Tensor,
        ra: Optional[torch.Tensor] = None,
        dec: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create spatial data dictionary from coordinate data.

        Parameters
        ----------
        coordinates : torch.Tensor
            Coordinate tensor [N, 3]
        ra : torch.Tensor, optional
            Right ascension
        dec : torch.Tensor, optional
            Declination
        distance : torch.Tensor, optional
            Distance values

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with spatial data
        """
        try:
            # Move to device
            coordinates = self.to_device(coordinates)

            spatial_data = {
                "coordinates": coordinates,
                "coordinate_system": "icrs",
                "unit": "Mpc",
            }

            if ra is not None:
                spatial_data["ra"] = self.to_device(ra)
            if dec is not None:
                spatial_data["dec"] = self.to_device(dec)
            if distance is not None:
                spatial_data["distance"] = self.to_device(distance)

            print(f"✅ Created spatial data: {coordinates.shape}")
            return spatial_data

        except Exception as e:
            print(f"❌ Error creating spatial data: {e}")
            return {}

    def create_survey_data(
        self,
        features: torch.Tensor,
        survey_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create survey data dictionary from feature data.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor
        survey_name : str
            Name of the survey
        metadata : Dict[str, Any], optional
            Additional metadata

        Returns
        -------
        Dict[str, Any]
            Dictionary with survey data
        """
        try:
            # Move to device
            features = self.to_device(features)

            # Create data dictionary
            survey_data = {
                "features": features,
                "survey_name": survey_name,
                "shape": features.shape,
                "device": str(features.device),
                **(metadata or {}),
            }

            print(f"✅ Created survey data for {survey_name}: {features.shape}")
            return survey_data

        except Exception as e:
            print(f"❌ Error creating survey data for {survey_name}: {e}")
            return {}

    def process_coordinate_data(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Process astronomical coordinate data.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing coordinate tensors

        Returns
        -------
        Dict[str, torch.Tensor]
            Processed coordinate data
        """
        processed = {}

        for key, tensor in data.items():
            # Move to device
            tensor = self.to_device(tensor)

            # Basic processing
            if "ra" in key.lower():
                # Ensure RA is in [0, 360] range
                processed[key] = torch.fmod(tensor, 360.0)
            elif "dec" in key.lower():
                # Ensure Dec is in [-90, 90] range
                processed[key] = torch.clamp(tensor, -90.0, 90.0)
            elif "distance" in key.lower() or "dist" in key.lower():
                # Ensure distances are positive
                processed[key] = torch.abs(tensor)
            else:
                processed[key] = tensor

        return processed

    def normalize_magnitudes(
        self, magnitudes: torch.Tensor, mag_zero: float = 0.0, mag_range: float = 30.0
    ) -> torch.Tensor:
        """
        Normalize magnitude values for ML processing.

        Parameters
        ----------
        magnitudes : torch.Tensor
            Magnitude tensor
        mag_zero : float
            Zero point for normalization
        mag_range : float
            Magnitude range for scaling

        Returns
        -------
        torch.Tensor
            Normalized magnitudes
        """
        magnitudes = self.to_device(magnitudes)

        # Handle missing values (typically 99.0 in astronomical data)
        mask = magnitudes < 90.0
        normalized = torch.zeros_like(magnitudes)

        # Normalize valid magnitudes to [0, 1]
        valid_mags = magnitudes[mask]
        if len(valid_mags) > 0:
            normalized[mask] = (valid_mags - mag_zero) / mag_range

        return torch.clamp(normalized, 0.0, 1.0)

    def compute_colors(
        self, magnitudes: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute astronomical colors from magnitudes.

        Parameters
        ----------
        magnitudes : Dict[str, torch.Tensor]
            Dictionary of magnitude tensors

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of color tensors
        """
        colors = {}

        # Common color combinations
        color_pairs = [
            ("u", "g"),
            ("g", "r"),
            ("r", "i"),
            ("i", "z"),
            ("g", "i"),
            ("r", "z"),
            ("u", "r"),
        ]

        for mag1, mag2 in color_pairs:
            if mag1 in magnitudes and mag2 in magnitudes:
                color_name = f"{mag1}_{mag2}"
                m1 = self.to_device(magnitudes[mag1])
                m2 = self.to_device(magnitudes[mag2])

                # Only compute colors for valid magnitudes
                mask = (m1 < 90.0) & (m2 < 90.0)
                color = torch.zeros_like(m1)
                color[mask] = m1[mask] - m2[mask]
                colors[color_name] = color

        return colors

    def save_tensors(
        self, tensors: Dict[str, Any], output_dir: Union[str, Path]
    ) -> Path:
        """
        Save processed tensors to disk.

        Parameters
        ----------
        tensors : Dict[str, Any]
            Tensor dictionary to save
        output_dir : Union[str, Path]
            Output directory

        Returns
        -------
        Path
            Path to saved tensors
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tensor_file = output_dir / "astro_tensors.pt"

        # Convert complex objects to regular tensors for saving
        save_dict = {}
        for name, tensor_obj in tensors.items():
            if isinstance(tensor_obj, dict) and "features" in tensor_obj:
                # Save the features tensor from survey data
                save_dict[name] = tensor_obj["features"]
            elif isinstance(tensor_obj, dict) and "coordinates" in tensor_obj:
                # Save the coordinates tensor from spatial data
                save_dict[name] = tensor_obj["coordinates"]
            elif isinstance(tensor_obj, torch.Tensor):
                save_dict[name] = tensor_obj
            else:
                # Skip non-tensor objects
                continue

        if save_dict:
            torch.save(save_dict, tensor_file)
            print(f"💾 Tensors saved to {tensor_file}")
            print(f"   - {len(save_dict)} tensor objects")
        else:
            print("⚠️  No tensors to save")

        return tensor_file

    def load_tensors(self, tensor_file: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load processed tensors from disk.

        Parameters
        ----------
        tensor_file : Union[str, Path]
            Path to tensor file

        Returns
        -------
        Dict[str, torch.Tensor]
            Loaded tensors
        """
        tensor_file = Path(tensor_file)

        if not tensor_file.exists():
            raise FileNotFoundError(f"Tensor file not found: {tensor_file}")

        tensors = torch.load(tensor_file, map_location=self.device)

        print(f"📂 Loaded tensors from {tensor_file}")
        print(f"   - {len(tensors)} tensor objects")

        return tensors


# Convenience functions
def create_simple_processor(device: str = "auto") -> AstroTensorProcessor:
    """Create a simple tensor processor."""
    config = ProcessingConfig(device=device)
    return AstroTensorProcessor(config=config)


def process_coordinate_dict(
    coordinates: Dict[str, torch.Tensor], device: str = "auto"
) -> Dict[str, torch.Tensor]:
    """Process a dictionary of coordinate tensors."""
    processor = create_simple_processor(device=device)
    return processor.process_coordinate_data(coordinates)


def normalize_astronomical_data(
    data: Dict[str, torch.Tensor], device: str = "auto"
) -> Dict[str, torch.Tensor]:
    """Normalize astronomical data for ML processing."""
    processor = create_simple_processor(device=device)

    normalized = {}
    for key, tensor in data.items():
        if "mag" in key.lower():
            # Normalize magnitudes
            normalized[key] = processor.normalize_magnitudes(tensor)
        elif any(coord in key.lower() for coord in ["ra", "dec", "distance"]):
            # Process coordinates
            coord_dict = {key: tensor}
            processed = processor.process_coordinate_data(coord_dict)
            normalized.update(processed)
        else:
            # Move to device without special processing
            normalized[key] = processor.to_device(tensor)

    return normalized


__all__ = [
    "ProcessingConfig",
    "AstroTensorProcessor",
    "create_simple_processor",
    "process_coordinate_dict",
    "normalize_astronomical_data",
]
