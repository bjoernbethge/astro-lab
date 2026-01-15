"""
Enhanced Image Processing - Package Combiner
===========================================

Orchestrates existing package features:
- PyVista for 3D visualization and plotting
- AstroPhot for astronomical image modeling
- PhotUtils for photometric analysis
- NumPy for array operations
"""

import logging
from typing import Any, Dict, List, Union

import astrophot as ap
import numpy as np
import pyvista as pv
import torch
from photutils import aperture, background

from astro_lab.tensors import (
    AnalysisTensorDict,
    ImageTensorDict,
    SpatialTensorDict,
)

from .tensor_bridge import AstronomicalTensorBridge

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Package orchestrator for astronomical image processing.

    Uses existing features from:
    - PyVista for visualization and plotting
    - AstroPhot for astronomical modeling
    - PhotUtils for photometric analysis
    """

    def __init__(self):
        self.plotter = None

    def create_pyvista_visualization(
        self,
        tensordict: Union[
            SpatialTensorDict, AnalysisTensorDict, dict, np.ndarray, torch.Tensor
        ],
        plot_type: str = "points",
        **kwargs,
    ) -> pv.Plotter:
        """
        Use PyVista's built-in plotting capabilities. Robust to input format.

        Args:
            tensordict: Spatial or analysis tensor data
            plot_type: Type of PyVista plot
            **kwargs: PyVista plot parameters

        Returns:
            PyVista plotter object
        """
        bridge = AstronomicalTensorBridge()
        features = bridge.extract_features(tensordict)
        coords = features.get("coordinates")
        if coords is None:
            coords = bridge.validate_coordinates(tensordict)
        coords_np = coords.cpu().numpy()
        cloud = pv.PolyData(coords_np)
        self.plotter = pv.Plotter()
        if plot_type == "points":
            self.plotter.add_points(cloud, **kwargs)
        elif plot_type == "mesh":
            mesh = cloud.delaunay_3d()
            self.plotter.add_mesh(mesh, **kwargs)
        elif plot_type == "volume":
            self.plotter.add_volume(cloud, **kwargs)
        return self.plotter

    def use_astrophot_models(
        self, image: np.ndarray, model_type: str = "gaussian", **kwargs
    ) -> Dict[str, Any]:
        """
        Use AstroPhot's existing model capabilities.

        Args:
            image: Input image array
            model_type: Type of AstroPhot model
            **kwargs: AstroPhot parameters

        Returns:
            AstroPhot model and results
        """
        # Use AstroPhot's target image creation
        target = ap.image.Target_Image(
            data=image.astype(np.float64),
            zeropoint=kwargs.get("zeropoint", 25.0),
            variance="auto",
        )

        # Use AstroPhot's model factory functions
        if model_type == "gaussian":
            model = ap.models.gaussian_model(name="gaussian_source", target=target)
        elif model_type == "sersic":
            model = ap.models.sersic_model(name="galaxy", target=target)
        elif model_type == "point_source":
            model = ap.models.point_source(name="point_source", target=target)
        else:
            model = ap.models.gaussian_model(name="gaussian_source", target=target)

        # Use AstroPhot's fitting
        result = ap.fit.LM(model, verbose=0).fit()

        return {"model": model, "result": result, "target": target}

    def use_photutils_analysis(
        self, image: np.ndarray, analysis_type: str = "aperture_photometry", **kwargs
    ) -> Dict[str, Any]:
        """
        Use PhotUtils' existing analysis capabilities.

        Args:
            image: Input image array
            analysis_type: Type of PhotUtils analysis
            **kwargs: PhotUtils parameters

        Returns:
            PhotUtils analysis results
        """
        if analysis_type == "aperture_photometry":
            # Use PhotUtils' aperture photometry
            positions = kwargs.get(
                "positions", [(image.shape[1] // 2, image.shape[0] // 2)]
            )
            apertures_obj = aperture.CircularAperture(
                positions, r=kwargs.get("radius", 5)
            )
            phot_table = aperture.aperture_photometry(image, apertures_obj)

            return {
                "apertures": apertures_obj,
                "photometry": phot_table,
                "positions": positions,
            }

        elif analysis_type == "background":
            # Use PhotUtils' background estimation
            bkg = background.Background2D(
                image,
                box_size=kwargs.get("box_size", 50),
                filter_size=kwargs.get("filter_size", 3),
            )

            return {
                "background": bkg.background,
                "background_rms": bkg.background_rms,
                "background_model": bkg,
            }

        elif analysis_type == "detection":
            # Use PhotUtils' source detection
            from photutils import detect_sources, detect_threshold

            threshold = detect_threshold(image, nsigma=kwargs.get("nsigma", 2.0))
            segm = detect_sources(image, threshold, npixels=kwargs.get("npixels", 5))

            return {"segmentation": segm, "threshold": threshold}

        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

    def use_numpy_operations(
        self, image: np.ndarray, operation: str = "normalize", **kwargs
    ) -> np.ndarray:
        """
        Use NumPy's existing array operations.

        Args:
            image: Input image
            operation: NumPy operation to apply
            **kwargs: NumPy parameters

        Returns:
            Processed image
        """
        if operation == "normalize":
            # Use NumPy's normalization
            min_val = kwargs.get("min_val", 0)
            max_val = kwargs.get("max_val", 1)
            return (image - image.min()) / (image.max() - image.min()) * (
                max_val - min_val
            ) + min_val

        elif operation == "log":
            # Use NumPy's logarithmic scaling
            return np.log(image + kwargs.get("offset", 1e-10))

        elif operation == "sqrt":
            # Use NumPy's square root scaling
            return np.sqrt(np.abs(image))

        elif operation == "histogram_equalization":
            # Use NumPy's histogram equalization
            hist, bins = np.histogram(
                image.flatten(), bins=256, range=(image.min(), image.max())
            )
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            return np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(
                image.shape
            )

        else:
            return image

    def orchestrate_pipeline(
        self,
        tensordict: Union[
            SpatialTensorDict, ImageTensorDict, dict, np.ndarray, torch.Tensor
        ],
        pipeline_steps: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Orchestrate processing pipeline using existing package features. Robust to input format.

        Args:
            tensordict: Input tensor data
            pipeline_steps: List of processing steps
            **kwargs: Pipeline parameters

        Returns:
            Pipeline results
        """
        bridge = AstronomicalTensorBridge()
        features = bridge.extract_features(tensordict)
        current_image = None
        results = {}

        for i, step in enumerate(pipeline_steps):
            step_name = step.get("step", f"step_{i}")
            step_type = step.get("type")

            if step_type == "pyvista_visualization":
                # Use PyVista's visualization
                plotter = self.create_pyvista_visualization(
                    features,
                    plot_type=step.get("plot_type", "points"),
                    **step.get("params", {}),
                )
                results[step_name] = {
                    "plotter": plotter,
                    "type": "pyvista_visualization",
                }

            elif step_type == "astrophot_analysis":
                # Use AstroPhot's analysis
                if current_image is None:
                    if isinstance(tensordict, ImageTensorDict):
                        current_image = tensordict["images"][0].cpu().numpy()
                    else:
                        # Create simple visualization for spatial data
                        coords = tensordict["coordinates"].cpu().numpy()
                        current_image = np.zeros((100, 100), dtype=np.float64)
                        for x, y in coords[:1000]:  # Limit for visualization
                            ix, iy = int(x * 50 + 50), int(y * 50 + 50)
                            if 0 <= ix < 100 and 0 <= iy < 100:
                                current_image[ix, iy] += 1

                astro_result = self.use_astrophot_models(
                    current_image,
                    model_type=step.get("model_type", "gaussian"),
                    **step.get("params", {}),
                )
                results[step_name] = astro_result

            elif step_type == "photutils_analysis":
                # Use PhotUtils' analysis
                if current_image is None:
                    if isinstance(tensordict, ImageTensorDict):
                        current_image = tensordict["images"][0].cpu().numpy()
                    else:
                        # Create simple visualization for spatial data
                        coords = tensordict["coordinates"].cpu().numpy()
                        current_image = np.zeros((100, 100), dtype=np.float64)
                        for x, y in coords[:1000]:  # Limit for visualization
                            ix, iy = int(x * 50 + 50), int(y * 50 + 50)
                            if 0 <= ix < 100 and 0 <= iy < 100:
                                current_image[ix, iy] += 1

                phot_result = self.use_photutils_analysis(
                    current_image,
                    analysis_type=step.get("analysis_type", "aperture_photometry"),
                    **step.get("params", {}),
                )
                results[step_name] = phot_result

            elif step_type == "numpy_processing":
                # Use NumPy's processing
                if current_image is None:
                    if isinstance(tensordict, ImageTensorDict):
                        current_image = tensordict["images"][0].cpu().numpy()
                    else:
                        # Create simple visualization for spatial data
                        coords = tensordict["coordinates"].cpu().numpy()
                        current_image = np.zeros((100, 100), dtype=np.float64)
                        for x, y in coords[:1000]:  # Limit for visualization
                            ix, iy = int(x * 50 + 50), int(y * 50 + 50)
                            if 0 <= ix < 100 and 0 <= iy < 100:
                                current_image[ix, iy] += 1

                processed_image = self.use_numpy_operations(
                    current_image,
                    operation=step.get("operation", "normalize"),
                    **step.get("params", {}),
                )
                results[step_name] = {
                    "image": processed_image,
                    "type": "numpy_processing",
                }

        return results


# Convenience functions that use existing package features
def create_pyvista_visualization(
    tensordict: Union[
        SpatialTensorDict, AnalysisTensorDict, dict, np.ndarray, torch.Tensor
    ],
    **kwargs,
) -> pv.Plotter:
    """Create PyVista visualization using PyVista's native features. Robust to input format."""
    processor = ImageProcessor()
    return processor.create_pyvista_visualization(tensordict, **kwargs)


def use_astrophot_models(image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Use AstroPhot's existing model capabilities."""
    processor = ImageProcessor()
    return processor.use_astrophot_models(image, **kwargs)


def use_photutils_analysis(image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Use PhotUtils' existing analysis capabilities."""
    processor = ImageProcessor()
    return processor.use_photutils_analysis(image, **kwargs)


def use_numpy_operations(image: np.ndarray, **kwargs) -> np.ndarray:
    """Use NumPy's existing array operations."""
    processor = ImageProcessor()
    return processor.use_numpy_operations(image, **kwargs)


def orchestrate_pipeline(
    tensordict: Union[
        SpatialTensorDict, ImageTensorDict, dict, np.ndarray, torch.Tensor
    ],
    pipeline_steps: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Orchestrate processing pipeline using existing package features. Robust to input format."""
    processor = ImageProcessor()
    return processor.orchestrate_pipeline(tensordict, pipeline_steps, **kwargs)


__all__ = [
    "ImageProcessor",
    "create_pyvista_visualization",
    "use_astrophot_models",
    "use_photutils_analysis",
    "use_numpy_operations",
    "orchestrate_pipeline",
]
