"""
Image TensorDict for AstroLab
=============================

TensorDict for astronomical image data with proper WCS and photometry support.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground

# Photometry and source detection
from photutils.detection import DAOStarFinder

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class ImageTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    TensorDict for astronomical image data.

    Features:
    - Proper WCS (World Coordinate System) handling
    - Source detection and aperture photometry
    - Background estimation and subtraction
    - PSF analysis and modeling
    - Multi-band image processing
    - Astrometric calibration support
    - Image quality assessment
    """

    def __init__(
        self,
        images: torch.Tensor,
        wcs: Optional[Union[WCS, List[WCS]]] = None,
        bands: Optional[List[str]] = None,
        pixel_scale: Optional[Union[float, List[float]]] = None,
        exposure_time: Optional[Union[float, torch.Tensor]] = None,
        zero_point: Optional[Union[float, torch.Tensor]] = None,
        image_type: str = "science",
        coordinates: Optional[SkyCoord] = None,
        **kwargs,
    ):
        """
        Initialize ImageTensorDict with proper astronomical metadata.

        Args:
            images: [N, C, H, W] Tensor with images (N=objects, C=bands, H/W=spatial)
            wcs: World Coordinate System(s) for astrometric calibration
            bands: Photometric band names
            pixel_scale: Pixel scale in arcsec/pixel
            exposure_time: Exposure time(s) in seconds
            zero_point: Photometric zero point(s)
            image_type: Type of image ('science', 'calibration', 'reference')
            coordinates: Central coordinates of images
        """
        if images.dim() != 4:
            raise ValueError(
                f"Images must be 4D tensor [N, C, H, W], got {images.shape}"
            )

        N, C, H, W = images.shape

        # Default bands if not provided
        if bands is None:
            bands = [f"band_{i}" for i in range(C)]
        elif len(bands) != C:
            raise ValueError(
                f"Number of bands ({len(bands)}) doesn't match channels ({C})"
            )

        # Handle WCS
        if wcs is not None:
            if isinstance(wcs, WCS):
                wcs = [wcs] * N  # Same WCS for all images
            elif len(wcs) != N:
                raise ValueError(
                    f"Number of WCS ({len(wcs)}) doesn't match images ({N})"
                )

        # Handle pixel scale
        if pixel_scale is not None:
            if isinstance(pixel_scale, (int, float)):
                pixel_scale = [pixel_scale] * N
            elif len(pixel_scale) != N:
                raise ValueError("Number of pixel scales doesn't match images")

        data = {
            "images": images,
            "meta": {
                "n_objects": N,
                "n_bands": C,
                "image_shape": (H, W),
                "bands": bands,
                "image_type": image_type,
                "pixel_scale": pixel_scale,
                "wcs_available": wcs is not None,
            },
        }

        if wcs is not None:
            data["wcs"] = wcs

        if exposure_time is not None:
            if isinstance(exposure_time, (int, float)):
                exposure_time = torch.tensor(exposure_time, dtype=torch.float32)
            data["exposure_time"] = exposure_time

        if zero_point is not None:
            if isinstance(zero_point, (int, float)):
                zero_point = torch.tensor(zero_point, dtype=torch.float32)
            data["zero_point"] = zero_point

        if coordinates is not None:
            data["coordinates"] = coordinates

        super().__init__(data, batch_size=(N,), **kwargs)

    @property
    def images(self) -> torch.Tensor:
        """Image data [N, C, H, W]."""
        return self["images"]

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Image spatial dimensions (H, W)."""
        return self._metadata["image_shape"]

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return self._metadata["n_bands"]

    @property
    def bands(self) -> List[str]:
        """Photometric band names."""
        return self._metadata["bands"]

    @property
    def image_type(self) -> str:
        """Image type identifier."""
        return self._metadata["image_type"]

    @property
    def pixel_scale(self) -> Optional[List[float]]:
        """Pixel scales in arcsec/pixel."""
        return self._metadata["pixel_scale"]

    @property
    def wcs_list(self) -> Optional[List[WCS]]:
        """List of WCS objects."""
        return self.get("wcs", None)

    def get_wcs(self, image_index: int = 0) -> Optional[WCS]:
        """Get WCS for specific image."""
        if "wcs" not in self:
            return None
        return self["wcs"][image_index]

    def pixel_to_sky(
        self,
        x_pix: Union[float, np.ndarray],
        y_pix: Union[float, np.ndarray],
        image_index: int = 0,
    ) -> Optional[SkyCoord]:
        """
        Convert pixel coordinates to sky coordinates.

        Args:
            x_pix: X pixel coordinate(s)
            y_pix: Y pixel coordinate(s)
            image_index: Which image's WCS to use

        Returns:
            Sky coordinates or None if no WCS available
        """
        wcs = self.get_wcs(image_index)
        if wcs is None:
            return None

        sky_coords = wcs.pixel_to_world(x_pix, y_pix)
        return sky_coords

    def sky_to_pixel(
        self, coordinates: SkyCoord, image_index: int = 0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert sky coordinates to pixel coordinates.

        Args:
            coordinates: Sky coordinates
            image_index: Which image's WCS to use

        Returns:
            Tuple of (x_pix, y_pix) arrays or None if no WCS
        """
        wcs = self.get_wcs(image_index)
        if wcs is None:
            return None

        x_pix, y_pix = wcs.world_to_pixel(coordinates)
        return x_pix, y_pix

    def estimate_background(
        self, method: str = "median", box_size: int = 50, filter_size: int = 3
    ) -> torch.Tensor:
        """
        Estimate image background using photutils.

        Args:
            method: Background estimation method ('median', 'mode', 'sigma_clip')
            box_size: Size of background estimation boxes
            filter_size: Size of median filter for background map

        Returns:
            Background estimate [N, C, H, W]
        """
        backgrounds = torch.zeros_like(self.images)

        for n in range(self.n_objects):
            for c in range(self.n_bands):
                image = self.images[n, c].detach().cpu().numpy()

                if method in ["median", "mode"]:
                    # Use photutils for robust background estimation
                    bkg_estimator = MedianBackground()
                    bkg = Background2D(
                        image,
                        box_size=(box_size, box_size),
                        filter_size=(filter_size, filter_size),
                        bkg_estimator=bkg_estimator,
                    )
                    background = bkg.background
                else:
                    # background estimation with sigma clipping
                    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
                    background = np.full_like(image, median)

                backgrounds[n, c] = torch.tensor(background, dtype=torch.float32)

        return backgrounds

    def subtract_background(
        self, method: str = "median", **kwargs
    ) -> "ImageTensorDict":
        """
        Subtract background from images.

        Args:
            method: Background estimation method
            **kwargs: Additional arguments for background estimation
        """
        backgrounds = self.estimate_background(method=method, **kwargs)
        background_subtracted = self.images - backgrounds

        result = ImageTensorDict(
            background_subtracted,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("subtract_background", method=method)
        return result

    def detect_sources(
        self, threshold: float = 5.0, fwhm: float = 3.0, band_index: int = 0
    ) -> List[Dict]:
        """
        Detect sources in images using DAOStarFinder.

        Args:
            threshold: Detection threshold in sigma above background
            fwhm: Expected FWHM of sources in pixels
            band_index: Which band to use for detection

        Returns:
            List of source catalogs (one per image)
        """
        source_catalogs = []

        for n in range(self.n_objects):
            image = self.images[n, band_index].detach().cpu().numpy()

            # Estimate background statistics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)

            # Use DAOStarFinder for robust source detection
            daofind = DAOStarFinder(
                fwhm=fwhm, threshold=threshold * std, exclude_border=True
            )
            sources = daofind(image - median)

            if sources is not None:
                catalog = {
                    "x": sources["xcentroid"].data,
                    "y": sources["ycentroid"].data,
                    "flux": sources["flux"].data,
                    "magnitude": sources["mag"].data
                    if "mag" in sources.colnames
                    else None,
                    "n_sources": len(sources),
                }
            else:
                catalog = {
                    "x": np.array([]),
                    "y": np.array([]),
                    "flux": np.array([]),
                    "magnitude": None,
                    "n_sources": 0,
                }

            source_catalogs.append(catalog)

        return source_catalogs

    def aperture_photometry(
        self,
        positions: List[Tuple[float, float]],
        aperture_radius: float = 5.0,
        annulus_inner: float = 10.0,
        annulus_outer: float = 15.0,
        image_index: int = 0,
        band_index: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Perform aperture photometry on detected sources.

        Args:
            positions: List of (x, y) source positions
            aperture_radius: Radius of photometry aperture in pixels
            annulus_inner: Inner radius of background annulus
            annulus_outer: Outer radius of background annulus
            image_index: Which image to analyze
            band_index: Which band to analyze

        Returns:
            Dictionary with photometry results
        """
        if not positions:
            return {
                "aperture_sum": np.array([]),
                "background": np.array([]),
                "net_flux": np.array([]),
                "magnitude": np.array([]),
            }

        image = self.images[image_index, band_index].detach().cpu().numpy()

        # Create apertures
        apertures = CircularAperture(positions, r=aperture_radius)
        annulus_apertures = CircularAnnulus(
            positions, r_in=annulus_inner, r_out=annulus_outer
        )

        # Perform photometry
        phot_table = aperture_photometry(image, apertures)
        bkg_table = aperture_photometry(image, annulus_apertures)

        # Calculate background per pixel
        bkg_mean = bkg_table["aperture_sum"] / annulus_apertures.area

        # Background-subtracted flux
        aperture_area = apertures.area
        total_bkg = bkg_mean * aperture_area
        net_flux = phot_table["aperture_sum"] - total_bkg

        # Convert to magnitudes if zero point available
        magnitudes = None
        if "zero_point" in self:
            zp = self["zero_point"]
            if isinstance(zp, torch.Tensor):
                zp = zp[image_index].item()
            magnitudes = -2.5 * np.log10(np.clamp(net_flux, 1e-10, None)) + zp

        return {
            "aperture_sum": phot_table["aperture_sum"].data,
            "background": total_bkg.data,
            "net_flux": net_flux.data,
            "magnitude": magnitudes.data if magnitudes is not None else None,
        }

    def estimate_psf_fwhm(
        self,
        source_positions: Optional[List[Tuple[float, float]]] = None,
        image_index: int = 0,
        band_index: int = 0,
        box_size: int = 25,
    ) -> float:
        """
        Estimate PSF FWHM from bright sources.

        Args:
            source_positions: List of source positions (if None, detect automatically)
            image_index: Which image to analyze
            band_index: Which band to analyze
            box_size: Size of cutout around each source

        Returns:
            Estimated FWHM in pixels
        """
        image = self.images[image_index, band_index].detach().cpu().numpy()

        if source_positions is None:
            # Auto-detect sources
            catalogs = self.detect_sources(band_index=band_index)
            if not catalogs[image_index]["n_sources"]:
                return 3.0  # Default fallback

            # Use brightest sources
            fluxes = catalogs[image_index]["flux"]
            x_coords = catalogs[image_index]["x"]
            y_coords = catalogs[image_index]["y"]

            # Sort by flux and take top 10
            sorted_indices = np.argsort(fluxes)[::-1][:10]
            source_positions = [(x_coords[i], y_coords[i]) for i in sorted_indices]

        fwhm_estimates = []

        for x, y in source_positions:
            # Extract cutout around source
            x_int, y_int = int(round(x)), int(round(y))
            half_box = box_size // 2

            x_min = max(0, x_int - half_box)
            x_max = min(image.shape[1], x_int + half_box)
            y_min = max(0, y_int - half_box)
            y_max = min(image.shape[0], y_int + half_box)

            cutout = image[y_min:y_max, x_min:x_max]

            if cutout.size < 25:  # Too small
                continue

            # Find peak and estimate FWHM
            peak_y, peak_x = np.unravel_index(np.argmax(cutout), cutout.shape)
            peak_val = cutout[peak_y, peak_x]

            # Background estimate
            background = np.median(cutout)
            signal = peak_val - background

            if signal <= 0:
                continue

            # Find half-maximum contour
            half_max = background + signal / 2.0

            # FWHM estimate along x and y axes
            x_profile = cutout[peak_y, :]
            y_profile = cutout[:, peak_x]

            # Find width at half maximum
            x_fwhm = self._estimate_fwhm_1d(x_profile, half_max)
            y_fwhm = self._estimate_fwhm_1d(y_profile, half_max)

            if x_fwhm > 0 and y_fwhm > 0:
                fwhm_estimates.append(np.sqrt(x_fwhm * y_fwhm))

        if fwhm_estimates:
            return float(np.median(fwhm_estimates))
        else:
            return 3.0  # Fallback default

    def _estimate_fwhm_1d(self, profile: np.ndarray, half_max: float) -> float:
        """Estimate FWHM from 1D profile."""
        above_half = profile > half_max
        if not np.any(above_half):
            return 0.0

        # Find edges of half-maximum region
        indices = np.where(above_half)[0]
        if len(indices) < 2:
            return 0.0

        return float(indices[-1] - indices[0])

    def convolve_psf(
        self, fwhm: float, kernel_size: Optional[int] = None
    ) -> "ImageTensorDict":
        """
        Convolve images with Gaussian PSF.

        Args:
            fwhm: FWHM of Gaussian kernel in pixels
            kernel_size: Size of convolution kernel (if None, auto-determine)

        Returns:
            Convolved ImageTensorDict
        """
        sigma = gaussian_fwhm_to_sigma * fwhm

        if kernel_size is None:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)

        # Create Gaussian kernel
        kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)
        kernel_array = kernel.array

        convolved_images = torch.zeros_like(self.images)

        for n in range(self.n_objects):
            for c in range(self.n_bands):
                image = self.images[n, c].detach().cpu().numpy()
                convolved = convolve(image, kernel_array, boundary="extend")
                convolved_images[n, c] = torch.tensor(convolved, dtype=torch.float32)

        result = ImageTensorDict(
            convolved_images,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("convolve_psf", fwhm=fwhm, kernel_size=kernel_size)
        return result

    def crop_center(self, crop_size: Tuple[int, int]) -> "ImageTensorDict":
        """Crop center region of images."""
        N, C, H, W = self.images.shape
        crop_h, crop_w = crop_size

        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        cropped = self.images[
            :, :, start_h : start_h + crop_h, start_w : start_w + crop_w
        ]

        result = ImageTensorDict(
            cropped,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("crop_center", crop_size=crop_size)
        return result

    def resize(self, target_size: Tuple[int, int]) -> "ImageTensorDict":
        """Resize images to target size."""
        import torch.nn.functional as F

        images = F.interpolate(
            self.images, size=target_size, mode="bilinear", align_corners=False
        )

        result = ImageTensorDict(
            images,
            self.wcs_list,
            self.bands,
            pixel_scale=self.pixel_scale,
            exposure_time=self.get("exposure_time", None),
            zero_point=self.get("zero_point", None),
            image_type=self.image_type,
            coordinates=self.get("coordinates", None),
        )
        result.add_history("resize", target_size=target_size)
        return result

    def extract_image_features(self) -> torch.Tensor:
        """
        Extract comprehensive image features for analysis.

        Returns:
            [N, F] Feature tensor with image properties
        """
        features = []

        for n in range(self.n_objects):
            img_features = []

            for c in range(self.n_bands):
                image = self.images[n, c]

                # Basic statistics
                img_features.extend(
                    [
                        torch.mean(image),
                        torch.std(image),
                        torch.median(image),
                        torch.min(image),
                        torch.max(image),
                    ]
                )

                # Image structure
                gradient_x = torch.diff(image, dim=1)
                gradient_y = torch.diff(image, dim=0)

                img_features.extend(
                    [
                        torch.mean(torch.abs(gradient_x)),  # Edge strength X
                        torch.mean(torch.abs(gradient_y)),  # Edge strength Y
                        torch.std(gradient_x),  # Texture X
                        torch.std(gradient_y),  # Texture Y
                    ]
                )

            features.append(torch.stack(img_features))

        return torch.stack(features)

    def validate(self) -> bool:
        """Validate image tensor data."""
        # Basic validation from parent
        if not super().validate():
            return False

        # Image-specific validation
        return (
            "images" in self
            and self.images.dim() == 4
            and self.images.shape[1] == len(self.bands)
            and self.images.shape[2] > 10  # Minimum image size
            and self.images.shape[3] > 10
            and torch.all(torch.isfinite(self.images))
        )
