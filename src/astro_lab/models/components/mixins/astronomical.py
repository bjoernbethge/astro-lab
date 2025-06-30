"""
Astronomical Domain Mixins for AstroLab Models
=============================================

Domain-specific mixins for astronomical data augmentation and loss functions.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class AstronomicalAugmentationMixin:
    """Astronomical data augmentation techniques."""

    def apply_photometric_noise(self, x: Tensor, noise_level: float = 0.02) -> Tensor:
        """Apply realistic photometric noise to astronomical features."""
        # Assume last few features are photometric measurements
        if x.size(-1) >= 3:
            # Apply higher noise to photometric features
            photometric_noise = torch.randn_like(x[:, -3:]) * noise_level * 2
            other_noise = torch.randn_like(x[:, :-3]) * noise_level

            noise = torch.cat([other_noise, photometric_noise], dim=-1)
        else:
            noise = torch.randn_like(x) * noise_level

        return x + noise

    def apply_coordinate_jitter(
        self, pos: Tensor, jitter_scale: float = 1e-5
    ) -> Tensor:
        """Apply small coordinate jitter (in degrees/arcsec)."""
        jitter = torch.randn_like(pos) * jitter_scale
        return pos + jitter

    def simulate_missing_observations(
        self, x: Tensor, missing_prob: float = 0.1
    ) -> Tensor:
        """Simulate missing astronomical observations."""
        mask = torch.rand_like(x) > missing_prob
        return x * mask.float()

    def apply_distance_uncertainty(
        self, x: Tensor, distance_col: int = 2, uncertainty: float = 0.1
    ) -> Tensor:
        """Apply realistic distance uncertainty to parallax/distance measurements."""
        if x.size(-1) > distance_col:
            x_copy = x.clone()
            distance_noise = torch.randn_like(x[:, distance_col]) * uncertainty
            x_copy[:, distance_col] = x[:, distance_col] + distance_noise
            return x_copy
        return x

    def apply_magnitude_noise(
        self, x: Tensor, magnitude_cols: list[int] = None
    ) -> Tensor:
        """Apply realistic magnitude-dependent noise."""
        if magnitude_cols is None:
            # Assume last 2-5 features are magnitudes
            magnitude_cols = list(range(max(0, x.size(-1) - 5), x.size(-1)))

        x_copy = x.clone()
        for col in magnitude_cols:
            if col < x.size(-1):
                # Brighter objects (lower magnitude) have lower noise
                magnitudes = x[:, col]
                # Noise scales as 1/sqrt(flux) ~ 10^(magnitude/5)
                noise_scale = 10.0 ** (magnitudes / 5.0) * 0.01
                noise = torch.randn_like(magnitudes) * noise_scale
                x_copy[:, col] = magnitudes + noise

        return x_copy

    def apply_redshift_uncertainty(self, x: Tensor, redshift_col: int = -1) -> Tensor:
        """Apply realistic redshift measurement uncertainty."""
        if x.size(-1) > abs(redshift_col):
            x_copy = x.clone()
            redshift = x[:, redshift_col]

            # Redshift uncertainty typically scales with redshift
            uncertainty = 0.001 * (1 + redshift)  # 0.1% baseline + redshift-dependent
            noise = torch.randn_like(redshift) * uncertainty

            x_copy[:, redshift_col] = redshift + noise
            return x_copy
        return x


class AstronomicalLossMixin:
    """Domain-specific loss functions for astronomical data."""

    def focal_loss(
        self, logits: Tensor, targets: Tensor, alpha: float = 1.0, gamma: float = 2.0
    ) -> Tensor:
        """Focal loss for imbalanced astronomical datasets."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def magnitude_aware_loss(
        self, logits: Tensor, targets: Tensor, magnitudes: Tensor
    ) -> Tensor:
        """Loss that accounts for magnitude-dependent noise in astronomical data."""
        # Brighter objects (lower magnitude) should have lower uncertainty
        weights = torch.exp(-0.4 * (magnitudes - magnitudes.min()))
        weights = weights / weights.mean()  # Normalize

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        weighted_loss = (ce_loss * weights).mean()
        return weighted_loss

    def distance_aware_regression_loss(
        self, predictions: Tensor, targets: Tensor, distances: Tensor
    ) -> Tensor:
        """Regression loss that accounts for distance-dependent uncertainties."""
        # More distant objects have higher uncertainty
        distance_weights = 1.0 / (1.0 + 0.1 * distances)
        distance_weights = distance_weights / distance_weights.mean()

        mse_loss = F.mse_loss(predictions, targets, reduction="none")
        weighted_loss = (mse_loss * distance_weights).mean()
        return weighted_loss

    def astronomical_uncertainty_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        uncertainties: Tensor,
        loss_type: str = "gaussian",
    ) -> Tensor:
        """Loss function that incorporates measurement uncertainties."""
        if loss_type == "gaussian":
            # Gaussian likelihood with known uncertainties
            log_likelihood = -0.5 * (
                ((predictions - targets) / uncertainties) ** 2
                + torch.log(2 * torch.pi * uncertainties**2)
            )
            return -log_likelihood.mean()

        elif loss_type == "huber":
            # Huber loss with uncertainty scaling
            delta = 1.0
            scaled_residuals = (predictions - targets) / uncertainties
            huber_loss = torch.where(
                torch.abs(scaled_residuals) <= delta,
                0.5 * scaled_residuals**2,
                delta * torch.abs(scaled_residuals) - 0.5 * delta**2,
            )
            return huber_loss.mean()

        else:
            # Default to weighted MSE
            weighted_mse = F.mse_loss(predictions, targets, reduction="none") / (
                uncertainties**2
            )
            return weighted_mse.mean()

    def redshift_consistency_loss(
        self,
        predicted_redshifts: Tensor,
        true_redshifts: Tensor,
        confidence: Optional[Tensor] = None,
    ) -> Tensor:
        """Loss that enforces redshift consistency constraints."""
        # Basic redshift loss
        redshift_loss = F.mse_loss(predicted_redshifts, true_redshifts)

        # Add physical constraints (redshifts should be positive)
        positivity_penalty = torch.mean(F.relu(-predicted_redshifts))

        # Add confidence weighting if available
        if confidence is not None:
            weighted_loss = (redshift_loss * confidence).mean()
        else:
            weighted_loss = redshift_loss

        return weighted_loss + 0.1 * positivity_penalty

    def photometric_consistency_loss(
        self,
        predicted_magnitudes: Tensor,
        true_magnitudes: Tensor,
        filters: list[str] = None,
    ) -> Tensor:
        """Loss that enforces photometric consistency across filters."""
        # Basic magnitude loss
        magnitude_loss = F.mse_loss(predicted_magnitudes, true_magnitudes)

        # Add color consistency if multiple filters
        if predicted_magnitudes.size(-1) > 1:
            # Compute colors (magnitude differences)
            pred_colors = predicted_magnitudes[:, 1:] - predicted_magnitudes[:, :-1]
            true_colors = true_magnitudes[:, 1:] - true_magnitudes[:, :-1]

            color_loss = F.mse_loss(pred_colors, true_colors)
            return magnitude_loss + 0.5 * color_loss

        return magnitude_loss
