"""
Tests for Refactored Tensor Modules
==================================

Tests for the cleaned-up tensor system.
"""

import pytest
import torch

from src.astro_lab.tensors.base import AstroTensorBase, ValidationMixin
from src.astro_lab.tensors.constants import (
    ASTRO,
    CONSTANTS,
    GRAVITY,
    PHOTOMETRY,
    SPECTROSCOPY,
    get_mu,
    get_planet_radius,
)
from src.astro_lab.tensors.factory import TensorFactory
from src.astro_lab.tensors.tensor_types import (
    PhotometricTensorProtocol,
    Spatial3DTensorProtocol,
    TensorProtocol,
)
from src.astro_lab.tensors.transformations import TransformationRegistry


class TestConstants:
    """Test the centralized constants."""

    def test_astro_constants(self):
        """Test astronomical constants."""
        assert ASTRO.AU_KM == pytest.approx(149597870.7)
        assert ASTRO.EARTH_RADIUS_KM == pytest.approx(6378.1)
        assert CONSTANTS.SPEED_OF_LIGHT_KM_S == pytest.approx(299792.458)

    def test_gravity_constants(self):
        """Test gravitational parameters."""
        assert GRAVITY.EARTH == pytest.approx(398600.4418)
        assert GRAVITY.SUN == pytest.approx(132712440018.0)

    def test_legacy_functions(self):
        """Test backward compatibility."""
        assert get_mu("earth") == GRAVITY.EARTH
        assert get_planet_radius("earth") == ASTRO.EARTH_RADIUS_KM


class TestTensorTypes:
    """Test tensor protocols."""

    def test_protocol_imports(self):
        """Test protocols can be imported."""
        assert PhotometricTensorProtocol is not None
        assert Spatial3DTensorProtocol is not None
        assert TensorProtocol is not None


class TestTransformations:
    """Test transformation registry."""

    def test_registry_has_transformations(self):
        """Test transformations are loaded."""
        assert len(TransformationRegistry._transformations) > 0

    def test_gaia_to_sdss_available(self):
        """Test specific transformation."""
        func = TransformationRegistry.get_transformation("gaia", "sdss")
        assert func is not None


class TestFactory:
    """Test tensor factory."""

    def test_factory_exists(self):
        """Test factory class exists."""
        assert TensorFactory is not None
        assert hasattr(TensorFactory, "create_spatial")
        assert hasattr(TensorFactory, "create_survey")


class TestRefactoredBase:
    """Test the refactored base tensor class."""

    def test_base_creation(self):
        """Test basic tensor creation."""
        data = torch.randn(10, 3)
        tensor = AstroTensorBase(data, test_metadata="value")

        assert torch.equal(tensor.data, data)
        assert tensor.get_metadata("test_metadata") == "value"

    def test_validation_mixin(self):
        """Test validation patterns."""
        data = torch.randn(10, 3)
        tensor = AstroTensorBase(data)

        # Should not raise
        tensor.validate_shape(2)
        tensor.validate_non_empty()
        tensor.validate_finite_values()

    def test_tensor_operations(self):
        """Test tensor operations preserve metadata."""
        data = torch.randn(10, 3)
        tensor = AstroTensorBase(data, survey="test")

        cloned = tensor.clone()
        assert cloned.get_metadata("survey") == "test"

        detached = tensor.detach()
        assert detached.get_metadata("survey") == "test"


class TestIntegration:
    """Test integration between modules."""

    def test_imports_work_together(self):
        """Test all modules can be imported together."""
        from src.astro_lab.tensors import (
            ASTRO,
            AstroTensorBase,
            SurveyTensor,
            TensorFactory,
            TransformationRegistry,
        )

        assert ASTRO is not None
        assert TensorFactory is not None
        assert TransformationRegistry is not None
        assert AstroTensorBase is not None
        assert SurveyTensor is not None

    def test_factory_creates_tensors(self):
        """Test factory can create tensors."""
        # Create simple spatial data with distance
        ra = [10.0, 20.0, 30.0]
        dec = [5.0, 15.0, 25.0]
        distance = [100.0, 200.0, 300.0]  # Add distance for 3D

        spatial = TensorFactory.create_spatial(ra, dec, distance)
        assert spatial is not None
        assert spatial.shape[0] == 3  # 3 objects
        assert spatial.shape[1] == 3  # ra, dec, distance


if __name__ == "__main__":
    pytest.main([__file__])
