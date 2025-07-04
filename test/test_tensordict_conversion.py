"""
Unit Tests for TensorDict Conversion Functions
==============================================

Tests for the new conversion functions that replace the legacy migration functions.
"""

import pytest
import numpy as np
import polars as pl
import torch
from tensordict import TensorDict
from torch_geometric.data import Data, HeteroData

from astro_lab.data.preprocessors.tensordict_migration import (
    data_to_tensordict,
    pyg_data_to_tensordict,
    heterodata_to_tensordict,
    dict_to_tensordict,
    dataframe_to_tensordict,
    ensure_tensordict_compatibility,
    batch_convert_files,
)
from astro_lab.tensors.base import AstroTensorDict
from astro_lab.tensors.spatial import SpatialTensorDict


class TestDataToTensorDict:
    """Test the main data_to_tensordict function."""
    
    def test_data_to_tensordict_with_pyg_data(self):
        """Test conversion of PyG Data to TensorDict."""
        # Create PyG Data
        pyg_data = Data(
            x=torch.randn(100, 5),
            pos=torch.randn(100, 3),
            edge_index=torch.randint(0, 100, (2, 200)),
            y=torch.randint(0, 3, (100,))
        )
        
        # Convert to TensorDict
        result = data_to_tensordict(pyg_data, survey_name="test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 100
        assert "coordinates" in result
        assert "features" in result
        assert result.meta["survey"] == "test"
        assert result.meta["source"] == "pyg_data_conversion"
    
    def test_data_to_tensordict_with_dict(self):
        """Test conversion of dictionary to TensorDict."""
        data_dict = {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
            "magnitude": [12.0, 13.0, 14.0],
        }
        
        result = data_to_tensordict(data_dict, survey_name="test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 3
        assert "coordinates" in result
        assert result.meta["survey"] == "test"
    
    def test_data_to_tensordict_with_dataframe(self):
        """Test conversion of DataFrame to TensorDict."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0], 
            "z": [7.0, 8.0, 9.0],
            "mag": [12.0, 13.0, 14.0],
            "color": [0.5, 0.6, 0.7],
        })
        
        result = data_to_tensordict(df, survey_name="test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 3
        assert "coordinates" in result
        assert result.meta["survey"] == "test"
    
    def test_data_to_tensordict_with_device(self):
        """Test device handling in conversion."""
        pyg_data = Data(
            x=torch.randn(10, 3),
            pos=torch.randn(10, 3),
        )
        
        # Test CPU device
        result_cpu = data_to_tensordict(pyg_data, device="cpu")
        assert result_cpu.device.type == "cpu"
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            result_cuda = data_to_tensordict(pyg_data, device="cuda")
            assert result_cuda.device.type == "cuda"
    
    def test_data_to_tensordict_with_memmap(self, tmp_path):
        """Test memory mapping functionality."""
        pyg_data = Data(
            x=torch.randn(50, 4),
            pos=torch.randn(50, 3),
        )
        
        memmap_path = str(tmp_path / "test_memmap")
        result = data_to_tensordict(
            pyg_data, 
            use_memmap=True, 
            memmap_path=memmap_path
        )
        
        assert isinstance(result, AstroTensorDict)
        # Check that memmap was applied (TensorDict has internal _is_memmap attribute)
        assert hasattr(result, '_is_memmap') or any(
            hasattr(tensor, '_is_memmap') for tensor in result.values() 
            if isinstance(tensor, torch.Tensor)
        )


class TestPygDataToTensorDict:
    """Test PyG Data to TensorDict conversion."""
    
    def test_basic_conversion(self):
        """Test basic PyG Data conversion."""
        data = Data(
            x=torch.randn(20, 4),
            pos=torch.randn(20, 3),
            edge_index=torch.randint(0, 20, (2, 40)),
            y=torch.randint(0, 2, (20,))
        )
        
        result = pyg_data_to_tensordict(data, survey_name="test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 20
        assert torch.equal(result["coordinates"], data.pos)
        assert torch.equal(result["features"], data.x)
        assert torch.equal(result["edge_index"], data.edge_index)
        assert torch.equal(result["labels"], data.y)
    
    def test_spatial_creation(self):
        """Test creation of SpatialTensorDict."""
        data = Data(
            x=torch.randn(15, 3),
            pos=torch.randn(15, 3),
        )
        
        result = pyg_data_to_tensordict(
            data, 
            survey_name="test", 
            create_spatial=True
        )
        
        assert isinstance(result, SpatialTensorDict)
        assert result.n_objects == 15
        assert torch.equal(result.coordinates, data.pos)
    
    def test_additional_attributes(self):
        """Test handling of additional attributes."""
        data = Data(
            x=torch.randn(10, 2),
            pos=torch.randn(10, 3),
            custom_attr=torch.randn(10, 1),
            non_tensor_attr="test"
        )
        
        result = pyg_data_to_tensordict(data, survey_name="test")
        
        assert "custom_attr" in result
        assert torch.equal(result["custom_attr"], data.custom_attr)
        # Non-tensor attributes should not be included
        assert "non_tensor_attr" not in result


class TestHeteroDataToTensorDict:
    """Test HeteroData to TensorDict conversion."""
    
    def test_basic_hetero_conversion(self):
        """Test basic heterogeneous data conversion."""
        hetero_data = HeteroData()
        
        # Add node types
        hetero_data['node_type_1'].x = torch.randn(10, 4)
        hetero_data['node_type_2'].x = torch.randn(15, 3)
        
        # Add edge types
        hetero_data['node_type_1', 'edge_type', 'node_type_2'].edge_index = torch.randint(0, 10, (2, 20))
        
        result = heterodata_to_tensordict(hetero_data, survey_name="test")
        
        assert isinstance(result, TensorDict)
        assert "node_type_1" in result
        assert "node_type_2" in result
        assert "node_type_1__to__node_type_2" in result
        assert result["meta"]["hetero"] is True
        assert result["meta"]["survey"] == "test"


class TestDictToTensorDict:
    """Test dictionary to TensorDict conversion."""
    
    def test_numpy_array_conversion(self):
        """Test conversion of numpy arrays."""
        data = {
            "features": np.random.randn(25, 5),
            "positions": np.random.randn(25, 3),
            "labels": np.random.randint(0, 3, 25),
        }
        
        result = dict_to_tensordict(data, survey_name="test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 25
        assert isinstance(result["features"], torch.Tensor)
        assert isinstance(result["positions"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)
    
    def test_coordinate_creation(self):
        """Test automatic coordinate tensor creation."""
        data = {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
            "other_feature": [0.1, 0.2, 0.3],
        }
        
        result = dict_to_tensordict(data, survey_name="test")
        
        assert "coordinates" in result
        assert result["coordinates"].shape == (3, 3)
        expected_coords = torch.tensor([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        assert torch.allclose(result["coordinates"], expected_coords)
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        data = {
            "numeric": [1.0, 2.0, 3.0],
            "string": ["a", "b", "c"],
            "tensor": torch.tensor([4.0, 5.0, 6.0]),
            "array": np.array([7.0, 8.0, 9.0]),
        }
        
        result = dict_to_tensordict(data, survey_name="test")
        
        assert isinstance(result["numeric"], torch.Tensor)
        assert isinstance(result["tensor"], torch.Tensor)
        assert isinstance(result["array"], torch.Tensor)
        # String data should be kept as-is
        assert result["string"] == ["a", "b", "c"]


class TestDataFrameToTensorDict:
    """Test DataFrame to TensorDict conversion."""
    
    def test_numeric_columns(self):
        """Test conversion of numeric columns."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
            "z": [9.0, 10.0, 11.0, 12.0],
            "magnitude": [13.0, 14.0, 15.0, 16.0],
            "color_index": [0.5, 0.6, 0.7, 0.8],
            "non_numeric": ["a", "b", "c", "d"],
        })
        
        result = dataframe_to_tensordict(df, survey_name="test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 4
        assert "coordinates" in result
        assert result["coordinates"].shape == (4, 3)
        
        # Check that features were created
        assert "features" in result
        assert result["features"].shape[1] > 0  # Should have at least some features
    
    def test_feature_tensor_creation(self):
        """Test automatic feature tensor creation."""
        df = pl.DataFrame({
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
            "feat1": [7.0, 8.0],
            "feat2": [9.0, 10.0],
        })
        
        result = dataframe_to_tensordict(df, survey_name="test")
        
        assert "features" in result
        # Features should exclude coordinate columns
        expected_feature_cols = ["feat1", "feat2"]
        assert result.meta["feature_names"] == expected_feature_cols
        assert result["features"].shape == (2, 2)
    
    def test_metadata_preservation(self):
        """Test that metadata is properly preserved."""
        df = pl.DataFrame({
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
            "magnitude": [12.0, 13.0],
        })
        
        result = dataframe_to_tensordict(
            df, 
            survey_name="test_survey",
            coordinate_system="galactic",
            unit="kpc",
            epoch="J2015.5"
        )
        
        assert result.meta["survey"] == "test_survey"
        assert result.meta["coordinate_system"] == "galactic"
        assert result.meta["unit"] == "kpc"
        assert result.meta["epoch"] == "J2015.5"
        assert result.meta["num_objects"] == 2


class TestEnsureTensorDictCompatibility:
    """Test the compatibility ensuring function."""
    
    def test_already_tensordict(self):
        """Test with already compatible TensorDict."""
        original = AstroTensorDict({
            "coordinates": torch.randn(5, 3),
            "features": torch.randn(5, 2),
        })
        
        result = ensure_tensordict_compatibility(original, "test")
        
        # Should return the same object
        assert result is original
        assert isinstance(result, AstroTensorDict)
    
    def test_conversion_needed(self):
        """Test conversion of non-TensorDict data."""
        data = {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
        }
        
        result = ensure_tensordict_compatibility(data, "test")
        
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 3
        assert result.meta["survey"] == "test"
    
    def test_device_and_memmap_handling(self, tmp_path):
        """Test device and memmap handling in compatibility function."""
        data = Data(
            x=torch.randn(10, 2),
            pos=torch.randn(10, 3),
        )
        
        memmap_path = str(tmp_path / "compat_memmap")
        result = ensure_tensordict_compatibility(
            data, 
            "test",
            device="cpu",
            use_memmap=True,
            memmap_path=memmap_path
        )
        
        assert isinstance(result, AstroTensorDict)
        assert result.device.type == "cpu"


class TestBatchConvertFiles:
    """Test batch file conversion functionality."""
    
    def test_batch_convert_files(self, tmp_path):
        """Test batch conversion of files."""
        # Create test input directory
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create test files
        test_data1 = Data(x=torch.randn(5, 2), pos=torch.randn(5, 3))
        test_data2 = Data(x=torch.randn(8, 3), pos=torch.randn(8, 3))
        
        torch.save(test_data1, input_dir / "test1.pt")
        torch.save(test_data2, input_dir / "test2.pt")
        
        # Run batch conversion
        batch_convert_files(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            file_pattern="*.pt",
            survey_name="test",
            device="cpu",
            use_memmap=False,
        )
        
        # Check output files were created
        assert (output_dir / "test1_tensordict.pt").exists()
        assert (output_dir / "test2_tensordict.pt").exists()
        
        # Load and verify converted files
        converted1 = torch.load(output_dir / "test1_tensordict.pt", weights_only=False)
        converted2 = torch.load(output_dir / "test2_tensordict.pt", weights_only=False)
        
        assert isinstance(converted1, AstroTensorDict)
        assert isinstance(converted2, AstroTensorDict)
        assert converted1.n_objects == 5
        assert converted2.n_objects == 8


class TestErrorHandling:
    """Test error handling in conversion functions."""
    
    def test_unsupported_data_type(self):
        """Test error handling for unsupported data types."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            data_to_tensordict("invalid_data", survey_name="test")
    
    def test_missing_coordinates(self):
        """Test error handling for missing coordinate information."""
        df = pl.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
        })
        
        # Should still work but without coordinates
        result = dataframe_to_tensordict(df, survey_name="test")
        assert isinstance(result, AstroTensorDict)
        # Coordinates might not be present or be zeros
        assert result.n_objects == 3


class TestLegacyCompatibility:
    """Test backward compatibility with legacy function names."""
    
    def test_legacy_function_warnings(self):
        """Test that legacy functions raise deprecation warnings."""
        from astro_lab.data.preprocessors.tensordict_migration import migrate_data_to_tensordict
        
        pyg_data = Data(x=torch.randn(5, 2), pos=torch.randn(5, 3))
        
        with pytest.warns(DeprecationWarning, match="migrate_data_to_tensordict is deprecated"):
            result = migrate_data_to_tensordict(pyg_data, survey_name="test")
        
        # Should still work correctly
        assert isinstance(result, AstroTensorDict)
        assert result.n_objects == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
