"""Integration tests for AstroLab - Testing complete workflows."""

import tempfile
from pathlib import Path

import pytest
import torch

from astro_lab.data.datamodule import AstroDataModule
from astro_lab.models import AstroSurveyGNN, ModelConfig
from astro_lab.training import AstroLightningModule, AstroTrainer, TrainingConfig


class TestTrainingIntegration:
    """Test complete training workflow."""

    def test_gaia_training_workflow(self):
        """Test complete workflow: data loading → model creation → training."""
        # 1. Create DataModule
        datamodule = AstroDataModule(
            survey="gaia",
            max_samples=100,  # Small dataset for testing
            batch_size=1,
            train_ratio=0.6,
            val_ratio=0.2,
        )

        # 2. Setup data (this detects number of classes)
        datamodule.setup()

        # Verify data was loaded correctly
        assert datamodule.num_classes is not None
        assert datamodule.num_classes > 1  # At least binary classification
        assert datamodule.num_features is not None

        # 3. Create model config with correct number of classes
        model_config = ModelConfig(
            name="test_gnn",
            hidden_dim=32,  # Small for testing
            output_dim=datamodule.num_classes,
            num_layers=2,
            conv_type="gcn",
            task="classification",
        )

        # 4. Create training config
        training_config = TrainingConfig(
            name="test_training",
            model=model_config,
            scheduler={"max_epochs": 1},  # Just 1 epoch for testing
            hardware={"accelerator": "cpu", "devices": 1},  # CPU for CI
        )

        # 5. Create trainer and train
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AstroTrainer(
                training_config=training_config,
                enable_checkpointing=False,  # Disable for test
            )

            # Train should not crash
            trainer.fit(datamodule=datamodule)

            # Verify model was created with correct dimensions
            assert trainer.astro_module is not None
            assert hasattr(trainer.astro_module, "model")

            # Check model output dimension matches data classes
            if hasattr(trainer.astro_module.model, "output_dim"):
                assert trainer.astro_module.model.output_dim == datamodule.num_classes

    def test_model_prediction_after_training(self):
        """Test that model can make predictions after training."""
        # Setup small dataset
        datamodule = AstroDataModule(
            survey="gaia",
            max_samples=50,
            batch_size=1,
        )
        datamodule.setup()

        # Create and train model
        model_config = ModelConfig(
            name="test_predictor",
            hidden_dim=16,
            output_dim=datamodule.num_classes,
            num_layers=2,
        )

        lightning_module = AstroLightningModule(
            model_config=model_config,
            num_classes=datamodule.num_classes,
        )

        # Get a batch of data
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Model should be able to forward pass
        with torch.no_grad():
            lightning_module.eval()
            outputs = lightning_module(batch)

        # Check outputs are valid
        assert outputs is not None
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

        # Output shape should match number of nodes and classes
        expected_shape = (batch.x.shape[0], datamodule.num_classes)
        assert outputs.shape == expected_shape

    def test_different_surveys(self):
        """Test that different surveys can be loaded and processed."""
        surveys_to_test = ["gaia"]  # Add more surveys as they become available

        for survey in surveys_to_test:
            # Skip if survey data not available
            try:
                datamodule = AstroDataModule(
                    survey=survey,
                    max_samples=10,  # Very small for testing
                    batch_size=1,
                )
                datamodule.setup()

                # Basic checks
                assert datamodule.num_classes is not None
                assert datamodule.num_features is not None
                assert datamodule._main_data is not None

                # Check masks are created
                assert hasattr(datamodule._main_data, "train_mask")
                assert hasattr(datamodule._main_data, "val_mask")
                assert hasattr(datamodule._main_data, "test_mask")

            except FileNotFoundError:
                pytest.skip(f"Survey data for {survey} not available")


class TestModelIntegration:
    """Test model components work together."""

    def test_encoder_in_model(self):
        """Test that encoders integrate properly with models."""
        from astro_lab.models.encoders import PhotometryEncoder

        # Create encoder on CPU
        photometry_encoder = PhotometryEncoder(output_dim=32, device="cpu")

        # Create model that uses encoder
        class ModelWithEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.photometry_encoder = photometry_encoder
                self.classifier = torch.nn.Linear(32, 4)

            def forward(self, x):
                # Assume first 5 features are photometry
                photometry_features = x[:, :5]
                encoded = self.photometry_encoder(photometry_features)
                return self.classifier(encoded)

        model = ModelWithEncoder()

        # Test forward pass
        batch_data = torch.randn(10, 20)  # 10 objects, 20 features
        output = model(batch_data)

        assert output.shape == (10, 4)
        assert not torch.isnan(output).any()

    def test_model_save_load(self):
        """Test that models can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model on CPU
            model = AstroSurveyGNN(
                input_dim=10,
                hidden_dim=32,
                output_dim=4,
                num_layers=2,
                device="cuda",  # Explicitly use CPU
            )

            # Save model
            save_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), save_path)

            # Create new model and load weights
            model2 = AstroSurveyGNN(
                input_dim=10,
                hidden_dim=32,
                output_dim=4,
                num_layers=2,
                device="cuda",  # Also on CPU
            )
            model2.load_state_dict(torch.load(save_path))

            # Check that parameters are the same
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestDataIntegration:
    """Test data processing pipeline."""

    def test_data_pipeline(self):
        """Test that data flows correctly through the pipeline."""
        # This tests the actual data loading and processing
        from astro_lab.data.core import AstroDataset

        try:
            dataset = AstroDataset(
                root="data",
                survey="gaia",
                max_samples=20,
            )

            # Get first data object
            data = dataset[0]

            # Check data object has required attributes
            assert hasattr(data, "x")  # Node features
            assert hasattr(data, "edge_index")  # Graph structure
            assert hasattr(data, "pos")  # Positions

            # Check data types and shapes
            assert data.x.dtype == torch.float32
            assert data.edge_index.dtype == torch.long
            assert data.edge_index.shape[0] == 2  # Source and target nodes

        except FileNotFoundError:
            pytest.skip("Gaia data not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
