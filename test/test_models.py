"""
Tests for astro_lab.models module.

Tests astronomical neural networks and model architectures.
"""

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

try:
    from astro_lab.models import (
        astro,
        encoders,
        tgnn,
        utils,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


class TestModelImports:
    """Test that model modules can be imported."""

    def test_import_astro_models(self):
        """Test importing astro models."""
        if not MODELS_AVAILABLE:
            pytest.skip("Models module not available")

        assert astro is not None

    def test_import_encoders(self):
        """Test importing encoder models."""
        if not MODELS_AVAILABLE:
            pytest.skip("Models module not available")

        assert encoders is not None

    def test_import_tgnn(self):
        """Test importing temporal graph neural networks."""
        if not MODELS_AVAILABLE:
            pytest.skip("Models module not available")

        assert tgnn is not None

    def test_import_utils(self):
        """Test importing model utilities."""
        if not MODELS_AVAILABLE:
            pytest.skip("Models module not available")

        assert utils is not None


class TestBasicNeuralNetworks:
    """Test basic neural network functionality."""

    def test_simple_mlp(self):
        """Test simple multi-layer perceptron."""
        input_dim = 10
        hidden_dim = 64
        output_dim = 5

        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Test forward pass
        x = torch.randn(32, input_dim)
        output = model(x)

        assert output.shape == (32, output_dim)
        assert not torch.isnan(output).any()

    def test_cnn_for_images(self):
        """Test CNN for astronomical images."""

        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 16 * 16, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 64 * 16 * 16)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        model = SimpleCNN(num_classes=5)

        # Test with batch of 64x64 images
        batch_size = 16
        x = torch.randn(batch_size, 1, 64, 64)
        output = model(x)

        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()

    def test_rnn_for_lightcurves(self):
        """Test RNN for lightcurve analysis."""

        class LightcurveRNN(nn.Module):
            def __init__(
                self, input_size=1, hidden_size=64, num_layers=2, output_size=1
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True
                )
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                # x shape: (batch, sequence, features)
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use last output
                last_output = lstm_out[:, -1, :]
                last_output = self.dropout(last_output)
                output = self.fc(last_output)
                return output

        model = LightcurveRNN()

        # Test with lightcurve data
        batch_size = 8
        sequence_length = 100
        x = torch.randn(batch_size, sequence_length, 1)
        output = model(x)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()


class TestAstronomicalModels:
    """Test astronomical-specific model components."""

    def test_coordinate_encoder(self):
        """Test coordinate encoding."""

        class CoordinateEncoder(nn.Module):
            def __init__(self, coord_dim=3, embed_dim=64):
                super().__init__()
                self.coord_proj = nn.Linear(coord_dim, embed_dim)
                self.pos_encoding = nn.Parameter(torch.randn(1, embed_dim))

            def forward(self, coords):
                # coords: (batch, 3) for (ra, dec, distance)
                embedded = self.coord_proj(coords)
                return embedded + self.pos_encoding

        encoder = CoordinateEncoder()

        # Test with coordinate data
        batch_size = 20
        coords = torch.randn(batch_size, 3)  # RA, Dec, Distance
        encoded = encoder(coords)

        assert encoded.shape == (batch_size, 64)
        assert not torch.isnan(encoded).any()

    def test_spectral_encoder(self):
        """Test spectral data encoder."""

        class SpectralEncoder(nn.Module):
            def __init__(self, n_wavelengths=1000, embed_dim=128):
                super().__init__()
                self.conv1d = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
                # Calculate the size after convolutions
                conv_output_size = n_wavelengths // 4 * 64
                self.fc = nn.Linear(conv_output_size, embed_dim)

            def forward(self, spectra):
                # spectra: (batch, n_wavelengths)
                x = spectra.unsqueeze(1)  # Add channel dimension
                x = self.conv1d(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc(x)
                return x

        encoder = SpectralEncoder(n_wavelengths=1000)

        # Test with spectral data
        batch_size = 12
        spectra = torch.rand(batch_size, 1000) + 1e-10  # Positive flux
        encoded = encoder(spectra)

        assert encoded.shape == (batch_size, 128)
        assert not torch.isnan(encoded).any()

    def test_multimodal_fusion(self):
        """Test multimodal data fusion."""

        class MultimodalAstroNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Different encoders for different data types
                self.coord_encoder = nn.Linear(3, 64)
                self.photom_encoder = nn.Linear(5, 64)  # 5 bands
                self.spectral_encoder = nn.Sequential(
                    nn.Linear(200, 128), nn.ReLU(), nn.Linear(128, 64)
                )

                # Fusion layer
                self.fusion = nn.Sequential(
                    nn.Linear(64 * 3, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),  # 10 classes
                )

            def forward(self, coords, photom, spectra):
                coord_feat = torch.relu(self.coord_encoder(coords))
                photom_feat = torch.relu(self.photom_encoder(photom))
                spectral_feat = self.spectral_encoder(spectra)

                # Concatenate features
                fused = torch.cat([coord_feat, photom_feat, spectral_feat], dim=1)
                output = self.fusion(fused)
                return output

        model = MultimodalAstroNet()

        # Test with multimodal data
        batch_size = 10
        coords = torch.randn(batch_size, 3)
        photom = torch.randn(batch_size, 5)
        spectra = torch.randn(batch_size, 200)

        output = model(coords, photom, spectra)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()


class TestModelUtilities:
    """Test model utility functions."""

    def test_parameter_counting(self):
        """Test parameter counting utility."""
        model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

    def test_model_summary(self):
        """Test model summary functionality."""
        model = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3)
        )

        # Test forward pass to get output shapes
        x = torch.randn(1, 5)
        output = model(x)

        assert output.shape == (1, 3)

        # Test model can be converted to string
        model_str = str(model)
        assert "Linear" in model_str
        assert "ReLU" in model_str

    def test_weight_initialization(self):
        """Test weight initialization utilities."""

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Apply initialization
        model.apply(init_weights)

        # Check that weights are not all zeros
        for param in model.parameters():
            if param.dim() > 1:  # Weight matrices
                assert not torch.allclose(param, torch.zeros_like(param))


class TestModelTraining:
    """Test model training functionality."""

    def test_loss_computation(self):
        """Test loss computation for different tasks."""
        batch_size = 16
        num_classes = 5

        # Classification loss
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predictions, targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

        # Regression loss
        predictions_reg = torch.randn(batch_size, 1)
        targets_reg = torch.randn(batch_size, 1)

        mse_loss = nn.MSELoss()
        loss_reg = mse_loss(predictions_reg, targets_reg)

        assert loss_reg.item() >= 0
        assert not torch.isnan(loss_reg)

    def test_gradient_computation(self):
        """Test gradient computation."""
        model = nn.Linear(10, 1)
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)

        # Forward pass
        predictions = model(x)
        loss = nn.MSELoss()(predictions, y)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_optimizer_step(self):
        """Test optimizer step."""
        model = nn.Linear(5, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(10, 5)
        y = torch.randn(10, 1)

        # Get initial weights
        initial_weight = model.weight.clone()

        # Training step
        optimizer.zero_grad()
        predictions = model(x)
        loss = nn.MSELoss()(predictions, y)
        loss.backward()
        optimizer.step()

        # Check weights changed
        assert not torch.equal(initial_weight, model.weight)


class TestModelEvaluation:
    """Test model evaluation functionality."""

    def test_model_evaluation_mode(self):
        """Test model evaluation mode."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Dropout(0.5), nn.Linear(20, 5))

        # Training mode
        model.train()
        assert model.training

        # Evaluation mode
        model.eval()
        assert not model.training

        # Test inference
        x = torch.randn(3, 10)
        with torch.no_grad():
            output = model(x)
            assert output.shape == (3, 5)

    def test_prediction_consistency(self):
        """Test prediction consistency in eval mode."""
        model = nn.Sequential(nn.Linear(5, 10), nn.Dropout(0.5), nn.Linear(10, 1))

        x = torch.randn(1, 5)

        # Set to eval mode
        model.eval()

        # Multiple predictions should be identical
        with torch.no_grad():
            pred1 = model(x)
            pred2 = model(x)
            pred3 = model(x)

        assert torch.allclose(pred1, pred2, atol=1e-6)
        assert torch.allclose(pred2, pred3, atol=1e-6)

    def test_accuracy_computation(self):
        """Test accuracy computation for classification."""
        batch_size = 20
        num_classes = 3

        # Mock predictions and targets
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Compute accuracy
        predicted_classes = predictions.argmax(dim=1)
        accuracy = (predicted_classes == targets).float().mean()

        assert 0 <= accuracy <= 1
        assert isinstance(accuracy.item(), float)
