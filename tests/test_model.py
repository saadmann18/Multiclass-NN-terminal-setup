"""
Tests for CNN model.
"""

import torch
import pytest
from src.mnist_cnn.model import CNN


class TestCNN:
    """Test cases for CNN model."""

    def test_model_initialization(self):
        """Test model can be initialized correctly."""
        model = CNN(n_channels=1)
        assert isinstance(model, torch.nn.Module)

        # Check if all layers are present
        assert hasattr(model, "hidden1")
        assert hasattr(model, "hidden2")
        assert hasattr(model, "hidden3")
        assert hasattr(model, "hidden4")

    def test_model_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = CNN(n_channels=1)
        model.eval()

        # Create dummy input (batch_size=2, channels=1, height=28, width=28)
        x = torch.randn(2, 1, 28, 28)

        with torch.no_grad():
            output = model(x)

        # Check output shape
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"

    def test_model_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = CNN(n_channels=1)
        model.eval()

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 28, 28)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 10)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = CNN(n_channels=1)
        model.train()

        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        target = torch.randint(0, 10, (2,))

        criterion = torch.nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None, "Gradient is None for a parameter"
            assert not torch.isnan(param.grad).any(), "Gradient contains NaN"

    def test_model_parameter_count(self):
        """Test model has expected number of parameters."""
        model = CNN(n_channels=1)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Expected parameters based on architecture
        # Conv1: (1*32*3*3) + 32 = 320
        # Conv2: (32*32*3*3) + 32 = 9248
        # FC1: (800*100) + 100 = 80100
        # FC2: (100*10) + 10 = 1010
        # Total: ~90,678

        assert total_params > 90000, f"Expected >90k parameters, got {total_params}"
        assert total_params == trainable_params, "All parameters should be trainable"

    def test_model_device_compatibility(self):
        """Test model can be moved to different devices."""
        model = CNN(n_channels=1)

        # Test CPU
        model_cpu = model.to("cpu")
        x_cpu = torch.randn(1, 1, 28, 28)
        output_cpu = model_cpu(x_cpu)
        assert output_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            x_cuda = torch.randn(1, 1, 28, 28).cuda()
            output_cuda = model_cuda(x_cuda)
            assert output_cuda.device.type == "cuda"

    def test_model_reproducibility(self):
        """Test model produces consistent outputs with same seed."""
        torch.manual_seed(42)
        model1 = CNN(n_channels=1)

        torch.manual_seed(42)
        model2 = CNN(n_channels=1)

        # Models should have identical parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Models are not identical with same seed"

        # Should produce identical outputs
        torch.manual_seed(123)
        x = torch.randn(1, 1, 28, 28)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2), "Outputs are not identical with same seed"
