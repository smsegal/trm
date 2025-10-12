""Tests for the PyTorch model."""

import pytest
import torch

from pytorch_project.model import SimpleCNN


def test_model_initialization() -> None:
    """Test that the model initializes with the correct architecture."""
    model = SimpleCNN(num_classes=10)
    
    # Test model architecture
    assert isinstance(model.conv1, torch.nn.Conv2d)
    assert isinstance(model.conv2, torch.nn.Conv2d)
    assert isinstance(model.fc1, torch.nn.Linear)
    assert isinstance(model.fc2, torch.nn.Linear)
    
    # Test output shape
    x = torch.randn(1, 3, 32, 32)  # Batch of 1, 3 channels, 32x32 image
    output = model(x)
    assert output.shape == (1, 10)  # Batch of 1, 10 classes


def test_model_forward_pass() -> None:
    """Test that the forward pass works with different batch sizes."""
    model = SimpleCNN(num_classes=10)
    
    # Test with batch size 1
    x1 = torch.randn(1, 3, 32, 32)
    output1 = model(x1)
    assert output1.shape == (1, 10)
    
    # Test with batch size 4
    x4 = torch.randn(4, 3, 32, 32)
    output4 = model(x4)
    assert output4.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__])
