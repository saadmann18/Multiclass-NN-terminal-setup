"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'n_channels': 1,
            'compile': False
        },
        'training': {
            'epochs': 5,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'batch_size_train': 64,
            'device': 'cpu'
        },
        'data': {
            'data_dir': '~/.torch/datasets/mnist',
            'batch_size_test': 256
        },
        'evaluation': {
            'show_plot': False,
            'save_confusion_matrix': True
        },
        'paths': {
            'artifacts_dir': 'artifacts',
            'model_filename': 'test_model.pth'
        }
    }


@pytest.fixture
def dummy_model():
    """Create a dummy CNN model for testing."""
    from src.mnist_cnn.model import CNN
    return CNN(n_channels=1)


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    # Create dummy MNIST-like data
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture(scope="session")
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_mnist_dataset():
    """Mock MNIST dataset for testing without downloading."""
    class MockMNIST:
        def __init__(self, root, train=True, download=True, transform=None):
            self.train = train
            self.transform = transform
            # Create dummy data
            if train:
                self.data = torch.randint(0, 255, (1000, 28, 28), dtype=torch.uint8)
                self.targets = torch.randint(0, 10, (1000,))
            else:
                self.data = torch.randint(0, 255, (200, 28, 28), dtype=torch.uint8)
                self.targets = torch.randint(0, 10, (200,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            image = self.data[idx].float() / 255.0  # Convert to float and normalize to [0,1]
            target = self.targets[idx]
            
            if self.transform:
                image = self.transform(image.unsqueeze(0)).squeeze(0)  # Add channel dim for transform
            
            return image, target
    
    return MockMNIST


# Skip tests that require CUDA if not available
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
