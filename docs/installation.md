# Installation Guide

This guide covers different ways to install and set up the MNIST CNN project.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (recommended)
- CUDA toolkit (optional, for GPU support)

## Installation Methods

### 1. Development Installation

For development and contributing:

```bash
# Clone the repository
git clone <repository-url>
cd Multiclass-NN-terminal-setup

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Package Installation

Install as a package:

```bash
pip install mnist-cnn
```

### 3. Docker Installation

Using Docker (recommended for production):

```bash
# Build the image
docker build -t mnist-cnn .

# Run training
docker run --rm -v $(pwd)/artifacts:/app/artifacts mnist-cnn python -m mnist_cnn.cli train

# Run evaluation
docker run --rm -v $(pwd)/artifacts:/app/artifacts mnist-cnn python -m mnist_cnn.cli evaluate
```

### 4. Docker Compose

For complete setup with GPU support:

```bash
docker-compose up --build
```

## GPU Support

### CUDA

For NVIDIA GPU support:

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon (MPS)

For Apple M1/M2 Macs:

```bash
# Install PyTorch with MPS support
pip install torch torchvision

# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Verification

Verify your installation:

```python
import mnist_cnn
print(mnist_cnn.__version__)

# Run a quick test
from mnist_cnn import CNN
model = CNN(n_channels=1)
print("Installation successful!")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the correct virtual environment
2. **CUDA Issues**: Verify CUDA version compatibility with PyTorch
3. **Memory Issues**: Reduce batch size in configuration
4. **Permission Issues**: Use `sudo` for system-wide installation (not recommended)

### Getting Help

- Check the [FAQ](faq.md)
- Open an issue on GitHub
- Check PyTorch installation guide for GPU-specific issues
