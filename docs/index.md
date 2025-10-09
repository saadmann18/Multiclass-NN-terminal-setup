# MNIST CNN Documentation

Welcome to the MNIST CNN project documentation. This project provides a complete PyTorch implementation for multiclass digit classification using Convolutional Neural Networks.

## Quick Start

```python
from mnist_cnn import run_training, run_evaluation

# Train the model
model, save_path = run_training(epochs=10, device_preference="auto")

# Evaluate the model
accuracy, loss, conf_mat, per_class_acc, plot_path = run_evaluation()
print(f"Test Accuracy: {accuracy:.4f}")
```

## Project Structure

```
mnist-cnn/
├── src/mnist_cnn/          # Main package
├── tests/                  # Test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── artifacts/              # Model outputs
└── requirements.txt        # Dependencies
```

## Features

- **Modern PyTorch Implementation**: Uses latest PyTorch features including mixed precision training
- **Device Agnostic**: Automatic device selection (CUDA, MPS, CPU) with optimizations
- **Configurable**: YAML-based configuration management
- **Well Tested**: Comprehensive test suite with >90% coverage
- **Dockerized**: Ready-to-use Docker setup
- **Documentation**: Complete API

## Navigation

- [Installation Guide](installation.md) - Get started quickly
- [Configuration Guide](configuration.md) - Customize your setup
- [API Reference](api/index.md) - Detailed function documentation
