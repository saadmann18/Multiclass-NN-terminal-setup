# MNIST CNN - A Modern PyTorch Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready PyTorch implementation for MNIST digit classification using Convolutional Neural Networks. This project demonstrates modern ML engineering practices with a complete, modular, and well-tested codebase.

## ✨ Features

- **🚀 Modern PyTorch**: Uses latest PyTorch features including mixed precision training and torch.compile
- **🔧 Device Agnostic**: Automatic device selection (CUDA, MPS, CPU) with optimizations
- **⚙️ Configurable**: YAML-based configuration management system
- **🧪 Well Tested**: Comprehensive test suite with >90% coverage
- **🐳 Dockerized**: Multi-stage Docker builds for development and production
- **📚 Documented**: Complete API documentation and tutorials
- **🛠️ Developer Friendly**: Pre-commit hooks, linting, formatting, and CI/CD ready

## 📁 Project Structure

```
mnist-cnn/
├── src/mnist_cnn/          # Main package source code
│   ├── __init__.py         # Package initialization
│   ├── model.py            # CNN model definition
│   ├── data.py             # Data loading utilities
│   ├── train.py            # Training pipeline
│   ├── eval.py             # Evaluation pipeline
│   ├── utils.py            # Utility functions
│   ├── config.py           # Configuration management
│   └── cli.py              # Command-line interface
├── tests/                  # Comprehensive test suite
│   ├── test_model.py       # Model tests
│   ├── test_data.py        # Data loading tests
│   ├── test_utils.py       # Utility tests
│   ├── test_config.py      # Configuration tests
│   └── conftest.py         # Test fixtures
├── config/                 # Configuration files
│   └── default.yaml        # Default configuration
├── docs/                   # Documentation
│   ├── index.md            # Main documentation
│   ├── installation.md     # Installation guide
│   ├── configuration.md    # Configuration guide
│   └── api/                # API documentation
├── scripts/                # Utility scripts
│   └── setup_environment.sh    # Environment setup
├── artifacts/              # Model outputs and results
├── logs/                   # Training and evaluation logs
├── experiments/            # Experiment results
├── requirements.txt        # Production dependencies (legacy/Docker)
├── pyproject.toml          # Package configuration
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose configuration
├── Makefile               # Development commands
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

The easiest way to run this project is with Docker:

```bash
# Train and evaluate with Docker Compose
docker-compose up pipeline

# Or run individual services
docker-compose up train  # Training only
docker-compose up eval   # Evaluation only
docker-compose up dev    # Development mode
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd Multiclass-NN-terminal-setup

# Quick setup (creates venv, installs dependencies, sets up pre-commit)
./scripts/setup_environment.sh

# Activate environment
source venv/bin/activate

# Train the model
make train
# or
mnist-train --epochs 10 --device auto

# Evaluate the model
make eval
# or
mnist-eval --device auto
```

### Option 3: Python API

```python
from mnist_cnn import run_training, run_evaluation

# Train the model
model, save_path = run_training(epochs=10, device_preference="auto")

# Evaluate the model
accuracy, loss, conf_mat, per_class_acc, plot_path = run_evaluation()
print(f"Test Accuracy: {accuracy:.4f}")
```

## 📋 Requirements

- Python 3.8 or higher
- PyTorch 2.0+ (automatically installed)
- CUDA toolkit (optional, for GPU support)
- Docker (optional, for containerized deployment)

## 🛠️ Development

### Setup Development Environment

```bash
# Install in development mode (includes all dev tools)
pip install -e ".[dev,docs]"
# or
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make check
```

### Configuration

The project uses YAML-based configuration. See `config/default.yaml` for all options:

```yaml
# Model configuration
model:
  n_channels: 1
  compile: false

# Training configuration
training:
  epochs: 10
  learning_rate: 0.01
  batch_size_train: 128
  device: "auto"  # auto, cuda, mps, cpu
```

Create custom configurations for different experiments:

```bash
# Use custom config
mnist-train --config config/my_experiment.yaml

# Override specific parameters
mnist-train --epochs 20 --learning-rate 0.001 --device cuda
```

## 🧪 Running Experiments

### Training
```bash
# Basic training
python trainer.py --epochs 10 --log-dir runs/exp1 --experiment-name exp1

# With model compilation (PyTorch 2.0+)
python trainer.py --epochs 20 --log-dir runs/exp2 --experiment-name exp2 --compile
```

### TensorBoard Visualization
```bash
# Start TensorBoard
tensorboard --logdir=runs

# Open in browser: http://localhost:6006
```

### Available Visualizations
- **HParams comparison** (parallel coordinates, scatter plots) ✨ NEW!
- **Training and validation metrics** (accuracy, loss, precision, recall, F1)
- **Model architecture** (computational graph)
- **Confusion matrices** (train/val)
- **ROC and PR curves** (multiclass)
- **Weight & gradient distributions/histograms**
- **Gradient norms** (monitor for vanishing/exploding gradients)
- **Weight statistics** (mean, std, min, max per layer)
- **Activation maps** (visualize layer outputs)
- **Feature space visualization** (T-SNE)
- **Hardware utilization** (CPU, GPU, memory)

📖 See [TENSORBOARD_FEATURES.md](TENSORBOARD_FEATURES.md) for complete documentation.

## 🐳 Docker Usage

### CPU Training
```bash
# Development
docker-compose up dev

# Production pipeline
docker-compose up pipeline

# Individual services
docker-compose up train
docker-compose up eval
```

### GPU Training
```bash
# Build GPU image
docker-compose -f docker-compose.gpu.yml build

# Run GPU training
docker-compose -f docker-compose.gpu.yml up train-gpu

# Run GPU pipeline
docker-compose -f docker-compose.gpu.yml up pipeline-gpu
```

### Jupyter Development
```bash
# Start Jupyter Lab
docker-compose up notebook

# Access at http://localhost:8888
```

## 📊 Model Architecture

The CNN model consists of:

- **Conv Block 1**: Conv2d(1→32, 3×3) → ReLU → MaxPool2d(2×2)
- **Conv Block 2**: Conv2d(32→32, 3×3) → ReLU → MaxPool2d(2×2)
- **FC Block 1**: Linear(800→100) → ReLU
- **Output**: Linear(100→10)

**Total Parameters**: ~90,678

## 🎯 Performance

Expected results on MNIST test set:
- **Accuracy**: ~98-99%
- **Training Time**: ~2-3 minutes (CPU), ~30 seconds (GPU)
- **Model Size**: ~350KB

## 🔧 Advanced Features

- **Mixed Precision Training**: Automatic on CUDA
- **Device Optimization**: Automatic device selection and optimization
- **Model Compilation**: `torch.compile` support for PyTorch 2.0+
- **Experiment Tracking**: Built-in experiment management
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: >90% test coverage

## 🔄 CI/CD Pipeline

Simple GitHub Actions workflows:

- **CI**: Tests and code style checks on every push/PR
- **Release**: Creates GitHub releases when you push a tag
- **Docs**: Builds documentation to verify it works

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `make install-dev`
4. Make your changes
5. Run tests: `make test`
6. Run quality checks: `make check`
7. Commit your changes: `git commit -am 'Add feature'`
8. Push to the branch: `git push origin feature-name`
9. Submit a pull request

That's it! Keep it simple.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- MNIST dataset creators
- Open source community for tools and libraries

## 📚 Documentation

For detailed documentation, see:
- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api/index.md)
- [Development Guide](docs/development.md)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in configuration
2. **Import errors**: Ensure you're in the correct virtual environment
3. **Permission errors**: Check Docker volume mount permissions
4. **Slow training**: Enable `torch.compile` for PyTorch 2.0+

### Getting Help

- Check the [documentation](docs/)
- Search existing [issues](https://github.com/yourusername/mnist-cnn/issues)
- Create a new issue with detailed information

---

**Made with ❤️ using PyTorch**
