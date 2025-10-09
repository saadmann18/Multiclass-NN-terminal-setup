# Configuration Guide

The MNIST CNN project uses YAML-based configuration for flexible parameter management.

## Configuration Files

### Default Configuration

The default configuration is located at `config/default.yaml`:

```yaml
# Model configuration
model:
  n_channels: 1
  compile: false

# Training configuration
training:
  epochs: 10
  learning_rate: 0.01
  momentum: 0.9
  batch_size_train: 128
  device: "auto"

# Data configuration
data:
  data_dir: "~/.torch/datasets/mnist"
  batch_size_test: 1024
  num_workers: null

# Evaluation configuration
evaluation:
  show_plot: true
  save_confusion_matrix: true

# Paths
paths:
  artifacts_dir: "artifacts"
  model_filename: "model_mnist_cnn.pth"
  confusion_matrix_filename: "confusion_matrix.png"
```

## Using Configuration

### Python API

```python
from mnist_cnn.config import Config

# Load default configuration
config = Config()

# Load custom configuration
config = Config('/path/to/custom/config.yaml')

# Access configuration values
epochs = config.get('training.epochs')
device = config.get('training.device', 'cpu')

# Update configuration
config.set('training.epochs', 20)
config.update({'training': {'learning_rate': 0.001}})

# Save configuration
config.save('/path/to/save/config.yaml')
```

### Configuration Sections

#### Model Configuration

```yaml
model:
  n_channels: 1        # Number of input channels (1 for grayscale)
  compile: false       # Enable torch.compile (PyTorch 2.0+)
```

#### Training Configuration

```yaml
training:
  epochs: 10           # Number of training epochs
  learning_rate: 0.01  # Learning rate for optimizer
  momentum: 0.9        # Momentum for SGD optimizer
  batch_size_train: 128 # Training batch size
  device: "auto"       # Device preference: auto, cuda, mps, cpu
```

#### Data Configuration

```yaml
data:
  data_dir: "~/.torch/datasets/mnist"  # Directory for MNIST data
  batch_size_test: 1024                # Test batch size
  num_workers: null                    # Data loader workers (null = auto)
```

#### Evaluation Configuration

```yaml
evaluation:
  show_plot: true              # Show confusion matrix plot
  save_confusion_matrix: true  # Save confusion matrix to file
```

#### Paths Configuration

```yaml
paths:
  artifacts_dir: "artifacts"                    # Output directory
  model_filename: "model_mnist_cnn.pth"        # Model checkpoint filename
  confusion_matrix_filename: "confusion_matrix.png"  # Plot filename
```

## Environment Variables

Some configuration values can be overridden with environment variables:

```bash
export DEVICE=cuda           # Override device selection
export EPOCHS=20            # Override number of epochs
export BATCH_SIZE=64        # Override batch size
```

## Custom Configurations

### Creating Custom Configs

Create custom configuration files for different scenarios:

```yaml
# config/gpu_training.yaml
training:
  device: "cuda"
  batch_size_train: 256
  epochs: 50
  
model:
  compile: true
```

```yaml
# config/quick_test.yaml
training:
  epochs: 2
  batch_size_train: 32
  
data:
  batch_size_test: 128
```

### Using Custom Configs

```python
# Use custom configuration
config = Config('config/gpu_training.yaml')
model, save_path = run_training(
    epochs=config.get('training.epochs'),
    device_preference=config.get('training.device')
)
```

## Configuration Validation

The configuration system includes validation for common parameters:

- Device values must be one of: `auto`, `cuda`, `mps`, `cpu`
- Batch sizes must be positive integers
- Learning rates must be positive floats
- Epochs must be positive integers

## Best Practices

1. **Use version control**: Keep configuration files in version control
2. **Environment-specific configs**: Create separate configs for dev/test/prod
3. **Document changes**: Comment configuration changes
4. **Validate configs**: Test configurations before deployment
5. **Use defaults**: Override only necessary parameters

## Advanced Usage

### Programmatic Configuration

```python
from mnist_cnn.config import Config

# Start with default config
config = Config()

# Modify for specific experiment
config.update({
    'training': {
        'epochs': 100,
        'learning_rate': 0.001
    },
    'model': {
        'compile': True
    }
})

# Save experiment configuration
config.save('experiments/exp_001.yaml')
```

### Configuration Inheritance

```python
# Load base configuration
base_config = Config('config/base.yaml')

# Create experiment-specific config
exp_config = Config('config/base.yaml')
exp_config.update({
    'training': {'epochs': 50},
    'experiment': {'name': 'high_lr_experiment'}
})

exp_config.save('experiments/high_lr.yaml')
```
