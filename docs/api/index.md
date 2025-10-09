# API Reference

This section provides detailed documentation for all modules and functions in the MNIST CNN package.

## Core Modules

::: mnist_cnn.model
    options:
      show_root_heading: true
      show_source: false

::: mnist_cnn.data
    options:
      show_root_heading: true
      show_source: false

::: mnist_cnn.train
    options:
      show_root_heading: true
      show_source: false

::: mnist_cnn.eval
    options:
      show_root_heading: true
      show_source: false

::: mnist_cnn.utils
    options:
      show_root_heading: true
      show_source: false

::: mnist_cnn.config
    options:
      show_root_heading: true
      show_source: false

## Quick Reference

```python
# Import main components
from mnist_cnn import CNN, prepare_data, run_training, run_evaluation

# Create model
model = CNN(n_channels=1)

# Prepare data
train_loader, test_loader = prepare_data('/path/to/data')

# Train model
model, save_path = run_training(epochs=10, device_preference='auto')

# Evaluate model
accuracy, loss, conf_mat, per_class_acc, plot_path = run_evaluation()
```
