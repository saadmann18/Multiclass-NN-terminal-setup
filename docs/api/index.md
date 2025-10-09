# API Reference

Complete API documentation for the MNIST CNN package.

## Core Modules

### Model (`mnist_cnn.model`)

- [`CNN`](model.md#cnn) - Convolutional Neural Network implementation

### Data (`mnist_cnn.data`)

- [`prepare_data`](data.md#prepare_data) - MNIST data preparation and loading

### Training (`mnist_cnn.train`)

- [`train_model`](train.md#train_model) - Core training function
- [`run_training`](train.md#run_training) - Complete training pipeline

### Evaluation (`mnist_cnn.eval`)

- [`evaluate_model`](eval.md#evaluate_model) - Model evaluation function
- [`run_evaluation`](eval.md#run_evaluation) - Complete evaluation pipeline
- [`plot_confusion_matrix`](eval.md#plot_confusion_matrix) - Visualization utilities

### Utilities (`mnist_cnn.utils`)

- [`select_device`](utils.md#select_device) - Device selection utility
- [`setup_device_optimizations`](utils.md#setup_device_optimizations) - Device optimization setup

### Configuration (`mnist_cnn.config`)

- [`Config`](config.md#config) - Configuration management class

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
