"""
MNIST CNN - A PyTorch implementation for multiclass digit classification.

This package provides a complete solution for training and evaluating 
a Convolutional Neural Network on the MNIST dataset.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model import CNN
from .data import prepare_data
from .train import train_model, run_training
from .eval import evaluate_model

__all__ = [
    "CNN",
    "prepare_data", 
    "train_model",
    "run_training",
    "evaluate_model"
]
