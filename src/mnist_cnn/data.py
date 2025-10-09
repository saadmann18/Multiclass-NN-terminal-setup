"""
Data preparation utilities for MNIST dataset.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


def prepare_data(
    path: str, 
    batch_size_train: int = 128, 
    batch_size_test: int = 1024
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare MNIST dataset with optimized data loaders.
    
    Args:
        path (str): Path to store/load the MNIST dataset
        batch_size_train (int): Batch size for training data loader
        batch_size_test (int): Batch size for test data loader
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders
    """
    # Define standardization transform
    transform = Compose([
        ToTensor(), 
        Normalize((0.5,), (1.0,))
    ])
    
    # Load datasets
    train_dataset = MNIST(path, train=True, download=True, transform=transform)
    test_dataset = MNIST(path, train=False, download=True, transform=transform)
    
    # Determine optimal data loader settings
    use_cuda = torch.cuda.is_available()
    num_workers = max(0, (os.cpu_count() or 0) - 1)
    pin_memory = use_cuda
    
    # Training data loader configuration
    train_kwargs = {
        'batch_size': batch_size_train,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
    }
    
    # Test data loader configuration
    test_kwargs = {
        'batch_size': batch_size_test,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    # Add performance optimizations for multi-worker setup
    if num_workers > 0:
        worker_optimizations = {
            'persistent_workers': True, 
            'prefetch_factor': 2
        }
        train_kwargs.update(worker_optimizations)
        test_kwargs.update(worker_optimizations)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    return train_loader, test_loader
