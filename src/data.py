"""
Data preparation utilities for MNIST dataset.
"""

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


def prepare_data(
    path: str, 
    batch_size_train: int = 1024,  # Increased default batch size
    batch_size_test: int = 2048,   # Increased test batch size
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare MNIST dataset with optimized data loaders.

    Args:
        path: Path to store/load the MNIST dataset
        batch_size_train: Batch size for training data loader
        batch_size_test: Batch size for test data loader
        num_workers: Number of worker processes for data loading. If None, uses (CPU cores - 1)
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
        persistent_workers: If True, the data loader will not shut down worker processes
                           after a dataset has been consumed once

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders
    """
    # Use more accurate normalization for MNIST
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load datasets with more efficient file system access
    train_dataset = MNIST(
        path, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = MNIST(
        path, 
        train=False, 
        download=True, 
        transform=transform
    )

    # Determine number of workers if not specified
    if num_workers is None:
        num_workers = min(8, (os.cpu_count() or 2) - 1)  # Use up to 8 workers

    # Common DataLoader arguments
    common_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers and num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else 2,
        'pin_memory_device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Training DataLoader with shuffling and drop_last
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch
        **common_kwargs
    )

    # Test DataLoader - no need to shuffle or drop last
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        drop_last=False,
        **common_kwargs
    )

    print(f"Data loaders created with {num_workers} worker processes")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader
