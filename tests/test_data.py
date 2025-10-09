"""
Tests for data preparation utilities.
"""

import os
import tempfile
import torch
import pytest
from torch.utils.data import DataLoader
from src.mnist_cnn.data import prepare_data


class TestDataPreparation:
    """Test cases for data preparation."""
    
    def test_prepare_data_returns_dataloaders(self):
        """Test that prepare_data returns DataLoader objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(temp_dir)
            
            assert isinstance(train_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)
    
    def test_data_loader_batch_sizes(self):
        """Test data loaders have correct batch sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(
                temp_dir, 
                batch_size_train=64, 
                batch_size_test=512
            )
            
            assert train_loader.batch_size == 64
            assert test_loader.batch_size == 512
    
    def test_data_loader_properties(self):
        """Test data loader properties are set correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(temp_dir)
            
            # Training loader should shuffle
            assert train_loader.shuffle is True
            assert train_loader.drop_last is True
            
            # Test loader should not shuffle
            assert test_loader.shuffle is False
            assert hasattr(test_loader, 'drop_last')  # May be False or not set
    
    def test_data_shapes_and_types(self):
        """Test data has correct shapes and types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(temp_dir)
            
            # Get a batch from each loader
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            
            for batch in [train_batch, test_batch]:
                images, labels = batch
                
                # Check image shape (batch_size, channels, height, width)
                assert len(images.shape) == 4
                assert images.shape[1] == 1  # Grayscale
                assert images.shape[2] == 28  # Height
                assert images.shape[3] == 28  # Width
                
                # Check label shape
                assert len(labels.shape) == 1
                assert labels.shape[0] == images.shape[0]  # Same batch size
                
                # Check data types
                assert images.dtype == torch.float32
                assert labels.dtype == torch.int64
                
                # Check value ranges
                assert images.min() >= -1.0  # Normalized
                assert images.max() <= 1.0   # Normalized
                assert labels.min() >= 0     # Valid class indices
                assert labels.max() <= 9     # Valid class indices
    
    def test_data_normalization(self):
        """Test that data is properly normalized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, _ = prepare_data(temp_dir)
            
            # Collect a few batches to check normalization
            all_images = []
            for i, (images, _) in enumerate(train_loader):
                all_images.append(images)
                if i >= 5:  # Just check first few batches
                    break
            
            all_images = torch.cat(all_images, dim=0)
            
            # With Normalize((0.5,), (1.0,)), the range should be roughly [-1, 1]
            # But exact mean/std will depend on the actual MNIST data distribution
            mean = all_images.mean()
            std = all_images.std()
            
            # Check that normalization was applied (not in [0, 1] range)
            assert mean < 0.4, f"Mean {mean} suggests normalization not applied"
            assert all_images.min() < 0, "Min value should be negative after normalization"
    
    def test_dataset_sizes(self):
        """Test that datasets have expected sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(temp_dir)
            
            # MNIST has 60,000 training and 10,000 test samples
            assert len(train_loader.dataset) == 60000
            assert len(test_loader.dataset) == 10000
    
    def test_data_loader_iteration(self):
        """Test that data loaders can be iterated multiple times."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(temp_dir)
            
            # First iteration
            train_batches_1 = list(train_loader)
            test_batches_1 = list(test_loader)
            
            # Second iteration
            train_batches_2 = list(train_loader)
            test_batches_2 = list(test_loader)
            
            # Should have same number of batches
            assert len(train_batches_1) == len(train_batches_2)
            assert len(test_batches_1) == len(test_batches_2)
            
            # Training data should be different (shuffled)
            first_train_batch_1 = train_batches_1[0][0]
            first_train_batch_2 = train_batches_2[0][0]
            
            # With high probability, shuffled data should be different
            # (This could rarely fail due to random chance)
            assert not torch.equal(first_train_batch_1, first_train_batch_2)
    
    def test_custom_batch_sizes(self):
        """Test data preparation with custom batch sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(
                temp_dir,
                batch_size_train=32,
                batch_size_test=256
            )
            
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            
            assert train_batch[0].shape[0] == 32
            assert test_batch[0].shape[0] == 256
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_pin_memory(self):
        """Test pin_memory is set correctly when CUDA is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_loader, test_loader = prepare_data(temp_dir)
            
            # When CUDA is available, pin_memory should be True
            assert train_loader.pin_memory is True
            assert test_loader.pin_memory is True
