"""
Tests for utility functions.
"""

import torch
import pytest
from unittest.mock import patch, MagicMock
from src.utils import select_device, setup_device_optimizations


class TestDeviceSelection:
    """Test cases for device selection utilities."""

    def test_select_device_cpu(self):
        """Test CPU device selection."""
        device = select_device("cpu")
        assert device.type == "cpu"

    @patch("torch.cuda.is_available")
    def test_select_device_cuda_available(self, mock_cuda_available):
        """Test CUDA device selection when available."""
        mock_cuda_available.return_value = True

        device = select_device("cuda")
        assert device.type == "cuda"

    @patch("torch.cuda.is_available")
    def test_select_device_cuda_unavailable(self, mock_cuda_available):
        """Test CUDA device selection when unavailable."""
        mock_cuda_available.return_value = False

        device = select_device("cuda")
        assert device.type == "cpu"  # Should fallback to CPU

    @patch("torch.backends.mps.is_available")
    @patch("src.mnist_cnn.utils.hasattr")
    def test_select_device_mps_available(self, mock_hasattr, mock_mps_available):
        """Test MPS device selection when available."""
        mock_hasattr.return_value = True  # MPS backend exists
        mock_mps_available.return_value = True

        device = select_device("mps")
        assert device.type == "mps"

    @patch("torch.cuda.is_available")
    def test_select_device_auto_cuda(self, mock_cuda_available):
        """Test auto device selection prefers CUDA."""
        mock_cuda_available.return_value = True

        device = select_device("auto")
        assert device.type == "cuda"

    @patch("torch.backends.mps.is_available")
    @patch("src.mnist_cnn.utils.hasattr")
    @patch("torch.cuda.is_available")
    def test_select_device_auto_mps(
        self, mock_cuda_available, mock_hasattr, mock_mps_available
    ):
        """Test auto device selection falls back to MPS."""
        mock_cuda_available.return_value = False
        mock_hasattr.return_value = True  # MPS backend exists
        mock_mps_available.return_value = True

        device = select_device("auto")
        assert device.type == "mps"

    @patch("src.mnist_cnn.utils.hasattr")
    @patch("torch.cuda.is_available")
    def test_select_device_auto_cpu(self, mock_cuda_available, mock_hasattr):
        """Test auto device selection falls back to CPU."""
        mock_cuda_available.return_value = False
        mock_hasattr.return_value = False  # No MPS available

        device = select_device("auto")
        assert device.type == "cpu"

    def test_select_device_default_auto(self):
        """Test default device selection is auto."""
        device = select_device()
        assert device.type in ["cpu", "cuda", "mps"]

    def test_select_device_case_insensitive(self):
        """Test device selection is case insensitive."""
        device_upper = select_device("CPU")
        device_lower = select_device("cpu")
        device_mixed = select_device("Cpu")

        assert device_upper.type == device_lower.type == device_mixed.type == "cpu"


class TestDeviceOptimizations:
    """Test cases for device optimization setup."""

    @patch("torch.backends.cudnn")
    @patch("torch.set_float32_matmul_precision")
    def test_setup_cuda_optimizations(self, mock_matmul_precision, mock_cudnn):
        """Test CUDA optimizations are set up correctly."""
        device = torch.device("cuda")

        setup_device_optimizations(device)

        # Should enable cuDNN benchmark
        assert mock_cudnn.benchmark is True

        # Should set high precision matmul
        mock_matmul_precision.assert_called_once_with("high")

    @patch("torch.set_num_threads")
    @patch("torch.set_num_interop_threads")
    @patch("os.cpu_count")
    def test_setup_cpu_optimizations(
        self, mock_cpu_count, mock_interop_threads, mock_threads
    ):
        """Test CPU optimizations are set up correctly."""
        mock_cpu_count.return_value = 8
        device = torch.device("cpu")

        setup_device_optimizations(device)

        # Should set thread counts
        mock_threads.assert_called_once_with(8)
        mock_interop_threads.assert_called_once_with(4)

    @patch("torch.set_num_threads")
    @patch("os.cpu_count")
    def test_setup_cpu_optimizations_low_core_count(self, mock_cpu_count, mock_threads):
        """Test CPU optimizations with low core count."""
        mock_cpu_count.return_value = 2
        device = torch.device("cpu")

        setup_device_optimizations(device)

        # Should use minimum of 1 thread
        mock_threads.assert_called_once_with(2)

    @patch("torch.set_num_threads")
    @patch("os.cpu_count")
    def test_setup_cpu_optimizations_high_core_count(
        self, mock_cpu_count, mock_threads
    ):
        """Test CPU optimizations with high core count."""
        mock_cpu_count.return_value = 16
        device = torch.device("cpu")

        setup_device_optimizations(device)

        # Should cap at 8 threads
        mock_threads.assert_called_once_with(8)

    @patch("torch.set_num_threads")
    @patch("os.cpu_count")
    def test_setup_cpu_optimizations_exception_handling(
        self, mock_cpu_count, mock_threads
    ):
        """Test CPU optimizations handle exceptions gracefully."""
        mock_cpu_count.return_value = 4
        mock_threads.side_effect = Exception("Thread setting failed")
        device = torch.device("cpu")

        # Should not raise exception
        setup_device_optimizations(device)

    def test_setup_mps_optimizations(self):
        """Test MPS device optimizations (should do nothing)."""
        device = torch.device("mps")

        # Should not raise any exceptions
        setup_device_optimizations(device)

    @patch("os.cpu_count")
    def test_setup_cpu_optimizations_no_cpu_count(self, mock_cpu_count):
        """Test CPU optimizations when cpu_count returns None."""
        mock_cpu_count.return_value = None
        device = torch.device("cpu")

        # Should not raise exception
        setup_device_optimizations(device)
