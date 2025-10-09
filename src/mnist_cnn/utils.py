"""
Utility functions for device selection and optimization.
"""

import os
import torch


def select_device(preference: str = "auto") -> torch.device:
    """
    Select the best available device for computation.
    
    Args:
        preference (str): Device preference - "auto", "cuda", "mps", or "cpu"
        
    Returns:
        torch.device: Selected device
    """
    pref = (preference or "auto").lower()
    
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif pref == "mps":
        return torch.device(
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
            else "cpu"
        )
    elif pref == "cpu":
        return torch.device("cpu")
    
    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_device_optimizations(device: torch.device) -> None:
    """
    Setup device-specific optimizations.
    
    Args:
        device (torch.device): Target device
    """
    if device.type == "cuda":
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Set high precision matrix multiplication (PyTorch 2.0+)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            
    elif device.type == "cpu":
        # Optimize CPU threading
        try:
            threads = max(1, min(8, (os.cpu_count() or 1)))
            torch.set_num_threads(threads)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, threads // 2))
        except Exception:
            pass
