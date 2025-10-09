"""
Training utilities for MNIST CNN model.
"""

import os
from typing import Tuple, Optional
from contextlib import nullcontext

import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .model import CNN
from .data import prepare_data
from .utils import select_device, setup_device_optimizations

# Handle different PyTorch versions for mixed precision
try:  # PyTorch >= 2.0
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    HAVE_TORCH_AMP = True
except Exception:  # PyTorch < 2.0
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    HAVE_TORCH_AMP = False


def train_model(
    train_loader, 
    model: torch.nn.Module, 
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 0.01,
    momentum: float = 0.9
) -> None:
    """
    Train the CNN model with mixed precision support.
    
    Args:
        train_loader: Training data loader
        model: CNN model to train
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
    """
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scaler = _GradScaler(enabled=(device.type == "cuda"))
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for inputs, targets in pbar:
            # Move data to device with optimizations
            if device.type == "cuda":
                inputs = inputs.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision context
            if HAVE_TORCH_AMP and device.type == "cuda":
                ctx = _autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            elif not HAVE_TORCH_AMP and device.type == "cuda":
                ctx = _autocast(enabled=True)
            else:
                ctx = nullcontext()
                
            # Forward pass
            with ctx:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n or 1):.4f}")
        
        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


def run_training(
    device_preference: str = "auto",
    epochs: int = 10,
    compile_model: bool = False,
    data_dir: Optional[str] = None
) -> Tuple[torch.nn.Module, str]:
    """
    Complete training pipeline for MNIST CNN.
    
    Args:
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        epochs: Number of training epochs
        compile_model: Whether to use torch.compile (PyTorch 2.0+)
        data_dir: Directory to store MNIST data
        
    Returns:
        Tuple[torch.nn.Module, str]: Trained model and save path
    """
    # Setup device
    device = select_device(device_preference)
    setup_device_optimizations(device)
    print(f"Using device: {device}")
    
    # Prepare data
    if data_dir is None:
        data_dir = os.path.expanduser('~/.torch/datasets/mnist')
    
    train_loader, test_loader = prepare_data(data_dir)
    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Initialize model
    model = CNN(n_channels=1).to(device)
    
    # Optimize memory layout for CUDA
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    
    # Optional model compilation (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed: {e}")
    
    # Train the model
    train_model(train_loader, model, device, epochs=epochs)
    
    # Save model
    save_dir = os.path.join(os.getcwd(), 'artifacts')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_mnist_cnn.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'n_channels': 1},
        'training_config': {
            'epochs': epochs,
            'device': str(device),
            'compiled': compile_model
        }
    }, save_path)
    
    print(f"Model saved to {save_path}")
    return model, save_path
