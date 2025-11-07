"""
Training utilities for MNIST CNN model.
"""

import os
import numpy as np
import warnings
from typing import Tuple, Optional, List, Dict, Any
from contextlib import nullcontext

import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .model import CNN
from .data import prepare_data
from .utils import select_device, setup_device_optimizations
from .callbacks import BaseCallback, TensorBoardLogger
import math

# Suppress false positive scheduler warning
warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')

# Handle different PyTorch versions for mixed precision
try:  # PyTorch >= 2.0
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler

    HAVE_TORCH_AMP = True
except Exception:  # PyTorch < 2.0
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore

    HAVE_TORCH_AMP = False


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth.tar'
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    # If this is the best model, save it separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_path)


def train_model(
    train_loader,
    val_loader,
    model: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    callbacks: Optional[List[BaseCallback]] = None,
    grad_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    use_amp: bool = True,
    checkpoint_dir: str = 'checkpoints',
) -> Dict[str, Any]:
    """
    Train the CNN model with mixed precision support and performance optimizations.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: CNN model to train
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
        weight_decay: Weight decay for regularization
        callbacks: List of callbacks to use during training
        grad_accumulation_steps: Number of steps for gradient accumulation
        clip_grad_norm: Maximum gradient norm for gradient clipping
        use_amp: Whether to use Automatic Mixed Precision (AMP)
        
    Returns:
        Dictionary containing training history and metrics
    """
    # Initialize model in training mode and move to device
    model.train()
    model = model.to(device, non_blocking=True)
    
    # Initialize loss function with label smoothing
    criterion = CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Initialize optimizer with Nesterov momentum and weight decay
    optimizer = SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=momentum, 
        weight_decay=weight_decay,
        nesterov=True
    )
    
    # Mixed precision training
    scaler = _GradScaler(enabled=use_amp and (device.type == "cuda"))
    
    # Learning rate scheduler with 5-epoch warmup and cosine decay
    total_steps = len(train_loader) * epochs
    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        # Linear warmup for first 5 epochs
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # Clamp progress to [0,1]
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda,
        last_epoch=-1
    )
    
    # Log scheduler info
    if callbacks:
        for cb in callbacks:
            if hasattr(cb, 'log_scalar'):
                cb.log_scalar('hparams/initial_learning_rate', learning_rate, 0)
                cb.log_scalar('hparams/total_steps', total_steps, 0)
                cb.log_scalar('hparams/warmup_steps', warmup_steps, 0)
                cb.log_scalar('hparams/warmup_epochs', warmup_epochs, 0)
    
    # Initialize callbacks
    callbacks = callbacks or []
    for cb in callbacks:
        cb.on_train_begin()
    
    # Initialize history and best metric tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rates': [],
    }
    
    # Track best validation accuracy
    best_val_accuracy = 0.0
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    # Log initial learning rate
    for cb in callbacks:
        if hasattr(cb, 'writer') and hasattr(cb, 'log_scalar'):
            cb.log_scalar('learning_rate', optimizer.param_groups[0]['lr'], 0)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct = 0
        total = 0
        
        # Configure progress bar
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs}", 
            leave=False,
            dynamic_ncols=True
        )
        
        # Notify callbacks of epoch start
        for cb in callbacks:
            cb.on_epoch_begin(epoch)
        
        # Process each batch
        optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device asynchronously
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with _autocast(device_type=device.type, enabled=use_amp and (device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / grad_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Gradient clipping
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=clip_grad_norm
                    )
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update learning rate
                scheduler.step()
                
                # Log learning rate to TensorBoard
                global_step = epoch * len(train_loader) + batch_idx
                for cb in callbacks:
                    if hasattr(cb, 'writer') and hasattr(cb, 'log_scalar'):
                        cb.log_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            
            # Update running loss and accuracy
            batch_size = inputs.size(0)
            running_loss += loss.item() * grad_accumulation_steps * batch_size
            total_samples += batch_size
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            avg_loss = running_loss / total_samples
            train_acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update progress bar
            # Get GPU memory usage if available
            gpu_mem = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                gpu_mem = f"{allocated:.0f}/{reserved:.0f}MB"
                
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{train_acc:.2f}%",
                'lr': f"{current_lr:.2e}",
                'GPU': gpu_mem if gpu_mem else "CPU"
            })
            
            # Update global step
            global_step = epoch * len(train_loader) + batch_idx + 1
            
            # Callbacks for batch end (reduced frequency)
            if callbacks and (batch_idx + 1) % 10 == 0:
                for cb in callbacks:
                    cb.on_batch_end(batch_idx, {
                        'outputs': outputs.detach(),
                        'targets': targets,
                        'loss': loss.item() * grad_accumulation_steps,
                        'lr': current_lr
                    })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        train_acc = 100.0 * correct / total
        
        # Validate the model
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, device, callbacks, use_amp
        )
        
        # Store metrics
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Check if this is the best model so far
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
            },
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename=f'checkpoint_epoch_{epoch+1}.pth.tar'
        )
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_acc*100:.2f}% - "
              f"LR: {current_lr:.2e}")
        
        # Call epoch end callbacks
        log_data = {
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'lr': current_lr,
            'epoch': epoch
        }
        
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs=log_data)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Training complete
    training_results = {
        'model': model,
        'best_val_loss': best_val_loss,
        'history': history,
        'best_model_state': best_model_state
    }
    
    for cb in callbacks:
        cb.on_train_end(logs=training_results)
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return training_results


def validate_model(
    model: torch.nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    callbacks: Optional[List[BaseCallback]] = None,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use for validation
        callbacks: Optional list of callbacks
        use_amp: Whether to use Automatic Mixed Precision
        
    Returns:
        Tuple of (val_loss, val_accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with _autocast(device_type=device.type, enabled=use_amp and (device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Call validation callbacks if provided
            if callbacks and (batch_idx + 1) % 10 == 0:  # Reduce callback frequency
                for cb in callbacks:
                    cb.on_batch_end(batch_idx, {
                        'outputs': outputs,
                        'targets': targets,
                        'loss': loss.item(),
                        'accuracy': correct / total
                    })
    
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    
    return val_loss, val_accuracy


def run_training(
    device_preference: str = "auto",
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    grad_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    use_amp: bool = True,
    data_dir: Optional[str] = None,
    log_dir: str = "runs",
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete training pipeline for MNIST CNN with optimized settings.

    Args:
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        momentum: Momentum for SGD optimizer
        grad_accumulation_steps: Number of gradient accumulation steps
        clip_grad_norm: Maximum gradient norm for gradient clipping
        use_amp: Whether to use Automatic Mixed Precision
        data_dir: Directory to store MNIST data
        log_dir: Directory to save logs
        experiment_name: Name of the experiment for logging

    Returns:
        Dictionary containing training results and model information
    """
    # Setup device
    device = select_device(device_preference)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    # Prepare data with optimized settings
    if data_dir is None:
        data_dir = os.path.expanduser("~/.torch/datasets/mnist")

    # Configure DataLoader for optimal performance
    num_workers = min(8, os.cpu_count() - 1) if os.cpu_count() > 1 else 0
    pin_memory = device.type == 'cuda'

    train_loader, test_loader = prepare_data(
        path=data_dir,
        batch_size_train=batch_size,
        batch_size_test=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Initialize model
    model = CNN(n_channels=1)

    # Setup experiment directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"mnist_cnn_{timestamp}"
    
    # Create a unique run directory for this experiment
    run_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving logs and checkpoints to: {run_dir}")

    # Save model architecture
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    # Prepare hyperparameters dictionary
    hparams = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'grad_accumulation_steps': grad_accumulation_steps,
        'clip_grad_norm': clip_grad_norm,
        'use_amp': use_amp,
        'epochs': epochs,
        'warmup_epochs': 5,  # Hardcoded to match lr_scheduler
        'optimizer': 'SGD',
        'model': model.__class__.__name__,
        'experiment_name': experiment_name,
        'device': device.type,
        'num_workers': train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0,
        'pin_memory': getattr(train_loader, 'pin_memory', False),
        'dropout': getattr(model, 'dropout_rate', 0.0)  # If your model has dropout
    }
    
    # Initialize callbacks
    callbacks = [
        TensorBoardLogger(
            log_dir=run_dir,
            model=model,
            input_shape=(1, 28, 28),
            log_interval=10,
            hparams=hparams
        )
    ]
    
    # Store validation loader in callbacks for embedding visualization
    for cb in callbacks:
        if hasattr(cb, 'val_loader'):
            cb.val_loader = test_loader  # Use test_loader for visualization
            cb.device = device

    # Train the model
    print("Starting training...")
    results = train_model(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        callbacks=callbacks,  # Use the callbacks list we defined earlier
        grad_accumulation_steps=grad_accumulation_steps,
        clip_grad_norm=clip_grad_norm,
        use_amp=use_amp
    )

    # Save the best model
    best_model_path = os.path.join(artifacts_dir, "best_model.pth")
    torch.save(results['best_model_state'], best_model_path)
    results['best_model_path'] = best_model_path

    # Save training configuration
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'grad_accumulation_steps': grad_accumulation_steps,
        'clip_grad_norm': clip_grad_norm,
        'use_amp': use_amp,
        'device': str(device),
        'best_val_loss': results['best_val_loss'],
        'best_val_accuracy': max(results['history']['val_accuracy']) if results['history']['val_accuracy'] else 0,
        'timestamp': timestamp
    }

    import json
    with open(os.path.join(artifacts_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return results
