"""Script to train the MNIST CNN model with optimized settings."""
import os
import sys
import torch
import datetime
from src.train import run_training

os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    
    # Training configuration
    config = {
        'device_preference': 'auto',  # 'auto', 'cuda', 'mps', or 'cpu'
        'epochs': 20,  # Reduced for faster training
        'batch_size': 256,  # Optimized batch size
        'learning_rate': 0.1,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'grad_accumulation_steps': 1,  # Reduced for simpler training
        'clip_grad_norm': 1.0,  # Gradient clipping
        'use_amp': True,  # Automatic Mixed Precision
        'data_dir': os.path.expanduser("~/.torch/datasets/mnist"),
        'log_dir': 'runs',
        'experiment_name': f'mnist_cnn_experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    results = run_training(**config)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy: {max(results['history']['val_accuracy']) * 100:.2f}%")
    print(f"Model saved to: {results['best_model_path']}")
    print("="*70)

if __name__ == "__main__":
    main()
