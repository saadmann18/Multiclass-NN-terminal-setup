"""
Command-line interface for MNIST CNN training and evaluation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import Config
from .train import run_training
from .eval import run_evaluation


def train_cli():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description="Train MNIST CNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device", "-d",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store MNIST data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.set("training.epochs", args.epochs)
    if args.device is not None:
        config.set("training.device", args.device)
    if args.batch_size is not None:
        config.set("training.batch_size_train", args.batch_size)
    if args.learning_rate is not None:
        config.set("training.learning_rate", args.learning_rate)
    if args.compile:
        config.set("model.compile", True)
    if args.data_dir is not None:
        config.set("data.data_dir", args.data_dir)
    
    try:
        # Run training
        model, save_path = run_training(
            device_preference=config.get("training.device", "auto"),
            epochs=config.get("training.epochs", 10),
            compile_model=config.get("model.compile", False),
            data_dir=config.get("data.data_dir")
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {save_path}")
        
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


def eval_cli():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST CNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", "--ckpt",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", "-d",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't show confusion matrix plot"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing MNIST data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Override config with command line arguments
    if args.device is not None:
        config.set("training.device", args.device)
    if args.no_plot:
        config.set("evaluation.show_plot", False)
    if args.data_dir is not None:
        config.set("data.data_dir", args.data_dir)
    
    try:
        # Run evaluation
        accuracy, avg_loss, conf_mat, per_class_acc, plot_path = run_evaluation(
            device_preference=config.get("training.device", "auto"),
            checkpoint_path=args.checkpoint,
            show_plot=config.get("evaluation.show_plot", True),
            data_dir=config.get("data.data_dir")
        )
        
        print(f"Evaluation completed successfully!")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Average Loss: {avg_loss:.4f}")
        
        if plot_path:
            print(f"Confusion matrix saved to: {plot_path}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: mnist-cnn <command> [options]")
        print("Commands: train, eval")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from argv
    
    if command == "train":
        train_cli()
    elif command == "eval":
        eval_cli()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
