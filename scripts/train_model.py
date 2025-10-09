#!/usr/bin/env python3
"""
Training script with advanced configuration and logging.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_cnn.config import Config
from mnist_cnn.train import run_training


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train MNIST CNN with advanced options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for logging and artifacts"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="Device")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file
    if args.experiment_name and not log_file:
        log_file = f"logs/{args.experiment_name}_train.log"
    
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override with command line arguments
        if args.epochs is not None:
            config.set("training.epochs", args.epochs)
        if args.batch_size is not None:
            config.set("training.batch_size_train", args.batch_size)
        if args.learning_rate is not None:
            config.set("training.learning_rate", args.learning_rate)
        if args.device is not None:
            config.set("training.device", args.device)
        if args.compile:
            config.set("model.compile", True)
        
        # Add experiment name to config
        if args.experiment_name:
            config.set("experiment.name", args.experiment_name)
        
        # Print configuration
        logger.info("Training Configuration:")
        logger.info(f"  Epochs: {config.get('training.epochs')}")
        logger.info(f"  Batch Size: {config.get('training.batch_size_train')}")
        logger.info(f"  Learning Rate: {config.get('training.learning_rate')}")
        logger.info(f"  Device: {config.get('training.device')}")
        logger.info(f"  Compile: {config.get('model.compile')}")
        
        if args.dry_run:
            logger.info("Dry run mode - exiting without training")
            return
        
        # Save experiment configuration
        if args.experiment_name:
            exp_config_path = f"artifacts/{args.experiment_name}_config.yaml"
            config.save(exp_config_path)
            logger.info(f"Saved experiment configuration to {exp_config_path}")
        
        # Run training
        logger.info("Starting training...")
        model, save_path = run_training(
            device_preference=config.get("training.device"),
            epochs=config.get("training.epochs"),
            compile_model=config.get("model.compile"),
            data_dir=config.get("data.data_dir")
        )
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {save_path}")
        
        # Rename model file if experiment name provided
        if args.experiment_name:
            exp_model_path = f"artifacts/{args.experiment_name}_model.pth"
            os.rename(save_path, exp_model_path)
            logger.info(f"Model renamed to: {exp_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
