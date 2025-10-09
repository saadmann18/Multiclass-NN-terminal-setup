#!/usr/bin/env python3
"""
Script to run multiple experiments with different hyperparameters.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_cnn.config import Config
from mnist_cnn.train import run_training
from mnist_cnn.eval import run_evaluation


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler()]
    )


def run_single_experiment(exp_config: Dict, exp_name: str, base_config: Config) -> Dict:
    """Run a single experiment with given configuration."""
    logger = logging.getLogger(__name__)
    
    # Create experiment-specific config
    config = Config(base_config.config_path)
    config.update(exp_config)
    
    # Create experiment directory
    exp_dir = f"experiments/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment config
    config_path = f"{exp_dir}/config.yaml"
    config.save(config_path)
    
    logger.info(f"Running experiment: {exp_name}")
    logger.info(f"Configuration: {exp_config}")
    
    try:
        # Train model
        model, save_path = run_training(
            device_preference=config.get("training.device"),
            epochs=config.get("training.epochs"),
            compile_model=config.get("model.compile"),
            data_dir=config.get("data.data_dir")
        )
        
        # Move model to experiment directory
        exp_model_path = f"{exp_dir}/model.pth"
        os.rename(save_path, exp_model_path)
        
        # Evaluate model
        accuracy, avg_loss, conf_mat, per_class_acc, plot_path = run_evaluation(
            device_preference=config.get("training.device"),
            checkpoint_path=exp_model_path,
            show_plot=False,
            data_dir=config.get("data.data_dir")
        )
        
        # Move confusion matrix plot
        if plot_path:
            exp_plot_path = f"{exp_dir}/confusion_matrix.png"
            os.rename(plot_path, exp_plot_path)
        
        # Prepare results
        results = {
            "experiment_name": exp_name,
            "config": exp_config,
            "accuracy": float(accuracy),
            "average_loss": float(avg_loss),
            "per_class_accuracy": per_class_acc.tolist() if per_class_acc is not None else None,
            "model_path": exp_model_path,
            "config_path": config_path,
            "status": "success"
        }
        
        logger.info(f"Experiment {exp_name} completed successfully!")
        logger.info(f"Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Experiment {exp_name} failed: {e}")
        results = {
            "experiment_name": exp_name,
            "config": exp_config,
            "error": str(e),
            "status": "failed"
        }
    
    return results


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run multiple MNIST CNN experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.yaml",
        help="Base configuration file"
    )
    parser.add_argument(
        "--experiments-file",
        type=str,
        help="JSON file with experiment configurations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load base configuration
    base_config = Config(args.config)
    logger.info(f"Loaded base configuration from {args.config}")
    
    # Define experiments
    if args.experiments_file:
        with open(args.experiments_file, 'r') as f:
            experiments = json.load(f)
    else:
        # Default experiments
        experiments = {
            "baseline": {
                "training": {"epochs": 10, "learning_rate": 0.01}
            },
            "high_lr": {
                "training": {"epochs": 10, "learning_rate": 0.05}
            },
            "low_lr": {
                "training": {"epochs": 10, "learning_rate": 0.001}
            },
            "more_epochs": {
                "training": {"epochs": 20, "learning_rate": 0.01}
            },
            "large_batch": {
                "training": {"epochs": 10, "learning_rate": 0.01, "batch_size_train": 256}
            },
            "small_batch": {
                "training": {"epochs": 10, "learning_rate": 0.01, "batch_size_train": 64}
            }
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Running {len(experiments)} experiments...")
    
    for exp_name, exp_config in experiments.items():
        exp_name_with_timestamp = f"{timestamp}_{exp_name}"
        results = run_single_experiment(exp_config, exp_name_with_timestamp, base_config)
        all_results.append(results)
    
    # Save summary results
    summary_path = f"{args.output_dir}/{timestamp}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    successful_experiments = [r for r in all_results if r["status"] == "success"]
    failed_experiments = [r for r in all_results if r["status"] == "failed"]
    
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print("\nSuccessful Experiments:")
        print("-" * 40)
        # Sort by accuracy
        successful_experiments.sort(key=lambda x: x["accuracy"], reverse=True)
        
        for i, result in enumerate(successful_experiments, 1):
            print(f"{i}. {result['experiment_name']}")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   Loss: {result['average_loss']:.4f}")
            
            # Print key config differences
            config_str = []
            if "training" in result["config"]:
                training = result["config"]["training"]
                if "learning_rate" in training:
                    config_str.append(f"LR={training['learning_rate']}")
                if "epochs" in training:
                    config_str.append(f"Epochs={training['epochs']}")
                if "batch_size_train" in training:
                    config_str.append(f"Batch={training['batch_size_train']}")
            
            if config_str:
                print(f"   Config: {', '.join(config_str)}")
            print()
    
    if failed_experiments:
        print("Failed Experiments:")
        print("-" * 40)
        for result in failed_experiments:
            print(f"- {result['experiment_name']}: {result['error']}")
    
    print(f"\nDetailed results saved to: {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
