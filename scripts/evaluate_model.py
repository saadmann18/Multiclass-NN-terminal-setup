#!/usr/bin/env python3
"""
Evaluation script with detailed analysis and reporting.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_cnn.config import Config
from mnist_cnn.eval import run_evaluation


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


def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    accuracy, avg_loss, conf_mat, per_class_acc, plot_path = results
    
    # Convert tensors to lists for JSON serialization
    results_dict = {
        "accuracy": float(accuracy),
        "average_loss": float(avg_loss),
        "confusion_matrix": conf_mat.tolist() if conf_mat is not None else None,
        "per_class_accuracy": per_class_acc.tolist() if per_class_acc is not None else None,
        "plot_path": plot_path
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)


def print_detailed_results(results):
    """Print detailed evaluation results."""
    accuracy, avg_loss, conf_mat, per_class_acc, plot_path = results
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Loss: {avg_loss:.4f}")
    
    if per_class_acc is not None:
        print("\nPer-Class Accuracy:")
        print("-" * 30)
        for i, acc in enumerate(per_class_acc):
            print(f"  Digit {i}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Find best and worst performing classes
        best_class = per_class_acc.argmax().item()
        worst_class = per_class_acc.argmin().item()
        print(f"\nBest performing digit: {best_class} ({per_class_acc[best_class]*100:.2f}%)")
        print(f"Worst performing digit: {worst_class} ({per_class_acc[worst_class]*100:.2f}%)")
    
    if conf_mat is not None:
        print(f"\nConfusion Matrix:")
        print("-" * 30)
        print("Predicted ->")
        print("Actual |")
        print("       v")
        
        # Print header
        print("     ", end="")
        for i in range(10):
            print(f"{i:4d}", end="")
        print()
        
        # Print matrix
        for i in range(10):
            print(f"{i:3d}: ", end="")
            for j in range(10):
                print(f"{conf_mat[i][j]:4d}", end="")
            print()
    
    if plot_path:
        print(f"\nConfusion matrix plot saved to: {plot_path}")
    
    print("="*50)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST CNN with detailed analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--checkpoint", "--model",
        type=str,
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for output files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file"
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
    
    # Evaluation parameters
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="Device")
    parser.add_argument("--no-plot", action="store_true", help="Don't show plots")
    parser.add_argument("--batch-size", type=int, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file
    if args.experiment_name and not log_file:
        log_file = f"logs/{args.experiment_name}_eval.log"
    
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Override with command line arguments
        if args.device is not None:
            config.set("training.device", args.device)
        if args.no_plot:
            config.set("evaluation.show_plot", False)
        if args.batch_size is not None:
            config.set("data.batch_size_test", args.batch_size)
        
        # Determine checkpoint path
        checkpoint_path = args.checkpoint
        if not checkpoint_path and args.experiment_name:
            checkpoint_path = f"{args.output_dir}/{args.experiment_name}_model.pth"
        
        logger.info("Evaluation Configuration:")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Device: {config.get('training.device')}")
        logger.info(f"  Batch Size: {config.get('data.batch_size_test')}")
        logger.info(f"  Show Plot: {config.get('evaluation.show_plot')}")
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = run_evaluation(
            device_preference=config.get("training.device"),
            checkpoint_path=checkpoint_path,
            show_plot=config.get("evaluation.show_plot"),
            data_dir=config.get("data.data_dir")
        )
        
        logger.info("Evaluation completed successfully!")
        
        # Print detailed results
        print_detailed_results(results)
        
        # Save results if requested
        if args.save_results:
            output_name = args.experiment_name or "evaluation"
            results_path = f"{args.output_dir}/{output_name}_results.json"
            save_results(results, results_path)
            logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
