"""
Evaluation utilities for MNIST CNN model.
"""

import os
import warnings
from typing import Tuple, Optional, Union
from contextlib import nullcontext

# Suppress multiprocessing import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .model import CNN
from .data import prepare_data
from .utils import select_device

# Handle different PyTorch versions for mixed precision
try:  # PyTorch >= 2.0
    from torch.amp import autocast as _autocast

    HAVE_TORCH_AMP = True
except Exception:  # PyTorch < 2.0
    from torch.cuda.amp import autocast as _autocast  # type: ignore

    HAVE_TORCH_AMP = False

# Optional matplotlib for visualization
try:
    import matplotlib

    try:
        from IPython import get_ipython  # type: ignore

        IN_IPY = get_ipython() is not None
    except Exception:
        IN_IPY = False

    if not IN_IPY:
        matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


@torch.no_grad()
def evaluate_model(
    test_loader, model: torch.nn.Module, device: torch.device
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate the CNN model on test data.

    Args:
        test_loader: Test data loader
        model: Trained CNN model
        device: Device to run evaluation on

    Returns:
        Tuple containing:
        - accuracy (float): Overall accuracy
        - avg_loss (float): Average loss
        - conf_mat (torch.Tensor): Confusion matrix
        - per_class_acc (torch.Tensor): Per-class accuracy
    """
    criterion = CrossEntropyLoss()
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0
    num_classes = None
    conf_mat = None

    pbar = tqdm(test_loader, desc="Evaluating", leave=False)

    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed precision context
        if HAVE_TORCH_AMP and device.type == "cuda":
            ctx = _autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        elif not HAVE_TORCH_AMP and device.type == "cuda":
            ctx = _autocast(enabled=True)
        else:
            ctx = nullcontext()

        with ctx:
            logits = model(inputs)
            loss = criterion(logits, targets)

        # Initialize confusion matrix lazily
        if num_classes is None:
            num_classes = int(logits.size(1))
            conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        # Calculate predictions and accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        running_loss += loss.item()

        # Update confusion matrix
        targets_cpu = targets.view(-1).to(torch.int64).cpu()
        preds_cpu = preds.view(-1).to(torch.int64).cpu()
        cm_batch = torch.bincount(
            targets_cpu * num_classes + preds_cpu, minlength=num_classes * num_classes
        )
        conf_mat += cm_batch.view(num_classes, num_classes)

        # Update progress bar
        avg_loss = running_loss / (pbar.n or 1)
        acc = 100.0 * correct / max(1, total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")

    # Calculate final metrics
    accuracy = correct / max(1, total)
    avg_loss = running_loss / max(1, len(test_loader))

    # Calculate per-class accuracy
    per_class_acc = None
    if conf_mat is not None:
        diag = conf_mat.diag().to(torch.float64)
        row_sum = conf_mat.sum(dim=1).to(torch.float64)
        per_class_acc = torch.where(row_sum > 0, diag / row_sum, torch.zeros_like(diag))

    return accuracy, avg_loss, conf_mat, per_class_acc


def plot_confusion_matrix(
    conf_mat: torch.Tensor, save_path: Optional[str] = None, show: bool = True
) -> Optional[str]:
    """
    Plot and optionally save confusion matrix.

    Args:
        conf_mat: Confusion matrix tensor
        save_path: Path to save the plot
        show: Whether to display the plot

    Returns:
        str: Path where plot was saved, or None
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return None

    cm = conf_mat.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Add labels
    ax.set(
        xticks=range(cm.shape[1]),
        yticks=range(cm.shape[0]),
        xticklabels=range(cm.shape[1]),
        yticklabels=range(cm.shape[0]),
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    if show and IN_IPY:
        plt.show()
    elif not show:
        plt.close()

    return save_path


def main(
    device_preference: str = "auto",
    checkpoint_path: Optional[str] = None,
    show_plot: bool = True,
    data_dir: Optional[str] = None,
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """
    Complete evaluation pipeline for MNIST CNN.

    Args:
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        checkpoint_path: Path to model checkpoint
        show_plot: Whether to show confusion matrix plot
        data_dir: Directory containing MNIST data

    Returns:
        Tuple containing accuracy, avg_loss, conf_mat, per_class_acc, and plot_path
    """
    # Setup device
    device = select_device(device_preference)
    print(f"Using device: {device}")

    # Prepare data
    if data_dir is None:
        data_dir = os.path.expanduser("~/.torch/datasets/mnist")

    _, test_loader = prepare_data(data_dir)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    model = CNN(n_channels=1).to(device)

    if checkpoint_path is None:
        # Look for the most recent training run
        runs_dir = os.path.join(os.getcwd(), "runs")
        if os.path.exists(runs_dir):
            experiments = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if experiments:
                # Get most recent experiment
                latest_exp = sorted(experiments)[-1]
                checkpoint_path = os.path.join(runs_dir, latest_exp, "artifacts", "best_model.pth")
            else:
                checkpoint_path = os.path.join(os.getcwd(), "artifacts", "best_model.pth")
        else:
            checkpoint_path = os.path.join(os.getcwd(), "artifacts", "best_model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the dict itself is the state dict
            model.load_state_dict(checkpoint)
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)

    # Evaluate model
    accuracy, avg_loss, conf_mat, per_class_acc = evaluate_model(
        test_loader, model, device
    )

    # Print results
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Loss: {avg_loss:.4f}")

    if per_class_acc is not None:
        print(f"\nPer-class Accuracy:")
        for i, acc in enumerate(per_class_acc):
            print(f"  Class {i}: {acc:.4f} ({acc*100:.2f}%)")

    # Plot confusion matrix
    plot_path = None
    if conf_mat is not None and HAS_MATPLOTLIB:
        # Create artifacts directory if it doesn't exist
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plot_confusion_matrix(conf_mat, save_path=plot_path, show=show_plot)

    return accuracy, avg_loss, conf_mat, per_class_acc, plot_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate MNIST CNN model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cuda, mps, or cpu (default: auto)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: latest in runs/)')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.expanduser('~/.torch/datasets/mnist'),
                        help='Directory containing MNIST data')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable confusion matrix plot')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MNIST CNN EVALUATION")
    print("=" * 70)
    print(f"Device: {args.device}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("Checkpoint: Auto-detect latest")
    print(f"Data directory: {args.data_dir}")
    print("-" * 70)
    
    try:
        accuracy, avg_loss, conf_mat, per_class_acc, plot_path = main(
            device_preference=args.device,
            checkpoint_path=args.checkpoint,
            show_plot=not args.no_plot,
            data_dir=args.data_dir
        )
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Average Loss: {avg_loss:.4f}")
        
        if per_class_acc is not None:
            print("\nPer-class Accuracy:")
            for i, acc in enumerate(per_class_acc):
                print(f"  Digit {i}: {acc * 100:6.2f}%")
        
        if plot_path:
            print(f"\nConfusion matrix saved to: {plot_path}")
        print("=" * 70)
            
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import sys
        sys.exit(1)