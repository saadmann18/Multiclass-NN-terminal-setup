"""
Evaluation utilities for MNIST CNN model.
"""

import os
from typing import Tuple, Optional, Union
from contextlib import nullcontext

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
        matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


@torch.no_grad()
def evaluate_model(
    test_loader, 
    model: torch.nn.Module, 
    device: torch.device
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
            targets_cpu * num_classes + preds_cpu, 
            minlength=num_classes * num_classes
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
        per_class_acc = torch.where(
            row_sum > 0, 
            diag / row_sum, 
            torch.zeros_like(diag)
        )
    
    return accuracy, avg_loss, conf_mat, per_class_acc


def plot_confusion_matrix(
    conf_mat: torch.Tensor, 
    save_path: Optional[str] = None,
    show: bool = True
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
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set(xticks=range(cm.shape[1]),
           yticks=range(cm.shape[0]),
           xticklabels=range(cm.shape[1]),
           yticklabels=range(cm.shape[0]),
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if show and IN_IPY:
        plt.show()
    elif not show:
        plt.close()
        
    return save_path


def run_evaluation(
    device_preference: str = "auto",
    checkpoint_path: Optional[str] = None,
    show_plot: bool = True,
    data_dir: Optional[str] = None
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
        data_dir = os.path.expanduser('~/.torch/datasets/mnist')
    
    _, test_loader = prepare_data(data_dir)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    model = CNN(n_channels=1).to(device)
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.getcwd(), 'artifacts', 'model_mnist_cnn.pth')
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise RuntimeError("Unexpected checkpoint format.")
    
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
        plot_path = os.path.join(os.getcwd(), 'artifacts', 'confusion_matrix.png')
        plot_confusion_matrix(conf_mat, save_path=plot_path, show=show_plot)
    
    return accuracy, avg_loss, conf_mat, per_class_acc, plot_path
