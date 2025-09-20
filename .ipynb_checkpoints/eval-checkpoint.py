import os
import argparse
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext
# Prefer torch.amp (PyTorch >= 2.0), fallback to torch.cuda.amp for older versions
try:  # PyTorch >= 2.0
    from torch.amp import autocast as _autocast
    HAVE_TORCH_AMP = True
except Exception:  # PyTorch < 2.0
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    HAVE_TORCH_AMP = False
from data import prepare_data
from model import CNN
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for saving figures
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

def select_device(preference: str = "auto") -> torch.device:
    pref = (preference or "auto").lower()
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = select_device(os.environ.get("DEVICE", "auto"))
print(f"Using device: {device}")


@torch.no_grad()
def evaluate_model(test_dl, model, device: torch.device):
    criterion = CrossEntropyLoss()
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0
    num_classes = None
    conf_mat = None  # Will initialize once num_classes is known

    pbar = tqdm(test_dl, desc="Evaluating", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if HAVE_TORCH_AMP:
            ctx = _autocast(device_type=device.type, dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16), enabled=(device.type == "cuda"))
        else:
            ctx = _autocast(enabled=(device.type == "cuda")) if device.type == "cuda" else nullcontext()
        with ctx:
            logits = model(inputs)
            loss = criterion(logits, targets)

        # Initialize confusion matrix lazily using model output size
        if num_classes is None:
            num_classes = int(logits.size(1))
            conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        running_loss += loss.item()

        # Update confusion matrix on CPU
        t = targets.view(-1).to(torch.int64).cpu()
        p = preds.view(-1).to(torch.int64).cpu()
        cm_batch = torch.bincount(t * num_classes + p, minlength=num_classes * num_classes)
        conf_mat += cm_batch.view(num_classes, num_classes)

        avg_loss = running_loss / (pbar.n or 1)
        acc = 100.0 * correct / max(1, total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")

    accuracy = correct / max(1, total)
    avg_loss = running_loss / max(1, len(test_dl))

    # Per-class accuracy
    per_class_acc = None
    if conf_mat is not None:
        diag = conf_mat.diag().to(torch.float64)
        row_sum = conf_mat.sum(dim=1).to(torch.float64)
        per_class_acc = torch.where(row_sum > 0, diag / row_sum, torch.zeros_like(diag))

    return accuracy, avg_loss, conf_mat, per_class_acc


def run_evaluation(device: str = "auto", ckpt_path: str | None = None):
    """
    Notebook-friendly entrypoint to evaluate the MNIST CNN.
    Returns (accuracy, avg_loss, conf_mat, per_class_acc, fig_path or None).
    """
    # Select device for this run (local variable)
    torch_device = select_device(device)
    print(f"Using device: {torch_device}")

    # prepare the data
    data_dir = os.path.expanduser('~/.torch/datasets/mnist')
    _, test_dl = prepare_data(data_dir)
    print(f"Test samples: {len(test_dl.dataset)}")

    # build model and load checkpoint
    model = CNN(1).to(torch_device)
    if ckpt_path is None:
        ckpt_path = os.path.join(os.getcwd(), 'artifacts', 'model_mnist_cnn.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")

    ckpt = torch.load(ckpt_path, map_location=torch_device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # Fallback if a full model was saved (not recommended)
        if hasattr(ckpt, 'state_dict'):
            model.load_state_dict(ckpt.state_dict())
        else:
            raise RuntimeError("Unexpected checkpoint format.")

    # Evaluate on the selected device without mutating module-global state
    acc, avg_loss, conf_mat, per_class_acc = evaluate_model(test_dl, model, device=torch_device)

    fig_path = None
    if conf_mat is not None:
        # Pretty print small confusion matrix
        with torch.no_grad():
            cm = conf_mat.numpy()
        # Save heatmap if matplotlib is available
        if plt is not None:
            save_dir = os.path.join(os.getcwd(), 'artifacts')
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, 'confusion_matrix.png')

            # Normalize rows to percentages for readability
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = cm / cm_sum.clip(min=1)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Confusion Matrix (row-normalized)')
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            num_classes = cm.shape[0]
            ticks = list(range(num_classes))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks)
            ax.set_yticklabels(ticks)
            # Rotate tick labels for x-axis
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            # Annotate cells with counts and percentages
            for i in range(num_classes):
                for j in range(num_classes):
                    count = cm[i, j]
                    pct = cm_norm[i, j] * 100.0
                    text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                    ax.text(j, i, f"{count}\n{pct:.1f}%", ha='center', va='center', color=text_color, fontsize=8)

            fig.tight_layout()
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

    return acc, avg_loss, conf_mat, per_class_acc, fig_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate MNIST CNN")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"], help="Compute device to use")
    # Use parse_known_args to ignore extraneous args injected by Jupyter/IPython (e.g., -f ...)
    args = parser.parse_known_args()[0]

    acc, avg_loss, conf_mat, per_class_acc, fig_path = run_evaluation(device=args.device)

    print(f"Test accuracy: {acc*100:.2f}%, avg loss: {avg_loss:.4f}")
    if per_class_acc is not None:
        try:
            for cls, pc in enumerate(per_class_acc.tolist()):
                print(f"Class {cls}: {pc*100:.2f}%")
        except Exception:
            print("Per-class accuracy (tensor):", per_class_acc)
    if conf_mat is not None:
        print("Confusion matrix:")
        with torch.no_grad():
            cm = conf_mat.numpy()
        for r in range(cm.shape[0]):
            print(" ".join(f"{int(v):4d}" for v in cm[r]))
    if fig_path:
        print(f"Saved confusion matrix heatmap to {fig_path}")
if __name__ == "__main__":
    main()
