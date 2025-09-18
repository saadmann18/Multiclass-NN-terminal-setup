import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
from data import prepare_data
from model import CNN
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for saving figures
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@torch.no_grad()
def evaluate_model(test_dl, model):
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

        with autocast(enabled=(device.type == "cuda")):
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


def main():
    # prepare the data
    data_dir = os.path.expanduser('~/.torch/datasets/mnist')
    _, test_dl = prepare_data(data_dir)
    print(f"Test samples: {len(test_dl.dataset)}")

    # build model and load checkpoint
    model = CNN(1).to(device)
    ckpt_path = os.path.join(os.getcwd(), 'artifacts', 'model_mnist_cnn.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # Fallback if a full model was saved (not recommended)
        if hasattr(ckpt, 'state_dict'):
            model.load_state_dict(ckpt.state_dict())
        else:
            raise RuntimeError("Unexpected checkpoint format.")

    acc, avg_loss, conf_mat, per_class_acc = evaluate_model(test_dl, model)
    print(f"Test accuracy: {acc*100:.2f}%, avg loss: {avg_loss:.4f}")

    # Print per-class accuracy and confusion matrix
    if per_class_acc is not None:
        try:
            # Assume MNIST labels 0-9
            for cls, pc in enumerate(per_class_acc.tolist()):
                print(f"Class {cls}: {pc*100:.2f}%")
        except Exception:
            print("Per-class accuracy (tensor):", per_class_acc)

    if conf_mat is not None:
        print("Confusion matrix:")
        # Pretty print small confusion matrix
        with torch.no_grad():
            cm = conf_mat.numpy()
        for r in range(cm.shape[0]):
            print(" ".join(f"{int(v):4d}" for v in cm[r]))

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
            print(f"Saved confusion matrix heatmap to {fig_path}")
        else:
            print("matplotlib not available; skipping heatmap save. Install it with: pip install matplotlib")


if __name__ == "__main__":
    main()
