# pytorch cnn for multiclass classification
import os
import argparse
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import CNN
from data import prepare_data
from contextlib import nullcontext

# Prefer torch.amp (PyTorch >= 2.0), fallback to torch.cuda.amp for older versions
try:  # PyTorch >= 2.0
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    HAVE_TORCH_AMP = True
except Exception:  # PyTorch < 2.0
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore
    HAVE_TORCH_AMP = False

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

# Enable backend optimizations when using CUDA
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def train_model(train_dl, model, epochs: int = 10):
    """
    Train the model with mixed precision on GPU when available.
    """
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = _GradScaler(enabled=(device.type == "cuda"))

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in pbar:
            # Prefer channels_last memory format on CUDA
            if device.type == "cuda":
                inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if HAVE_TORCH_AMP:
                ctx = _autocast(device_type=device.type, dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16), enabled=(device.type == "cuda"))
            else:
                # Older API only supports CUDA autocast
                ctx = _autocast(enabled=(device.type == "cuda")) if device.type == "cuda" else nullcontext()
            with ctx:
                yhat = model(inputs)
                loss = criterion(yhat, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n or 1):.4f}")

        epoch_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train MNIST CNN")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "auto"), choices=["auto", "cuda", "mps", "cpu"], help="Compute device to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (PyTorch 2.x, CUDA/CPU)")
    args = parser.parse_args()

    # re-select device in case flag provided
    global device
    device = select_device(args.device)
    print(f"Using device: {device}")
    # prepare the data
    data_dir = os.path.expanduser('~/.torch/datasets/mnist')
    train_dl, test_dl = prepare_data(data_dir)
    print(len(train_dl.dataset), len(test_dl.dataset))

    # define the network
    model = CNN(1).to(device)
    # Use channels_last for better memory access on CUDA
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Optional compile for speed (PyTorch 2.x)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore[attr-defined]
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile unavailable or failed: {e}")

    # Tune threading for CPU
    if device.type == "cpu":
        try:
            threads = max(1, min(8, (os.cpu_count() or 1)))
            torch.set_num_threads(threads)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, threads // 2))
            print(f"CPU threads set to {threads}")
        except Exception:
            pass

    # train the model
    train_model(train_dl, model, epochs=args.epochs)

    # Save model state dict
    save_dir = os.path.join(os.getcwd(), 'artifacts')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_mnist_cnn.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")

    ##########

    # evaluate the model (optional)


if __name__ == "__main__":
    main()
