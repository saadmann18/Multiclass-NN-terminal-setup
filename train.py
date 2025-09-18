# pytorch cnn for multiclass classification
import os
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import CNN
from data import prepare_data
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    scaler = GradScaler(enabled=(device.type == "cuda"))

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
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
    # prepare the data
    data_dir = os.path.expanduser('~/.torch/datasets/mnist')
    train_dl, test_dl = prepare_data(data_dir)
    print(len(train_dl.dataset), len(test_dl.dataset))

    # define the network
    model = CNN(1).to(device)

    # train the model
    train_model(train_dl, model, epochs=10)

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
