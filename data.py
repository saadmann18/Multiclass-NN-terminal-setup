from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import torch
import os


# prepare the dataset
def prepare_data(path):
    """
    MNIST data preparation
    """
    # define standardization
    trans = Compose([ToTensor(), Normalize((0.5,), (1.0,))])
    # load dataset
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False, download=True, transform=trans)
    # prepare data loaders
    use_cuda = torch.cuda.is_available()
    # Heuristics for performant data loading
    num_workers = max(0, (os.cpu_count() or 0) - 1)
    pin = True if use_cuda else False

    train_kwargs = dict(
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    test_kwargs = dict(
        batch_size=1024,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    if num_workers > 0:
        train_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
        test_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_dl = DataLoader(train, **train_kwargs)
    test_dl = DataLoader(test, **test_kwargs)
    return train_dl, test_dl
