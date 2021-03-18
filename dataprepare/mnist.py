'''
Load Mnist Data as dataloader
'''

import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, ConcatDataset, DataLoader

transform = transforms.Compose(
        [transforms.ToTensor()]
    )

abspath = os.path.abspath('..')
print(abspath)

def load_mnist_ds():
    ds_train = MNIST(root=os.path.join(abspath,'data/mnist'), train=True, transform=transform, download=True)
    ds_test = MNIST(root=os.path.join(abspath,'data/mnist'), train=False, transform=transform, download=True)
    ds_all = ConcatDataset((ds_train, ds_test))
    x_train, y_train = ds_train.data, ds_train.targets
    x_test, y_test = ds_test.data, ds_test.targets
    x_all = torch.cat((x_train, x_test), 0)
    y_all = torch.cat((y_train, y_test), 0)
    x_all = np.divide(x_all, 255.)
    return x_train, y_train, x_test, y_test, x_all, y_all, ds_all

def reshape_mnist_mlp(batch: torch.Tensor) -> torch.Tensor:
    new = batch.view(batch.shape[0], -1)
    return new

def reshape_mnist_cnn(batch: torch.Tensor) -> torch.Tensor:
    new = batch.view(-1, 1, 28, 28)
    return new

def reshape_mnist_rnn(batch: torch.Tensor) -> torch.Tensor:
    pass

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_all, y_all, ds_all = load_mnist_ds()