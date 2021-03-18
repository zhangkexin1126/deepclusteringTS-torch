'''
Convolutional Autoencoder
'''
import os
import sys
sys.path.append(os.path.abspath('..'))

import utils
from dataprepare import mnist

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tqdm import tqdm
from typing import Callable, Iterable, Optional, Tuple, List, Any

class CNNEncoder(nn.Module):
    def __init__(self, in_channel, embedded_dim=5):
        super(CNNEncoder, self).__init__()
        self.embedded_dim = embedded_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True))
        self.dense = nn.Linear(1152, self.embedded_dim)
        self.encoder = nn.Sequential(self.conv1, self.conv2, self.conv3)

    def forward(self, x):
        z = self.encoder(x)
        h = self.dense(z.view(z.shape[0], -1))
        return h

class CNNDecoder(nn.Module):
    def __init__(self, out_channel):
        super(CNNDecoder, self).__init__()
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channel, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        self.dense = nn.Linear(5, 1152)
        self.decoder = nn.Sequential(self.deconv3, self.deconv2, self.deconv1)

    def forward(self, x):
        z = self.dense(x)
        z = torch.reshape(z, (z.shape[0], -1, 3, 3))
        out = self.decoder(z)
        return out

class CAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model = nn.Sequential(encoder, decoder)
    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        return self.model(x)


def CAE_train(model, train_x, train_y, validation: Optional[Dataset],
              train_epochs: int,
              batch_size: int,
              device: str,
              optimizer: optim.Optimizer,
              loss_func: nn = nn.MSELoss(),
              scheduler: Any = None,
              cuda: bool = True,
              sampler: Optional[torch.utils.data.sampler.Sampler] = None,
              silent: bool = False):

    train_dl = utils.generate_dataloader(train_x,
                                         batch_size=batch_size,
                                         pin_memory=False,
                                         sampler = None,
                                         shuffle=True if sampler is None else False)
    if validation is not None:
        validation_dl = utils.generate_dataloader(validation,
                                             batch_size=batch_size,
                                             pin_memory=False,
                                             sampler=None,
                                             shuffle=True if sampler is None else False)
    else:
        validation_dl = None
        validation_loss_value = -1
    model.to(device) # GPU or CPU
    for epo in range(train_epochs):
        data_iterator = tqdm(
            train_dl,
            leave=True,
            unit="batch",
            postfix={"epo": epo, "training_loss": "%.6f" % 0.0, "validation_loss": "%.6f" % -1},
            disable=silent)

        for idx, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]):
                batch = batch[0]
            batch = batch.to(device)
            batch = mnist.reshape_mnist_cnn(batch).float()
            output = model(batch)
            #print(output.dtype, batch.dtype)
            loss = loss_func(output, batch)
            # accuracy
            train_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_iterator.set_postfix(
                epo=epo, training_loss="%.4f" % train_loss, validation_loss="%.4f" % validation_loss_value,
            )

        if scheduler is not None:
            scheduler.step()


def CAE_predict(model, x, y,
                batch_size: int,
                device: str,
                encode: bool=True,
                silent: bool = False):
    test_dl = utils.generate_dataloader(x,
                                         batch_size=batch_size,
                                         pin_memory=False,
                                         sampler=None,
                                         shuffle=False)
    model.to(device)
    model.eval()
    data_iterator = tqdm(test_dl, leave=False, unit="batch", disable=silent)
    embedded_layers = []
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if (isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]):
            batch = batch[0]
        batch = batch.to(device)
        batch = mnist.reshape_mnist_cnn(batch).float()
        if encode:
            output = model.encode(batch)
        else:
            output = model(batch)
        embedded_layers.append(output.detach().cpu())
    feat = torch.cat(embedded_layers, dim=0)
    model.train()
    return feat



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CAE')
    args = parser.parse_args()
    print(args)

    '''x = torch.rand(10,2,28,28)
    cae = CAE(CNNEncoder(in_channel=2), CNNDecoder(out_channel=2))
    print('Input shape -->', x.shape)
    h = cae.encode(x)
    print('Embedded shape -->', h.shape)
    rec = cae(x)
    print('Rec shape -->', rec.shape)'''
    ## Train
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-loginfo -> ', device)
    train_x, train_y, test_x, test_y, x_all, y_all, ds_all = mnist.load_mnist_ds()
    model = CAE(CNNEncoder(in_channel=1), CNNDecoder(out_channel=1))
    model_optim = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    CAE_train(model=model,
              train_x=train_x,
              train_y=train_y,
              validation=None,
              train_epochs=5,
              batch_size=256,
              device=device,
              optimizer=model_optim,
              loss_func=loss_func,
              scheduler=lr_scheduler.StepLR(model_optim, 50, gamma=0.1))

    pred = CAE_predict(model=model,
                       x=test_x,
                       y=test_y,
                       batch_size=256,
                       device=device,
                       encode=True)
    print(pred.shape)


