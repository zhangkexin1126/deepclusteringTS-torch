'''
- PyTorch implementation of Denoising AutoEncoder:
- Stacked Denoising Autoencoders Learning Useful Representations in a
  Deep Network with a Local Denoising Criterion. JMLR2010
'''

import os
import sys
sys.path.append(os.path.abspath('..'))

import click
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

from sklearn.cluster import KMeans

from dataprepare import mnist
from models import sdae

import utils
from utils import cluster_accuracy


@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False)", type=bool, default=True)
@click.option(
    "--batch_size", help="training batch size (default 128).", type=int, default=256)
@click.option(
    "--pretrain_epochs", help="number of pretraining epochs (default 300).", type=int, default=50)
@click.option(
    "--train_epochs", help="number of training epochs (default 300).", type=int, default=75)

def experiment(cuda, batch_size, pretrain_epochs, train_epochs):
    print('Use CUDA:', cuda)
    print('Batch Size:', batch_size)
    print('Pretrain Epochs:', pretrain_epochs)

    x_train, y_train, x_test, y_test, x_all, y_all, ds_all = mnist.load_mnist_ds()

    daenet = sdae.denoising_mlp_ae([784, 500, 500, 2000, 10], final_activation=None)
    if cuda:
        daenet.cuda()
    print('-log info -> Start pretraining AE')
    sdae.pretrain_mlp_dae(
        x_all,
        daenet,
        pretrain_epochs,
        batch_size,
        optimizer=lambda model: optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 50, gamma=0.1),
        validation=None,
        corruption = 0.2,
        cuda=cuda,
        silent=False,
        update_freq = 5
    )
    print('-log info -> Finish pretraining AE')
    #savepath =
    print("-log info -> Start Training AE")
    ae_optimizer = optim.SGD(params=daenet.parameters(), lr=0.1, momentum=0.9)
    sdae.train_mlp_dae(
        x_all,
        daenet,
        train_epochs,
        batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        validation=None,
        corruption=0.2,
        cuda=cuda,
        silent=False,
        update_freq=1
    )
    print("-log info -> Finish Training AE")
    print("-log info -> Clustering Stage")
    kmeans = KMeans(n_clusters=10, n_init=20)
    daenet.eval()
    features = []
    dl = utils.generate_dataloader(x_all, batch_size=1024, shuffle=False)
    for batch in dl:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch = batch  # if we have a prediction label, separate it to actual
        if cuda:
            batch = batch.cuda(non_blocking=True)
        batch = batch.view(batch.shape[0], -1)
        features.append(daenet.encoder(batch).detach().cpu())
    actual = y_all.numpy()
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    accuracy = cluster_accuracy(actual, predicted)
    print("Final k-Means accuracy: %s" % accuracy)

if __name__ == "__main__":
    experiment()




