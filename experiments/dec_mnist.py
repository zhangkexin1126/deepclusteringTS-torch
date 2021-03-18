import os
import sys
sys.path.append(os.path.abspath('..'))

import click
from torch import optim
from torch.optim.lr_scheduler import StepLR

from dataprepare import mnist
from models import sdae
from models import dec
from utils import cluster_accuracy

@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False)", type=bool, default=True)
@click.option(
    "--batch_size", help="training batch size (default 128).", type=int, default=256)
@click.option(
    "--ae_pretrain_epochs", help="number of pretraining epochs (default 300).", type=int, default=50)
@click.option(
    "--ae_train_epochs", help="number of training epochs (default 300).", type=int, default=50)
@click.option(
    "--cluster_num", help="number of clusters (default 2).", type=int, default=10)
@click.option(
    "--hidden_dim", help="dimension of hidden (default 2).", type=int, default=10)
@click.option(
    "--dec_train_epochs", help="dimension of hidden (default 2).", type=int, default=50)

def experiment(cuda,
               batch_size,
               ae_pretrain_epochs,
               ae_train_epochs,
               cluster_num,
               hidden_dim,
               dec_train_epochs):
    print('Use CUDA:', cuda)
    print('Batch Size:', batch_size)
    print('Pretrain Epochs:', ae_pretrain_epochs)

    x_train, y_train, x_test, y_test, x_all, y_all, ds_all = mnist.load_mnist_ds()
    #print(y_all)
    daenet = sdae.denoising_mlp_ae([784, 500, 500, 2000, 10], final_activation=None)
    if cuda:
        daenet.cuda()
    print('-log info: Start pretraining AE')
    sdae.pretrain_mlp_dae(
        x_all,
        daenet,
        ae_pretrain_epochs,
        batch_size,
        optimizer=lambda model: optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 50, gamma=0.1),
        validation=None,
        corruption=0.2,
        cuda=cuda,
        silent=False,
        update_freq=5
    )
    print('-log info: Finish pretraining AE')
    print("-log info: Start Training AE")
    ae_optimizer = optim.SGD(params=daenet.parameters(), lr=0.1, momentum=0.9)
    sdae.train_mlp_dae(
        x_all,
        daenet,
        ae_train_epochs,
        batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        validation=None,
        corruption=0.2,
        cuda=cuda,
        silent=False,
        update_freq=1
    )
    print("-log ->: Finish Training AE")
    print("-log ->: Deep Embedding Clustering Stage")
    decnet = dec.dec(cluster_num=cluster_num, hidden_dim=hidden_dim, encoder=daenet.encoder)
    if cuda:
        decnet.cuda()
    dec_optimizer = optim.SGD(decnet.parameters(), lr=0.01, momentum=0.9)
    dec.dectrain(
        data=ds_all,
        label=y_all,
        model=decnet,
        epochs=dec_train_epochs,
        batch_size=batch_size,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    print("-log ->: Finish Training DEC")
    predicted, actual = dec.decpredict(
        ds_all, decnet, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)

if __name__ == "__main__":
    experiment()