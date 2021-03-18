'''
PyTorch implementation of the paper:
    Unsupervised Deep Embedding for Clustering Analysis.
Ref:
    https://github.com/vlukiyanov/pt-dec
    https://github.com/CharlesNord/DEC-pytorch
'''
import numpy as np

import torch
import torch.nn as nn
from typing import Optional, Callable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from sklearn.cluster import KMeans
import utils
from utils import cluster_accuracy

class cluster_assignment(nn.Module):
    def __init__(
            self,
            cluster_num: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        super(cluster_assignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_num
        self.alpha = alpha
        if cluster_centers is None:
            init_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(init_cluster_centers)
        else:
            init_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(init_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        assign = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return assign

class dec(nn.Module):
    def __init__(self,
                 cluster_num: int,
                 hidden_dim: int,
                 encoder: nn.Module,
                 alpha: float = 1.0):
        super(dec, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dim
        self.cluster_number = cluster_num
        self.alpha = alpha
        self.assignment = cluster_assignment(
            cluster_num, self.hidden_dimension, alpha
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.assignment(self.encoder(batch))

def dectrain(
        data: Dataset,
        label,
        model: nn.Module,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        stopping_delta: Optional[float] = None,
        collate_fn=default_collate,
        cuda: bool = True,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        silent: bool = False,
        update_freq: int = 10,
        evaluate_batch_size: int = 1024,
        update_callback: Optional[Callable[[float, float], None]] = None,
        epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
):
    static_dataloader = utils.generate_dataloader(data,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    pin_memory=False,
                                    sampler=sampler,
                                    shuffle=False)
    train_dataloader = utils.generate_dataloader(data,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    pin_memory=False,
                                    sampler=sampler,
                                    shuffle=True)
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
        },
        disable=silent,
    )
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    actual = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
        if cuda:
            batch = batch.cuda(non_blocking=True)
        batch = batch.view(batch.shape[0], -1)
        features.append(model.encoder(batch.double()).detach().cpu())
    #print(actual.shape)
    actual = torch.cat(actual).long()
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    loss_function = nn.KLDivLoss()
    delta_label = None
    for epoch in range(epochs):
        features = []
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "acc": "%.4f" % (accuracy or 0.0),
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % (delta_label or 0.0),
            },
            disable=silent,
        )
        model.train()
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                batch
            ) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch = batch.cuda(non_blocking=True)
            batch = batch.view(batch.shape[0], -1)
            batch = batch.double()
            output = model(batch)
            target = target_distribution(output).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % float(loss.item()),
                dlb="%.4f" % (delta_label or 0.0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            features.append(model.encoder(batch).detach().cpu())
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % loss_value,
                    dlb="%.4f" % (delta_label or 0.0),
                )
                if update_callback is not None:
                    update_callback(accuracy, loss_value, delta_label)
        predicted, actual = decpredict(
            data,
            model,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            return_actual=True,
            cuda=cuda,
        )

        delta_label = (
                float((predicted != predicted_previous).float().sum().item())
                / predicted_previous.shape[0]
        )
        if stopping_delta is not None and delta_label < stopping_delta:
            print(
                'Early stopping as label delta "%1.5f" less than "%1.5f".'
                % (delta_label, stopping_delta)
            )
            break
        predicted_previous = predicted
        accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
        data_iterator.set_postfix(
            epo=epoch,
            acc="%.4f" % (accuracy or 0.0),
            lss="%.8f" % 0.0,
            dlb="%.4f" % (delta_label or 0.0),
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model)

def decpredict(
    data: Dataset,
    model: nn.Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    cuda: bool = True,
    silent: bool = False,
    return_actual: bool = False,
):
    dataloader = utils.generate_dataloader(
        dataset=data, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent)
    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError(
                "Dataset has no actual value to unpack, but return_actual is set."
            )
        if cuda:
            batch = batch.cuda(non_blocking=True)
        batch = batch.view(batch.shape[0], -1)
        features.append(
            model(batch.double()).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features).max(1)[1]


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()