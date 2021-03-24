import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional

def generate_dataloader(dataset,
                batch_size,
                shuffle=True,
                sampler=None,
                collate_fn=None,
                pin_memory=False,
                drop_last=False):
    # Wrap Dataset to Dataloader
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      sampler=sampler,
                      collate_fn=collate_fn,
                      pin_memory=pin_memory,
                      drop_last=drop_last)

def classification_accuracy(real, pred):
    return accuracy_score(real, pred)

def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    #reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return accuracy

def t_student(batch, cluster_centers, alpha):
    '''
    use the Student’s t-distribution as a kernel to measure the similarity
    between embedded point zi and centroid j
    :param z:
    :param u:
    :return:
    '''
    norm_squared = torch.sum((batch.unsqueeze(1) - cluster_centers) ** 2, 2)
    numerator = 1.0 / (1.0 + (norm_squared / alpha))
    power = float(alpha + 1) / 2
    numerator = numerator ** power
    assign = numerator / torch.sum(numerator, dim=1, keepdim=True)
    return assign

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")
