import time
import numpy as np
from dtaidistance import dtw
from dataprepare import isdb
import math

def generate_segments_random_idx(length, size, seed=0):
    np.random.seed(seed)
    interval = math.floor(length / (size))
    #print('-loginfo -> interval:', interval)
    remainder = length % size
    seq_idx = np.array(range(0, length - remainder, interval))
    sublength = np.random.randint(interval, interval * 5, len(seq_idx))
    seq_end = seq_idx + sublength
    for i in range(len(seq_end)):
        if seq_end[i] > length:
            seq_end[i] = length
        else:
            seq_end[i] = seq_end[i]
    return seq_idx, seq_end

def generate_segments_fixed_idx(length, size):
    interval = math.floor(length/(size))
    #print('-loginfo -> interval:', interval)
    remainder = length % size
    seq_idx = np.array(range(0, length-remainder, interval))
    seq_end = seq_idx + interval
    for i in range(len(seq_end)):
        if seq_end[i] > length:
            seq_end[i] = length
        else:
            seq_end[i] = seq_end[i]
    return seq_idx, seq_end


def univariate_dtwmatrix_fast(s, dtwsize, dtwtpye='fixed', dtwseed=0):
    '''
    Generate the dtwmatrix of a given univariate series
    :param s: given series
    :param size: expected matrix size
    :param seed: random size
    :return: a dtw matrix of size*size
    '''
    l = len(s)
    M_dtw = np.zeros((dtwsize, dtwsize))
    if dtwtpye == 'fixed':
        sidx, send = generate_segments_fixed_idx(l, size=dtwsize)
    elif dtwtpye == 'random':
        sidx, send = generate_segments_random_idx(l, size=dtwsize, seed=dtwseed)

    for i in range(dtwsize):
        s1 = np.array(s[sidx[i]:send[i]], dtype=np.float)
        for j in range(dtwsize):
            if j > i:
                s2 = np.array(s[sidx[j]:send[j]], dtype=np.float)
                d = dtw.distance_fast(s1, s2) #use_pruning=True
                d = round(d, 4)
                M_dtw[i, j] = d
    M = M_dtw.T + M_dtw
    nanflag = np.any(np.isnan(M))
    if nanflag:
        print('There are wrong dtw in matrix')
    #print('Check Nan', np.any(np.isnan(M)))
    return np.array(M)

def multivariate_dtwmatrix_fast(ss, mdtwsize, dtwtpye='fixed',mdtwseed=0):
    '''
    :param ss: a given multivariate series
    :param mdtwsize:
    :param mdtwseed:
    :return: Dim * size *size dtw_matrix
    '''
    N, Dim = ss.shape
    Ms_dtw = np.zeros((Dim, mdtwsize, mdtwsize))
    for i in range(Dim):
        s = ss[:, i]
        m = univariate_dtwmatrix_fast(s, dtwsize=mdtwsize, dtwtpye=dtwtpye, dtwseed=mdtwseed)
        Ms_dtw[i, :, :] = m
    return Ms_dtw

if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: dtwmatrix_conversion.py')
    print('---------------------------------------')

    sticdict, normdict, distdict = isdb.load_isdb()
    # chemicals_loop5
    ss = sticdict['chemicals_loop5'][:, 0:2]
    dtwmatrix = multivariate_dtwmatrix_fast(ss, mdtwsize=28,
                                            dtwtpye='fixed', mdtwseed=0)
    print(dtwmatrix)

    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')

