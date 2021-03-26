import os
abspath = os.path.abspath('..')
#print(abspath)
import sys
sys.path.append(abspath)
import time
import shutil
import numpy as np
from dataprepare import uea
import dtwmatrix
from sklearn import preprocessing
from datapreprocess import dtwmatrix

def train_convert():
    rawdatadir = os.path.join(abspath, 'data/uea/raw')
    rawdatalist = os.listdir(rawdatadir)
    rawdatalist.sort()
    for dataname in rawdatalist:
        #print(dataname)
        data = uea.load_uea(dataname[0:-4])
        C = len(list(set(data['train_y'])))
        savepath = os.path.join(abspath, 'data/uea/dtwmatrix', dataname[0:-4])
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        os.makedirs(os.path.join(savepath, 'train'))

        train_x = data['train_x']
        train_y = data['train_y']
        N_train = train_x.shape[0]
        datalength = len(train_x.iloc[0, 0])
        dim = train_x.shape[1]
        for i in range(N_train):
            d = np.zeros((datalength, dim),dtype=np.float32)
            for k in range(dim):
                d[:,k] = train_x.iloc[i, k]
            mat = dtwmatrix.multivariate_dtwmatrix_fast(d,
                                                        mdtwsize=28,
                                                        dtwtpye='fixed')
            traindata = (mat, train_y[i])
            id = str(i)
            label = str(train_y[i])
            savename = 'train_' + id + '_' + label + '.npy'
            np.save(os.path.join(savepath, 'train', savename), traindata)
            print('-[info]:', dataname[0:-4], 'train:', id, label)

        os.makedirs(os.path.join(savepath, 'test'))
        test_x = data['test_x']
        test_y = data['test_y']
        N_test = test_x.shape[0]
        datalength = len(test_x.iloc[0, 0])
        dim = test_x.shape[1]
        for i in range(N_test):
            d = np.zeros((datalength, dim), dtype=np.float32)
            for k in range(dim):
                d[:, k] = test_x.iloc[i, k]
            mat = dtwmatrix.multivariate_dtwmatrix_fast(d,
                                                        mdtwsize=28,
                                                        dtwtpye='fixed')
            testdata = (mat, test_y[i])
            id = str(i)
            label = str(test_y[i])
            savename = 'test_' + id + '_' + label + '.npy'
            np.save(os.path.join(savepath, 'test', savename), testdata)
            print('-[info]:', dataname[0:-4], 'test:', id, label)


if __name__ == '__main__':
    train_convert()