import os
abspath = os.path.abspath('..')
#print(abspath)
import sys
sys.path.append(abspath)
import time
import shutil
import numpy as np
import pandas as pd
from dataprepare import isdb
import dtwmatrix
from sklearn import preprocessing
from datapreprocess import dtwmatrix

def data_filter(rawdata, scaler = 'minmax', start_point = 0):
    if scaler == 'minmax':
        datascaler = preprocessing.MinMaxScaler()
    elif scaler == 'standard':
        datascaler = preprocessing.StandardScaler()
    elif scaler == 'norm':
        datascaler = preprocessing.Normalizer()
    datanew = datascaler.fit_transform(rawdata)
    return datanew

def faultfreetrain_convert(dtwsize=48, dtwtype='fixed'):
    datapath = os.path.join(abspath, 'data/te/faultfreetrain')
    savepath = os.path.join(abspath, 'data/te/dtwmatrix/faultfreetrain')
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
        os.makedirs(savepath)
    else:
        os.makedirs(savepath)
    filelist = os.listdir(datapath)
    filelist.sort()
    print('---------------------------------')
    for i, filename in enumerate(filelist):
        print('-[Convert Faultfreetrain]:', filename)
        file = os.path.join(datapath, filename)
        data = np.array(pd.read_csv(file))
        filtered = data_filter(data)
        M_dtw = dtwmatrix.multivariate_dtwmatrix_fast(filtered, mdtwsize=dtwsize,
                                                      dtwtpye=dtwtype, mdtwseed=0)
        newfilename = filename[0:-4]
        np.save(os.path.join(savepath, newfilename), M_dtw)

def faultfreetest_convert(dtwsize=48, dtwtype='fixed'):
    datapath = os.path.join(abspath, 'data/te/faultfreetest')
    savepath = os.path.join(abspath, 'data/te/dtwmatrix/faultfreetest')
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
        os.makedirs(savepath)
    else:
        os.makedirs(savepath)
    filelist = os.listdir(datapath)
    filelist.sort()
    print('---------------------------------')
    for i, filename in enumerate(filelist):
        print('-[Convert Faultfreetest]:', filename)
        file = os.path.join(datapath, filename)
        data = np.array(pd.read_csv(file))
        filtered = data_filter(data)
        M_dtw = dtwmatrix.multivariate_dtwmatrix_fast(filtered, mdtwsize=dtwsize,
                                                      dtwtpye=dtwtype, mdtwseed=0)
        newfilename = filename[0:-4]
        np.save(os.path.join(savepath, newfilename), M_dtw)


def faulttrain_convert(dtwsize=48, dtwtype='fixed'):
    datapath = os.path.join(abspath, 'data/te/faulttrain')
    savepath = os.path.join(abspath, 'data/te/dtwmatrix/faulttrain')
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
        os.makedirs(savepath)
    else:
        os.makedirs(savepath)
    filelist = os.listdir(datapath)
    filelist.sort()
    print('---------------------------------')
    for i, filename in enumerate(filelist):
        print('-[Convert Faulttrain]:', filename)
        file = os.path.join(datapath, filename)
        data = np.array(pd.read_csv(file))
        filtered = data_filter(data)
        M_dtw = dtwmatrix.multivariate_dtwmatrix_fast(filtered, mdtwsize=dtwsize,
                                                      dtwtpye=dtwtype, mdtwseed=0)
        newfilename = filename[0:-4]
        np.save(os.path.join(savepath, newfilename), M_dtw)

def faulttest_convert(dtwsize=48, dtwtype='fixed'):
    datapath = os.path.join(abspath, 'data/te/faulttest')
    savepath = os.path.join(abspath, 'data/te/dtwmatrix/faulttest')
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
        os.makedirs(savepath)
    else:
        os.makedirs(savepath)
    filelist = os.listdir(datapath)
    filelist.sort()
    print('---------------------------------')
    for i, filename in enumerate(filelist):
        print('-[Convert Faulttest]:', filename)
        file = os.path.join(datapath, filename)
        data = np.array(pd.read_csv(file))
        filtered = data_filter(data)
        M_dtw = dtwmatrix.multivariate_dtwmatrix_fast(filtered, mdtwsize=dtwsize,
                                                      dtwtpye=dtwtype, mdtwseed=0)
        newfilename = filename[0:-4]
        np.save(os.path.join(savepath, newfilename), M_dtw)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert TE')
    parser.add_argument('-freetrain', '--tefreetrain_convert', action='store_true')
    parser.add_argument('-freetest', '--tefreetest_convert', action='store_true')
    parser.add_argument('-faulttrain', '--tefaulttrain_convert', action='store_true')
    parser.add_argument('-faulttest', '--tefaulttest_convert', action='store_true')
    args = parser.parse_args()

    if args.tefreetrain_convert:
        faultfreetrain_convert(dtwsize=48, dtwtype='fixed')
    if args.tefreetest_convert:
        faultfreetest_convert(dtwsize=48, dtwtype='fixed')
    if args.tefaulttrain_convert:
        faulttrain_convert(dtwsize=48, dtwtype='fixed')
    if args.tefaulttest_convert:
        faulttest_convert(dtwsize=48, dtwtype='fixed')
