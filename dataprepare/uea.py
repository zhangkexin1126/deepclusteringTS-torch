import os
import gc
import time
import pandas as pd
import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe

abspath = os.path.abspath('..')
print(abspath)

def uea_prepare():
    ueapath = '/home/kexin/data/uea/Multivariate2018_ts'
    uealist = os.listdir(ueapath)
    uealist.sort()
    savepath = os.path.join(abspath, 'data/uea/raw')
    for uea in uealist:
        data = {}
        savename = uea + '.npy'
        print(os.path.join(savepath, savename))
        if (uea != 'InsectWingbeat') and (not os.path.exists(os.path.join(savepath,savename))):
            trainfile = uea + '_TRAIN.ts'
            train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(ueapath, uea, trainfile))
            data['train_y'] = train_y
            data['train_x'] = train_x.applymap(lambda x: x.astype(np.float32))
            testfile = uea + '_TEST.ts'
            test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(ueapath, uea, testfile))
            data['test_y'] = test_y
            data['test_x'] = test_x.applymap(lambda x: x.astype(np.float32))
            np.save(os.path.join(savepath, savename), data)

def load_uea(dataset_name):
    '''
    :param dataset_name: ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    'Cricket', 'DuckDuckGeese', 'ERing', 'EigenWorms', 'Epilepsy', 'EthanolConcentration',
    'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    'InsectWingbeat', 'JapaneseVowels', 'LSST', 'Libras', 'MotorImagery', 'NATOPS', 'PEMS-SF',
    'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']
    :return: train_x, train_y, test_x, test_y
    '''
    filename = dataset_name + '.npy'
    datapath = os.path.join(abspath, 'data/uea/raw', filename)
    #print(datapath)
    data = np.load(datapath, allow_pickle=True).item()
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    print('---------------------------')
    print('Dataset Name:', dataset_name)
    print('Dataset Dim:', train_x.shape[1])
    print('Dataset Length:', len(train_x.iloc[0, 0]))
    print('Number of Train Samples:', train_x.shape[0])
    print('Number of Test Samples:', test_x.shape[0])
    print('Number of classs:', len(list(set(data['train_y']))))
    return data

if __name__ == '__main__':
    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: uea.py')
    print('---------------------------------------')

    uea_prepare()

    dataname = 'FaceDetection'
    data = load_uea(dataname)


    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')