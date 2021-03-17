import os
import time
import pandas as pd
import numpy as np
import shutil

'''
The dataprepare is downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
Each dataframe contains 55 columns:

Column 1 ('faultNumber') ranges from 1 to 20 in the “Faulty” datasets and represents the fault type in the TEP. 
The “FaultFree” datasets only contain fault 0 (i.e. normal operating conditions).

Column 2 ('simulationRun') ranges from 1 to 500 and represents a different random number generator state from which 
a full TEP dataset was generated (Note: the actual seeds used to generate training and testing datasets were non-overlapping).

Column 3 ('sample') ranges either from 1 to 500 (“Training” datasets) or 1 to 960 (“Testing” datasets). 
The TEP variables (columns 4 to 55) were sampled every 3 minutes for a total duration of 25 hours and 48 hours respectively. 
Note that the faults were introduced 1 (20th point) and 8 (160the point) hours into the Faulty Training and Faulty Testing datasets, respectively.

Columns 4 to 55 contain the process variables; the column names retain the original variable names.

removenum = ['3', '9', '15'] becasue they are not recognitive
'''

abspath = os.path.abspath('..')
print(abspath)

def load_te_chunker():
    '''
    Because of the dataprepare size is very large, use chunksize
    :return:
    '''
    # fault free training
    path = r'/home/kexin/data/teprocess/TEPS/faultfreetrain.csv'
    faultfreetrain_ck = pd.read_csv(path, chunksize=500) # total size 500*500

    # fault free testing
    path = r'/home/kexin/data/teprocess/TEPS/faultfreetest.csv'
    faultfreetest_ck = pd.read_csv(path, chunksize=960)  # total size 500*960

    # fault training
    path = r'/home/kexin/data/teprocess/TEPS/faulttrain.csv'
    faulttrain_ck = pd.read_csv(path, chunksize=10000)  # total size 20*500*500

    # fault testing
    path = r'/home/kexin/data/teprocess/TEPS/faulttest.csv'
    faulttest_ck = pd.read_csv(path, chunksize=9600)  # total size 20*500*960

    return faultfreetrain_ck, faultfreetest_ck, faulttrain_ck, faulttest_ck


def te_prepare():
    # XVMS
    path = r'/home/kexin/data/teprocess/TEPS/faultfreetrain.csv'
    ck = pd.read_csv(path, chunksize=500)  # total size 500*500
    savepath = os.path.join(abspath,'data/te/faultfreetrain')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    #shutil.rmtree(savepath)
    #os.makedirs(savepath)
        for i, piece in enumerate(ck):
            print('1 faultfreetrain:', i)
            faultnumber = piece['faultNumber'].values[0]
            fn = str(faultnumber)
            simulationRun = piece['simulationRun'].values[0]
            sr = str(simulationRun)
            itemname = 'f' + fn + 'r' + sr + '_train.csv'
            # print(itemname)
            piece.drop('sample', axis=1, inplace=True)
            piece.drop('faultNumber', axis=1, inplace=True)
            piece.drop('simulationRun', axis=1, inplace=True)
            piece.to_csv(os.path.join(savepath, itemname), index = False)
    else:
        print('faultfreetrain/ exists')


    path = r'/home/kexin/data/teprocess/TEPS/faulttrain.csv'
    ck = pd.read_csv(path, chunksize=500)  # total size 500*500
    savepath = os.path.join(abspath,'data/te/faulttrain')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        for i, piece in enumerate(ck):
            print('2 faulttrain:', i)
            faultnumber = piece['faultNumber'].values[0]
            fn = str(faultnumber)
            simulationRun = piece['simulationRun'].values[0]
            sr = str(simulationRun)
            itemname = 'f' + fn + 'r' + sr + '_train.csv'
            # print(itemname)
            piece.drop('sample', axis=1, inplace=True)
            piece.drop('faultNumber', axis=1, inplace=True)
            piece.drop('simulationRun', axis=1, inplace=True)
            piece.to_csv(os.path.join(savepath,itemname), index = False)
    else:
        print('faulttrain/ exists')

    path = r'/home/kexin/data/teprocess/TEPS/faultfreetest.csv'
    ck = pd.read_csv(path, chunksize=960)  # total size 500*500
    savepath = os.path.join(abspath,'data/te/faultfreetest')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        for i, piece in enumerate(ck):
            print('3 faultfreetest:', i)
            faultnumber = piece['faultNumber'].values[0]
            fn = str(faultnumber)
            simulationRun = piece['simulationRun'].values[0]
            sr = str(simulationRun)
            itemname = 'f' + fn + 'r' + sr + '_test.csv'
            # print(itemname)
            piece.drop('sample', axis=1, inplace=True)
            piece.drop('faultNumber', axis=1, inplace=True)
            piece.drop('simulationRun', axis=1, inplace=True)
            piece.to_csv(os.path.join(savepath, itemname), index=False)
    else:
        print('faultfreetest/ exists')

    path = r'/home/kexin/data/teprocess/TEPS/faulttest.csv'
    ck = pd.read_csv(path, chunksize=960)  # total size 500*500'
    savepath = os.path.join(abspath, 'data/te/faulttest')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        for i, piece in enumerate(ck):
            print('4 faulttest:', i)
            faultnumber = piece['faultNumber'].values[0]
            fn = str(faultnumber)
            simulationRun = piece['simulationRun'].values[0]
            sr = str(simulationRun)
            itemname = 'f' + fn + 'r' + sr + '_test.csv'
            # print(itemname)
            piece.drop('sample', axis=1, inplace=True)
            piece.drop('faultNumber', axis=1, inplace=True)
            piece.drop('simulationRun', axis=1, inplace=True)
            piece.to_csv(os.path.join(savepath, itemname), index=False)
    else:
        print('faulttest/ exists')

if __name__ == '__main__':
    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: te.py')
    print('---------------------------------------')

    te_prepare()

    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')