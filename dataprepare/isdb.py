import os
import re
import time
import numpy as np
import pandas as pd
from scipy import signal

abspath = os.path.abspath('..')
print(abspath)

def isdb_prepare():
    # ------------读取粘滞回路数据和回路名: stictiondata/stictionloop---------------
    filepath = r'/home/kexin/data/isdb/stiction_loops'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    stictiondata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        stictiondata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                      engine='python',
                                                                      names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    stictionloop = list(map(lambda x: x[0:-3], list(stictiondata.keys())))
    stictionloop = list(set(stictionloop))
    print('Number of Stiction Loops: ', len(stictionloop))

    # 读取正常回路数据和回路名:normaldata/nomalloop
    filepath = r'/home/kexin/data/isdb/normal_loops'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    normaldata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        normaldata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                    engine='python',
                                                                    names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    normalloop = list(map(lambda x: x[0:-3], list(normaldata.keys())))
    normalloop = list(set(normalloop))
    print('Number of Normal Loops: ', len(normalloop))

    # 读取正常回路数据和回路名:normaldata/nomalloop
    filepath = r'/home/kexin/data/isdb/disturbance_loops'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    disturbancedata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        disturbancedata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                     engine='python',
                                                                     names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    disturbanceloop = list(map(lambda x: x[0:-3], list(disturbancedata.keys())))
    disturbanceloop = list(set(disturbanceloop))
    print('Number of Disturbance Loops: ', len(disturbanceloop))

    # ----读取回路信息，包括回路样本数量和采样周期: loop1info-------
    filepath = r'/home/kexin/data/isdb/loopinfo'
    filelist = os.listdir(filepath)
    loopinfo = pd.DataFrame(columns=['loopname', 'ts', 'samplenum'])
    pattern1 = re.compile('Ts: \d.+|Ts: \d+')
    pattern2 = re.compile(r'\d.+|\d+')
    pattern3 = re.compile(r'PV: \[\d+')
    pattern4 = re.compile(r'\d+')
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        with open(data_path, 'r') as f:
            tstemp = f.readlines()
            numtemp = tstemp.copy()
            # 查询ts
            tstemp = list(map(lambda x: pattern1.search(x), tstemp))
            tstemp = list(filter(None, tstemp))[0]
            ts = pattern2.search(tstemp.group(0)).group(0)
            # 查询样本数量
            numtemp = list(map(lambda x: pattern3.search(x), numtemp))
            numtemp = list(filter(None, numtemp))[0]
            num = pattern4.search(numtemp.group(0)).group(0)
        loopinfo.loc[i, 'loopname'] = os.path.basename(data_path)[0:-4]
        loopinfo.loc[i, 'ts'] = ts
        loopinfo.loc[i, 'samplenum'] = num
    loopinfo.to_csv(os.path.join(abspath, 'data/isdb/loopinfo.csv'))

    # 将粘滞回路数据按照dict格式存放：stictiondata
    datalist = list(stictiondata.keys())
    # stictionloop
    stictiondatadict = {}
    for loop in stictionloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = stictiondata[loop + '.OP'].values.flatten()
        df.pv = stictiondata[loop + '.PV'].values.flatten()
        df.sp = stictiondata[loop + '.SP'].values.flatten()
        stictiondatadict[loop] = df.values
    print('Number of Siction Loops:', len(list(stictiondatadict.keys())))
    stictioninfo = loopinfo.loc[loopinfo['loopname'].isin(stictionloop)]

    # 将正常回路数据按照dict格式存放： normaldatadict
    datalist = list(normaldata.keys())
    # stictionloop
    normaldatadict = {}
    for loop in normalloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = normaldata[loop + '.OP'].values.flatten()
        df.pv = normaldata[loop + '.PV'].values.flatten()
        df.sp = normaldata[loop + '.SP'].values.flatten()
        normaldatadict[loop] = df.values
    print('Number of Normal Loops:', len(list(normaldatadict.keys())))
    normalinfo = loopinfo.loc[loopinfo['loopname'].isin(normalloop)]

    # 将扰动回路数据按照dict格式存放： disturbancedatadict
    datalist = list(disturbancedata.keys())
    # stictionloop
    disturbancedatadict = {}
    for loop in disturbanceloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = disturbancedata[loop + '.OP'].values.flatten()
        df.pv = disturbancedata[loop + '.PV'].values.flatten()
        df.sp = disturbancedata[loop + '.SP'].values.flatten()
        disturbancedatadict[loop] = df.values
    print('Number of Distrubance Loops:', len(list(disturbancedatadict.keys())))
    disturbanceinfo = loopinfo.loc[loopinfo['loopname'].isin(disturbanceloop)]

    # Save as np
    np.save(os.path.join(abspath, 'data/isdb/sticdict.npy'), stictiondatadict)
    np.save(os.path.join(abspath, 'data/isdb/normdict.npy'), normaldatadict)
    np.save(os.path.join(abspath, 'data/isdb/distdict.npy'), disturbancedatadict)


def isdb_prepare_filter(maxlength=600):
    path = os.path.join(abspath, 'data/isdb/sticdict.npy')
    sticdict = np.load(path, allow_pickle=True).item()
    path = os.path.join(abspath, 'data/isdb/normdict.npy')
    normdict = np.load(path, allow_pickle=True).item()
    path = os.path.join(abspath, 'data/isdb/distdict.npy')
    distdict = np.load(path, allow_pickle=True).item()

    sticf = {}
    for k, v in sticdict.items():
        if len(v) >= maxlength:
            data = v[0:maxlength, 0:2]
        else:
            data = v
        b, a = signal.butter(3, 0.15, 'lowpass')
        dataf = signal.filtfilt(b, a, data, axis=0)
        sticf[k] = dataf

    normf = {}
    for k, v in normdict.items():
        if len(v) >= maxlength:
            data = v[0:maxlength, 0:2]
        else:
            data = v
        b, a = signal.butter(3, 0.15, 'lowpass')
        dataf = signal.filtfilt(b, a, data, axis=0)
        normf[k] = dataf

    distf = {}
    for k, v in distdict.items():
        if len(v) >= maxlength:
            data = v[0:maxlength, 0:2]
        else:
            data = v
        b, a = signal.butter(3, 0.15, 'lowpass')
        dataf = signal.filtfilt(b, a, data, axis=0)
        distf[k] = dataf

    np.save(os.path.join(abspath, 'data/isdb/sticf.npy'), sticf)
    np.save(os.path.join(abspath, 'data/isdb/normf.npy'), normf)
    np.save(os.path.join(abspath, 'data/isdb/distf.npy'), distf)

def load_matlab(maxlength=600):
    sticpath = os.path.join(abspath,'data/isdb/stic.csv')
    df = pd.read_csv(sticpath)
    idx = np.array(range(0, len(df), 1))
    sticdata = df.iloc[idx, 0:2000]

    weakpath = os.path.join(abspath, 'data/isdb/weakstic.csv')
    df = pd.read_csv(weakpath)
    idx = np.array(range(0, len(df), 1))
    weakdata = df.iloc[idx, 0:2000]

    sticdata = pd.concat((sticdata, weakdata), axis=0)
    op = sticdata.iloc[:, 0:maxlength]
    pv = sticdata.iloc[:, 1000:1000+maxlength]
    sticdata = pd.concat((op, pv), axis=1)

    normpath = os.path.join(abspath, 'data/isdb/norm.csv')
    df = pd.read_csv(normpath)
    idx = np.array(range(0, len(df), 2))
    normdata = df.iloc[idx, 0:2000]
    op = normdata.iloc[:, 0:maxlength]
    pv = normdata.iloc[:, 1000:1000 + maxlength]
    normdata = pd.concat((op, pv), axis=1)
    return sticdata, normdata

def load_isdb():
    sticpath = os.path.join(abspath, 'data/isdb/sticf.npy')
    sticdict = np.load(sticpath, allow_pickle=True).item()
    distpath = os.path.join(abspath, 'data/isdb/distf.npy')
    distdict = np.load(distpath, allow_pickle=True).item()
    normpath = os.path.join(abspath, 'data/isdb/normf.npy')
    normdict = np.load(normpath, allow_pickle=True).item()
    return sticdict, normdict, distdict

if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: isdb_dtwmatrix.py')
    print('---------------------------------------')

    #### Main_func
    isdb_prepare()
    isdb_prepare_filter(maxlength=600)
    sticdict, normdict, distdict = load_isdb()
    sticdata, normdata = load_matlab(maxlength=600)
    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')