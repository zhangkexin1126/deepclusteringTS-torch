import pandas as pd
import numpy as np
import os

abspath = os.path.abspath('..')
print(abspath)

def ucr_prepare():
    '''
    :return: Dict(ucrdata, [train_x,train_y,test_x,test_y]), saved as '/Data/zkx/ucrdata/ucrdict.npy'
    '''
    ucrpath = '/home/kexin/data/ucr/UCRArchive_2018'
    ucrlist = os.listdir(ucrpath)
    ucrlist.sort()
    savepath = os.path.join(abspath, 'data/ucr/raw')
    for ucrdata in ucrlist:
        data = {}
        savename = ucrdata + '.npy'
        print(os.path.join(savepath,savename))
        trainfile = ucrdata + '_TRAIN.tsv'
        traindata = pd.read_csv(os.path.join(ucrpath, ucrdata, trainfile), sep='\t', header=None)
        data['train_y'] = traindata.iloc[:,0].values
        data['train_x'] = traindata.drop([0],axis=1).values
        testfile = ucrdata + '_TEST.tsv'
        testdata = pd.read_csv(os.path.join(ucrpath, ucrdata, testfile), sep='\t', header=None)
        data['test_y'] = testdata.iloc[:,0].values
        data['test_x'] = testdata.drop([0],axis=1).values
        np.save(os.path.join(savepath, savename), data)

def load_ucr(dataset_name: str):
    '''
    :param dataset_name: ['ACSF1', 'Adiac', 'AllGestureWiimoteX',
    'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 'BME', 'Beef',
    'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration',
    'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ',
    'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay',
    'DodgerLoopGame', 'DodgerLoopWeekend', 'ECG200', 'ECG5000', 'ECGFiveDays',
    'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'ElectricDevices',
    'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA',
    'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1',
    'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2',
    'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring',
    'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain',
    'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
     'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
     'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
      'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain',
      'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf',
      'OliveOil', 'PLAID', 'PhalangesOutlinesCorrect', 'Phoneme',
      'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
      'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
      'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
      'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
      'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
      'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
      'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
      'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
      'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
      'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer',
       'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']
    :return: train_x, train_y, test_x, test_y
    '''
    filename = dataset_name + '.npy'
    datapath = os.path.join(abspath, 'data/ucr/raw', filename)
    print(datapath)
    data = np.load(datapath, allow_pickle=True).item()
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    print('---------------------------')
    print('Dataset Name:', dataset_name)
    print('Length:', train_x.shape[1])
    print('Number of Train Samples:', train_x.shape[0])
    print('Number of Test Samples:', test_x.shape[0])
    print('Number of classs:', len(list(set(data['train_y']))))
    return data

if __name__ == '__main__':
    # ucr_prepare()
    # ACSF1
    dataname = 'ACSF1'
    load_ucr(dataname)
    dataname = 'MixedShapesRegularTrain'
    load_ucr(dataname)


