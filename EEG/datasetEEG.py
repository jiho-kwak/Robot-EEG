import os
import time
import math
import datetime
import random
import scipy.io as spio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pyriemann.estimation import Covariances


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DatasetAffectRobot(Dataset):
    def __init__(self, path, names, cov=False, train=True):
        self._path = path
        self._names = names
        self._spd = cov
        self._train = train
        self._state = []
        self._spd_width = 32 # EEG Channel numbers
        self._spd_height = self._spd_width  # SPD matrix

        if self._train:
            # get matrix size
            load_fname = self._path + self._names[0]  # read a dummy input matrix
            dummy_data = np.load(load_fname)
            dummy_eeg = dummy_data[:, 1:33].astype(float).transpose()
            dummy_cov = Covariances('oas').fit_transform(np.expand_dims(dummy_eeg, axis=0))

            self._spd_mat = np.zeros((2 * len(self._names), self._spd_width, self._spd_height))
            self._spd_labels = np.zeros(2 * len(self._names))
            labels = np.loadtxt(self._path + "labels_VA.csv", delimiter= ",", skiprows=1)


            # retrieve all data
            fdx = 0
            for fname in self._names:
                neutral = False
                data = np.load(self._path + fname)
                # IAPS - Positive vs Negative
                if int(fname[:4]) > 1010:
                    # labeling
                    valence = labels[labels[:, 0].astype('int').astype('str') == fname[:4]][0, 1]  # valence
                    label = 0
                    if float(valence) < 3.5:
                        label = 1   # Negative
                    else:
                        label = 2   # Positive
                    # 2 seconds segments, window size = 1/2, each of 500 Hz
                    eeg1 = data[:1000, 1:33].astype(float).transpose()
                    eeg2 = data[500:1500, 1:33].astype(float).transpose()
                    cov1 = np.squeeze(Covariances('oas').fit_transform(np.expand_dims(eeg1, axis=0)))
                    cov2 = np.squeeze(Covariances('oas').fit_transform(np.expand_dims(eeg2, axis=0)))
                    self._spd_mat[fdx] = cov1
                    self._spd_labels[fdx] = label
                    fdx += 1
                    self._spd_mat[fdx] = cov2
                    self._spd_labels[fdx] = label
                    fdx += 1
                # else: # Emotional Regression for personlization
                #
                # eeg = data[:, 1:33].astype(float).transpose()
                # cov = np.squeeze(Covariances('oas').fit_transform(np.expand_dims(eeg, axis=0)))
                # print("FDX")
                # print(fdx)
                # print(fname[:4])
                # self._spd_mat[fdx, :, :] = cov
                #
                # # labeling
                # valence = labels[labels[:, 0].astype('int').astype('str') == fname[:4]][0, 1]  # valence
                # control = labels[labels[:, 0].astype('int').astype('str') == fname[:4]][0, 2]  # control
                #
                # if round(float(valence)) == 1.0:
                #     self._spd_labels[fdx] = 1
                # elif round(float(valence)) == 2.0:
                #     self._spd_labels[fdx] = 2
                # elif round(float(valence)) == 3.0:
                #     self._spd_labels[fdx] = 3
                # elif round(float(valence)) == 4.0:
                #     self._spd_labels[fdx] = 4
                # elif round(float(valence)) == 5.0:
                #     self._spd_labels[fdx] = 5
                # elif round(float(valence)) == 6.0:
                #     self._spd_labels[fdx] = 6
                # fdx += 1
        else: # For testing (Robot)
            print(self._path + self._names[0])
            data = np.load(self._path + self._names[0])
            _spd_ = []
            #self._spd_mat = np.zeros((len(np.unique(data[:, 0].astype(str))), self._spd_width, self._spd_height))
            tdx = 0
            for sdx in np.unique(data[:, 0].astype(str)):
                eeg = data[data[:, 0].astype(str) == sdx][:, 1:33].astype(float).transpose()  # numpy : channel x samples
                # 2 seconds. no windows. EEG to Cov
                edx = 0
                while edx < eeg.shape[1]:
                    cov = np.squeeze(Covariances('oas').fit_transform(np.expand_dims(eeg[:, edx: edx + 2 * 500], axis=0)))
                    _spd_.append(cov)               # data
                    self._state.append(sdx)         # label
                    # print(self._state)
                    edx += 1000
                    tdx += 1
            self._spd_mat = np.zeros((len(_spd_), self._spd_width, self._spd_height))
            for sdx in range(len(_spd_)):
                self._spd_mat[sdx, :, :] = _spd_[sdx]

    def __len__(self):
        if self._train:
            ret = len(self._names)
        else:
            ret = len(self._state)
        return ret

    def __getitem__(self, item):
        if self._train:
            x = torch.from_numpy(self._spd_mat[item,:,:]).double()  # tensor : 1 x cov.x x cov.y
            y = torch.from_numpy(np.array(self._spd_labels[item])).long()  # tensor : n_labels
        else:
            x = torch.from_numpy(self._spd_mat[item, :, :]).double()
            y = self._state[item]
        return x, y


class DataLoaderAffectRobot:
    def __init__(self, data_path, k_fold, valid_size, batch_size, cov=False, train=True):
        fnames = []
        self._train = train
        self._k_fold = k_fold
        self._valid_size = valid_size
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file[len(file)-4:] == '.npy':
                    fnames.append(file)
        self.data_path = data_path
        self.cov = cov
        self.batch_size = batch_size

        random.shuffle(fnames)
        self.all_fnames = fnames
        self.n_test = len(fnames) // self._k_fold
        self.test_fnames = fnames[:self.n_test]
        trainset = fnames[self.n_test:]
        n_validset = int(len(trainset) * valid_size)

        # training and validation dataset
        self.train_fnames = trainset[n_validset:]
        self.valid_fnames = trainset[:n_validset]


    def getAllData(self):
        return DataLoader(DatasetAffectRobot(self.data_path, self.all_fnames, False, self._train), batch_size=1, shuffle='True')

    def getTrainset(self):
        return DataLoader(DatasetAffectRobot(self.data_path, self.train_fnames, self.cov,self._train), batch_size=self.batch_size,
                          shuffle='True')

    def getTestset(self):
        return DataLoader(DatasetAffectRobot(self.data_path, self.test_fnames, self.cov,self._train), batch_size=self.batch_size,
                          shuffle='False')

    def getValidset(self):
        return DataLoader(DatasetAffectRobot(self.data_path, self.valid_fnames, self.cov,self._train), batch_size=self.batch_size,
                          shuffle='True')