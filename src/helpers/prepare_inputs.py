"""
Helpers to prepare input data for models
stack, shuffle, normalize

__author__: Tati Gabru
"""
import os
import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy import stats
from scipy.stats import norm
from sklearn.utils import shuffle

from helpers import load_ctg_npy, load_fhr_npy

warnings.filterwarnings('ignore')
#from ..constants import DATA_DIR, META_FILE, RESULTS_DIR



def combine_fhr_data(file_path: str, num_set: List) -> Tuple[np.array, np.array]:
    """
    Combines fhr signals and y labels for a set of patients
    """
    fhr, y = load_fhr_npy(file_path, num_set[0])
    for patient in num_set[1:]:
        fhr_, y_  = load_fhr_npy(file_path, patient)
        fhr = np.hstack((fhr, fhr_))        
        y = np.hstack((y , y_))

    print(f'fhr.shape: {fhr.shape}, y.shape: {y.shape}')

    return fhr_train, y_train


def combine_ctg_data(file_path: str, num_set: List) -> Tuple[np.array, np.array, np.array]:
    """
    Combines fhr signals and y labels for a set of patients
    """
    fhr, uc, time, y = load_ctg_npy(file_path, num_set[0])
    for patient in num_set[1:]:
        fhr_, uc_, time_, y_  = load_ctg_npy(file_path, patient)
        fhr = np.hstack((fhr, fhr_))
        uc = np.hstack((uc, uc_))
        y = np.hstack((y , y_))    
        
    print(f'fhr.shape: {fhr.shape}, y.shape: {y.shape}') 

    return fhr, uc, time, y


def reshape_shuffle_fhr(fhr: np.array, y: np.array, shuffle: bool = True) -> Tuple[np.array, np.array]:
    """
    Reshape and optionally shuffle inputs and targets for the keras/tf input in model

    Args:
        fhr, y : (np.arrays) combines batch of fhr signals and targets
        shuffle: (bool) is True the data are shuffled together

    Output: np.arrays of reshaped and (optionally) suffled data    
    """
    N = fhr.shape[1]  # number of samples after concat
    length = fhr.shape[0]
    fhr = np.reshape(fhr[:, :N], (N, length, 1))
    y = np.reshape(y, (N, 1))
    
    if shuffle:
        fhr, y = shuffle(fhr, y) # shuffles both in unison along the first axis

    return fhr, y


def reshape_shuffle_ctg(fhr: np.array, uc: np.array, y: np.array, time: Optional[np.array]) -> Tuple[np.array, np.array, np.array, Optional[np.array], Optional[np.array]]:
    """
    Reshape and optionally shuffle inputs and targets for the keras/tf input in model

    Args:
        fhr, y : (np.arrays) combines batch of fhr signals and targets
        shuffle: (bool) is True the data are shuffled together

    Output: np.arrays of reshaped and (optionally) suffled data    
    """
    N = fhr.shape[1]  # number of samples after concat
    length = fhr.shape[0]
    fhr = np.reshape(fhr[:, :N], (N, length, 1))
    uc = np.reshape(uc[:, :N], (N, length, 1))
    y = np.reshape(y, (N, 1))
    if time: 
        time = np.reshape(time[:, :N], (N, length, 1))
        if shuffle:
            fhr, uc, time, y = shuffle(fhr, uc, time, y)
        return fhr, uc, time, y    
    elif shuffle:
        fhr, uc, y = shuffle(fhr, uc, y) # shuffles all in unison along the first axis
        return fhr, uc, y


def normalize_meanstd(data: np.array) -> Tuple[np.array, float, float]:
    mean = np.mean(data, axis=0)
    std = np.std(data)

    signal_norm = data - mean


def normalize(data: np.array) -> Tuple[np.array, float, float]:

