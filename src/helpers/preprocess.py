"""
Functions for data cleaning and preprocessing

__author__: Tato Gabru

"""
import csv
import os
import os.path
import sys
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler

from src.helpers.helpers import (get_ctg_sizes, load_patient_data, plot_ctg, plot_fhr,
                     plot_fhr_time)


def load_patient_data(data_dir: str, num: int) -> Tuple[np.array, np.array, np.array]:
    """
    Loads FHR and UC data and time from a csv file to numpy arrays

    Args: 
        data_dir: (str) directory with the csv files
        num: (int) patient number

    Output: fhr, uc, time as np.arrays
    """
    file_name = os.path.join(data_dir, f"{str(num)}.csv") #file name    
    data = pd.read_csv(file_name)
    fhr = data['FHR'].values
    uc = data['UC'].values
    time = data['seconds'].values
    
    return fhr, uc, time


def time_span(signal: np.array, fs:float = 4.0) -> np.array:    
    """
    Generates time span in the physical units
    Args:
        signal: (np.array) 1D signal
        fs: (float) sampling rate. Default = 4 Hz
        
    Output: (np.array) time in minutes    
    """
    time = np.arange(signal.shape[0])
    time = time/fs # time in seconds
    time_min = time/60 # time in minutes    
    
    return time_min


def remove_artefacts(signal: np.array, low_limit: int = 40, high_limit: int = 210) -> np.array:
    """
    Replace artefacts [ultra-low and ultra-high values] with zero

        Args: 
            signal: (np.array) 1D signal
            low_limit: (int) filter values below it
            high_limit: (int) filter values above it

        Output: (np.array) filtered signal   
    """
    # replace artefacts with zero
    signal_new = signal.astype('float')
    signal_new[signal < low_limit] = 0 #replace ultra-zmall values with 0
    signal_new[signal > high_limit] = 0 #replace ultra-large values with 0
    
    return signal_new


def remove_global_outliers(signal: np.array, coef: float = 0.3) -> np.array:
    """
    Removes abrupt changes in signal and replaces them with zeros

    Args: 
        signal: (np.array) 1D signal
        coef: (float) percent of the relative change. Default = 0.3
    
    Output: (np.array) filtered signal         
    """
    # baseline without zeros
    baseline = signal[signal != 0].mean()
    #print(f'baseline: {baseline}')   
    signal_new = signal.copy()
    signal_new[np.abs(signal - baseline)> coef*baseline] = 0 
    
    return signal_new


def zero_to_nan(signal:np.array) -> np.array:
    """
    Replace zeros with nan
    
    Args: 
        signal: (np.array) 1D signal       
    
    Output: (np.array) filtered signal
    """
    signal_nan = signal.astype('float')
    signal_nan[signal == 0] = np.nan

    return signal_nan


def remove_local_outliers(signal: np.array, window_size: int = 4, coef: float = 0.3) -> np.array:
    """
    Removes local outliers in signal and replaces them with zeros

    Args: 
        signal: (np.array) 1D signal
        coef: (float) percent of the relative change. Defualt = 0.3
    
    Output: (np.array) filtered signal         
    """
    signal_new = signal.copy()
    for k in range(window_size, len(signal)-window_size):
        # local baseline without zeros
        sig = signal[k-window_size:k+window_size]
        baseline = sig[sig != 0].mean()
        #print(f'baseline: {baseline}')   
        if np.absolute(signal[k] - baseline) > coef*baseline:
            signal_new[k] = 0  
                 
    return signal_new


"""

Smoothing

"""
def moving_average(signal: np.array, window_size: int=5) -> np.array:
    """
    Сaclulate moving average for signal with zeros, 
    replaces zero values with the moving average

    Args:
        signal: (np.array) 1D signal
        window size: (int) takes 2*window_size points for moving average; 5 by default (10 points)
        
    Output: (np.array) corrected by moving mean signal    
    """
    n = window_size
    #calculate baseline without zeros
    signal_mean = signal[signal != 0].mean()
    #create masked array for nan values
    signal = np.ma.masked_array(signal, signal==0)
    #calculted moving average with window size = n
    ave = np.cumsum(signal.filled(0))
    ave[n:] = ave[n:] - ave[:-n]
    counts = np.cumsum(~signal.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ave[~signal.mask] /= counts[~signal.mask]
    ave[signal.mask] = signal_mean # replces nan with baseline mean taken without nan 
    
    signal = np.flip(ave, 0) #reverces signal to make a symmetrical moving averaging
    #calculate baseline without zeros
    signal_mean = signal[signal != 0].mean()
    signal = np.ma.masked_array(signal, signal==0)
    #calculted moving average with window size = n
    ave = np.cumsum(signal.filled(0))
    ave[n:] = ave[n:] - ave[:-n]
    counts = np.cumsum(~signal.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ave[~signal.mask] /= counts[~signal.mask]
    ave[signal.mask] = signal_mean # replaces zeros with baseline mean taken without nan 
    #reverse back
    ave = np.flip(ave, 0) #reverces signal back
    np.nan_to_num(ave)
    
    return ave


def cycle_moving_average(signal: np.array, window_size: int=5, num_cycles: int=5) -> np.array:
    """
    Apply moving average to signal for several cycles

    Args:
        signal: (np.array) 1D signal
        window size: (int) takes 2*window_size points for moving average; 5 by default (10 points)
        num_cycles: (int) how many cycles ro average. Default = 5
        
    Output: (np.array) smoothed signal    
    """
    n = window_size    
    # copy signal
    signal_ave = signal.astype('float')
    for i in range(num_cycles):
        signal_ave = moving_average(signal_ave, n) # averaging
    
    return signal_ave


def one_side_moving_average(signal: np.array, window_size: int=5) -> np.array:
    """
    Сaclulate moving average of signal without zeros, 
    replaces zero values with the moving average

    Args:
        signal: (np.array) 1D signal
        window size: (int) takes 2*window_size points for moving average; 5 by default (10 points)
        
    Output: (np.array) filtered signal    
    """
    n = window_size
    #calculate baseline without zeros
    signal_mean = signal[signal != 0].mean()
    #create masked array for zero values
    signal = np.ma.masked_array(signal, signal==0)
    #calculted moving average with window size = n
    ave = np.cumsum(signal.filled(0))
    ave[n:] = ave[n:] - ave[:-n]
    counts = np.cumsum(~signal.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ave[~signal.mask] /= counts[~signal.mask]
    ave[signal.mask] = signal_mean # replces nan with baseline mean taken without nan 

    return ave


"""

Missing values 

"""
def remove_maternal_heart_rate(fhr: np.array, uc: np.array, time: np.array, num: int = 240, criterion: float = 0.8) -> Tuple[np.array, np.array, np.array]:
    """
    Remove parts of signals with long consecutive zeros in FHR signal > num

    Args:  
        fhr, uc signals: (np.arrays) 1D signals
        num: (int) max duration of nans to be removed. Default = 240 --> 1 min with fs = 4

    Output: (Tuple[np.array, np.array, np.array]) FHR, UC and time filtered signals  
    """
    # signal baseline 
    baseline = fhr[fhr != 0].mean()
    # Calculate non-zero regions, filter out streaks that are too short, apply global mask
    nonzero_spots = np.where(fhr!=0)
    diff = np.diff(nonzero_spots)[0]
    streaks = np.split(nonzero_spots[0], np.where(diff != 1)[0]+1)
    
    # get short nonzero regions between zeros
    short_streaks =[streak for streak in streaks if len(streak) <= num]
    if len(short_streaks) > 0:
        short_streaks = set(np.hstack(short_streaks))
    # maternal heart rate streaks
    MHR_streaks = [streak for streak in short_streaks if fhr[streak].mean() < criterion*baseline]
    if len(MHR_streaks) > 0:
        MHR_streaks = set(np.hstack(MHR_streaks))
    
    mask = [item not in MHR_streaks for item in range(len(fhr))]
    fhr = fhr[mask]
    # remove same parts in corresponding uc and time arrays
    uc = uc[mask] 
    time = time[mask]
    
    return (fhr, uc, time)


def remove_between_zeros_ctg(fhr: np.array, uc: np.array, time: np.array, num: int = 10) -> Tuple[np.array, np.array, np.array]:
    """
    Remove short islands of the signal between nans/zeros if length < num

    Args:  
        fhr, uc signals: (np.arrays) 1D signals
        num: (int) dutration of nans to be remoed. Default = 20 --> 5 sec with fs = 4

    Output: (Tuple[np.array, np.array, , np.array]) FHR, UC and time filtered signals  
    """
    # Calculate non-zero regions, filter out streaks that are too short, apply global mask
    nonzero_spots = np.where(fhr!=0)
    diff = np.diff(nonzero_spots)[0]
    streaks = np.split(nonzero_spots[0], np.where(diff != 1)[0]+1)
    
    # get short nonzero regions between zeros
    short_streaks =[streak for streak in streaks if len(streak) <= num]
    if len(short_streaks) > 0:
        short_streaks = set(np.hstack(short_streaks))
        
    mask = [item not in short_streaks for item in range(len(fhr))]
    fhr = fhr[mask]
    # remove same parts in corresponding uc and time arrays
    uc = uc[mask] 
    time = time[mask]
    
    return (fhr, uc, time)


def remove_longzeros_ctg(fhr: np.array, uc: np.array, time: np.array, num: int = 240) -> Tuple[np.array, np.array, np.array]:
    """
    Remove parts of signals with long consecutive zeros in FHR signal > num

    Args:  
        fhr, uc, time signals: (np.arrays) 1D signals
        num: (int) duration of zeros to be removed. Default = 20 --> 5 sec with fs = 4

    Output: (Tuple[np.array, np.array, np.array]) FHR, UC and time filtered signals 
    """
    #time = np.arange(len(fhr))
    # Calculate streaks, filter out streaks that are too long, apply global mask
    zero_spots = np.where(fhr==0)
    diff = np.diff(zero_spots)[0]
    streaks = np.split(zero_spots[0], np.where(diff != 1)[0]+1)
    
    long_streaks = [streak for streak in streaks if len(streak) > num]
    if len(long_streaks) > 0:
        long_streaks = set(np.hstack(long_streaks))        
        
    mask = [item not in long_streaks for item in range(len(fhr))]
    fhr = fhr[mask]    
    uc = uc[mask] 
    time = time[mask]
    
    return (fhr, uc, time)


def remove_longnan_uc(uc: np.array, fhr: np.array, num: int = 20) -> Tuple[np.array, np.array]:
    """
    Remove parts of signals with long consecutive nans in UC signal > num
    
    Args:  
        uc, fhr signals: (np.arrays) 1D signals
        num: (int) duration of nans to be remoed. Default = 20 --> 5 sec with fs = 4

    Output: (Tuple[np.array, np.array]) UC and FHR filtered signals  
    """
    # Calculate streaks, filter out streaks that are too short, apply global mask
    nan_spots = np.where(np.isnan(uc))
    diff = np.diff(nan_spots)[0]
    streaks = np.split(nan_spots[0], np.where(diff != 1)[0]+1)
    long_streaks = set(np.hstack([streak for streak in streaks if len(streak) > num]))
    mask = [item not in long_streaks for item in range(len(uc))]
    #print("Filtered (without long streaks): ", uc[mask])
    uc = uc[mask]
    # remove same parts in corresponding uc and time arrays
    fhr = fhr[mask] 
            
    return (uc, fhr)


def fill_with_local_mean(signal: np.array, window_size: int = 5) -> np.array:
    """
    Removes local outliers in signal and replaces them with zeros

    Args: 
        signal: (np.array) 1D signal
            
    Output: (np.array) filtered signal         
    """
    signal_new = signal.copy()
    signal_mean = signal[signal != 0].mean()
    signal_new[signal==0] = signal_mean
                 
    return signal_new


def drop_zeros_fhr(signal: np.array) -> np.array:
    """
    Removes all missingvalues, encoded as zeros

    Args: 
        signal: (np.array) 1D signal
            
    Output: (np.array) filtered signal         
    """
    signal_new = signal[signal != 0].copy()
                     
    return signal_new


def drop_zeros_ctg(fhr: np.array, uc: np.array, time: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Removes all missingvalues, encoded as zeros

    Args: 
        uc, fhr signals: (np.arrays) 1D signals
        time: (np.array) 1D time signal 
            
    Output: Tuple(np.arrays) filtered FHR and UC signals and time         
    """
    mask = [fhr != 0]
    fhr = fhr[mask]
    uc = uc[mask]
    time = time[mask]    
                     
    return (fhr, uc, time)


def interpolate_nans(signal: np.array) -> np.array:
    """
    Replace nans/zeros with linear interpolation
    
    Output: (np.array) signal with interpolated nans     
    """
    x = zero_to_nan(signal).astype('float')
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))

    return np.interp(indices, indices[not_nan], x[not_nan])


def assess_signal(signal: np.array, threshold: float = 0.30) -> Tuple[bool, float]:
    """
    Signal quality assessment based on the amount of missing data 

    Args: 
        signal: (np.array) 1D signal
        time: (np.array) 1D time signal 
   
    Output: Tuple[bool, float] returns True if amount of missing data > threshold and the proportion of missing data 
    """
    signal_assessment = False # "good" signal
    # number of nonzero and non NaN elements 
    count_nan = np.count_nonzero(np.isnan(signal))
    count_zeros = len(signal[signal == 0])
    missing_fraction = (count_nan+count_zeros)/len(signal)

    if missing_fraction > threshold:
        signal_assessment = True
    
    return (signal_assessment, missing_fraction) 


"""

Downsampling

"""
def downsample(signal: np.array, factor: int = 16, ftype: str ='iir') -> np.array:
    """    
    Downsample the signal after applying an anti-aliasing filter.
    By default, an order 8 Chebyshev type I filter is used, 'iir'.
    A 30 point FIR filter with Hamming window is used if ftype is ‘fir’.
    When using IIR downsampling, it is recommended to call decimate multiple times for downsampling factors higher than 13.

    Args:
        signal: (np.array) 1D signal
        factor: (int) downsample factor. Default = 16 (0.25 Hz from sampling frequency 4 Hz)
        ftype: (str) filter type. Default ='iir', optional 'fir'
                
    Output: (np.array) downsampled signal   
    """
    if ftype != 'iir' and ftype != 'fir': raise ValueError("Unknown filter. Possible options are 'iir' and 'fir'") 
    
    if factor > 12: 
        num = int(np.log(factor)/np.log(2))
        if num%1 != 0: raise ValueError("Factor above 12 must be 2**n; otherwise apply downsampling a few times")
        for i in range(num):
            signal =  decimate(signal, 2, ftype=ftype)   
    else:
        signal =  decimate(signal, factor, ftype=ftype)    
    
    return signal


"""

Get needed part

"""    
def crop_last(signal: np.array, crop_size: int=14400) -> np.array:
    """    
    Crop the last N points of the signal

    Args:
        signal: (np.array) 1D signal
        crop_size: (int) takes only the last N points of the signal. Default = 14400 -> 60 min
                
    Output: (np.array) cut signals batch   
    """
    if crop_size >= len(signal): raise ValueError("Crop size must be smaller than the original signal length")
    
    return signal[len(signal)-crop_size:]


def pad_zeros(signal: np.array, crop_size: int = 9600) -> np.array:
    """
    Adds zeros padding in the beginning of the signal to match the desired size 
    
    Args: 
        signal: (np.array) 1D signal
        crop_size: (int) desired size
                    
    Output: (np.array) zero-padded signal         
    """
    if len(signal) < crop_size:
        signal_ = np.zeros(crop_size)
        signal_[crop_size - len(signal):] = signal
        signal = signal_
    
    return signal 


def get_I_stage(signal: np.array, df: pd.DataFrame, patient: int) -> np.array:
    """
    Take only the I stage of labour for a given patient
    
    Args:
        signal: (np.array) 1D signal
        df: (pd.DataFrame) DataFrame with mate data
        patient: (int) patient ID number
                
    Output: (np.array) cut signals batch
    
    """
    second_stage = df['II.stage'][df['patient'] == patient].values[0].clip(0, 30)
    # second stage in samples
    second_stage = int(second_stage*60*4)             
    
    return signal[:len(signal)-second_stage]


def get_II_stage(signal: np.array, df: pd.DataFrame, patient: int) -> np.array:
    """
    Take only the II stage of labour for a given patient
    
    Args:
        signal: (np.array) 1D signal
        df: (pd.DataFrame) DataFrame with mate data
        patient: (int) patient ID number
                
    Output: (np.array) cut signals batch
    
    """
    second_stage = df['II.stage'][df['patient'] == patient].values[0].clip(0, 30)
    # second stage in samples
    second_stage = int(second_stage*60*4)           
    
    return signal[len(signal)-second_stage:]
 

"""

Create batches ad preprocessing all together

"""  
def split_to_batch(signal: np.array, crop_size: int=7200, shift: int=240, batch_size: Optional[int] = None) -> List[List[int]]:
    """
    Split original signal in batches of sub-signals with size and shifted by shift

    Args:
        signal: (np.array) 1D signal
        crop_size: (int) duration of signal crop. Default = 7200 -> 1200 sec = 30 min
        shift: (int) duration of the sub-sugnals shift. Default = 240 -> 60 sec = 1 min
        
    Output: List(np.array) signals batch   
    """
    if crop_size > len(signal): raise ValueError("Crop size must be smaller than the original signal length")

    signal = np.flip(signal, 0) # flip to start from the end        
    num = (len(signal)-crop_size)//shift # how many batches fit completely within signal    
    if batch_size: num = min(batch_size, num)
        
    batch_signals = np.empty([num, crop_size], dtype = float)         
    batch_signals = [signal[k*shift:k*shift+crop_size] for k in range(num)]
    batch_signals = np.flip(batch_signals, 1) # flip back   

    return batch_signals


def preprocess_ctg(file_path: str, patient: int, df: pd.DataFrame, stage: Optional[int] = None) -> Tuple[np.array, np.array, np.array]:
    """
    CTG data preprocessing 
    Cleans the signal from noise
    
    Args:
        file_path: (str) directory with patients csv data 
        patient: (int): patient number 
        df: (pd.DataFrame) dataframe with meta data 
        stage: stage of labour. Stage must be integer 1 (I stage), 2 (II stage), or None for both stages together
        
    Output: (Tuple[np.array, np.array, np.array]) preprocessed fhr, UC and time signals
    """
    fhr, uc, time = load_patient_data(file_path, int(patient))
    
    # select stage
    if stage == 1:
        fhr1, uc1, time1 = get_I_stage(fhr, df, patient), get_I_stage(uc, df, patient), get_I_stage(time, df, patient)
    elif stage == 2:
        fhr1, uc1, time1 = get_II_stage(fhr, df, patient), get_II_stage(uc, df, patient), get_II_stage(time, df, patient)
    elif stage == None:
        fhr1, uc1, time1 = fhr, uc, time 
    else:
        raise ValueError("Stage must be integer 1 (I stage), 2 (II stage), or None for both stages together")
            
    # remove noise
    fhr_ = remove_artefacts(fhr1) 
    fhr_ = remove_global_outliers(fhr_, 0.4) 
    fhr_ = remove_local_outliers(fhr_, window_size = 5, coef = 0.4)
    
    # remove maternal heart rate  
    fhr_, uc_, time_ = remove_maternal_heart_rate(fhr_, uc1, time1, num=120)
    fhr_, uc_, time_ = remove_between_zeros_ctg(fhr_, uc_, time_, num=10)
    
    return (fhr_, uc_, time_)


def preprocess_ctg_nn(file_path: str, patient: int, df: pd.DataFrame, stage: int = 1, crop_size: int = 9600) -> Tuple[np.array, np.array, np.array, bool, float]:
    """
    CTG data preprocessing for neural nets
    Cleans the signal from noise and interpolates missing values 
    
    Args:
        file_path: (str) directory with patients csv data 
        patient: (int): patient number 
        df: (pd.DataFrame) dataframe with meta data 
        stage: stage of labour. Stage must be integer 1 (I stage), 2 (II stage), or None for both stages together
        crop_size: (int) cize of the crop from the signal end
        
    Output: (Tuple[np.array, np.array, np.array, bool, float]) preprocessed fhr, UC and time signals, signal assessment and missing part
    """
    
    fhr_, uc_, time_ = preprocess_ctg(file_path, patient, df, stage)
    # assess signal
    assess, missing_part = assess_signal(fhr_, 0.3)

    # discard long/short missing values
    fhr_, uc_, time_ = remove_longzeros_ctg(fhr_, uc_, time_, 1000) # 4 min missing    
    # interpolate short missing values
    fhr_= interpolate_nans(fhr_)    
    # pad if needed 
    fhr_, uc_, time_ = pad_zeros(fhr_, crop_size), pad_zeros(uc_, crop_size), pad_zeros(time_, crop_size)

    # downsample
    fhr, uc, time = downsample(fhr_), downsample(uc_), downsample(time_) 

    return (fhr, uc, time, assess, missing_part)


def preprocess_ctg_feat(file_path, patient: int, df: pd.DataFrame, stage: int = 1) -> Tuple[np.array, np.array, np.array, bool, float]:
    """
    CTG data preprocessing for features extraction
    Cleans the singal from noise and interpolated missing values 
    
    Args:
        file_path: (str) directory with patients csv data 
        patient: (int): patient number 
        df: (pd.DataFrame) dataframe with meta data 
        stage: stage of labour. Stage must be integer 1 (I stage), 2 (II stage), or None for both stages together
                
    Output: (Tuple[np.array, np.array, np.array, bool, float]) preprocessed fhr, UC and time signals, signal assessment and missing part
    """
    
    fhr_, uc_, time_ = preprocess_ctg(file_path, patient, df, stage)
    # assess signal
    assess, missing_part = assess_signal(fhr_, 0.3)

    # discard long/short missing values
    fhr, uc, time = drop_zeros_ctg(fhr_, uc_, time_)
    
    return (fhr, uc, time, assess, missing_part)


def preprocess_batch(file_path, patient: int, labels_df: pd.DataFrame, stage: int = 1, crop_size: int = 9600, shift: int = 4) -> Tuple[np.array, np.array, np.array]:
    """
    CTG data preprocessing 
    Cleans the singal from noise and interpolated missing values 
    """
    fhr_, uc_, time_ = preprocess_ctg(file_path, patient, df, stage)
    # assess signal
    assess, missing_part = assess_signal(fhr_, 0.3)
    
    if assess:
        print(f'Patient to remove: {patient}')
    
    # discard long/short missing values
    fhr_, uc_, time_ = remove_longzeros_ctg(fhr_, uc_, time_, 1000) # 4 min missing    
    # interpolate short missing values
    fhr_= interpolate_nans(fhr_)    
    # pad if needed 
    fhr_, uc_, time_ = pad_zeros(fhr_, crop_size+shift), pad_zeros(uc_, crop_size+shift), pad_zeros(time_, crop_size+shift)

    # get patient batch
    fhr_batch = split_to_batch(fhr_, crop_size=crop_size, shift=shift) 
    uc_batch = split_to_batch(uc_, crop_size=crop_size, shift=shift) 
    time_batch = split_to_batch(time_, crop_size=crop_size, shift=shift) 
    
    fhr_batch2 = np.empty([len(fhr_batch), crop_size//16])
    uc_batch2 = np.empty([len(fhr_batch), crop_size//16])
    time_batch2 = np.empty([len(fhr_batch), crop_size//16])
    
    for i in range(len(fhr_batch)):  
        fhr_ = fhr_batch[i, :]
        uc_ = uc_batch[i, :]
        time_ = time_batch[i, :] 
        # downsample
        fhr, uc, time = downsample(fhr_), downsample(uc_), downsample(time_) 
        fhr_batch2[i, :], uc_batch2[i, :], time_batch2[i, :] = fhr, uc, time
    print('fhr_batch shape:', fhr_batch2.shape) 
    
    return (fhr_batch2, uc_batch2, time_batch2, assess, missing_part)


def preprocess_patient(file_path: str, patient: int, df: pd.DataFrame, stage: int = 1, crop_size: int = 9600) -> Tuple[np.array, np.array, np.array]:
    # read labels
    target = df['target'][df['patient'] == patient].values[0]
    target = int(target)
       
    if target == 0:
        fhrb, ucb, timeb, assess, missing_part = preprocess_batch(file_path, patient, df, stage = 1, crop_size = crop_size, shift = 1)
        yb = np.zeros((1, len(fhrb))) #match labels dim to samples size        
    elif target == 1:
        #divide signals with fs = stride1     
        fhrb, ucb, timeb, assess, missing_part = preprocess_batch(file_path, patient, df, stage = 1, crop_size = crop_size, shift = 1)
        yb = np.ones((1, len(fhrb)))
    elif target == 2:
        #divide signals with fs = stride2     
        fhrb, ucb, timeb, assess, missing_part = preprocess_batch(file_path, patient, df, stage = 1, crop_size = crop_size, shift = 1)
        yb = np.ones((1, len(fhrb)))*2    
    
    print('y batch shape:', yb.shape)
    
    return (fhrb, ucb, timeb, yb, assess, missing_part)


def save_patient_npy(save_dir: str, missings: list, dropped_patients: list, batch_sizes: list, file_path: str, patient: int, df: pd.DataFrame, stage: int = 1, crop_size: int = 9600) -> Tuple[list, list, list]:
    """
    Save preprocessed patient signals to npy arrays
    
    Args:
    
    """
    fhrb, ucb, timeb, yb, assess, missing_part = preprocess_patient(file_path, patient, df, stage, crop_size)
    print(f'missing_part: {missing_part}')
    missings.append(missing_part)
    if assess:
        print(f'Patient to remove: {patient}')
        dropped_patients.append(patient)
    else:
        batch_sizes.append(len(fhrb))
        # save signals/batches to numpy array
        np.save(save_dir+str(patient)+'_fhr.npy', fhrb)
        np.save(save_dir+str(patient)+'_uc.npy', ucb)
        np.save(save_dir+str(patient)+'_t.npy', timeb) 
        np.save(save_dir+str(patient)+'_y.npy', yb) 
        
    return (dropped_patients, missings, batch_sizes)


def preprocess_dataset(save_dir: str):
    """
    Preprocess dataset for features extraction
    """
    os.makedirs(save_dir, exist_ok = True)
    
    missings = []
    # whole dataset preprocess and save; to test use patients[:3] 
    for patient in patients:  
        print('current patient: ', patient)
        fhrb, ucb, timeb, _, missing_part = preprocess_ctg_feat(DATA_DIR, patient, df, stage = 1)    
        assert len(fhrb)==len(ucb)==len(timeb)
        print(f'missing_part: {missing_part}')
        missings.append(missing_part)
        # save signals to numpy array
        np.save(save_dir+str(patient)+'_fhr.npy', fhrb)
        np.save(save_dir+str(patient)+'_uc.npy', ucb)
        np.save(save_dir+str(patient)+'_t.npy', timeb)     
    df = pd.DataFrame()
    df['patient'] = patients
    df['missing_part'] = missings


def preprocess_dataset4nn(save_dir: str):
    """
    Preprocess dataset for neural nets
    """
    os.makedirs(save_dir, exist_ok = True) 

    missings, dropped_patients, saved_patients = [], [], []
    batch_sizes = []
    # whole dataset preprocess and save; to test use patients[:3] 
    for patient in patients:  
        print('current patient: ', patient)
        fhrb, ucb, timeb, yb, assess, missing_part = preprocess_patient(DATA_DIR, patient, df, stage=None, crop_size= 14400)
        assert len(fhrb)==len(ucb)==len(timeb)
        
        print(f'missing_part: {missing_part}')
        missings.append(missing_part)
        if assess:
            print(f'Patient to remove: {patient}')
            dropped_patients.append(patient)
        else:
            batch_sizes.append(len(fhrb))
            saved_patients.append(patient)
            # save signals/batches to numpy array
            np.save(save_dir+str(patient)+'_fhr.npy', fhrb)
            np.save(save_dir+str(patient)+'_uc.npy', ucb)
            np.save(save_dir+str(patient)+'_t.npy', timeb) 
            np.save(save_dir+str(patient)+'_y.npy', yb) 

    np.save(save_dir+'dropped_patients.npy', np.asarray(dropped_patients)) # 102/552 bad records dropped for now, 18%
    np.save(save_dir+'saved_patients.npy', np.asarray(saved_patients))
    df = pd.DataFrame()
    df['patient'] = patients
    df['missing_part'] = missings
    df.to_csv(os.path.join(save_dir, "df_missing_parts.csv"), index=False)
    


if __name__=="__main__":   
    DATA_DIR = '../../data/database/database/signals'
    META_FILE = '../meta.csv'
    RESULTS_DIR = '../../output/pics'
    df = pd.read_csv(META_FILE)
    patients = df['patient'].values

    # preprocessed I stage for 60 min window (14400/16 = 900 points), made pH based target labels
    save_dir = '../../data/preprocessed/npy900_3_ph/'
    # preprocessed I stage for 60 min window (14400/16 = 900 points), made pH based target labels
    save_dir = '../../data/preprocessed/npy900_3_ph/'
    # preprocessed I stage for 30 min window (7200/16 = 450 points), made pH based target labels
    save_dir = '../../data/preprocessed/npy450_3_ph/'
    # preprocess for features, drop zeros
    save_dir = '../../data/preprocessed/npy4features_keepzeros/'
    
    print('Done.')