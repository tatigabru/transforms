# -*- coding: utf-8 -*-
"""
Helpers to load and vizualise data 
from CTG open dataset

__author__: Tati Gabru
"""
import os
import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import seaborn as sns
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings('ignore')

#from ..constants import DATA_DIR, META_FILE, RESULTS_DIR


"""

Loading data

"""
def load_CTU_to_arrays(data_dir: str, num: int)->Tuple[np.array, np.array]:
    """
    Loads FHR and UC data from the csv files to numpy arrays

    Args: 
        data_dir: (str) directory with the csv files
        num: (int) patient number

    Output: fhr and uc as np.arrays
    """
    file_name = os.path.join(data_dir, f"{str(num)}_FHR.dat") #file name    
    #load the data in the .dat file into a numpy array
    fhr = np.loadtxt(file_name)
    file_name = os.path.join(data_dir, f"{str(num)}_UC.dat") #file name    
    #load the data in the .dat file into a numpy array
    uc = np.loadtxt(file_name)
    
    return fhr, uc


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


def load_patient_data_to_df(data_dir: str, num: int) -> pd.DataFrame:
    """
    Loads FHR and UC data and time from a csv file to numpy arrays

    Args: 
        data_dir: (str) directory with the csv files
        num: (int) patient number

    Output: patient CTG signals as a DataFrame
    """
    file_name = os.path.join(data_dir, f"{str(num)}.csv") #file name    
    data = pd.read_csv(file_name)
        
    return data   


def gbk_to_utf8(input_file: str, output_file: str) -> None:
    # Load Files
    input_file_opened = open(input_file, 'r', encoding='gb18030') #cp936
    input_file_read = input_file_opened.read()
    output_file_opened = open(output_file, 'x', encoding='utf-8', newline='\n')
    # Transcode
    print('Transcoding…')
    output_file_opened.write(input_file_read)
    input_file_opened.close()
    output_file_opened.close()
    print('\nDone.')


def test_gbk_to_utf8(file_path: str) -> None:
    #tester gk converter
    num = 1002
    input_file = os.path.join(file_path, str(num)+".dat")
    output_file = os.path.join(file_path, str(num)+"_utf-8.dat")
    gbk_to_utf8(input_file, output_file)


def load_dat2array(data_dir: str, num: int) -> np.array:
    """
    Loads raw data from the given path

    Args: 
        data_dir: (str) directory with the csv files
        num: (int) patient number

    Output: CTG data as a np.array    
    """
    file_name = os.path.join(data_dir, str(num)+".dat") #file name
    data = np.fromfile(file_name, dtype=float)   
    
    return data


def load_ctg_y(file_path: str, patient: int) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Load preprocessed fhr, uc, time signals and y labels for a patient  
    """
    fhr = np.load(file_path+str(patient)+'_fhr.npy')
    uc = np.load(file_path+str(patient)+'_uc.npy')
    time = np.load(file_path+str(patient)+'_t.npy')
    y = np.load(file_path+str(patient)+'_y.npy')
           
    return (fhr, uc, time, y)


def load_ctg_time(file_path: str, patient: int) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Load preprocessed fhr, uc, time signals and y labels for a patient  
    """
    fhr = np.load(file_path+str(patient)+'_fhr.npy')
    uc = np.load(file_path+str(patient)+'_uc.npy')
    time = np.load(file_path+str(patient)+'_t.npy')
               
    return (fhr, uc, time)


def load_fhr_y(file_path: str, patient: int) -> Tuple[np.array, np.array]:
    """
    Load preprocessed fhr signals and y labels for a patient
    """
    fhr = np.load(file_path+str(patient)+'_fhr.npy')
    y = np.load(file_path+str(patient)+'_y.npy')

    return fhr, y


<<<<<<< HEAD
=======
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


def combine_ctg_data(file_path: str, num_set: List, batch_size: Optional[int] = None) -> Tuple[np.array, np.array, np.array]:
    """
    Combines fhr signals and y labels for a set of patients
    """
    fhr, uc, time, y = load_ctg_npy(file_path, num_set[0], batch_size)    
    for patient in num_set[1:]:
        fhr_, uc_, time_, y_  = load_ctg_npy(file_path, patient, batch_size)
        fhr = np.vstack((fhr, fhr_))
        uc = np.vstack((uc, uc_))
        #time = np.hstack((time, time_))
    print(f'fhr.shape: {fhr.shape}, y.shape: {y.shape}') 

    return fhr, uc


def combine_ctg_data(file_path: str, num_set: List, batch_size: Optional[int] = None) -> Tuple[np.array, np.array, np.array]:
    """
    Combines fhr signals and y labels for a set of patients
    """
    fhr, uc, time, y = load_ctg_npy(file_path, num_set[0], batch_size)    
    for patient in num_set[1:]:
        fhr_, uc_, time_, y_  = load_ctg_npy(file_path, patient, batch_size)
        fhr = np.vstack((fhr, fhr_))
        uc = np.vstack((uc, uc_))
        #time = np.hstack((time, time_))
    print(f'fhr.shape: {fhr.shape}, y.shape: {y.shape}') 

    return fhr, uc

>>>>>>> tanya
"""

Plot data

"""
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


def plot_ctg(fhr: np.array, uc: np.array, patient_num: Optional[int] = None) -> plt.figure:
    """
    Creates CTG plots with FHR and UC subplots
    """
    time = np.arange(len(fhr))  
    fig = plt.figure(1, figsize=(15, 8))

    plt.subplot(211)
    plt.title(f'Patient: {patient_num}')
    plt.plot(time, fhr, 'b')
    plt.ylabel('FHR')    
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(time, uc, 'r')
    plt.ylabel('UC')
    plt.xlabel('Time (min)')
    plt.grid(True)
    plt.tight_layout()
        
    return fig


def plot_fhr(fhr: np.array, fig_num: int = 1, patient_num: Optional[int] = None, color = 'b') -> plt.figure:
    """
    Creates CTG plots with FHR data
    """
    time = np.arange(len(fhr))  
  
    fig = plt.figure(fig_num, figsize=(12, 5))
    plt.title(f'Patient: {patient_num}')
    plt.plot(time, fhr, color)
    plt.ylabel('FHR')     
    plt.xlabel('Time (min)')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()    
    
    return fig


def plot_fhr_time(fhr: np.array, time: np.array, fig_num: int = 1, patient_num: Optional[int] = None, color = 'b') -> plt.figure:
    """
    Creates CTG plots with FHR data and time
    """
    fig = plt.figure(fig_num, figsize=(12, 5))
    plt.title(f'Patient: {patient_num}')
    plt.plot(time, fhr, color)
    plt.ylabel('FHR')     
    plt.xlabel('Time (min)')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()    
    
    return fig


"""

Analyse data

"""
def get_ctg_sizes(data_dir: str, df_train: pd.DataFrame):
    df = df_train.copy()
    df['len_ctg'] = 0    
    len_ctg = []
    patients = df_train['patient'].values   
    for patient in patients:
        fhr, uc, _ = load_patient_data(data_dir, int(patient))
        #print(len(fhr), len(uc))
        if len(fhr) != len(uc): raise ValueError("FHR and UC length must match")
        len_ctg.append(len(fhr)) 
        
    df['len_ctg'] = len_ctg    
    
    return df


def count_classes(df_train: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Classes distribution 
    """
    print(df_train['pH'].describe())
    normal = df_train['pH'][df_train['pH'] >= 7.15].count()
    susp = df_train['pH'][(df_train['pH'] < 7.15)&(df_train['pH'] >= 7.05)].count()
    pathol = df_train['pH'][df_train['pH'] < 7.05].count()
    
    return normal, susp, pathol


def plot_hystogram(df_train: pd.DataFrame) -> None:
    """
    Plot hystogram plot of series in dataframe
    """
    #histogram
    plt.figure(figsize=(8,5))
    sns.distplot(df_train['pH'])
    plt.show()


def plot_patients_ctg(patients: List[int], save_fig: bool = True, save_dir: Optional[str] = None) -> None:
    """
    Plot patients ctgs

    Args: 
        patients: (List[int]) list of patients
        save_fig: (bool) if True saves the picture to the disc in save_dir. Default = False
        save_dir: (str) directory fpr saving pictures. Default = RESULTS_DIR 
    """
    if save_fig: os.makedirs(save_dir, exist_ok=True)
    for patient in patients:
        fhr, uc, _ = load_patient_data(DATA_DIR, int(patient))
        fig = plot_ctg(fhr, uc, patient_num = patient)       
        if save_fig: fig.savefig(f"{save_dir}/{patient}.png", dpi=300) 
        else: plt.show()   
        plt.close()                       
        # zoomed plot
        fig2 = plot_ctg(fhr[0:3000], uc[0:3000], patient_num = patient)    
        if save_fig: fig2.savefig(f"{save_dir}/{patient}_zoom.png", dpi=300) 
        else: plt.show()    
        plt.close()         
           
        
def plot_corr(df_train: pd.DataFrame, num_factors: int = 10) -> None:
    """
    Plot correlation of df_train columns
    Plot the most correlated num_factors factors to pH
    """
    plt.figure(figsize=(10,10)) 
    corrmat = df_train.corr()
    cols = abs(corrmat).nlargest(num_factors, 'pH')['pH'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


"""
Create labels

"""
def pH_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        pH >= 7.15 - normal, label = 0
        pH >= 7.05 and  < 7.15 - intermediate, label = 2
        pH < 7.05 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1   
    df_train[target_col][df_train['pH'] >= 7.2] = 0 # 375       
    df_train[target_col][(df_train['pH'] < 7.2)&(df_train['pH'] >= 7.15)] = 1  # 72  
    df_train[target_col][(df_train['pH'] < 7.15)&(df_train['pH'] >= 7.1)] = 2 # 49
    df_train[target_col][(df_train['pH'] < 7.1)&(df_train['pH'] >= 7.0)] = 3  # 36   
    df_train[target_col][df_train['pH'] < 7.0] = 4 # 20
    
    return df_train


def argar1_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        Apgar1 > 7 - normal, label = 0
        Apgar1 == 7 - intermediate, label = 2
        Apgar1 < 7 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1  
    df_train[target_col][(df_train['Apgar1'] > 8)] = 0          
    df_train[target_col][(df_train['Apgar1'] == 7)|(df_train['Apgar1'] == 8)] = 1  
    df_train[target_col][(df_train['Apgar1'] == 5)|(df_train['Apgar1'] == 6)] = 2
    df_train[target_col][df_train['Apgar1'] < 5] = 3  

    return df_train


def argar5_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        Apgar5 > 7 - normal, label = 0
        Apgar1 == 7 - intermediate, label = 2
        Apgar1 < 7 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1  
    df_train[target_col][(df_train['Apgar5'] > 8)] = 0  
    df_train[target_col][(df_train['Apgar5'] == 7)|(df_train['Apgar5'] == 8)] = 1        
    df_train[target_col][(df_train['Apgar5'] == 5)|(df_train['Apgar5'] == 6)] = 2
    df_train[target_col][df_train['Apgar5'] < 5] = 3    

    return df_train


def filter_c_section_patients(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out patients with C-section 
    
    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df without filtered patients
    """
    df = df_train[df_train['Deliv. type'] == 1]  

    return df


if __name__=="__main__":
     
    DATA_DIR = '../../../data/database/database/signals'
    META_FILE = '../../meta.csv'
    RESULTS_DIR = '../../../output/pics'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_train = pd.read_csv(META_FILE)
    print(f'\nAnnotations: \n{df_train.head()}')
    print(f'\nCount classes: {count_classes(df_train)}')

    # normal
    patients = df_train['patient'][df_train['pH'] >= 7.15].values
    plot_patients_ctg(patients, save_fig=True, save_dir='../../output/pics/normal')
    
    # intermediate
    patients = df_train['patient'][(df_train['pH'] < 7.15) & (df_train['pH'] >= 7.05)].values
    plot_patients_ctg(patients, save_fig=True, save_dir=f'{RESULTS_DIR}/inter')

    # pathological
    patients = df_train['patient'][df_train['pH'] < 7.05].values
    plot_patients_ctg(patients, save_fig=True, save_dir=f'{RESULTS_DIR}/pathol')



