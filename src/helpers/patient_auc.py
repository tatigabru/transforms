# -*- coding: utf-8 -*-
"""
Created on Tue Sep 4 19:17:08 2018

one patient AUC

"""
from keras.models import model_from_json, load_model
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import csv
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import os.path
import csv
import keras
from keras.layers import Input, Dense, concatenate, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, \
    GlobalMaxPool1D, Flatten, GlobalAveragePooling1D, AveragePooling1D, Lambda, LSTM
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers, initializers
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
#for GPU
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"


def load_data(file_path, patient):
    """
    loads signals and y labels for all samples in patient n  
    """
    fhr = np.load(file_path+str(patient)+'_fhr.npy')
    y = np.load(file_path+str(patient)+'_y.npy')
    
    return fhr, y


def combine_data(file_path, num_set):
    """
    loads fhr, uc signals and y labels for patients in train_set
    """
    fhr_train, y_train = load_data(file_path, num_set[0])

    for patient in num_set[1:]:
        fhr, y = load_data(file_path, patient)
        fhr_train = np.hstack((fhr_train, fhr))
        y_train = np.hstack((y_train, y))

    return fhr_train, y_train


def input_format(fhr, y):
    """
    shuffle and reshape data for the input in model
    """
    N = fhr.shape[1]  # number of samples after concat
    length = fhr.shape[0]
    fhr = np.reshape(fhr[:, :N], (N, length, 1))
    y = np.reshape(y, (N, 1))
    #fhr, y = shuffle(fhr, y) # shuffles both in unison along the first axis
    y = keras.utils.to_categorical(y) # one-hot encoding
    print('y one hot: ', y)

    return fhr, y


def plot_probs(probs, fig_name = '', fig_num = 1):
    """
    creates plot to visualize probabilities
    """
    fig = plt.figure(fig_num, figsize=(9, 8))
    
    plt.plot(probs[:,0], 'r')
    plt.plot(probs[:,1], 'b')
    plt.ylabel('Probabilities')     
    plt.xlabel('Samples')
    plt.grid()
    plt.show()
    fig.savefig(fig_name)
    
    return fig


def patient_probs(file_path, patient):
    """
    calculate averaged probabilities and a true label for a single patient
    """
    fhr, y = load_data(file_path, patient)  
    y_true = int(y[0,0])
    fhr, y = input_format(fhr, y)
    probs = model.predict(fhr)
    prob_0 = np.mean(probs[:,0])
    prob_1 = np.mean(probs[:,1])
     
    print('averaged probs for patient, 0: ', prob_0, ' 1: ', prob_1,' y_true: ', y_true)
        
    return probs, y_true  


def plot_roc_auc(fpr, tpr):
    """
    plot roc_curve

    Input: fpr_set, tpr_set
    Output: figure

    """
    fig = plt.figure(2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    return fig


# load model
model_path = '/home/psollai/tanya/ctg/results_300_ave80_cont/'
model_name = 'FCN10ave_24_epoch610_240.00013'
model = load_model(model_path + model_name + '.h5')
model.summary()
print("\nModel successfully loaded from disk! ")


# load data, test val set
#file_path = 'C:/Users/New/Documents/Datasets/ctg_stage1/npy_300_20_60min_1_lin_ave80/'
file_path = '/media/ExtHDD02/Tanya/npy_300_20_60min_1_lin_ave80/'
val_set = np.load(file_path + 'val_ave80_24_15.npy')
train_set = np.load(file_path + 'train_ave80_24_85.npy')
#print('train set: ', train_set)
#print('\nval set: ', val_set)


# test pobabilities
#k = 1
#patient_set = [1002] #, 1008, 1017, 1022, 1070, 1100, 1359, 1025, 1370, 1234, 1495, 1418, 1506, 2004, 2045]
#for patient in patient_set:
#    probs, y_true = patient_probs(file_path, patient)
#    fig_name = file_path + str(patient) + '_probs.png'
#    plot_probs(probs, fig_name, k)


# test combine data
fhr, y = combine_data(file_path, train_set)
fhr_v, y_v = combine_data(file_path, val_set)

# test input format data
fhr, y = input_format(fhr, y)
fhr_v, y_v = input_format(fhr_v, y_v)
print('fhr: ', fhr.shape, ', y: ', y.shape)
print('fhr val: ', fhr_v.shape, ', y val: ', y_v.shape)


# auc for train and validation data
y_pred_train = model.predict(fhr)
score_train = roc_auc_score(y, y_pred_train)

y_pred_val = model.predict(fhr_v)
score_val = roc_auc_score(y_v, y_pred_val)
print("roc_auc_score train: ", score_train)
print(" \nroc_auc_score val: ", score_val)

# train
fpr, tpr, thresholds = roc_curve(y[:,0], y_pred_train[:,0]) # for pathological class
fig4 = plot_roc_auc(fpr, tpr)

fpr, tpr, thresholds = roc_curve(y_v[:,0], y_pred_val[:,0])
fig5 = plot_roc_auc(fpr, tpr)

# test confusion matrix data
tn, fp, fn, tp = confusion_matrix(y_v[:,1], y_pred_val[:,1])
print(tn, fp, fn, tp)


