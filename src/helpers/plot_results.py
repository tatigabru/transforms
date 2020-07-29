"""
read outcomes of training/ models tunings

@author: Tanya
"""


#usual imports
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import matplotlib.pyplot as plt
import os.path
import csv



def load_results(file_path, model_name, num, lr):
    """
    loads results from the given path to dataframe
    """
    file_name = os.path.join(file_path, model_name + str(num) + str(lr) + "_params.csv")  # parameters 
    params_df = pd.read_csv(file_name) # read data as csv to a dataframe
    
    file_name = os.path.join(file_path, model_name + str(num) + str(lr) + "_history.csv")  # train history 
    history_df = pd.read_csv(file_name) # read data as csv to a dataframe

    return params_df, history_df


def load_params(params_df):
    """
    read models parameters from a dataframe
    """
    decay = float(params_df.loc[:,'decay'])
    lr = float(params_df.loc[:,'lr'])
    batch_size = int(params_df.loc[:,'batch_size'])
    kernel_size = int(params_df.loc[:,'kernel_size'])
    num_filters = int(params_df.loc[:,'num_filters'])
    #print(decay)
    
    return decay, lr, batch_size, kernel_size, num_filters

    
#file_path = '/home/psollai/tanya/ctg/models_results/FCN_5_tuning/' 
file_path = 'C:/Users/New/Documents/Datasets/results_300_ave80/'
model_name = 'FCN10ave_24_'
#model_name = 'FCN_RS_5_tune'

# test load training results
params, history = load_results(file_path, model_name, 24, 0.0005)
print('params: ', params,' history: ', history)
decay, lr, batch_size, kernel_size, num_filters = load_params(params)


def plot_acc(file_path, model_name, num_filters, lr):
    """
    plots accuracy for various models, returns fig
    """
    fig = plt.figure(1, figsize=(9, 7))
    plt.ylabel('Accuracy', fontsize = 16)
    plt.xlabel('# epochs', fontsize = 16)    
    plt.title('Accuracy, '+model_name, fontsize = 18)
    info = [] #list of strings
    colors = ['r', 'b', 'g', 'm', 'k', 'c', '--r', '--b', '--g', '--m', '--k','--c', \
              'r', 'b', 'g', 'm', 'k', 'c', '--r', '--b', '--g', '--m', '--k','--c']
    
    for num in num_filters:
        params, history = load_results(file_path, model_name, num, lr)
        color = colors[1]
        acc = history.loc[:,'acc']
        plt.plot(acc, color)
        color = colors[0]
        val_acc = history.loc[:,'val_acc']
        plt.plot(val_acc, color)
        #add a legend with parameters
        decay, learning_rate, batch_size, kernel_size, num_filters = load_params(params)
        info.append('decay: '+str(decay)+', batch: '+str(batch_size)+', kernel: '+ \
                    str(kernel_size)+', # filters:'+str(num_filters)) #append a line to a legend list  
        plt.legend(info, loc='lower right', bbox_to_anchor=(1, 0.0))
        
    # plt.text(6, 0.75, 'learning rate: '+str(learning_rate), fontsize = 14)
    plt.grid() 
    
    return fig


def plot_auc(file_path, model_name, num_filters, lr):
    """
    plots accuracy for various models, returns fig
    """
    fig = plt.figure(2, figsize=(9, 7))
    plt.ylabel('AUC', fontsize = 16)
    plt.xlabel('# epochs', fontsize = 16)
    plt.grid() 
    plt.title('AUC, '+model_name, fontsize = 18)
    info = [] #list of strings
    colors = ['r', 'b', 'g', 'm', 'k', 'c', '--r', '--b', '--g', '--m', '--k','--c', \
              'r', 'b', 'g', 'm', 'k', 'c', '--r', '--b', '--g', '--m', '--k','--c']
    
    for num in num_filters:
        params, history = load_results(file_path, model_name, num, lr)
        color = colors[1]
        auc = history.loc[:,'auc']
        plt.plot(auc, color)
        color = colors[0]
        val_auc = history.loc[:,'val_auc']
        plt.plot(val_auc, color)
        #add a legend with parameters
        decay, learning_rate, batch_size, kernel_size, num_filters = load_params(params)
        info.append('decay: '+str(decay)+', batch: '+str(batch_size)+', kernel: '+ \
                    str(kernel_size)+', # filters:'+str(num_filters)) #append a line to a legend list  
        plt.legend(info, loc='lower right', bbox_to_anchor=(1, 0.0))
        
    #plt.text(6, 0.75, 'learning rate: '+str(learning_rate), fontsize = 14)
        
    return fig


def plot_loss(file_path, model_name, num_filters, lr):
    """
    plots accuracy for various models, returns fig
    """
    fig = plt.figure(3, figsize=(9, 7))
    plt.ylabel('Loss (binary crossentrophy)', fontsize = 16)
    plt.xlabel('# epochs', fontsize = 16)
    plt.grid() 
    plt.title('Loss, '+model_name, fontsize = 18)
    info = [] #list of strings
    colors = ['r', 'b', 'g', 'm', 'k', 'c', '--r', '--b', '--g', '--m', '--k','--c', \
              'r', 'b', 'g', 'm', 'k', 'c', '--r', '--b', '--g', '--m', '--k','--c']
    
    for num in num_filters:
        params, history = load_results(file_path, model_name, num, lr)
        color = colors[1]
        loss = history.loc[:,'loss']
        plt.plot(loss, color)
        color = colors[0]
        val_loss = history.loc[:,'val_loss']
        plt.plot(val_loss, color)
        #add a legend with parameters
        decay, learning_rate, batch_size, kernel_size, num_filters = load_params(params)
        info.append('decay: '+str(decay)+', batch: '+str(batch_size)+', kernel: '+ \
                    str(kernel_size)+', # filters:'+str(num_filters)) #append a line to a legend list  
        plt.legend(info, loc='lower right', bbox_to_anchor=(1, 0.0))
        
    plt.text(6, 0.75, 'learning rate: '+str(learning_rate), fontsize = 14)
        
    return fig


# test plot results for different model parameters    
lr = 0.0005
num = [24]
plot_models_results = True

if plot_models_results:
    #test fig_acc    
    fig_acc = plot_acc(file_path, model_name, num, lr) 
    fig_acc.savefig(file_path + str(model_name) + str(num) + '_acc.png')
 
    fig_auc = plot_auc(file_path, model_name, num, lr) 
    fig_auc.savefig(file_path + str(model_name) + str(num) + '_auc.png')
    
    fig_loss = plot_loss(file_path, model_name, num, lr) 
    fig_loss.savefig(file_path + str(model_name) + str(num) + '_loss.png')
    

     
        

