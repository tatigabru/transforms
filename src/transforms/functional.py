"""
Functions for signal transformations

__author__: Tati Gabru

"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from typing import Iterable, List, Optional, Tuple
from scipy import signal
from scipy.signal import decimate, butter, sosfilt

"""
Time Domain

"""

def crop_last(signal: np.array, cut_size: int=14400) -> np.array:
    """    
    Crop the last N points of the signal

    Args:
        signal: (np.array) 1D signal
        cut_size: (int) takes only the last N points of the signal. Default = 14400 -> 60 min
                
    Output: (np.array) cut signals batch   
    """
    # cut the last signal part of length cut
    return signal[len(signal)-cut_size:]


def crop_first(signal: np.array, cut_size: int=14400) -> np.array:
    """    
    Crop the first  N points of the signal

    Args:
        signal: (np.array) 1D signal
        cut_size: (int) takes only the last N points of the signal. Default = 14400 -> 60 min
                
    Output: (np.array) cut signals batch   
    """
    # cut the last signal part of length cut
    return signal[cut_size:]


def crop_translate(signal: np.array, crop_size: int=7200, shift: Optional[int] = None) -> np.array:
    """
    Crop part of the signal with length of crop_size and translate by shift

    Args:
        signal: (np.array) 1D signal
        crop_size: (int) duration of sub-signal. Default = 7200 -> 1200 sec = 30 min
        shift: (int) the crop shift. Shift must be smaller than signal length - crop_size. 
                It is clipped between 0 and len(signal)-crop_size
                If None: random shift is applied within limits. Default = None         
        
    Output: (np.array) signal crop with translate
    """
    if crop_size >= len(signal): raise ValueError("Crop size must be smaller than the original signal length")
    if shift == None:
        shift = np.random.randint(0, len(signal)-crop_size)
    else:
        shift = np.clip(shift, 0, len(signal)-crop_size)

    return signal[shift:shift+crop_size] 


def horizontal_flip(signal: np.array) -> np.array:
    """
    Flip original signal horizontally

    Args:
        signal: (np.array) 1D signal
            
    Output: (np.array) flipped signal
    """
    signal = np.flip(signal, 0) # flip to start from the end  

    return signal   


def vertical_flip(signal: np.array) -> np.array:
    """
    Flip original signal vertically around a baseline

    Args:
        signal: (np.array) 1D signal
            
    Output: (np.array) flipped signal
    """
    baseline = np.nanmean(signal)
    # flip vertically 
    signal = -signal + 2*baseline

    return signal 


def shift_up(signal: np.array, max_shift: float = 1) -> np.array:
    """
    Shift all signal up by a random value from a uniform distribution, within a shift amplitude 
    
    Args:
        signal: (np.array) 1D signal
        max_shift: (float) the maximum up-shift. Default = 1
               
    Output: (np.array) shifted signal
    """
    # “continuous uniform” distribution [0.0 1.0)
    shift = np.random.rand(len(signal))

    return signal + shift*max_shift


def shift_down(signal: np.array, max_shift: float = 1) -> np.array:
    """
    Shift all signal up by a random value from a uniform distribution, within a shift amplitude 
    
    Args:
        signal: (np.array) 1D signal
        max_shift: (float) the maximum down-shift. Default = 1
               
    Output: (np.array) shifted signal
    """
    # “continuous uniform” distribution [0.0 1.0)
    shift = np.random.rand(len(signal))

    return signal - shift*max_shift


"""

Time domain noise

"""
def gauss_noise(signal: np.array, max_noise: float=1) -> np.array:
    """
    Add gaussian noise to the signal with defined amplitude
    
    Args:
        signal: (np.array) 1D signal
        max_noise: (float) maximum value of the noise              
        
    Output: (np.array) signal with added Gaussian noise
    """
    noise = np.random.randn(len(signal))*max_noise
        
    return signal + noise


def uniform_noise(signal: np.array, max_noise: float=1) -> np.array:
    """
    Add noise to the signal from thw uniform distribution with defined amplitude
    
    Args:
        signal: (np.array) 1D signal
        max_noise: (float) maximum value of the noise              
        
    Output: (np.array) signal with added uniform noise
    """
    noise = np.random.rand(len(signal)) - 0.5
        
    return signal + noise*max_noise


def window_warp(signal: np.array, window_size: int = 4800, scale: float = 0.5) -> np.array:
    """
    Selects a random time range, then compresses (downsample) or extends (up sample) it, 
    while keeps other time range unchanged. 
    This change the total length of the original time series. 
    It should conducted along with window cropping that uses and downsampling/upsampling.
    """
    signal = signal
    return signal


def random_outliers(signal: np.array, min_value: Optional[float], max_value: Optional[float], number: int = 10) -> np.array:
    """
    Injects outliers from the uniform distribution into the signal, replacing the original values at random locations
    """
    if not min_value: min_value = np.nanmean(signal[signal != 0])
    if not max_value: max_value = 1.2*min_value
    if max_value < min_value: raise ValueError("max value for outliers should exceed minimum value")

    outliers = (max_value - min_value + 1)*np.random.rand(number) + min_value
    locations = np.random.randint(0, len(signal), number)
    signal[locations] = outliers

    return signal


"""
Frequency domain

"""
def signal_fft(signal: np.array) -> np.array:
    """    
    Gets FFT of the signal
    Assumes uniformly sampled signal with fs sampling frequency

    """


    return freqs
    

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


def upsample():

    return signal


"""

Frequency filters

"""
def _butter_lowpass(cutoff: float, fs: float = 4, order: int = 5) -> np.array:
    """
    Butterworth digital filter design.
    Design an Nth-order bandpass Butterworth filter and return the filter coefficients.

    Args:
        cutoff: (float) high frequency cut-off
        fs: (int) sampling frequency. Default = 4 Hz
        order: (int) filter order. Default = 5

    Output: (ndarray) second-order sections representation of the IIR filter
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    sos = butter(order, norm_cutoff, btype="lowpass", analog=False, output="sos")
    
    return sos 

    
def _butter_highpass(cutoff: float, fs: float = 4, order: int = 5) -> np.array:
    """
    Butterworth digital filter design.
    Design an Nth-order bandpass Butterworth filter and return the filter coefficients.

    Args:
        cutoff: (float) low frequency cut-off
        fs: (int) sampling frequency. Default = 4 Hz
        order: (int) filter order. Default = 5

    Output: (ndarray) second-order sections representation of the IIR filter
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    sos = butter(order, norm_cutoff, btype="highpass", analog=False, output="sos")
    
    return sos 


def _butter_bandpass(lowcut: float, highcut: float, fs: float = 4, order: int = 5) -> np.array:
    """
    Butterworth digital and analog filter design.
    Design an Nth-order bandpass Butterworth filter and return the filter coefficients.

    Args:
        lowcut: (float) low cut-off frequency 
        highcut: (float) high cut-off frequency 
        fs: (int) sampling frequency. Default = 4 Hz
        order: (int) filter order. Default = 5

    Output: (ndarray) second-order sections representation of the IIR filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    
    return sos     


def bandpass_filter(signal: np.array, lowcut: float, highcut: float, fs: float = 4, order: int = 5):
    """
    Apply Butterworth bandpath filter to a signal.
    
    Args:
        signal: (np.array) 1D signal
        lowcut: (float) low cut-off frequency 
        highcut: (float) high cut-off frequency 
        fs: (int) sampling frequency. Default = 4 Hz
        order: (int) filter order. Default = 5

    Output: (ndarray) filtered signal
    """
    sos = _butter_bandpass(lowcut, highcut, fs, order)
    filtered = sosfilt(sos, signal).astype(np.float32)
    
    return filtered     


def lowpass_filter(signal: np.array, cutoff: float, fs: float = 4, order: int = 5):
    """
    Apply Butterworth bandpath filter to a signal.
    
    Args:
        signal: (np.array) 1D signal
        lowcut: (float) low cut-off frequency 
        highcut: (float) high cut-off frequency 
        fs: (int) sampling frequency. Default = 4 Hz
        order: (int) filter order. Default = 5

    Output: (ndarray) filtered signal
    """
    sos = _butter_lowpass(cutoff, fs, order)
    filtered = sosfilt(sos, signal).astype(np.float32)
    
    return filtered


def highpass_filter(signal: np.array, cutoff: float, fs: float = 4, order: int = 5):
    """
    Apply Butterworth bandpath filter to a signal.
    
    Args:
        signal: (np.array) 1D signal
        lowcut: (float) low cut-off frequency 
        highcut: (float) high cut-off frequency 
        fs: (int) sampling frequency. Default = 4 Hz
        order: (int) filter order. Default = 5

    Output: (ndarray) filtered signal
    """
    sos = _butter_highpass(cutoff, fs, order)
    filtered = sosfilt(sos, signal).astype(np.float32)
    
    return filtered


def freq_amplitude_noise():

    return signal


def phase_noise():

    return signal


def doppler_shift():

    return signal
    



def test_filters():    
    filtered = apply_bandpass_filter(signal, lowcut=0.3, highcut=10, fs = 4, order = 5)
    ax2.plot(t, filtered)
    ax2.set_title('After 15 Hz high-pass filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.show()