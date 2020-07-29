import numpy as np
import scipy
from scipy import stats
from scipy import signal
from scipy.signal import butter, iirnotch
from scipy.fftpack import fft, ifft
import pywt
from statsmodels.robust import mad


def high_pass_filter(signal, low_cutoff: float = 0.15, fs: int = 4):
    """
    High pass filter
    """
    
    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * fs
    norm_low_cutoff = low_cutoff / nyquist
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, signal)

    return filtered_sig


def maddest(signal: np.array) -> float:
    """
    Median Absolute Deviation
    Args:
        signal: (np.array) 1D signal

    Output: (float) Median Absolute Deviation
    """    
    return np.median(np.absolute(signal - np.median(signal)))


def wavelet_denoising(signal: np.array, threshold: Optional = None, wavelet: str = 'db4', level: int = 1) -> np.array:
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    # calculate a threshold
    sigma = threshold or maddest(coeff[-level])
    # changing this threshold also changes the behavior
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    filtered = pywt.waverec(coeff, wavelet, mode="per")
    
    return filtered

