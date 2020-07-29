"""
Add Gaussian Noise to signal

__author__: Tati Gabru

"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import pandas as pd
from typing import Iterable, List, Optional, Tuple
from scipy import signal
from functional import gauss_noise


class GaussNoise(BasicTransform):
    """
    Add Gaussian noise to the signal"""

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, samples, sample_rate):
        noise = np.random.randn(len(samples)).astype(np.float32)
        samples = samples + self.parameters["amplitude"] * noise
        return samples
