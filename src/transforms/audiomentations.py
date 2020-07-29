"""
Transforms from audiomentations libruary
https://github.com/iver56/audiomentations/tree/master/audiomentations

"""
import functools
import random

import librosa
import numpy as np
from scipy.signal import butter, sosfilt, convolve


class BasicTransform:
    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.are_parameters_frozen = False

    def __call__(self, samples, sample_rate):
        if not self.are_parameters_frozen:
            self.randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and len(samples) > 0:
            return self.apply(samples, sample_rate)
        return samples

    def randomize_parameters(self, samples, sample_rate):
        self.parameters["should_apply"] = random.random() < self.p

    def apply(self, samples, sample_rate):
        raise NotImplementedError

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        return self.parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False


class FrequencyMask(BasicTransform):
    """
    Mask some frequency band on the spectrogram.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """

    def __init__(self, min_frequency_band=0.0, max_frequency_band=0.5, p=0.5):
        """
        :param min_frequency_band: Minimum bandwidth, float
        :param max_frequency_band: Maximum bandwidth, float
        :param p:
        """
        super().__init__(p)
        self.min_frequency_band = min_frequency_band
        self.max_frequency_band = max_frequency_band

    def __butter_bandstop(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="bandstop", output="sos")
        return sos

    def __butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.__butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data).astype(np.float32)
        return y

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["bandwidth"] = random.randint(
                self.min_frequency_band * sample_rate // 2,
                self.max_frequency_band * sample_rate // 2,
            )
            self.parameters["freq_start"] = random.randint(
                16, sample_rate // 2 - self.parameters["bandwidth"] - 1
            )

    def apply(self, samples, sample_rate):
        bandwidth = self.parameters["bandwidth"]
        freq_start = self.parameters["freq_start"]
        samples = self.__butter_bandstop_filter(
            samples, freq_start, freq_start + bandwidth, sample_rate, order=6
        )
        return samples


class TimeMask(BasicTransform):
    """
    Mask some time band on the spectrogram.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """

    def __init__(self, min_band_part=0.0, max_band_part=0.5, fade=False, p=0.5):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Float.
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Float.
        :param fade: Bool, Add linear fade in and fade out of the silent part.
        :param p:
        """
        super().__init__(p)
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[0]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(self, samples, sample_rate):
        new_samples = samples.copy()
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t)
        if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            mask[0:fade_length] = np.linspace(1, 0, num=fade_length)
            mask[-fade_length:] = np.linspace(0, 1, num=fade_length)
        new_samples[t0 : t0 + t] *= mask
        return new_samples


class AddGaussianSNR(BasicTransform):
    """Add gaussian noise to the samples with random Signal to Noise Ratio (SNR)"""

    def __init__(self, min_SNR=0.001, max_SNR=1.0, p=0.5):
        """
        :param min_SNR: Minimum signal-to-noise ratio
        :param max_SNR: Maximum signal-to-noise ratio
        :param p:
        """
        super().__init__(p)
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            std = np.std(samples)
            self.parameters["noise_std"] = random.uniform(
                self.min_SNR * std, self.max_SNR * std
            )

    def apply(self, samples, sample_rate):
        noise = np.random.normal(
            0.0, self.parameters["noise_std"], size=len(samples)
        ).astype(np.float32)
        return samples + noise


class AddGaussianNoise(BasicTransform):
    """Add gaussian noise to the samples"""

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["amplitude"] = random.uniform(
                self.min_amplitude, self.max_amplitude
            )

    def apply(self, samples, sample_rate):
        noise = np.random.randn(len(samples)).astype(np.float32)
        samples = samples + self.parameters["amplitude"] * noise
        return samples


class TimeStretch(BasicTransform):
    """Time stretch the signal without changing the pitch"""

    def __init__(self, min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.5):
        super().__init__(p)
        assert min_rate > 0.1
        assert max_rate < 10
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.leave_length_unchanged = leave_length_unchanged

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            """
            If rate > 1, then the signal is sped up.
            If rate < 1, then the signal is slowed down.
            """
            self.parameters["rate"] = random.uniform(self.min_rate, self.max_rate)

    def apply(self, samples, sample_rate):
        time_stretched_samples = librosa.effects.time_stretch(
            samples, self.parameters["rate"]
        )
        if self.leave_length_unchanged:
            # Apply zero padding if the time stretched audio is not long enough to fill the
            # whole space, or crop the time stretched audio if it ended up too long.
            padded_samples = np.zeros(shape=samples.shape, dtype=samples.dtype)
            window = time_stretched_samples[: samples.shape[0]]
            actual_window_length = len(window)  # may be smaller than samples.shape[0]
            padded_samples[:actual_window_length] = window
            time_stretched_samples = padded_samples
        return time_stretched_samples
