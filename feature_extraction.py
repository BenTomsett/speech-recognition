# feature_extraction.py
# Extracts features from audio files for use in training/testing/using a neural network for speech recognition
# Author: Ben Tomsett

import numpy as np
from scipy.fft import dct
import soundfile as sf


def framed_signal(signal, sample_rate, frame_length=20, overlap=10):
    """
    Splits a signal into frames of a specified length, with a specified overlap.
    :param signal: A one-dimensional array containing the audio signal to be split into frames
    :param sample_rate: The sample rate of the audio signal
    :param frame_length: The length of each frame in milliseconds
    :param overlap: The overlap between each frame in milliseconds
    :return: A two-dimensional array containing the frames of the signal
    """

    samples_per_frame = int(frame_length / 1000 * sample_rate)
    overlap_samples = int(overlap / 1000 * sample_rate)

    step_size = samples_per_frame - overlap_samples
    num_frames = int(np.ceil((len(signal) - samples_per_frame) / step_size)) + 1

    frames = []

    for i in range(num_frames):
        start = i * step_size
        end = start + samples_per_frame
        frame = signal[start:end]

        # Pads the last frame with zeroes if it is shorter than the rest
        if len(frame) < samples_per_frame:
            frame = np.append(frame, np.zeros(samples_per_frame - len(frame)))
        frames.append(frame)

    return np.array(frames)


def windowed_signal(frames):
    """
    Applies a Hamming window to each frame in a set of frames.
    :param frames: A two-dimensional array containing the frames to be windowed
    :return: A two-dimensional array containing the windowed frames
    """

    return frames * np.hamming(len(frames[0]))


def power_spectrum(frames, NFFT):
    """
    Calculates the power spectrum of each frame in a set of frames.
    :param frames: A two-dimensional array containing the frames to be windowed
    :param NFFT: The number of samples to be used in the FFT
    :return: A two-dimensional array containing the power spectrum of each frame
    """
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames


def hz_to_mel(hz):
    """
    Converts a frequency in Hz to a frequency in the mel scale.
    :param hz: The frequency in Hz to be converted
    :return: The frequency in the mel scale
    """
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    """
    Converts a frequency in the mel scale to a frequency in Hz.
    :param mel: The frequency in the mel scale to be converted
    :return: The frequency in Hz
    """
    return 700 * (10**(mel / 2595) - 1)


def create_linear_filters(n_filters, n_fft):
    samples_per_channel = int(n_fft / n_filters)

    fbank = np.zeros((n_filters, n_fft))

    for i in range(n_filters):
        fbank[i, i * samples_per_channel:(i + 1) * samples_per_channel] = 1

    return fbank


def create_mel_filters(num_filters, n_fft, sample_rate):
    """
    Creates a set of mel filters to be applied to a magnitude spectrum.
    :param num_filters: The number of filters to create
    :param n_fft: The number of samples to be used in the FFT
    :param sample_rate: The sample rate of the audio signal
    :return: A two-dimensional array containing the mel filters
    """
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate // 2)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # +2 for the edges
    hz_points = mel_to_hz(mel_points)

    indices = np.floor((n_fft // 2 + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, int(np.floor(n_fft // 2 + 1))))

    for i in range(1, num_filters + 1):
        left = indices[i - 1]
        center = indices[i]
        right = indices[i + 1]

        for j in range(left, center):
            filters[i - 1, j] = (j - indices[i - 1]) / (indices[i] - indices[i - 1])
        for j in range(center, right):
            filters[i - 1, j] = (indices[i + 1] - j) / (indices[i + 1] - indices[i])

    return filters


def log_signal(signal):
    """
    Calculates the log of each value in a signal.
    :param signal: The signal to be logged
    :return: The logged signal
    """
    return np.log10(signal)


def dct_signal(signal):
    """
    Calculates the discrete cosine transform of a signal.
    :param signal: The signal to be transformed
    :return: The transformed signal
    """
    return dct(signal, type=2, axis=1, norm='ortho')


def generate_mfccs(audio_signal, sample_rate, filters, ceps):
    """
    Generates the mel-frequency cepstral coefficients of an audio signal.
    :param audio_signal: A one-dimensional array containing the audio signal
    :param sample_rate: The sample rate of the audio signal
    :param filters: A two-dimensional array containing the mel filters to be applied to the signal
    :param ceps: The number of cepstral coefficients to be returned
    :param nfft: The number of samples to be used in the FFT
    :return: A two-dimensional array containing the mel-frequency cepstral coefficients of the signal
    """
    frames = framed_signal(audio_signal, sample_rate)
    windowed = windowed_signal(frames)
    pow_spec = power_spectrum(windowed, 512)
    filtered_signal = np.dot(filters, pow_spec.T)
    log = log_signal(filtered_signal)
    mfcc = dct(log, type=2, axis=1, norm='ortho')

    return mfcc[:, 1: (ceps + 1)]  # Discards the first coefficient

