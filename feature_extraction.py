# feature_extraction.py
# Extracts features from audio files for use in training/testing/using a neural network for speech recognition
# Author: Ben Tomsett

import glob
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.fft import dct


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


def magnitude_spectrum(frames):
    """
    Calculates the magnitude spectrum of each frame in a set of frames. Removes the redundant half of the spectrum.
    :param frames: A two-dimensional array containing the frames to be windowed
    :return: A two-dimensional array containing the magnitude spectrum of each frame
    """

    mag_spec = np.abs(np.fft.fft(frames))
    return mag_spec[:, 0:int(len(mag_spec[0]) / 2)]


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)


def create_linear_filters(n_filters, n_fft):
    samples_per_channel = int(n_fft / n_filters)

    fbank = np.zeros((n_filters, n_fft))

    for i in range(n_filters):
        fbank[i, i * samples_per_channel:(i + 1) * samples_per_channel] = 1

    return fbank


def create_mel_filters(num_filters, num_samples, sample_rate):
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate // 2)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # +2 for the edges
    hz_points = mel_to_hz(mel_points)

    indices = np.floor((num_samples + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, int(np.floor(num_samples / 2))))

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
    return np.log10(signal)


filters = create_mel_filters(40, 320, 16000)

for audio_file in sorted(glob.glob("audio/training/*.wav")):
    r, sr = sf.read(audio_file)

    print(f'Processing {Path(audio_file).stem}...')

    frames = framed_signal(r, sr)
    windowed = windowed_signal(frames)
    mag_spec = magnitude_spectrum(windowed)
    filtered_signal = np.dot(filters, mag_spec.T)
    log = log_signal(filtered_signal)
    mfcc = dct(log, type=2, axis=1, norm='ortho')[:, 1: (12 + 1)]

    np.save(f'mfccs/training/{Path(audio_file).stem}.npy', mfcc)

    print("Done.")
