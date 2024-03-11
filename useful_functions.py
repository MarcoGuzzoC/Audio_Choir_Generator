import os
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
import librosa as lib
import random as rd
import pickle
import matplotlib.pyplot as plt
import scipy

def load_from_file(file: str):
    
    """
    Import data from file.

    Parameters:
    --------------------------
    file : str
        The .wave file you want to import

    Outputs:
    --------------------------
    sig : ndarray
        Array containing signal amplitude value
    sr : int
        Value of the sampling rate
    t : int
        5*sr to generate signal of lenght 5s 
    """

    sig, sr = lib.load(file)
    t = 5*sr
    return sig, sr, t

def slices(sig, t: int):
    """
    Make audio slices.

    Parameters:
    --------------------------
    sig : ndarray
        Array containing signal amplitude value
    t : int
        Multiple of the sampling rate to have a specific time length

    Outputs:
    --------------------------
    li : list[ndarray]
        List of different slices
    """


    li = []
    for i in range(0,len(sig)-t,t):
        li.append(np.asarray(sig[i:i+t]))
    return li

def pitch_mod(data, sampling_rate:int, pitch_factor:int):

    """
    Modulate the signal amplitude (for data augmentation purposes)

    Parameters:
    --------------------------
    data: ndarray
        Array containing signal amplitude value
    sampling_rate : int
        Sampling rate of the signal
    pitch_factor : int
        Number of semitones used for modulation
    
    Outputs:
    --------------------------
    pitch_shifted_data : ndarray
        Array containing modulated signal amplitude value
    """

    return lib.effects.pitch_shift(y=data, bins_per_octave=12, sr=sampling_rate, n_steps=pitch_factor)

def noising(data,noise_factor:float):

    """
    Add noise to the signal (for data augmentation purposes)

    Parameters:
    --------------------------
    data: ndarray
        List containing signal amplitude value
    noise_factor : float
        Hyperparameter used to specify the importance of the noise
    
    Outputs:
    --------------------------
    noised_data : ndarray
        Array containing noisy signal amplitude value
    """

    noise = np.random.randn(len(data))
    noisy_data = data + noise_factor * noise
    # Cast back to same data type
    noisy_data = noisy_data.astype(type(data[0]))
    return np.asarray(noisy_data)

def gen_spectrogramm(li: list, sr: int):

    """
    Generate mel-spectrograms from slices.

    Parameters:
    --------------------------
    li: list[ndarray]
        List containing slices
    sr : int
        Sampling rate
    
    Outputs:
    --------------------------
    spec : list[ndarray]
        List of generated spectrograms
    """    

    hl = 512 # number of samples per time-step in spectrogram
    hi = 216 # Height of image
    wi = 384 # Width of image

    spec = []
    for el in li:
        S = lib.feature.melspectrogram(y=el, sr=sr, n_mels=hi,hop_length=hl)
        spec.append(S)

    return spec

def preprocess_pipeline(list_of_wavefile:list)->list:
    
    list_of_signals = []
    for e in list_of_wavefile:
        (s, sr, t) = load_from_file(e)
        list_of_signals.extend(s)
    
    array_of_signals = np.asarray(list_of_signals)
    list_of_slices = slices(array_of_signals,t)
    list_of_spectrograms = gen_spectrogramm(list_of_slices,sr)

    array_of_noisy_signals = noising(list_of_signals,0.1)
    list_of_noisy_slices = slices(array_of_noisy_signals,t)
    list_of_noisy_spectrograms = gen_spectrogramm(list_of_noisy_slices,sr)

    array_of_pitched_signals = pitch_mod(array_of_signals,sr,3)
    list_of_pitched_slices = slices(array_of_pitched_signals,t)
    list_of_pitched_spectrograms = gen_spectrogramm(list_of_pitched_slices,sr)

    spec = []
    spec.extend(list_of_spectrograms)
    spec.extend(list_of_noisy_spectrograms)
    spec.extend(list_of_pitched_spectrograms)
    rd.shuffle(spec)

    return spec

def data_wave_generation(list_of_wavefile:list)->list:
    """
    Generate wave files

    """

        
    list_of_signals = []
    for e in list_of_wavefile:
        (s, sr, t) = load_from_file(e)
        list_of_signals.extend(s)
    
    array_of_signals = np.asarray(list_of_signals)
    list_of_slices = slices(array_of_signals,t)
    array_of_noisy_signals = noising(list_of_signals,0.1)
    list_of_noisy_slices = slices(array_of_noisy_signals,t)
    array_of_pitched_signals = pitch_mod(array_of_signals,sr,3)
    list_of_pitched_slices = slices(array_of_pitched_signals,t)

    sound = []
    sound.extend(list_of_slices)
    sound.extend(list_of_noisy_slices)
    sound.extend(list_of_pitched_slices)

    return sound

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def plot_response(w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)
