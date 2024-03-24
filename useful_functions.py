# **************************************************************************** #
#                                                                              #
#    useful_functions.py                                                       #
#                                                                              #
#    By: MarcoGuzzoC <marco.guzzocholet@ensea.fr                               #
#                                                                              #
#    Created: 2024/03/14 16:20:03 by MarcoGuzzoC                               #
#    Updated: 2024/03/14 19:28:23 by MarcoGuzzoC                               #
#                                                                              #
# **************************************************************************** #

import librosa as lib

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import random as rd
import pickle
import scipy
from scipy.signal import hilbert
import time
from datetime import timedelta as td

import torch
from torch.utils.data import DataLoader

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


########## Miscellaneaous ##########

def get_time_stamp():
  secondsSinceEpoch = time.time()
  timeObj = time.localtime(secondsSinceEpoch)
  x = ('%d_%d_%d_%d%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min))
  return x

def plot_response(w:np.ndarray, h:filter, title:str):
    """
    Plot the frequency response of a filter.

    Parameters
    ----------
    w : np.ndarray
        Frequency array
    h : filter
        Filter used
    title : str
        Graph title
    """
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)


def lowpass(data: np.ndarray, cutoff: int, sample_rate: int, poles: int = 5)->np.ndarray:
    """
    Apply a low-pass filter to a sequence.

    Parameters
    ----------
    data : np.ndarray
        Input signal
    cutoff : int
        Cutoff frequency
    sample_rate : int
        Sample rate of the input signal
    poles : int, optional
        Number of poles (i.e. order of the filter), by default 5

    Returns
    -------
    filtered_data : np.ndarray
        Filtered signal
    """
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def f_high(y:np.ndarray,sr:int,cutoff:int)->np.ndarray:
    """
    Apply a high pass filter.

    Parameters
    ----------
    y : np.ndarray
        Input signal
    sr : int
        Sampling rate of the input file
    cutoff : int
        Cutoff frequency of the filter

    Returns
    -------
    yf : np.ndarray
        Filtered data.
    """
    b,a = scipy.signal.butter(10, cutoff/(sr/2), btype='highpass')
    yf = scipy.signal.lfilter(b,a,y)
    return yf


########## Feature extraction ##########

torch.random.manual_seed(0)

def plot_waveform(waveform:torch.tensor, sr:int, title="Waveform", ax=None):
    """
    Allow to plot waveform
    
    Parameters
    ----------
    waveform : torch.tensor
        Input signal
    sr : int
        Sampling rate of the input
    title : str, optional
        Title of the graph, by default "Waveform"
    ax : _type_, optional
       Specified axis, by default None
    """

    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram:torch.tensor, title=None, ylabel="freq_bin", ax=None):
    """
    Allow to plot spectrogram.

    Parameters
    ----------
    specgram : torch.tensor
        Input signal
    title : str, optional
        Title of the graph, by default None
    ylabel : str, optional
        Y-axis label, by default "freq_bin"
    ax : _type_, optional
        Specified axis, by default None
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(lib.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):

    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

def plot_pitch(waveform:torch.tensor, sr:int, pitch:torch.tensor):
    """
    Plot the pitch value during the entire sample.

    Parameters
    ----------
    waveform : torch.tensor
        Input data
    sr : int
        Sampling rate of the input
    pitch : torch.tensor
        Calculated pitch of the input
    """

    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="blue")

    axis2.legend(loc=0)

def plot_enveloppe(waveform:torch.tensor,sr:int):
    """
    Plot the enveloppe of the input signal

    Parameters
    ----------
    waveform : torch.tensor
        Input data
    sr : int
        Sampling rate of the input
    """
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Enveloppe Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    sig = waveform[0]
    analytical_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytical_signal)

    smooth_env = lowpass(amplitude_envelope,50,sr)

    axis.plot(time_axis, smooth_env, linewidth = 0.5, color="green", label="Enveloppe")
    
    axis.legend(loc=0)


########## Preprocess pipeline ##########
    
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


def data_augmentation(data:list, sampling_rate:int, t:int, pitch_factor:int, noise_factor:float, type:str)->list:
    """
    Allows to increase the size of the dataset for better results

    Args:
        data (list): List of array containing signal magnitude or spectrogram
        t (int) : Length of slices in seconds
        sampling_rate (int): Value of the data sampling rate
        pitch_factor (int): Desired pitch factor (number of semitones added) (0<_<12)
        noise_factor (float): Coefficient of the noise
        type (str) : Type of data
    """
    sr = sampling_rate

    array_of_noisy_signals = noising(np.array(data),noise_factor)
    list_of_noisy_slices = slices(array_of_noisy_signals,t)
    list_of_noisy_spectrograms = gen_spectrogramm(list_of_noisy_slices,sr)

    array_of_pitched_signals = pitch_mod(np.array(data),sr,pitch_factor)
    list_of_pitched_slices = slices(array_of_pitched_signals,t)
    list_of_pitched_spectrograms = gen_spectrogramm(list_of_pitched_slices,sr)

    if type=="spec":
        data.extend(list_of_noisy_spectrograms)
        data.extend(list_of_pitched_spectrograms)
    else:
        data.extend(list_of_noisy_slices)
        data.extend(list_of_pitched_slices) 

    return data


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

    hl = 864 # number of samples per time-step in spectrogram
    hi = 256 # Height of image

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

    return spec


def data_wave_generation(list_of_wavefile:list, isAugmented:bool)->list:
    """
    Generate wave files

    """
    list_of_signals = []
    for e in list_of_wavefile:
        (s, sr, t) = load_from_file(e)
        list_of_signals.extend(s)
    
    array_of_signals = np.asarray(list_of_signals)
    list_of_slices = slices(array_of_signals,t)

    sound = []
    sound.extend(list_of_slices)
    if isAugmented:
        sound = data_augmentation(sound,sr,t,3,0.1,"wave")

    # rd.shuffle(sound)
    return sound

########## Denoising ##########

# Inspired from https://github.com/timsainb/noisereduce


def _stft(y, n_fft, hop_length, win_length):
    return lib.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return lib.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return lib.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return lib.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram_den(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.plasma,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """
    Remove noise from audio based upon a clip containing only noise

    Parameters:
    ----------------
    audio_clip : np.ndarray
        The first parameter, noisy audio.
    noise_clip : np.ndarray
        The second parameter, full noise similar to the original.
    n_grad_freq : int 
        How many frequency channels to smooth over with the mask.
    n_grad_time : int 
        How many time channels to smooth over with the mask.
    n_fft : int 
        Number audio of frames between STFT columns.
    win_length : int 
        Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
    hop_length : int
        Number audio of frames between STFT columns.
    n_std_thresh : int 
        How many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
    prop_decrease : float 
        To what extent should you decrease noise (1 = all, 0 = none)
    visual : bool 
        Whether to plot the steps of the algorithm

    Output:
    ------------
    recoverred_signal : array
        The recovered signal with noise subtracted

    """

    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram_den(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram_den(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram_den(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram_den(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram_den(recovered_spec, title="Recovered spectrogram")
    return recovered_signal





def generate_sinewave(n:int, duration:int, freq:int = 440)->np.ndarray:
    """
    Generate a n-long array of sine waves

    Parameters
    ----------
    n : int
        Length of our data, number of generated signal
    duration : int
        Duration of one sample
    freq : int
        Frequency of the signal

    Returns
    -------
    sine_wave_array : np.ndarray
        Array of n generated signal
    """
    li = []
    sr = 22050
    samples = np.linspace(0, duration, int(sr*duration), endpoint=False)
    for i in range(n):
        sig = np.sin(2*np.pi*(freq+np.random.randint(500))*samples)
        li.append(sig)
    return np.array(li)


def generate_squarewave(n:int, duration:int, freq:int=440, duty:float=0.5)->np.ndarray:
    """
    Generate a n-long array of square waves

    Parameters
    ----------
    n : int
        Length of our data, number of generated signal
    duration : int
        Duration of one sample
    freq : int
        Frequency of the signal

    Returns
    -------
    square_wave_array : np.ndarray
        Array of n generated signal
    """
    li = []
    sr = 22050
    samples = np.linspace(0, duration, int(sr*duration), endpoint=False)

    for i in range(n):
        pwm = scipy.signal.square(2 * np.pi * (freq+np.random.randint(500)) * samples,duty=duty)
        li.append(pwm)
    return np.array(li)

def generate_sawtooth(n:int, duration:int, freq:int=440)->np.ndarray:
    """
    Generate a n-long array of sawtooth waves

    Parameters
    ----------
    n : int
        Length of our data, number of generated signal
    duration : int
        Duration of one sample
    freq : int
        Frequency of the signal

    Returns
    -------
    sawtooth_wave_array : np.ndarray
        Array of n generated signal
    """
    li = []
    sr = 22050
    samples = np.linspace(0, duration, int(sr*duration), endpoint=False)

    for i in range(n):
        saw = scipy.signal.sawtooth(2*np.pi*(freq+np.random.randint(500))*samples)    
        li.append(saw)
    return np.array(li)

def generate_trianglewave(n:int,duration:int,freq:int)->np.ndarray:
    """
    Generate a n-long array of triangle waves

    Parameters
    ----------
    n : int
        Length of our data, number of generated signal
    duration : int
        Duration of one sample
    freq : int
        Frequency of the signal

    Returns
    -------
    triangle_wave_array : np.ndarray
        Array of n generated signal
    """
    li = []
    sr = 22050
    samples = np.linspace(0, duration, int(sr*duration), endpoint=False)

    for i in range(n):
        saw = scipy.signal.sawtooth(2*np.pi*(freq+np.random.randint(500))*samples,width=0.5)    
        li.append(saw)
    return np.array(li)

def modulated_sine(n:int,duration:int,freq_carrier:int=10,freq_in:int=440)->np.ndarray:
    """
    Generate a n-long vector of a sine at frequency freq_in carried by a cosine at frequency freq_carrier

    Parameters
    ----------
    n : int
        Length of our data, number of generated signal
    duration : int
        Duration of one sample
    freq_carrier : int, optional
        Frequency of the carrier wave, by default 10
    freq_in : int, optional
        Frequency of the carried wave, by default 440

    Returns
    -------
    modulated_sine_array : np.ndarray
        Array of n generated signal
    """
    li = []
    sr = 22050
    samples = np.linspace(0, duration, int(sr*duration), endpoint=False)

    for i in range(n):
        sig = np.sin(2 * np.pi * (freq_in+np.random.randint(500)) * samples) * np.cos(2* np.pi * (freq_carrier+np.random.randint(10)) * samples)
        li.append(sig)
    return np.array(li)