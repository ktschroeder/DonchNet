import os, pickle
import unicodedata, re
import config
import numpy as np
from numpy import *


def makeSafeFilename(name):  # taken from Django's slugify function
    # Normalizes string, converts to lowercase, removes non-alpha characters, and converts spaces to underscores.
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name.lower())
    return re.sub(r'[-\s]+', '_', name).strip('-_')

def addContext(x):
    # x.reshape(())   (config.audioLengthMaxSeconds, 40)?

    padding = np.full((1,40), -500, dtype=float32)
    context = 7
    window = 2*context + 1 # prepend and append context
    # want to create input_shape=(max_sequence_length,15,40) from (max_sequence_length, 40)
    out = np.zeros((len(x),window,40))
    for i in range (len(x)):
        bookended = np.zeros((window,40), dtype=float32)
        for j in range (context*-1, context+1):
            indexToGet = i + j  # if at start of audio this is negative in first half, if at end this is out of bounds positive in second half
            if indexToGet < 0 or indexToGet >= len(x):
                bookended[j + context] = padding
            else:
                bookended[j + context] = x[i + j]  # TODO this implementation uses 14x more storage than perhaps necessary (but will speed up training over time)
        out[i] = bookended
    return out




# @misc{fayek2016,
#       title   = "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between",
#       author  = "Haytham M. Fayek",
#       year    = "2016",
#       url     = "https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"
#     }
def makeFeats(file, targetDir = None, songFolder = None):  # './sample_maps/1061593 katagiri - Urushi/audio.wav'
    import numpy
    import scipy.io.wavfile
    from scipy.fftpack import dct

    sample_rate, signal = scipy.io.wavfile.read(file)  # TODO .mp3/.ogg files, also inconsistent bitrates? Probably best to pre-process, convert all to .wav. Manage bitrates?
    signal = signal[0:int(config.audioLengthMaxSeconds * sample_rate)]  # Keep the first audioLengthMaxSeconds seconds
    #
    pre_emphasis = 0.97
    signalLength = len(signal)
    # emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])  # original line, this doubles length of signal, seemingly in error perhaps for testing
    emphasized_signal = numpy.append(signal[0:1], signal[1:] - pre_emphasis * signal[:-1])  # my modified line

    emphasized_signal = signal
    emphasized_signal[1:] = signal[1:] - pre_emphasis * signal[:-1]

    # print(len(signal[0]))
    # print(len(signal[1:] - pre_emphasis * signal[:-1]))

    # print(signalLength, len(emphasized_signal))
    assert len(emphasized_signal) == signalLength
    # assert 1 == 0
    # emphasized_signal = signal  # see article but can maybe just skip this anyway
    #
    frame_size = 0.046  # was 0.025: 25 ms
    frame_stride = 0.01  # 10 ms

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    # print(len(signal), signal_length, frame_length, frame_step, num_frames) 

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    #
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    #
    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    #
    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    #
    num_ceps = 12

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  #*
    #
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    #
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)  # TODO one thing to test in model will be mel bands versus mfcc
    #
    # title, diff = getTitleAndDiff(songFolder, jsons)

    filter_banks = addContext(filter_banks)  # new shape is (max_sequence_length,15,40)

    
    
    if not targetDir:  # if no target directory then return the actual filter banks
        return filter_banks

    filepath = os.path.join(targetDir, songFolder) + '.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(filter_banks, f)
    return 1