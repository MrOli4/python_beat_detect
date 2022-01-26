import librosa
import matplotlib.pyplot as plt
import librosa.display

import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.fftpack
from librosa._cache import cache
from librosa import core
from librosa import util
from librosa.util.exceptions import ParameterError
from librosa.feature.spectral import melspectrogram
from librosa.feature import tempogram, fourier_tempogram
from librosa.onset import onset_strength
from librosa.core.fft import get_fftlib

__all__ = ["onset_detect", "onset_strength_multi"]  # So other functions can recognize

def peak_pick(onset_env, pre_max, post_max, pre_avg, post_avg, delta, wait):

    """
       Parameters
    ----------
    x         : np.ndarray [shape=(n,)]
        input signal to peak picks from

    pre_max   : int >= 0 [scalar]
        number of samples before ``n`` over which max is computed

    post_max  : int >= 1 [scalar]
        number of samples after ``n`` over which max is computed

    pre_avg   : int >= 0 [scalar]
        number of samples before ``n`` over which mean is computed

    post_avg  : int >= 1 [scalar]
        number of samples after ``n`` over which mean is computed

    delta     : float >= 0 [scalar]
        threshold offset for mean

    wait      : int >= 0 [scalar]
        number of samples to wait after picking a peak
    """

    # Ensure valid index types
    pre_max = int(np.ceil(pre_max)) # Ceil is just a round down
    post_max = int(np.ceil(post_max))
    pre_avg = int(np.ceil(pre_avg))
    post_avg = int(np.ceil(post_avg))
    wait = int(np.ceil(wait))

    # print(pre_max, post_max, pre_avg, post_avg, wait)

    # print("Onset env")
    # print(onset_env)

    # Get the maximum of the signal over a sliding window, how long is the sliding window tho?
    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max)) # Now is just 0 but different samples could mess with this
    # Using mode='constant' and cval = x.min() effectively truncates the sliding window at the boundaries
    # This next unit calculates the max over a certain size, determined by int(max_length), thus the premax, and postmax
    mov_max = scipy.ndimage.filters.maximum_filter1d(
        onset_env, int(max_length), mode="constant", origin=int(max_origin), cval=onset_env.min()
    )

    # print("After MovMax filter")
    # print(mov_max)

    # Get the mean of the signal over a sliding window
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    # Here, there is no mode which results in the behavior we want,
    # so we'll correct below.
    mov_avg = scipy.ndimage.filters.uniform_filter1d(
        onset_env, int(avg_length), mode="nearest", origin=int(avg_origin)
    )

    # print("Mov avg")
    # print(mov_avg)

    # Commanded out for ease of use, could return if necessary ------------------------------ #
    # # Correct sliding average at the beginning
    # n = 0
    # # Only need to correct in the range where the window needs to be truncated
    # while n - pre_avg < 0 and n < onset_env.shape[0]:  # Shape is the num of elements in each dimension
    #     # This just explicitly does mean(x[n - pre_avg:n + post_avg])
    #     # with truncation
    #     start = n - pre_avg
    #     start = start if start > 0 else 0
    #     mov_avg[n] = np.mean(onset_env[start : n + post_avg])
    #     n += 1
    #
    # # Correct sliding average at the end
    # n = onset_env.shape[0] - post_avg
    # # When post_avg > x.shape[0] (weird case), reset to 0
    # n = n if n > 0 else 0
    # while n < onset_env.shape[0]:
    #     start = n - pre_avg
    #     start = start if start > 0 else 0
    #     mov_avg[n] = np.mean(onset_env[start : n + post_avg])
    #     n += 1
    # Commanded out for ease of use, could return if necessary ------------------------------ #

    # First mask out all entries not equal to the local max, using the onset_env
    detections = onset_env * (onset_env == mov_max)

    # print("Detect peaks 1.0 thing, some filter of sorts")
    # print(detections)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            # print("i = ", i)
            # print("Other thing", last_onset+wait)

            peaks.append(i)
            # Save last reported onset
            last_onset = i

            # print("Last onset: ", last_onset)

    # print("Final peaks")
    # print(peaks)

    return np.array(peaks)

def calc_bpm(
        y = None,
        hop_length=512,
        onsets = None,
        sr=44100,
):
    time_stamps = core.samples_to_time(onsets, sr=sr)

    # print("Time stamps", time_stamps)

    avg_list = []  # Empty list

    for idx, val in enumerate(time_stamps):
        if idx < len(time_stamps)-1:
            difference = time_stamps[idx+1] - time_stamps[idx]
            avg_list.append(difference)

    # Cluster the difference onset data points based on difference
    sorting_list = []
    avg_factor = 0.05 # Change if needed

    # Already add the first item to provide a reference
    sorting_list.append([avg_list[0]])

    for idx, val in enumerate(avg_list[1:]):
        create_new = False

        for sorting_list_item in sorting_list:
            # if sorting_list_check[0] + avg_factor > avg_list[idx] < sorting_list_check[0] + avg_factor:
            if sorting_list_item[0] - avg_factor < avg_list[idx] < sorting_list_item[0] + avg_factor:

                sorting_list_item.append(avg_list[idx])
                # We use the first item out of the
                create_new = False
                break
            else:
                create_new = True

        if create_new:
            sorting_list.append([avg_list[idx]])

    print(sorting_list)

    # Determine the biggest data batch, longest array of difference
    longest_array = sorting_list[0]

    for index, item in enumerate(sorting_list[1:]):
        if index < len(sorting_list)-1:
            if len(sorting_list[index]) >= len(longest_array):
                longest_array = sorting_list[index]

    print("Array with longest list: ", longest_array)

    avg_time_val = np.average(longest_array)

    bpm = 60/avg_time_val

    return bpm

def onset_detect(
    y=None,
    sr=44100,
    onset_envelope=None,
    hop_length=512,

    units="frames",
    normalize=True,
    **kwargs,
):
    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset_strength_multi(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    if normalize:
        # Normalize onset strength function to [0, 1] range
        onset_envelope = onset_envelope - onset_envelope.min()
        # Max-scale with safe division
        onset_envelope /= np.max(onset_envelope) + util.tiny(onset_envelope)

    # Do we have any onsets to grab?
    if not onset_envelope.any() or not np.all(np.isfinite(onset_envelope)):
        onsets = np.array([], dtype=np.int)

    else:
        # These parameter settings found by large-scale search and are used by the peak picking algorithm,
        # could be adjusted for specific music genre
        kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
        kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
        kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
        kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("delta", 0.2)

        # Peak pick the onset envelope
        onsets = peak_pick(onset_envelope, **kwargs)

        # Optionally backtrack the events
        """
        if backtrack:
            if energy is None:
                energy = onset_envelope

            onsets = onset_backtrack(onsets, energy)
        """

    if units == "frames":
        pass
    elif units == "samples":
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == "time":
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError("Invalid unit type: {}".format(units))

    # Look at each of the beats after they are picked


    # We should use sample based thing for the onset allocation in the song
    sample_onsets = core.frames_to_samples(onsets, hop_length=hop_length)

    # print("Sample_onsets")
    # print(sample_onsets)

    offset = 10
    snippitsize = 4000

    kick_output = []

    for x in sample_onsets:
        begin_snippit = x
        if y is not None:
            audio_snippit = y[begin_snippit-offset:begin_snippit + snippitsize]

            # Next part is from https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python/27191172
            w = np.fft.fft(audio_snippit)
            freqs = np.fft.fftfreq(len(w))
            # print(freqs.min(), freqs.max())
            # (-0.5, 0.499975)

            # Find the peak in the coefficients
            idx = np.argmax(np.abs(w))
            freq = freqs[idx]
            freq_in_hertz = abs(freq * 41000)
            # print("Freq in Hz:", freq_in_hertz)

            if freq_in_hertz < 200 and freq_in_hertz > 80:
                # Now I am highly cherry-picking
                kick_output.append(x)

    # print("Kick Output")
    # print(kick_output)

    bpm = calc_bpm(onsets=kick_output, sr=sr)
    print("BPM: ", bpm)

    onsets = core.samples_to_frames(kick_output, hop_length=hop_length)

    return onsets


@cache(level=30)
def onset_strength_multi(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    channels=None,
    **kwargs,
):

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault("fmax", 11025.0)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, (int, np.integer)):
        raise ParameterError("lag must be a positive integer")

    if max_size < 1 or not isinstance(max_size, (int, np.integer)):
        raise ParameterError("max_size must be a positive integer")

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))

        # Convert to dBs
        S = core.power_to_db(S)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.

    if max_size == 1:
        ref = S
    else:
        ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=0)

    # Compute difference to the reference, spaced by lag
    onset_env = S[:, lag:] - ref[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if aggregate:
        onset_env = util.sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=0)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]), mode="constant")

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[:, : S.shape[1]]

    return onset_env[0]  # Only need to return one channel

def time_to_frames(times, sr=22050, hop_length=512, n_fft=None):
    # Convert the type

    samples = (np.asanyarray(times) * sr).astype(int) # Should work just fine

    offset = 0

    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.asanyarray(samples)

    frames = np.floor((samples - offset) // hop_length).astype(int)

    return frames

@cache(level=20)
def autocorrelate(y, max_size=None, axis=-1):
    """Bounded-lag auto-correlation

    Parameters
    ----------
    y : np.ndarray
        array to autocorrelate

    max_size  : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to ``y.shape[axis]`` (unbounded)

    axis : int
        The axis along which to autocorrelate.
        By default, the last axis (-1) is taken.

    """

    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    # Compute the power spectrum along the chosen axis
    # Pad out the signal to support full-length auto-correlation.
    fft = get_fftlib()
    powspec = np.abs(fft.fft(y, n=2 * y.shape[axis] + 1, axis=axis)) ** 2

    # Convert back to time domain
    autocorr = fft.ifft(powspec, axis=axis)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr = autocorr[tuple(subslice)]

    if not np.iscomplexobj(y):
        autocorr = autocorr.real

    return autocorr

# -- Rhythmic features -- #
def tempogram(
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    win_length=384,
    center=True,
    norm=np.inf,
):
    """
    Compute the tempogram: local autocorrelation of the onset strength envelope.

    [#] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    """
    # Returns a window, in this case, a Hann window
    ac_window = scipy.signal.get_window("hann", win_length, fftbins=True)
    onset_envelope = np.ascontiguousarray(onset_envelope)

    # Center the autocorrelation windows
    n = len(onset_envelope)

    if center:
        onset_envelope = np.pad(onset_envelope, int(win_length // 2), mode="linear_ramp", end_values=[0, 0])
        # This Numpy function call,

    # print("Onset after temp thingie")
    # print(onset_envelope)

    # Carve onset envelope into frames
    odf_frame = util.frame(onset_envelope, frame_length=win_length, hop_length=1)

    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[:, :n]

    # Window, autocorrelate, and normalize
    window_correlated_normalized = util.normalize(
        autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0), norm=norm, axis=0
    )

    return window_correlated_normalized

@cache(level=30)
def tempo(
    y=None,
    sr=44100,
    onset_envelope=None,
    hop_length=512,
    start_bpm=120,
    std_bpm=1.0,
    ac_size=8.0,
    max_tempo=180.0,
    aggregate=np.mean,
    prior=None,
):
    """Estimate the tempo (beats per minute)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of the time series

    onset_envelope    : np.ndarray [shape=(n,)]
        pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length of the time series

    start_bpm : float [scalar]
        initial guess of the BPM

    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution

    ac_size : float > 0 [scalar]
        length (in seconds) of the auto-correlation window

    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold

    aggregate : callable [optional]
        Aggregation function for estimating global tempo.
        If `None`, then tempo is estimated independently for each frame.

    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a pseudo-log-normal prior is used.
        If given, ``start_bpm`` and ``std_bpm`` will be ignored.

    Returns
    -------
    tempo : np.ndarray [scalar]
        estimated tempo (beats per minute)

    """

    if start_bpm <= 0:
        raise ParameterError("start_bpm must be strictly positive")

    win_length = time_to_frames(ac_size, sr=sr, hop_length=hop_length).item()

    tg = tempogram(
        y=y,
        sr=sr,
        onset_envelope=onset_envelope,
        hop_length=hop_length,
        win_length=win_length,
    )  # What actually is this?


    # Eventually, we want this to work for time-varying tempo
    if aggregate is not None:

        tg = np.mean(tg, axis=1, keepdims=True)  # Take the mean of all of the tgs

    # Get the BPM values for each bin, skipping the 0-lag bin
    bin_frequencies = np.zeros(int(tg.shape[0]), dtype=float)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, tg.shape[0]))
    bpms = bin_frequencies

    # Weight the autocorrelation by a log-normal distribution
    if prior is None:
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    else:
        logprior = prior.logpdf(bpms)

    # Kill everything above the max tempo
    if max_tempo is not None:
        max_idx = np.argmax(bpms < max_tempo)
        logprior[:max_idx] = -np.inf

    # Get the maximum, weighted by the prior
    # Using log1p here for numerical stability
    best_period = np.argmax(np.log1p(1e6 * tg) + logprior[:, np.newaxis], axis=0)

    return bpms[best_period]

# files = librosa.ex('brahms')
# y, sr = librosa.load(files)

# The code that is actually run
y, sr = librosa.load("audio/125bpm_kick16bit.wav", offset= 2, duration=2)

t = np.linspace(0, 1, 1000, False)
# Create the filter, since dance music is of importance simple low pass filter, with fcut at 200 Hz
sos = scipy.signal.butter(30, 200, 'lp', fs=sr, output='sos')
filtered_signal = scipy.signal.sosfilt(sos, y)  # Apply the filter

y = filtered_signal

tempo_lib, beats = librosa.beat.beat_track(y=y, sr=sr)

o_env = onset_strength_multi(filtered_signal, sr=sr)
onset_frames = onset_detect(y=y, onset_envelope=o_env, sr=sr)

times = librosa.times_like(o_env, sr=sr)

tempo = tempo(onset_envelope=o_env, sr=sr)

# print("Liberosa native tempo: ", tempo_lib)
# print("Tempo: ", tempo)

D = np.abs(librosa.stft(y))
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()

ax[1].plot(times, o_env, label='Onset strength')
ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
ax[1].legend()

plt.show()
