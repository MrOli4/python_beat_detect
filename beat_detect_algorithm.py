import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.fftpack
from scipy.io.wavfile import write
from librosa._cache import cache
from librosa import core
from librosa import util
from librosa.util.exceptions import ParameterError
from librosa.feature.spectral import melspectrogram

__all__ = ["onset_detect", "onset_strength"]  # So other functions can recognize

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

    # Get the maximum of the signal over a sliding window
    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max)) # Now is just 0 but different samples could mess with this
    # Using mode='constant' and cval = x.min() effectively truncates the sliding window at the boundaries
    # This next unit calculates the max over a certain size, determined by int(max_length), thus the premax, and postmax
    mov_max = scipy.ndimage.filters.maximum_filter1d(
        onset_env, int(max_length), mode="constant", origin=int(max_origin), cval=onset_env.min()
    )

    # Get the mean of the signal over a sliding window
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    # Here, there is no mode which results in the behavior we want,
    # so we'll correct below.
    mov_avg = scipy.ndimage.filters.uniform_filter1d(
        onset_env, int(avg_length), mode="nearest", origin=int(avg_origin)
    )

    # First mask out all entries not equal to the local max, using the onset_env
    detections = onset_env * (onset_env == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    # print("Final peaks")
    # print(peaks)

    return np.array(peaks)

def calc_bpm(
        onsets = None,
        sr=44100,
):
    bpm_boundries = [60, 200]
    time_stamps = core.samples_to_time(onsets, sr=sr)

    # print("Time stamps", time_stamps)

    avg_list = []  # Empty list

    for idx, val in enumerate(time_stamps):
        if idx < len(time_stamps)-1:
            difference = time_stamps[idx+1] - time_stamps[idx]

            # Filter out the entries with an unlogical
            if 60/bpm_boundries[1] < difference < 60/bpm_boundries[0]:
                avg_list.append(difference)

    # Cluster the difference onset data points based on difference
    sorting_list = []
    avg_factor = 0.05  # Change if needed

    # print(avg_list)

    if avg_list:
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
            # First check if the length makes sense, more than x in a minute is weird

            if index < len(sorting_list)-1:
                if len(sorting_list[index]) >= len(longest_array):
                    longest_array = sorting_list[index]

        # print("Array with longest list: ", longest_array)

        avg_time_val = np.average(longest_array)

        print(avg_time_val)

        if avg_time_val != 0:
            bpm = 60/avg_time_val
        else:
            bpm = 0

    else:
        # There is no beat detected so the bpm will be set to 0
        print("No bpm could be detected")
        bpm = 0

    print(bpm)

    return bpm

def onset_detect(
    y=None,
    sr=44100,
    onset_envelope=None,
    hop_length=512,
    normalize=True,
    **kwargs,
):

    # Shift onset envelope up to be non-negative
    # a common normalization step to make the threshold more consistent
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
        kwargs.setdefault("delta", 0.3)

        # Peak pick the onset envelope
        onsets = peak_pick(onset_envelope, **kwargs)

    # We should use sample based thing for the onset allocation in the song
    sample_onsets = core.frames_to_samples(onsets, hop_length=hop_length)

    return onsets, sample_onsets

@cache(level=30)
def onset_strength(
    y=None,
    sr=22050,
    ** kwargs
):
    n_fft = 2048
    hop_length = 512
    lag = 1
    center = True
    channels = None

    kwargs.setdefault("fmax", 11025.0)

    # First, compute mel spectrogram
    S = np.abs(melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))
    # Convert to dBs
    S = core.power_to_db(S)
    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute difference to a reference, spaced by lag
    ref = S
    onset_env = S[:, lag:] - ref[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    onset_env = util.sync(onset_env, channels, aggregate=np.mean, pad=pad, axis=0)

    # Compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]), mode="constant")

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

