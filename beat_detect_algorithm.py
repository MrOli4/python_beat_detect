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
    avg_factor = 0.05  # Change if needed

    print(avg_list)
    #
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
            if index < len(sorting_list)-1:
                if len(sorting_list[index]) >= len(longest_array):
                    longest_array = sorting_list[index]

        print("Array with longest list: ", longest_array)

        avg_time_val = np.average(longest_array)

        bpm = 60/avg_time_val
    else:
        # There is no beat detected so the bpm will be set to 0
        print("No bpm could be detected")
        bpm = 0

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
        kwargs.setdefault("wait", 0.3 * sr // hop_length)  # 300ms
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
    freq_list = []

    for x in sample_onsets:
        begin_snippit = x
        if y is not None:
            audio_snippit = y[begin_snippit-offset:begin_snippit + snippitsize]

            # Next part is from https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python/27191172
            w = np.fft.fft(audio_snippit)
            freqs = np.fft.fftfreq(len(w))

            # Find the peak in the coefficients of frequencies
            idx = np.argmax(np.abs(w))
            freq = freqs[idx]
            freq_in_hertz = abs(freq * 41000)
            print("Freq in Hz:", freq_in_hertz)

            freq_list.append(freq_in_hertz)
            kick_output.append(x)

    sorting_list = []
    avg_factor = 7  # Avg factor for grouping the freq data

    # Now find the most common dominant frequency in freq list
    if freq_list:
        # Already add the first item to provide a reference

        sorting_list.append([freq_list[0]])

        for idx, val in enumerate(freq_list[1:]):
            create_new = False

            for sorting_list_item in sorting_list:
                # if sorting_list_check[0] + avg_factor > avg_list[idx] < sorting_list_check[0] + avg_factor:
                if sorting_list_item[0] - avg_factor < freq_list[idx] < sorting_list_item[0] + avg_factor:

                    sorting_list_item.append(freq_list[idx])
                    # We use the first item out of the
                    create_new = False
                    break
                else:
                    create_new = True

            if create_new:
                sorting_list.append([freq_list[idx]])

        print(sorting_list)

        # Determine the biggest data batch, longest array of difference
        longest_array = sorting_list[0]

        for index, item in enumerate(sorting_list[1:]):
            if index < len(sorting_list)-1:
                if len(sorting_list[index]) >= len(longest_array):
                    longest_array = sorting_list[index]

        print("Longest freq list: ", longest_array)

        for item in longest_array:
            for id, x in enumerate(freq_list):
                if x == item:
                    kick_output.append(sample_onsets[id])

    else:
        kick_output = []


    # Get the estimated bpm and round off to the closed whole number
    bpm = round(calc_bpm(onsets=kick_output, sr=sr))
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

