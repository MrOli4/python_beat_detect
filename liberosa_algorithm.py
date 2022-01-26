import librosa
import matplotlib.pyplot as plt
import librosa.display

import time
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.fftpack
from scipy.io.wavfile import write

from beat_detect_algorithm import onset_detect
from beat_detect_algorithm import onset_strength_multi

import PySimpleGUI as sg
import sounddevice as sd

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Press the button to record the record')],
            [sg.Button('Record BPM'), sg.Button('Cancel')],
            [sg.Text('Amount of time: ', key='timer')] ]

# Create the Window
window = sg.Window('Beat Detection Program', layout)

# Set the sampling frequency of the operation and amount of channels
fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2
#  This is the recording device, Find recording device with command
sd.default.device = 1  # This is the Realtek audio mic
duration = 10  # seconds

# define the countdown func. From https://www.geeksforgeeks.org/how-to-create-a-countdown-timer-using-python/
def countdown(time_sec):
    while time_sec:
        mins, secs = divmod(time_sec, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat)
        time.sleep(1)
        window['timer'].update(timeformat)
        time_sec -= 1

def record_audio():

    myrecording = sd.rec(int(duration * fs))

    # Countdown for the amount of seconds that it has to record
    countdown(duration)

    sd.wait()  # Wait until recording is finished, you do not have to wait but can run other commands, like a thread
    print("Done recording")

    write('audio/output.wav', fs, myrecording)  # Save as WAV file

    # This sleep is placed to allow the program some time before starting analysis
    time.sleep(1)

def analyse_audio():
    # The code that is actually run
    y, sr = librosa.load("audio/output.wav", offset=2, duration=2)

    t = np.linspace(0, 1, 1000, False)
    # Create the filter, since dance music is of importance simple low pass filter, with fcut at 200 Hz
    sos = scipy.signal.butter(30, 200, 'lp', fs=sr, output='sos')
    filtered_signal = scipy.signal.sosfilt(sos, y)  # Apply the filter

    y = filtered_signal

    # tempo_lib, beats = librosa.beat.beat_track(y=y, sr=sr)

    o_env = onset_strength_multi(filtered_signal, sr=sr)
    onset_frames = onset_detect(y=y, onset_envelope=o_env, sr=sr)

    times = librosa.times_like(o_env, sr=sr)

    D = np.abs(librosa.stft(y))
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()

    ax[1].plot(times, o_env, label='Onset strength')
    ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[1].legend()

    plt.show()

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break

    if event == 'Record BPM':
        print('Recording started')
        record_audio()
        analyse_audio()

window.close()

# Hello Cheffo
print("Ojalele")


