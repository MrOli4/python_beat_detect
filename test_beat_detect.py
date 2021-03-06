import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display

import time
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.fftpack
from scipy.io.wavfile import write
import threading
from beat_detect_algorithm import onset_detect
from beat_detect_algorithm import onset_strength_multi

import PySimpleGUI as sg
import sounddevice as sd

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('Press the button to record the record')],
            [sg.Button('Start Recording'), sg.Button('Stop Recording'), sg.Button('Stop Application')],
            [sg.Text('BPM: '), sg.Text('Not yet determined')],
            [sg.Text('BPM to match'), sg.InputText(), sg.Button('Go to BPM')],
            [sg.Text('Amount of time: ', key='timer')],
            [sg.Canvas(key='figCanvas')]]

# VARS CONSTS:
_VARS = {'window': False}
_VARS['window'] = sg.Window('Such Window',
                            layout,
                            finalize=True,
                            resizable=True,
                            element_justification="left")

THREAD_EVENT = '-THREAD-'

# Setup the figure
fig, ax = plt.subplots(nrows=2, sharex=True)
figure_canvas_agg = FigureCanvasTkAgg(fig, _VARS['window']['figCanvas'].TKCanvas)
figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

duration = 5  # seconds
sample_wait = 8

# define the countdown func. From https://www.geeksforgeeks.org/how-to-create-a-countdown-timer-using-python/
def countdown(time_sec):
    while time_sec:
        mins, secs = divmod(time_sec, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat)
        time.sleep(1)
        # _VARS['window']['timer'].update(timeformat)
        time_sec -= 1

def record_audio():

    myrecording = sd.rec(int(duration * fs))

    # Countdown for the amount of seconds that it has to record
    countdown(duration)

    sd.wait()  # Wait until recording is finished, you do not have to wait but can run other commands, like a thread
    print("Done recording")

    write('audio/house_ex.wav', fs, myrecording)  # Save as WAV file

    _VARS['window'].write_event_value('-THREAD-', 'analyse')

def analyse_audio():
    # The code that is actually run
    y, sr = librosa.load("audio/good_things.wav", offset=14, duration=duration)

    t = np.linspace(0, 1, 1000, False)
    # Create the filter, since dance music is of importance simple low pass filter, with fcut at 200 Hz
    sos = scipy.signal.butter(30, 200, 'lp', fs=sr, output='sos')
    filtered_signal = scipy.signal.sosfilt(sos, y)  # Apply the filter

    y = filtered_signal

    # filtered_signal = y

    # tempo_lib, beats = librosa.beat.beat_track(y=y, sr=sr)

    o_env = onset_strength_multi(filtered_signal, sr=sr)
    onset_frames = onset_detect(y=y, onset_envelope=o_env, sr=sr)

    times = librosa.times_like(o_env, sr=sr)

    D = np.abs(librosa.stft(y))

    # Figure is defined outside of this function to remain same when recursively calling
    ax[0].cla()
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()

    ax[1].cla()
    ax[1].plot(times, o_env, label='Onset strength')
    ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[1].legend()

    # _VARS['window']['figCanvas'].TKCanvas.delete("all")
    figure_canvas_agg.draw()

    # plt.show()

# Set the sampling frequency of the operation and amount of channels
fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2
#  This is the recording device, Find recording device with command
sd.default.device = 1  # This is the Realtek audio mic


# First the setup bit, create thread that keeps track of the time
recording_enabled = False
analysed = False

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = _VARS['window'].read()

    if event == sg.WIN_CLOSED or event == 'Stop Application':  # if user closes window or clicks cancel
        break

    if event == 'Start Recording':
        # record_audio()
        # Only want to run this once now
        analyse_audio()

_VARS['window'].close()




