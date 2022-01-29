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

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

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

    write('audio/output.wav', fs, myrecording)  # Save as WAV file

    _VARS['window'].write_event_value('-THREAD-', 'analyse')

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

    # _VARS['window']['figCanvas'].TKCanvas.delete("all")
    draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig)

    # plt.show()

def threading_function():
    """This function is the thread that keeps track of the current time in the system, updates every 5 seconds
    It is used to take samples every period, now 5 seconds"""

    first_run = True
    # Setup overall timer
    start_time = time.time()

    while True:
        if recording_enabled:
            time_tracker = time.time()
            # print(time_tracker)
            if time_tracker > start_time + 10 or first_run:

                start_time = time_tracker

                # A event is writen to the window that we need to record and analyse audio
                _VARS['window'].write_event_value('-THREAD-', 'start_rec')

                # Variable used to detect and pass the first run should be set to False now
                first_run = False

            time.sleep(1)  # Timer otherwise the tracker wont work
        else:
            time.sleep(1)  # 1 sec sleeper until next variable check

# Set the sampling frequency of the operation and amount of channels
fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2
#  This is the recording device, Find recording device with command
sd.default.device = 1  # This is the Realtek audio mic
duration = 5  # seconds

# First the setup bit, create thread that keeps track of the time
recording_enabled = False
analysed = False
timer_thread = threading.Thread(target=threading_function)

timer_thread.start()  # Start the thread

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = _VARS['window'].read()

    if event == sg.WIN_CLOSED or event == 'Stop Application':  # if user closes window or clicks cancel
        break

    if event == 'Start Recording':
        print('Recording enabled')
        recording_enabled = True

    if event == 'Stop Recording':
        print('Recording stopped')
        recording_enabled = False

    if event == THREAD_EVENT:
        if values[THREAD_EVENT] == 'start_rec':
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()  # Let this thing start, and let it notify the window when done
        if values[THREAD_EVENT] == 'analyse':
            print("Analyse")
            record_thread.join()

            # Only want to run this once now
            if analysed is False:
                analyse_audio()
                analysed = True

timer_thread.join()
_VARS['window'].close()




