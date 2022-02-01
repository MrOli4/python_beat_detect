import time
import threading
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.fftpack
import librosa
import librosa.display
from scipy.io.wavfile import write
import PySimpleGUI as sg
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from beat_detect_algorithm import onset_detect
from beat_detect_algorithm import onset_strength_multi
from beat_detect_algorithm import calc_bpm

""" 
    This application has been created for the Premaster Linear systems, signals & control course 5LIU0,
    of the University of Twente.
    ---
    Created by: Olivier Mathijssen, and Tijn XXX
    Date: 1-02-2022
    ---
    This program deals with the bpm analysis of an audio signal and virtual adjustment at which this audio signal
    is played back. 
"""

# First some varibles of the PySimpleGUI are specified.
sg.theme('DarkAmber')
# All GUI components
layout = [  [sg.Text('Press the button to record the record')],
            [sg.Button('Start Recording'), sg.Button('Stop Recording'), sg.Button('Stop Application')],
            [sg.Text('BPM: '), sg.Text('Not yet determined', key='bpm_showcase')],
            [sg.Text('BPM to match'), sg.InputText(key='textbox'), sg.Button('Go to BPM')],
            [sg.Text('Record timer: '), sg.Text('No timer started yet', key='timer')],
            [sg.Canvas(key='figCanvas')]]
# VARS Constants:
_VARS = {'window': False}
_VARS['window'] = sg.Window('Such Window',
                            layout,
                            finalize=True,
                            resizable=True,
                            element_justification="left")
THREAD_EVENT = '-THREAD-'

# Some elements that setup the plots that can be seen in the application
fig, ax = plt.subplots(nrows=3, sharex=False)
figure_canvas_agg = FigureCanvasTkAgg(fig, _VARS['window']['figCanvas'].TKCanvas)
figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

# Some global variables used to control the response of the whole application
sample_wait = 10  # Time to wait until taking a next sample
duration = 5  # Duration of the recording of a sample
testmode = True  # Boolean that tells if the program is in test mode or not.


def countdown(time_sec):
    # Simple countdown function,
    # adapted from: https://www.geeksforgeeks.org/how-to-create-a-countdown-timer-using-python/
    while time_sec:
        mins, secs = divmod(time_sec, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat)
        time.sleep(1)
        _VARS['window']['timer'].update(timeformat)
        time_sec -= 1

    _VARS['window']['timer'].update("Timer finished")

def record_audio():
    """Method to record audio, should only record audio if it is not in testmode, in testmode the application uses
    predefined audio files located in the audio folder. """
    countdown(duration)
    if not testmode:

        myrecording = sd.rec(int(duration * fs))

        # Countdown for the amount of seconds that it has to record
        countdown(duration)

        print("Wait until done")

        sd.wait()  # Wait until recording is finished, you do not have to wait but can run other commands, like a thread
        write('audio/output.wav', fs, myrecording)  # Save as WAV file

        print("Wrote to file")

    # Signal that the audio can be analysed
    _VARS['window'].write_event_value('-THREAD-', 'analyse')


def analyse_audio(speed):
    """Function to analyse a certain audio file and determine its bpm
    The signal is first filtered to remain only low frequency domain, the main aspect of beat importance in dance
    music. Then this whole signal is analysed using its onsets by a function is another file.
    >> The BPM is returned by this function.
    ---
    This function now uses audio files for the test phase of the project, by setting the boolean X to Y, the system
    will be in record mode and not use these test files for BPM analysing.-"""

    # Check if program is in testmode
    if testmode:
        # Forming the filename for the audio file with the input speed variable
        filename = "audio/tamborine_" + str(speed) + "bpm.wav"

    else:
        # Use the just recorded audio
        filename = "audio/output.wav"

    y, sr = librosa.load(filename, offset=0, duration=duration)

    # Create the filter, since dance music is of importance simple low pass filter, with fcut at 200 Hz
    sos = scipy.signal.butter(30, 200, 'lp', fs=sr, output='sos')
    filtered_signal = scipy.signal.sosfilt(sos, y)  # Apply the filter

    y = filtered_signal

    # First create a full onset envelope
    o_env = onset_strength_multi(filtered_signal, sr=sr)
    # Get the onset points picked out form the onset envelope (after peak picking), in both frames and samples
    onset_frames, onset_samples = onset_detect(y=y, onset_envelope=o_env, sr=sr)
    # Get the BPM from the onset points after peak picking, use onset points in samples
    bpm = round(calc_bpm(onsets=onset_samples, sr=sr))

    times = librosa.times_like(o_env, sr=sr)

    # Figure is defined outside of this function to remain same when recursively calling
    ax[0].cla()
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()

    ax[1].cla()
    ax[1].plot(times, o_env, label='Onset strength')
    ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    ax[1].legend()

    # _VARS['window']['figCanvas'].TKCanvas.delete("all")
    # figure_canvas_agg.draw()

    return bpm

def pid_function(setpoint, output, reset, pre_error):
    kP = 0.5
    tauI = 1
    tauD = 1
    error = setpoint - output
    reset = reset + (kP / tauI) * error
    output = kP * error + reset + ((pre_error - error) * (kP / tauD))

    print(output)
    return output, reset, pre_error

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
            if time_tracker > start_time + sample_wait or first_run:
                # Variable used to detect and pass the first run should be set to False now
                if first_run:
                    first_run = False

                start_time = time_tracker

                # A event is writen to the window that we need to record and analyse audio
                _VARS['window'].write_event_value('-THREAD-', 'start_rec')

            time.sleep(1)  # Timer otherwise the tracker wont work
        else:
            time.sleep(1)  # 1 sec sleeper until next variable check


# Set the sampling frequency of the operation and amount of channels
fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2
#  This is the recording device, Find recording device with command: python -m souddevice
sd.default.device = 1

# First the setup bit, create thread that keeps track of the time
recording_enabled = False
timer_thread = threading.Thread(target=threading_function)
timer_thread.start()  # Start the thread

# PID values
output = 0
pre_error = 0
reset = 0
adapt = False
output_array = [output]
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

    if event == 'Go to BPM':
        adapt = True

    if event == THREAD_EVENT:
        if values[THREAD_EVENT] == 'start_rec':
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()  # Let this thing start, and let it notify the window when done
        if values[THREAD_EVENT] == 'analyse':
            print("Analyse")

            # record_thread.join()

            # The analyse audio function takes a speed variable, from 20 to 240, this is the BPM of the PID controller
            # The function return the bpm after analysis of the audio that comes with the PID controller bpm
            bpm_after_analysis = analyse_audio(217)

            print("BPM: ", bpm_after_analysis)

            _VARS['window']['bpm_showcase'].update(bpm_after_analysis)

            if adapt == True:
                try:
                    setpoint = int(values[0])
                except:
                    setpoint = output

                if (abs((output - setpoint)) < (setpoint * 0.05)):
                    adapt = False
                else:
                    output, reset, pre_error = pid_function(int(values[0]), output, reset, pre_error)
            output_array.append(output)
            ax[2].cla()
            ax[2].plot(output_array)
            figure_canvas_agg.draw()

_VARS['window'].close()




