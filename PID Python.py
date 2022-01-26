import time

inputBPM = 124 #BPM current music
setPointBPM = 130 #BPM that must be reached

#PID set values
kP = 0.5
tauI = 1
tauD = 1

#pre loop variables
reset = 0
preError = 0
output = inputBPM
while(True):
    error = setPointBPM - output
    reset = reset + (kP / tauI) * error
    output = kP * error + reset + ((preError - error) * (kP / tauD))
    
    preError = error
    print("output:", output, "error:", error)
    time.sleep(5)
