import winsound

def play_alarm():
    duration = 1000  # milliseconds
    freq = 1000      # Hz
    winsound.Beep(freq, duration)