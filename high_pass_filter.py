import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd
import time

# Setting up the constants for the low pass filter,
# the cutoff frequency is 0.5 times the Nyquist frequency, which is 500 Hz
CUTOFF_FREQ = 0.5

# Creating the low-pass filter
cutoff_frequency = CUTOFF_FREQ
order = 6
b, a = signal.butter(order, cutoff_frequency, 'high', analog=False)

# Sampling frequency
fs = 1000
# Create a time vector from 0 to 1 second
t = np.linspace(0, 1, fs, endpoint=False)

# Creating the input signal with low and high frequency components
input_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

# Filtering the signal
filtered_signal = signal.lfilter(b, a, input_signal)

plt.figure()
plt.plot(t, input_signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.title(f'High Pass Filter with cutoff frequency {CUTOFF_FREQ}')
plt.legend()
plt.savefig(f'high_pass_filter_{CUTOFF_FREQ}.png')
# plt.show()

# Play the original signal
sd.play(input_signal, fs)
sd.wait()  # Wait until the sound is played

time.sleep(1)  # Wait for 1 second

# Play the filtered signal
sd.play(filtered_signal, fs)
sd.wait()  # Wait until the sound is played

# plot the frequency response of the filter
w, h = signal.freqz(b, a)
# freqz() is a function in scipy.signal that computes the frequency response of a filter
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))

plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.savefig(f'high_pass_filter_{CUTOFF_FREQ}_freq_response.png')
plt.show()
