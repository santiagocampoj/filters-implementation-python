import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.fft import fft, fftfreq

# the cutoff frequency is 0.5 times the Nyquist frequency, which is 500 Hz
CUTOFF_FREQ = 0.5

# Creating the low-pass filter
cutoff_frequency = CUTOFF_FREQ
# The order refers to the complexity of the filter
order = 6
b, a = signal.butter(order, cutoff_frequency, 'low', analog=False)

# Sampling frequency
fs = 1000
# Create a time vector from 0 to 1 second
t = np.linspace(0, 1, fs, endpoint=False)

# Creating the input signal with low and high frequency components
input_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 400 * t)

# converting input signal to stereo
# input_signal = np.vstack((input_signal, input_signal)).T

# Filtering the signal
filtered_signal = signal.lfilter(b, a, input_signal)

# Play the original and filtered signals
sd.play(input_signal, fs)
sd.wait()
time.sleep(1)
sd.play(filtered_signal, fs)
sd.wait()


# Plotting the signals and frequency spectrum
# Start of the subplot plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plotting the time-domain signals
axs[0].plot(t, input_signal, label='Original Signal')
axs[0].plot(t, filtered_signal, label='Filtered Signal')
axs[0].set_title('Time Domain')
axs[0].set_xlabel('Time [seconds]')
axs[0].set_ylabel('Amplitude')
axs[0].legend()

# Plotting the frequency response
w, h = signal.freqz(b, a)
axs[1].plot(w, 20 * np.log10(abs(h)))
axs[1].set_title('Frequency Response')
axs[1].set_xlabel('Frequency [radians / second]')
axs[1].set_ylabel('Amplitude [dB]')
axs[1].grid(which='both', axis='both')

# Plotting the frequency spectrum
input_fft = fft(input_signal)
filtered_fft = fft(filtered_signal)
freqs = fftfreq(fs, 1/fs)
axs[2].plot(freqs, np.abs(input_fft), label='Input Signal')
axs[2].plot(freqs, np.abs(filtered_fft), label='Filtered Signal')
axs[2].set_title('Frequency Spectrum')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Amplitude')
axs[2].set_xlim(0, fs/2)
axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f'low_pass_filter_freq_resp_{CUTOFF_FREQ}.png')
plt.show()


# Plotting the spectrogram
# Creating a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Compute and plot the spectrogram for the original signal
f, t, Sxx = signal.spectrogram(input_signal, fs)
pcm = axs[0].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
axs[0].set_ylabel('Frequency [Hz]')
axs[0].set_title('Spectrogram of Original Signal')

# Compute and plot the spectrogram for the filtered signal
f, t, Sxx = signal.spectrogram(filtered_signal, fs)
axs[1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_xlabel('Time [sec]')
axs[1].set_title('Spectrogram of Filtered Signal')

# Define a new axes for the colorbar
# The values are [left, bottom, width, height] in figure coordinate space
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # Adjust these values as needed

# Adding colorbar to the new axes
fig.colorbar(pcm, cax=cbar_ax, label='Magnitude (dB)')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.9)  # Make room for the new colorbar
plt.savefig(f'low_pass_filter_pectrogram_{CUTOFF_FREQ}.png')
plt.show()