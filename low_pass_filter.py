import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Setting up the constants for the low pass filter
CUTOFF_FREQ = 0.5

# the cutoff frequency is 0.3 times the Nyquist frequency:
# The cutoff frequency should be a value between 0 and 0.5, (0 and 1)
# where 0.5 corresponds to the Nyquist frequency. (half the sampling rate)
# The Nyquist frequency is half the sampling rate of a discrete signal processing system.
cutoff_frequency = CUTOFF_FREQ

# The order of the filter is equal to the length of the filter minus one.
# The higher the order, the sharper the transition is at the cutoff frequency.
# which means that the higher the order, the better the filter is at removing noise.
# However, a high order also results in a slower computation.
# The order should be an integer greater than 0. In a range between 1 and 10 is good.
order = 6

# The butter() function returns the filter coefficients: b and a.
# The difference equation is a mathematical equation that defines the filter (to calculate the output of the filter from its input)
# y[n] = b[0] * x[n] + b[1] * x[n-1] + ... + b[M] * x[n-M] - a[1] * y[n-1] - ... - a[N] * y[n-N]

# where:
# y[n] is the output signal
# x[n] is the input signal
# M is the order of the feedforward part of the filter
# N is the order of the feedback part of the filter
# b and a are the filter coefficients
# The filter coefficients are calculated with the butter() function.
b, a = signal.butter(order, cutoff_frequency, 'low', analog=False)


fs = 1000  # Sampling frequency
# the linspace() function returns evenly spaced numbers over a specified interval.
# The first argument is the starting point of the sequence.
# The second argument is the end point of the sequence.
# The third argument is the number of values to generate.
# The fourth argument is the endpoint argument. The default value of this argument is True.
# If the endpoint argument is set to True, the stop value is included in the sequence.
# If the endpoint argument is set to False, the stop value is not included in the sequence.
t = np.linspace(0, 1, fs, endpoint=False)

# We can use the lfilter() function to apply the filter to a signal.
# The lfilter() function takes the filter coefficients and the signal values as input and applies the filter to the signal values.
# The mathematical for creating the sin signal is: y(t) = A * sin(2 * pi * f * t)
# where:
# A is the amplitude of the signal
# f is the frequency of the signal
# t is the time
# The amplitude of the signal is 1, the frequency is 5 Hz and 50 Hz.
# The signal is sampled 1000 times per second.
# The signal is sampled for 1 second.
input_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)  # Low and high frequency

# We can use the lfilter() function to apply the filter to a signal.
# The lfilter() function takes the filter coefficients and the signal values as input and applies the filter to the signal values.
filtered_signal = signal.lfilter(b, a, input_signal)

# We now can plot the signal and the filtered signal in the time domain.
plt.figure()
plt.plot(t, input_signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.title(f'Low Pass Filter with cutoff frequency {CUTOFF_FREQ}')
plt.legend()
plt.savefig(f'low_pass_filter_{CUTOFF_FREQ}.png')
plt.show()
