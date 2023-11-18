import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.fft import fft, fftfreq
import os
from logger_config import setup_logger

def create_signal(t: np.ndarray, freq_components: list, logger) -> np.ndarray:
    """Creates a signal with the given time vector and frequency components
    Args:
        t (np.ndarray): Time vector
        freq_components (list): List of frequency components of the signal
    """
    try:
        signal = sum(np.sin(2 * np.pi * freq * t) for freq in freq_components)
        logger.info(f'Created signal with frequency components {freq_components}')
        return signal
    except Exception as e:
        logger.error(f'Error creating signal: {e}')

def apply_Low_pass_filter(input_signal: np.ndarray, b: np.ndarray, a: np.ndarray, logger) -> np.ndarray:
    """Applies a low pass filter to the input signal
    Args:
        input_signal (np.ndarray): Input signal
        b (np.ndarray): Numerator coefficients of the filter
        a (np.ndarray): Denominator coefficients of the filter
    """
    try:
        filter_signal = signal.lfilter(b, a, input_signal)
        logger.info(f'Applied low pass filter to input signal')
        return filter_signal
    except Exception as e:
        logger.error(f'Error applying low pass filter: {e}')

def play_signals(signal_to_play: np.ndarray, fs: int, filtered_signal: np.ndarray, 
                 time_to_wait: int = 1, fs_filtered: int = None, logger=None) -> None:
    """Plays the given signals
    Args:
        signal_to_play (np.ndarray): Signal to play
        fs (int): Sampling frequency of the signal
        time_to_wait (int, optional): Time to wait between playing the signals. Defaults to 1.
        filtered_signal (np.ndarray): Filtered signal to play.
        fs_filtered (int, optional): Sampling frequency of the filtered signal. Defaults to None.
        logger: Logger object for logging messages
    """
    try:
        sd.play(signal_to_play, fs)
        sd.wait()
        time.sleep(time_to_wait)
        logger.info(f'Played signal with sampling frequency {fs}')

        sd.play(filtered_signal, fs_filtered or fs)
        sd.wait()
        logger.info(f'Played filtered signal with sampling frequency {fs_filtered or fs}')
    except Exception as e:
        logger.error(f'Error playing signals: {e}')

def save_figure(fig: plt.Figure, name: str, cutoff_frequency: float, logger) -> None:
    """Saves the given figure
    Args:
        fig (plt.Figure): Figure to save
        name (str): Name of the figure
        cutoff_frequency (float): Cutoff frequency of the filter
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, f'{name}_{cutoff_frequency}.png'))
        logger.info(f'Saved figure {name}_{cutoff_frequency}.png')
    except Exception as e:
        logger.error(f'Error saving figure: {e}')

def plot_signal_spectrum(t: np.ndarray, input_signal: np.ndarray, filtered_signal: np.ndarray,
                         b: np.ndarray, a: np.ndarray, fs: int, cutoff_frequency: float, logger) -> plt.Figure:
    """Plots the time domain signal, frequency response and frequency spectrum
    Args:
        t (np.ndarray): Time vector
        input_signal (np.ndarray): Input signal
        filtered_signal (np.ndarray): Filtered signal
        b (np.ndarray): Numerator coefficients of the filter
        a (np.ndarray): Denominator coefficients of the filter
        fs (int): Sampling frequency
        cutoff_frequency (float): Cutoff frequency of the filter
    """
    try:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Plotting the time-domain signals
        axs[0].plot(t, input_signal, label='Original Signal')
        axs[0].plot(t, filtered_signal, label='Filtered Signal')
        axs[0].set_title('Time Domain')
        axs[0].set_xlabel('Time [seconds]')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()
        logger.info(f'Plotted time domain signals')

        # Plotting the frequency response
        w, h = signal.freqz(b, a)
        axs[1].plot(w, 20 * np.log10(abs(h)))
        axs[1].set_title('Frequency Response')
        axs[1].set_xlabel('Frequency [radians / second]')
        axs[1].set_ylabel('Amplitude [dB]')
        axs[1].grid(which='both', axis='both')
        logger.info(f'Plotted frequency response')

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
        logger.info(f'Plotted frequency spectrum')

        # Adjust layout
        plt.tight_layout()
        save_figure(fig, 'low_pass_filter', cutoff_frequency, logger)
        logger.info(f'Saved figure low_pass_filter_{cutoff_frequency}.png')
        return fig
    except Exception as e:
        logger.error(f'Error plotting signals: {e}')

def plot_spectrogram(input_signal: np.ndarray, filtered_signal: np.ndarray, fs: int, cutoff_frequency: float, logger) -> plt.Figure:
    """Plots the spectrogram of the original and filtered signals
    Args:
        input_signal (np.ndarray): Input signal
        filtered_signal (np.ndarray): Filtered signal
        fs (int): Sampling frequency
        cutoff_frequency (float): Cutoff frequency of the filter
    """
    try:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        f, t, Sxx = signal.spectrogram(input_signal, fs)
        pcm = axs[0].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
        axs[0].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
        axs[0].set_ylabel('Frequency [Hz]')
        axs[0].set_title('Spectrogram of Original Signal')
        logger.info(f'Plotted spectrogram of original signal')

        f, t, Sxx = signal.spectrogram(filtered_signal, fs)
        axs[1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
        axs[1].set_ylabel('Frequency [Hz]')
        axs[1].set_xlabel('Time [sec]')
        axs[1].set_title('Spectrogram of Filtered Signal')
        logger.info(f'Plotted spectrogram of filtered signal')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(pcm, cax=cbar_ax, label='Magnitude (dB)')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)

        save_figure(fig, 'low_pass_filter_spectrogram', cutoff_frequency, logger)
        return fig
    except Exception as e:
        logger.error(f'Error plotting spectrogram: {e}')

def main():
    logger = setup_logger('low_pass_filter.log')
    logger.info('Starting low pass filter')

    try:
        duration, freq_components, fs = 1, [5, 50], 1000  
        cutoff_frequency, order = 0.5, 6 
        logger.info(f'Created low pass filter with cutoff frequency {cutoff_frequency} and order {order}')
        
        b, a = signal.butter(order, cutoff_frequency, 'low', analog=False)
        logger.info(f"Created filter's numerator coefficients {b} and denominator coefficients {a}")
    
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        input_signal = create_signal(t, freq_components, logger)
        filtered_signal = apply_Low_pass_filter(input_signal, b, a, logger)

        play_signals(input_signal, fs, filtered_signal=filtered_signal, fs_filtered=fs, logger=logger)

        plot_signal_spectrum(t, input_signal, filtered_signal, b, a, fs, cutoff_frequency, logger)
        plot_spectrogram(input_signal, filtered_signal, fs, cutoff_frequency, logger)
        plt.show()

    except Exception as e:
        logger.error(f'Error creating low pass filter: {e}')
    
if __name__ == '__main__':
    main()