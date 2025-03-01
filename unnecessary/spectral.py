import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
import pywt

# No changes needed for this file as it only imports standard libraries


def calculate_power_spectrum(data, times, method='periodogram', 
                             detrend=True, pad=True, window='hann'):
    """
    Calculate the power spectrum of time series data.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    times : array-like
        Time points (in kyr)
    method : str, default='periodogram'
        Method to use for spectral estimation. Options:
        - 'periodogram': Standard periodogram
        - 'welch': Welch's method with overlapping windows
        - 'lombscargle': Lomb-Scargle periodogram for unevenly sampled data
    detrend : bool, default=True
        Whether to remove linear trend before analysis
    pad : bool, default=True
        Whether to pad the signal to the next power of 2
    window : str, default='hann'
        Window function to use. Options depend on method.
        
    Returns:
    --------
    frequencies : array-like
        Frequencies (in cycles/kyr)
    power : array-like
        Power spectral density
    """
    # Convert to numpy arrays
    data = np.asarray(data)
    times = np.asarray(times)
    
    # Check if time points are regularly spaced 
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    # Sampling frequency (samples per kyr)
    if is_regular:
        fs = 1.0 / dt[0]
    else:
        # Average sampling rate for irregular data
        fs = (len(times) - 1) / (times[-1] - times[0])
    
    # Detrend if requested
    if detrend:
        # Remove linear trend
        from scipy import signal
        data = signal.detrend(data)
    
    # Calculate spectrum based on method
    if method == 'periodogram':
        if not is_regular:
            raise ValueError("Periodogram requires regularly spaced data. Use method='lombscargle' for irregular data.")
        
        # Apply window if specified
        if window is not None:
            win = signal.get_window(window, len(data))
            data = data * win
        
        # Pad if requested
        if pad:
            n_fft = int(2 ** np.ceil(np.log2(len(data))))
        else:
            n_fft = len(data)
        
        # Calculate periodogram
        f, Pxx = signal.periodogram(data, fs=fs, nfft=n_fft, scaling='density')
        
    elif method == 'welch':
        if not is_regular:
            raise ValueError("Welch's method requires regularly spaced data. Use method='lombscargle' for irregular data.")
        
        # Calculate power spectrum using Welch's method
        nperseg = min(256, len(data) // 4)
        f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, window=window, scaling='density')
        
    elif method == 'lombscargle':
        # Lomb-Scargle periodogram for irregularly sampled data
        f = np.linspace(1/times.ptp(), fs/2, 1000)  # frequency grid
        Pxx = signal.lombscargle(times, data, 2*np.pi*f)
        
    else:
        raise ValueError(f"Unknown method: {method}. Expected one of: 'periodogram', 'welch', 'lombscargle'")
    
    return f, Pxx


def milankovitch_filter(data, times, cycles=None, bandwidth=0.2, method='fft'):
    """
    Filter the time series to isolate Milankovitch cycles.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    times : array-like
        Time points (in kyr)
    cycles : list, optional
        List of cycles to isolate (in kyr). If None, use standard
        Milankovitch cycles: [100, 41, 23]
    bandwidth : float, default=0.2
        Relative bandwidth for filtering (fraction of center frequency)
    method : str, default='fft'
        Filtering method. Options:
        - 'fft': Fast Fourier Transform (requires regular sampling)
        - 'butter': Butterworth bandpass filter (requires regular sampling)
        - 'gaussian': Gaussian bandpass filter in frequency domain
        
    Returns:
    --------
    components : dict
        Dictionary of filtered components for each cycle
    """
    # Convert to numpy arrays
    data = np.asarray(data)
    times = np.asarray(times)
    
    # Default cycles if not specified
    if cycles is None:
        cycles = [100, 41, 23]  # kyr
    
    # Check if time points are regularly spaced 
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular and method in ['fft', 'butter']:
        # For irregular data, interpolate to regular grid
        t_reg = np.linspace(times.min(), times.max(), len(times))
        interp_func = interpolate.interp1d(times, data, kind='cubic')
        data_reg = interp_func(t_reg)
        
        # Use regular grid for processing
        times_proc = t_reg
        data_proc = data_reg
        fs = 1.0 / (t_reg[1] - t_reg[0])
    else:
        # Use original data for processing
        times_proc = times
        data_proc = data
        
        if is_regular:
            fs = 1.0 / dt[0]
        else:
            # Average sampling rate for irregular data
            fs = (len(times) - 1) / (times[-1] - times[0])
    
    # Initialize results dictionary
    components = {}
    
    if method == 'fft':
        # FFT-based filtering
        n = len(data_proc)
        
        # Compute FFT
        y_fft = fft(data_proc)
        freqs = fftfreq(n, 1/fs)
        
        # Create filtered components for each cycle
        for cycle in cycles:
            # Cycle frequency
            cycle_freq = 1.0 / cycle
            
            # Bandwidth in frequency
            bw = cycle_freq * bandwidth
            
            # Create frequency domain filter
            filt = np.zeros(n, dtype=complex)
            
            # Positive frequencies
            mask = (freqs >= cycle_freq - bw/2) & (freqs <= cycle_freq + bw/2)
            filt[mask] = y_fft[mask]
            
            # Negative frequencies (for real signal)
            mask = (freqs <= -cycle_freq + bw/2) & (freqs >= -cycle_freq - bw/2)
            filt[mask] = y_fft[mask]
            
            # Inverse FFT to get filtered component
            component = np.real(np.fft.ifft(filt))
            
            # Save component
            components[cycle] = component
            
            # If we used interpolation, interpolate back to original time points
            if not is_regular and method in ['fft', 'butter']:
                interp_func = interpolate.interp1d(times_proc, component, kind='cubic')
                components[cycle] = interp_func(times)
    
    elif method == 'butter':
        # Butterworth bandpass filtering 
        from scipy.signal import butter, filtfilt
        
        for cycle in cycles:
            # Cycle frequency
            cycle_freq = 1.0 / cycle
            
            # Bandwidth in frequency
            bw = cycle_freq * bandwidth
            
            # Normalized frequencies for Butterworth filter (Nyquist = 1)
            low = max(0, (cycle_freq - bw/2) / (fs/2))
            high = min(1, (cycle_freq + bw/2) / (fs/2))
            
            # Design filter
            b, a = butter(4, [low, high], btype='band')
            
            # Apply filter
            component = filtfilt(b, a, data_proc)
            
            # Save component
            components[cycle] = component
            
            # If we used interpolation, interpolate back to original time points
            if not is_regular:
                interp_func = interpolate.interp1d(times_proc, component, kind='cubic')
                components[cycle] = interp_func(times)
    
    elif method == 'gaussian':
        # Gaussian bandpass filter (works with irregular data)
        f, Pxx = calculate_power_spectrum(data, times, method='lombscargle')
        
        for cycle in cycles:
            # Cycle frequency
            cycle_freq = 1.0 / cycle
            
            # Bandwidth in frequency
            bw = cycle_freq * bandwidth
            
            # Create Gaussian filter in frequency domain
            sigma = bw / 2.355  # convert FWHM to sigma
            gauss_filter = np.exp(-0.5 * ((f - cycle_freq) / sigma)**2)
            
            # Scale factor to ensure filter has unit gain at center frequency
            scale = 1.0 / np.max(gauss_filter)
            gauss_filter *= scale
            
            # Filter power spectrum
            Pxx_filt = Pxx * gauss_filter
            
            # Reconstruct signal using inverse Fourier transform (approximate)
            # This is a simplified approach and may not be accurate for highly irregular data
            component = np.zeros_like(data)
            for i, t in enumerate(times):
                # Reconstruct sinusoidal components
                comp = np.sum(np.sqrt(Pxx_filt) * np.cos(2*np.pi*f*t + np.angle(Pxx_filt)))
                component[i] = comp
            
            # Save component
            components[cycle] = component
    
    else:
        raise ValueError(f"Unknown method: {method}. Expected one of: 'fft', 'butter', 'gaussian'")
    
    return components


def wavelet_analysis(data, times, wavelet='morlet', num_scales=100, min_period=10, max_period=500):
    """
    Perform wavelet analysis on time series data.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    times : array-like
        Time points (in kyr)
    wavelet : str, default='morlet'
        Wavelet type. Options: 'morlet', 'paul', 'dog' (derivative of Gaussian)
    num_scales : int, default=100
        Number of scales to use
    min_period : float, default=10
        Minimum period to consider (kyr)
    max_period : float, default=500
        Maximum period to consider (kyr)
        
    Returns:
    --------
    periods : array-like
        Periods corresponding to scales (kyr)
    power : 2D array
        Wavelet power spectrum (time x scale)
    coi : array-like
        Cone of influence
    """
    # Convert to numpy arrays
    data = np.asarray(data)
    times = np.asarray(times)
    
    # Normalize data
    data_norm = (data - np.mean(data)) / np.std(data)
    
    # Check if time points are regularly spaced 
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular:
        # For irregular data, interpolate to regular grid
        t_reg = np.linspace(times.min(), times.max(), len(times))
        interp_func = interpolate.interp1d(times, data_norm, kind='cubic')
        data_proc = interp_func(t_reg)
        dt_mean = (times.max() - times.min()) / (len(times) - 1)
    else:
        data_proc = data_norm
        dt_mean = dt[0]
    
    # Set up scales for continuous wavelet transform
    # Convert periods to scales (depends on wavelet)
    if wavelet == 'morlet':
        # For Morlet, scale ~= period / 1.03
        scales = np.logspace(np.log10(min_period/1.03), np.log10(max_period/1.03), num_scales)
    elif wavelet == 'paul':
        # For Paul, scale ~= period / 4.0
        scales = np.logspace(np.log10(min_period/4.0), np.log10(max_period/4.0), num_scales)
    elif wavelet == 'dog':
        # For DOG, scale ~= period / 2.0
        scales = np.logspace(np.log10(min_period/2.0), np.log10(max_period/2.0), num_scales)
    else:
        raise ValueError(f"Unknown wavelet: {wavelet}. Expected one of: 'morlet', 'paul', 'dog'")
    
    # Perform continuous wavelet transform
    coef, freqs = pywt.cwt(data_proc, scales, wavelet, dt_mean)
    
    # Convert scales to periods
    if wavelet == 'morlet':
        periods = 1.03 / freqs
    elif wavelet == 'paul':
        periods = 4.0 / freqs
    elif wavelet == 'dog':
        periods = 2.0 / freqs
    
    # Calculate power (squared magnitude)
    power = np.abs(coef)**2
    
    # Calculate cone of influence
    # Approximate COI as e-folding time of the wavelet at each scale
    t = np.arange(len(data_proc)) * dt_mean
    coi = np.zeros_like(t)
    
    if wavelet == 'morlet':
        # For Morlet wavelet with omega0=6, e-folding time is sqrt(2)*scale
        c = np.sqrt(2)
    elif wavelet == 'paul':
        # For Paul wavelet with order=4, e-folding time is scale
        c = 1.0
    elif wavelet == 'dog':
        # For DOG wavelet with order=2, e-folding time is scale/sqrt(2)
        c = 1/np.sqrt(2)
    
    # Calculate COI at each time point
    for i in range(len(t)):
        if i < len(t)/2:
            coi[i] = c * dt_mean * (i + 0.5)
        else:
            coi[i] = c * dt_mean * (len(t) - i - 0.5)
    
    return periods, power, coi


def plot_spectrum(frequencies, power, title='Power Spectrum', 
                 highlight_milankovitch=True, figsize=(10, 6)):
    """
    Plot the power spectrum with optional Milankovitch cycle highlights.
    
    Parameters:
    -----------
    frequencies : array-like
        Frequencies (in cycles/kyr)
    power : array-like
        Power spectral density
    title : str, default='Power Spectrum'
        Plot title
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Convert frequencies to periods for better readability
    periods = 1.0 / frequencies[1:]  # Skip zero frequency
    power_nonzero = power[1:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot power spectrum
    ax.semilogy(periods, power_nonzero, 'b-', linewidth=1.5)
    
    # Highlight Milankovitch cycles if requested
    if highlight_milankovitch:
        milankovitch_cycles = {
            'Eccentricity': 100,
            'Obliquity': 41,
            'Precession': 23
        }
        
        for name, period in milankovitch_cycles.items():
            idx = np.argmin(np.abs(periods - period))
            ax.axvline(period, color='r', linestyle='--',
                      alpha=0.7, label=f'{name} ({period} kyr)')
            
            # Mark peak if exists
            peak_idx = signal.find_peaks(power_nonzero, distance=20)[0]
            if np.any(np.abs(periods[peak_idx] - period) < period*0.1):
                peak_i = peak_idx[np.argmin(np.abs(periods[peak_idx] - period))]
                ax.plot(periods[peak_i], power_nonzero[peak_i], 'ro', markersize=8)
    
    # Set axis properties
    ax.set_xscale('log')
    ax.set_xlabel('Period (kyr)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, which='both', alpha=0.3)
    
    if highlight_milankovitch:
        ax.legend()
    
    # Set x-tick labels for common periods
    ax.set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
    ax.set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    plt.tight_layout()
    
    return fig


def plot_wavelet(times, periods, power, coi, data=None, 
                highlight_milankovitch=True, figsize=(12, 8)):
    """
    Plot the wavelet power spectrum with optional time series.
    
    Parameters:
    -----------
    times : array-like
        Time points (in kyr)
    periods : array-like
        Periods corresponding to scales (kyr)
    power : 2D array
        Wavelet power spectrum (time x scale)
    coi : array-like
        Cone of influence
    data : array-like, optional
        Original time series data to plot above wavelet
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Create figure
    if data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot original time series
        ax1.plot(times, data, 'k-')
        ax1.set_ylabel('Value')
        ax1.set_title('Time Series')
        ax1.grid(True)
        
        # Wavelet plot
        wavelet_ax = ax2
    else:
        fig, wavelet_ax = plt.subplots(figsize=figsize)
    
    # Normalize power for better visualization
    norm_power = power / np.max(power)
    
    # Plot wavelet power spectrum
    levels = np.linspace(0, 1, 20)
    contour = wavelet_ax.contourf(times, periods, norm_power, levels=levels,
                                 cmap='viridis', extend='both')
    
    # Plot cone of influence
    # Convert COI values to periods
    coi_line = wavelet_ax.plot(times, coi, 'w--', linewidth=2)
    
    # Fill outside COI
    wavelet_ax.fill_between(times, coi, np.max(periods), 
                           color='white', alpha=0.5)
    
    # Highlight Milankovitch cycles if requested
    if highlight_milankovitch:
        milankovitch_cycles = {
            'Eccentricity': 100,
            'Obliquity': 41,
            'Precession': 23
        }
        
        for name, period in milankovitch_cycles.items():
            wavelet_ax.axhline(period, color='r', linestyle='--',
                             alpha=0.7, label=f'{name} ({period} kyr)')
    
    # Set y-axis to log scale
    wavelet_ax.set_yscale('log')
    
    # Set axis limits
    wavelet_ax.set_ylim(np.min(periods), np.max(periods))
    wavelet_ax.set_xlim(np.min(times), np.max(times))
    
    # Add labels
    wavelet_ax.set_xlabel('Time (kyr)')
    wavelet_ax.set_ylabel('Period (kyr)')
    wavelet_ax.set_title('Wavelet Power Spectrum')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=wavelet_ax)
    cbar.set_label('Normalized Power')
    
    # Add legend if Milankovitch cycles are highlighted
    if highlight_milankovitch:
        wavelet_ax.legend(loc='upper right')
    
    # Set y-tick labels
    wavelet_ax.set_yticks([10, 20, 23, 41, 50, 100, 200, 500])
    wavelet_ax.set_yticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    plt.tight_layout()
    
    return fig


def cross_spectral_analysis(data1, data2, times, method='periodogram'):
    """
    Perform cross-spectral analysis between two time series.
    
    Parameters:
    -----------
    data1 : array-like
        First time series
    data2 : array-like
        Second time series
    times : array-like
        Time points (in kyr)
    method : str, default='periodogram'
        Method for spectral estimation. Options: 'periodogram', 'welch', 'lombscargle'
        
    Returns:
    --------
    frequencies : array-like
        Frequencies (in cycles/kyr)
    coherence : array-like
        Magnitude-squared coherence
    phase : array-like
        Phase spectrum (in radians)
    """
    # Convert to numpy arrays
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    times = np.asarray(times)
    
    # Check if time points are regularly spaced 
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular and method != 'lombscargle':
        # For irregular data with methods requiring regular sampling,
        # interpolate to regular grid
        t_reg = np.linspace(times.min(), times.max(), len(times))
        interp_func1 = interpolate.interp1d(times, data1, kind='cubic')
        interp_func2 = interpolate.interp1d(times, data2, kind='cubic')
        data1_proc = interp_func1(t_reg)
        data2_proc = interp_func2(t_reg)
        times_proc = t_reg
        fs = 1.0 / (t_reg[1] - t_reg[0])
    else:
        data1_proc = data1
        data2_proc = data2
        times_proc = times
        
        if is_regular:
            fs = 1.0 / dt[0]
        else:
            # Average sampling rate for irregular data
            fs = (len(times) - 1) / (times[-1] - times[0])
    
    if method == 'welch':
        # Use Welch's method for cross-spectral estimates
        nperseg = min(256, len(data1_proc) // 4)
        f, Pxy = signal.csd(data1_proc, data2_proc, fs=fs, nperseg=nperseg, scaling='density')
        f, Pxx = signal.welch(data1_proc, fs=fs, nperseg=nperseg, scaling='density')
        f, Pyy = signal.welch(data2_proc, fs=fs, nperseg=nperseg, scaling='density')
        
        # Calculate coherence
        coherence = np.abs(Pxy)**2 / (Pxx * Pyy)
        
        # Calculate phase
        phase = np.angle(Pxy)
        
    elif method == 'periodogram':
        # Use periodogram for cross-spectral estimates
        f, Pxx = signal.periodogram(data1_proc, fs=fs, scaling='density')
        _, Pyy = signal.periodogram(data2_proc, fs=fs, scaling='density')
        
        # Cross spectrum
        cross_spectrum = np.fft.rfft(data1_proc) * np.conjugate(np.fft.rfft(data2_proc))
        Pxy = np.abs(cross_spectrum) / len(data1_proc)
        
        # Calculate coherence
        coherence = np.abs(cross_spectrum)**2 / (np.abs(np.fft.rfft(data1_proc))**2 * 
                                               np.abs(np.fft.rfft(data2_proc))**2)
        
        # Calculate phase
        phase = np.angle(cross_spectrum)
        
    elif method == 'lombscargle':
        # For Lomb-Scargle, calculate separate periodograms and approximate cross-spectral values
        f = np.linspace(1/times.ptp(), fs/2, 1000)  # frequency grid
        
        # Calculate Lomb-Scargle periodograms
        pgram1 = signal.lombscargle(times, data1, 2*np.pi*f)
        pgram2 = signal.lombscargle(times, data2, 2*np.pi*f)
        
        # Approximate coherence (simplified approach)
        coherence = np.ones_like(f) * 0.5  # placeholder, more advanced methods required
        
        # Approximate phase (simplified approach)
        phase = np.zeros_like(f)  # placeholder
    
    else:
        raise ValueError(f"Unknown method: {method}. Expected one of: 'periodogram', 'welch', 'lombscargle'")
    
    return f, coherence, phase


def plot_cross_spectrum(frequencies, coherence, phase, title='Cross-Spectral Analysis',
                      highlight_milankovitch=True, figsize=(12, 8)):
    """
    Plot cross-spectral analysis results.
    
    Parameters:
    -----------
    frequencies : array-like
        Frequencies (in cycles/kyr)
    coherence : array-like
        Magnitude-squared coherence
    phase : array-like
        Phase spectrum (in radians)
    title : str, default='Cross-Spectral Analysis'
        Plot title
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Convert frequencies to periods for better readability
    periods = 1.0 / frequencies[1:]  # Skip zero frequency
    coherence_nonzero = coherence[1:]
    phase_nonzero = phase[1:]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot coherence
    ax1.plot(periods, coherence_nonzero, 'b-', linewidth=1.5)
    ax1.set_ylabel('Coherence')
    ax1.set_title('Magnitude-Squared Coherence')
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    
    # Plot phase
    ax2.plot(periods, phase_nonzero, 'g-', linewidth=1.5)
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title('Phase Spectrum')
    ax2.grid(True)
    ax2.set_ylim(-np.pi, np.pi)
    
    # Set x-axis properties
    ax2.set_xscale('log')
    ax2.set_xlabel('Period (kyr)')
    
    # Add super title
    plt.suptitle(title, fontsize=16)
    
    # Highlight Milankovitch cycles if requested
    if highlight_milankovitch:
        milankovitch_cycles = {
            'Eccentricity': 100,
            'Obliquity': 41,
            'Precession': 23
        }
        
        for name, period in milankovitch_cycles.items():
            for ax in [ax1, ax2]:
                ax.axvline(period, color='r', linestyle='--',
                          alpha=0.7, label=f'{name} ({period} kyr)')
    
    # Add legend
    if highlight_milankovitch:
        ax1.legend(loc='upper right')
    
    # Set x-tick labels for common periods
    ax2.set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
    ax2.set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig