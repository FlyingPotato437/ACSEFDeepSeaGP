"""
synthetic_proxy_data.py - Generate synthetic SST data with corresponding proxies

This module creates synthetic Sea Surface Temperature (SST) data with seasonal
variations and generates corresponding proxy datasets (δ18O and UK'37) with 
realistic noise to simulate paleoclimate reconstruction scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_sst(n_points=200):
    """
    Generate synthetic SST data with seasonal variations and realistic noise.
    
    Parameters:
        n_points (int): Number of time points to generate
        
    Returns:
        tuple: (time_points, sst_values)
    """
    # Create time points (years)
    time_points = np.linspace(0, 20, n_points)
    
    # Generate baseline seasonal cycle (annual)
    annual_cycle = 3 * np.sin(2 * np.pi * time_points)
    
    # Add multi-year cycles (similar to ENSO, PDO, etc.)
    enso_cycle = 1.5 * np.sin(2 * np.pi * time_points / 3.7)  # ~3-4 year cycle
    decadal_cycle = 2 * np.sin(2 * np.pi * time_points / 10)  # ~10 year cycle
    
    # Add long-term trend
    trend = 0.1 * time_points
    
    # Combine components
    sst_base = annual_cycle + enso_cycle + decadal_cycle + trend
    
    # Add realistic noise (mix of white noise and red noise)
    white_noise = np.random.normal(0, 0.5, n_points)
    
    # Generate red noise (AR(1) process)
    red_noise = np.zeros(n_points)
    alpha = 0.7  # autocorrelation coefficient
    red_noise[0] = np.random.normal(0, 0.3)
    for i in range(1, n_points):
        red_noise[i] = alpha * red_noise[i-1] + np.random.normal(0, 0.3)
    
    # Add more complex noise pattern (combination of different frequencies)
    complex_noise = np.zeros(n_points)
    for freq in [0.1, 0.5, 1.5, 3.0]:
        complex_noise += 0.2 * np.random.normal(0, 1, n_points) * np.sin(freq * time_points)
    
    # Combine all noise components
    noise = 0.4 * white_noise + 0.4 * red_noise + 0.2 * complex_noise
    
    # Final SST with noise
    sst_values = sst_base + noise
    
    # Ensure realistic SST range (10°C to 30°C)
    sst_values = (sst_values - np.min(sst_values)) * (30 - 10) / (np.max(sst_values) - np.min(sst_values)) + 10
    
    return time_points, sst_values

def generate_d18o_proxy(sst_values):
    """
    Generate δ18O proxy data from SST values using the equation:
    δ18O = 16.9 - 4.38 * SST + ε
    
    Parameters:
        sst_values (array): SST values in °C
        
    Returns:
        array: δ18O proxy values
    """
    # Apply the δ18O calibration equation
    d18o_values = 16.9 - 4.38 * sst_values
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.2, len(sst_values))
    
    # Add some measurement bias and non-linear effects to make it more realistic
    bias = 0.1 * np.sin(np.linspace(0, 4 * np.pi, len(sst_values)))
    nonlinear = 0.05 * (sst_values - np.mean(sst_values))**2
    
    return d18o_values + noise + bias + nonlinear

def generate_uk37_proxy(sst_values):
    """
    Generate UK'37 proxy data from SST values using the equation:
    UK'37 = 0.033 * SST + 0.044 + ε
    
    Parameters:
        sst_values (array): SST values in °C
        
    Returns:
        array: UK'37 proxy values
    """
    # Apply the UK'37 calibration equation
    uk37_values = 0.033 * sst_values + 0.044
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.15, len(sst_values))
    
    # Add some measurement bias and non-linear effects
    bias = 0.02 * np.cos(np.linspace(0, 3 * np.pi, len(sst_values)))
    nonlinear = 0.01 * (sst_values - np.mean(sst_values))**2
    
    # Ensure UK'37 values stay in realistic range (0 to 1)
    result = uk37_values + noise + bias + nonlinear
    return np.clip(result, 0.0, 1.0)

def plot_synthetic_data(time_points, sst_values, d18o_values, uk37_values):
    """
    Plot the synthetic SST data and corresponding proxies.
    
    Parameters:
        time_points (array): Time points (x-axis)
        sst_values (array): SST values
        d18o_values (array): δ18O proxy values
        uk37_values (array): UK'37 proxy values
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot SST
    axes[0].plot(time_points, sst_values, 'b-', linewidth=2)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Synthetic Sea Surface Temperature (SST)')
    axes[0].grid(True)
    
    # Plot δ18O proxy
    axes[1].plot(time_points, d18o_values, 'g-', linewidth=2)
    axes[1].set_ylabel('δ¹⁸O (‰)')
    axes[1].set_title('δ¹⁸O Proxy (Negatively Correlated with SST)')
    axes[1].grid(True)
    
    # Plot UK'37 proxy
    axes[2].plot(time_points, uk37_values, 'r-', linewidth=2)
    axes[2].set_xlabel('Time (years)')
    axes[2].set_ylabel('UK\'37 Index')
    axes[2].set_title('UK\'37 Proxy (Positively Correlated with SST)')
    axes[2].grid(True)
    
    # Add correlations to titles
    d18o_corr = np.corrcoef(sst_values, d18o_values)[0, 1]
    uk37_corr = np.corrcoef(sst_values, uk37_values)[0, 1]
    axes[1].set_title(f'δ¹⁸O Proxy (Correlation with SST: {d18o_corr:.2f})')
    axes[2].set_title(f'UK\'37 Proxy (Correlation with SST: {uk37_corr:.2f})')
    
    plt.tight_layout()
    return fig

def calculate_power_spectrum(signal, dt=0.1):
    """
    Calculate and return the power spectrum of a signal.
    
    Parameters:
        signal (array): Time series signal
        dt (float): Time step between measurements
        
    Returns:
        tuple: (frequencies, power)
    """
    # Detrend the signal to remove linear trends
    detrended_signal = detrend(signal)
    
    # Calculate the FFT
    n = len(detrended_signal)
    fft_values = np.fft.rfft(detrended_signal)
    
    # Get the power spectrum
    power = np.abs(fft_values)**2
    
    # Calculate the frequencies
    freq = np.fft.rfftfreq(n, dt)
    
    return freq[1:], power[1:]  # Skip the first point (DC component)

def plot_power_spectra(time_points, sst_values, d18o_values, uk37_values):
    """
    Plot power spectra of the synthetic data and proxies.
    
    Parameters:
        time_points (array): Time points
        sst_values (array): SST values
        d18o_values (array): δ18O proxy values
        uk37_values (array): UK'37 proxy values
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Calculate time step (assume uniform sampling)
    dt = time_points[1] - time_points[0]
    
    # Calculate power spectra
    freq_sst, power_sst = calculate_power_spectrum(sst_values, dt)
    freq_d18o, power_d18o = calculate_power_spectrum(d18o_values, dt)
    freq_uk37, power_uk37 = calculate_power_spectrum(uk37_values, dt)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot SST spectrum
    axes[0].loglog(freq_sst, power_sst, 'b-', linewidth=2)
    axes[0].set_ylabel('Power')
    axes[0].set_title('Power Spectrum: Sea Surface Temperature (SST)')
    axes[0].grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Plot δ18O spectrum
    axes[1].loglog(freq_d18o, power_d18o, 'g-', linewidth=2)
    axes[1].set_ylabel('Power')
    axes[1].set_title('Power Spectrum: δ¹⁸O Proxy')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Plot UK'37 spectrum
    axes[2].loglog(freq_uk37, power_uk37, 'r-', linewidth=2)
    axes[2].set_xlabel('Frequency (1/year)')
    axes[2].set_ylabel('Power')
    axes[2].set_title('Power Spectrum: UK\'37 Proxy')
    axes[2].grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add vertical lines at key frequencies
    for ax in axes:
        # Annual cycle (1/year)
        ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Annual')
        # ENSO-like cycle (~1/3.7 years)
        ax.axvline(x=1/3.7, color='m', linestyle='--', alpha=0.5, label='ENSO-like')
        # Decadal cycle (1/10 years)
        ax.axvline(x=1/10, color='c', linestyle='--', alpha=0.5, label='Decadal')
    
    axes[0].legend(loc='best')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate and plot synthetic data."""
    print("Generating synthetic paleoclimate data...")
    
    # Create output directory if it doesn't exist
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic SST data
    time_points, sst_values = generate_synthetic_sst(n_points=200)
    
    # Generate proxy data
    d18o_values = generate_d18o_proxy(sst_values)
    uk37_values = generate_uk37_proxy(sst_values)
    
    # Create and save time series plot
    fig_time = plot_synthetic_data(time_points, sst_values, d18o_values, uk37_values)
    fig_time.savefig(os.path.join(output_dir, "synthetic_proxy_data.png"), dpi=300)
    
    # Create and save power spectra plot
    fig_power = plot_power_spectra(time_points, sst_values, d18o_values, uk37_values)
    fig_power.savefig(os.path.join(output_dir, "synthetic_proxy_spectra.png"), dpi=300)
    
    # Show correlation statistics
    d18o_corr = np.corrcoef(sst_values, d18o_values)[0, 1]
    uk37_corr = np.corrcoef(sst_values, uk37_values)[0, 1]
    
    print(f"Generated {len(time_points)} time points covering {time_points[-1] - time_points[0]} years")
    print(f"SST range: {np.min(sst_values):.2f}°C to {np.max(sst_values):.2f}°C")
    print(f"δ18O correlation with SST: {d18o_corr:.2f}")
    print(f"UK'37 correlation with SST: {uk37_corr:.2f}")
    print(f"Plots saved to {output_dir}")
    
    # Display plots
    plt.show()

if __name__ == "__main__":
    main()