"""
visualization.py - Visualization utilities for paleoclimate data

This module provides functions for visualizing paleoclimate time series data,
model reconstructions, and spectral properties with robust handling of both
regular and irregular time sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, interpolate
import pandas as pd
from scipy.fft import fft, fftfreq


def plot_time_series(times, values, title="Time Series", figsize=(12, 6),
                      xlabel="Time (kyr)", ylabel="Value", color='b',
                      marker='.', linestyle='-', alpha=0.7, errors=None):
    """
    Plot a paleoclimate time series with optional error bars.
    
    Parameters:
    -----------
    times : array-like
        Time points
    values : array-like
        Data values
    title : str, default="Time Series"
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
    xlabel : str, default="Time (kyr)"
        X-axis label
    ylabel : str, default="Value" 
        Y-axis label
    color : str, default='b'
        Line color
    marker : str, default='.'
        Marker style
    linestyle : str, default='-'
        Line style
    alpha : float, default=0.7
        Transparency
    errors : array-like, optional
        Error values for error bars
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if errors is not None:
        ax.errorbar(times, values, yerr=errors, fmt=f'{marker}{linestyle}',
                   color=color, alpha=alpha, capsize=3, label='Data')
    else:
        ax.plot(times, values, f'{marker}{linestyle}', color=color, 
                alpha=alpha, label='Data')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # If only a few points, show markers
    if len(times) < 100:
        ax.plot(times, values, 'o', color=color, alpha=0.5, markersize=4)
    
    return fig, ax


def plot_model_comparison(times, true_values, model_predictions, model_names=None,
                         title="Model Comparison", figsize=(12, 6), 
                         xlabel="Time (kyr)", ylabel="Value"):
    """
    Plot multiple model predictions against true values.
    
    Parameters:
    -----------
    times : array-like
        Time points
    true_values : array-like
        True data values
    model_predictions : list of array-like
        List of model predictions
    model_names : list of str, optional
        List of model names for legend
    title : str, default="Model Comparison"
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
    xlabel : str, default="Time (kyr)"
        X-axis label
    ylabel : str, default="Value"
        Y-axis label
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot true values
    ax.plot(times, true_values, 'k-', linewidth=2, label='True')
    
    # Plot model predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_predictions)))
    
    for i, pred in enumerate(model_predictions):
        label = model_names[i] if model_names is not None else f"Model {i+1}"
        ax.plot(times, pred, '-', color=colors[i], linewidth=1.5, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig, ax


def plot_residuals(times, true_values, model_predictions, model_names=None,
                  title="Residuals", figsize=(12, 8), xlabel="Time (kyr)"):
    """
    Plot residuals for multiple models.
    
    Parameters:
    -----------
    times : array-like
        Time points
    true_values : array-like
        True data values
    model_predictions : list of array-like
        List of model predictions
    model_names : list of str, optional
        List of model names for subplot titles
    title : str, default="Residuals"
        Main plot title
    figsize : tuple, default=(12, 8)
        Figure size
    xlabel : str, default="Time (kyr)"
        X-axis label
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_models = len(model_predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharex=True)
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    # Plot residuals for each model
    for i, pred in enumerate(model_predictions):
        residuals = true_values - pred
        label = model_names[i] if model_names is not None else f"Model {i+1}"
        
        axes[i].plot(times, residuals, 'o-', markersize=3)
        axes[i].axhline(y=0, color='k', linestyle='--')
        axes[i].set_ylabel('Residual')
        axes[i].set_title(f'{label} Residuals')
        axes[i].grid(True, alpha=0.3)
    
    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel(xlabel)
    
    # Add main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_power_spectrum(times, values, title="Power Spectrum", figsize=(12, 6),
                       highlight_milankovitch=True, resample=True, resample_points=None):
    """
    Plot power spectrum of a time series with robust handling of irregular sampling.
    
    Parameters:
    -----------
    times : array-like
        Time points
    values : array-like
        Data values
    title : str, default="Power Spectrum"
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    resample : bool, default=True
        Whether to resample irregular data to regular grid
    resample_points : int, optional
        Number of points for resampling. Default is len(times)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Check if time points are regularly spaced
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    # Handle irregular sampling
    if not is_regular:
        if not resample:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Cannot compute spectrum for irregularly sampled data.\nSet resample=True to enable resampling.',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig, ax
        
        # Resample to regular grid
        if resample_points is None:
            resample_points = len(times)
        
        regular_times = np.linspace(min(times), max(times), resample_points)
        interp_func = interpolate.interp1d(times, values, kind='cubic', 
                                         bounds_error=False, fill_value='extrapolate')
        regular_values = interp_func(regular_times)
        
        # Use resampled data
        t_use = regular_times
        v_use = regular_values
        dt_use = np.diff(t_use)[0]
    else:
        # Use original data
        t_use = times
        v_use = values
        dt_use = dt[0]
    
    # Compute sample spacing
    fs = 1.0 / dt_use  # samples per kyr
    
    # Compute power spectrum
    f, Pxx = signal.periodogram(v_use, fs=fs, scaling='spectrum')
    
    # Convert to period for plotting
    period = 1.0 / f[1:]  # Skip zero frequency
    power = Pxx[1:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot power spectrum
    ax.semilogy(period, power, 'b-', linewidth=1.5)
    
    # Highlight Milankovitch cycles if requested
    if highlight_milankovitch:
        milankovitch_cycles = {
            'Eccentricity': 100,
            'Obliquity': 41,
            'Precession': 23
        }
        
        for name, p in milankovitch_cycles.items():
            ax.axvline(p, color='r', linestyle='--', 
                      alpha=0.7, label=f'{name} ({p} kyr)')
    
    # Set axis properties
    ax.set_xscale('log')
    ax.set_xlabel('Period (kyr)')
    ax.set_ylabel('Power')
    
    if not is_regular and resample:
        title = f"{title}\n(Resampled from irregular data)"
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, which='both', alpha=0.3)
    
    if highlight_milankovitch:
        ax.legend()
    
    # Set common x-tick labels
    ax.set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
    ax.set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    return fig, ax


def plot_model_spectra(times, true_values, model_predictions, model_names=None,
                      title="Model Power Spectra", figsize=(12, 10),
                      highlight_milankovitch=True, resample=True):
    """
    Plot power spectra for multiple models with robust handling of irregular data.
    
    Parameters:
    -----------
    times : array-like
        Time points
    true_values : array-like
        True data values
    model_predictions : list of array-like
        List of model predictions
    model_names : list of str, optional
        List of model names for subplot titles
    title : str, default="Model Power Spectra"
        Main plot title
    figsize : tuple, default=(12, 10)
        Figure size
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    resample : bool, default=True
        Whether to resample irregular data to regular grid
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_models = len(model_predictions)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=figsize, sharex=True)
    
    # Check if time points are regularly spaced
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular and not resample:
        for ax in axes:
            ax.text(0.5, 0.5, 'Cannot compute spectrum for irregularly sampled data.\nSet resample=True to enable resampling.',
                   ha='center', va='center', transform=ax.transAxes)
        axes[0].set_title("True Data Power Spectrum")
        plt.suptitle(title, fontsize=16)
        return fig
    
    # Handle irregular sampling
    if not is_regular and resample:
        # Resample to regular grid
        regular_times = np.linspace(min(times), max(times), len(times))
        
        # Resample true values
        interp_func = interpolate.interp1d(times, true_values, kind='cubic',
                                         bounds_error=False, fill_value='extrapolate')
        regular_true = interp_func(regular_times)
        
        # Resample model predictions
        regular_preds = []
        for pred in model_predictions:
            interp_func = interpolate.interp1d(times, pred, kind='cubic',
                                             bounds_error=False, fill_value='extrapolate')
            regular_preds.append(interp_func(regular_times))
        
        # Use resampled data
        t_use = regular_times
        v_true = regular_true
        v_preds = regular_preds
    else:
        # Use original data
        t_use = times
        v_true = true_values
        v_preds = model_predictions
    
    # Compute sample spacing
    fs = 1.0 / np.diff(t_use)[0]  # samples per kyr
    
    # Plot true data spectrum
    f, Pxx = signal.periodogram(v_true, fs=fs, scaling='spectrum')
    period = 1.0 / f[1:]  # Skip zero frequency
    
    axes[0].semilogy(period, Pxx[1:], 'k-', linewidth=2)
    axes[0].set_title("True Data Power Spectrum")
    
    # Highlight Milankovitch cycles
    if highlight_milankovitch:
        for i, (name, p) in enumerate(zip(['Eccentricity', 'Obliquity', 'Precession'], 
                                         [100, 41, 23])):
            for ax in axes:
                color = ['r', 'g', 'b'][i]
                ax.axvline(p, color=color, linestyle='--', 
                          alpha=0.7, label=f'{name} ({p} kyr)')
    
    # Plot model spectra
    for i, pred in enumerate(v_preds):
        f, Pxx = signal.periodogram(pred, fs=fs, scaling='spectrum')
        
        label = model_names[i] if model_names is not None else f"Model {i+1}"
        axes[i+1].semilogy(period, Pxx[1:], '-', linewidth=2)
        axes[i+1].set_title(f"{label} Power Spectrum")
    
    # Set common properties
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
        
    # Add legend to first axis
    if highlight_milankovitch:
        axes[0].legend()
    
    # Set properties for bottom axis
    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('Period (kyr)')
    
    # Common y label
    fig.text(0.04, 0.5, 'Power', va='center', rotation='vertical')
    
    # Common x-tick labels
    axes[-1].set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
    axes[-1].set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    # Add note about resampling if applicable
    if not is_regular and resample:
        plt.figtext(0.5, 0.01, "Note: Data was resampled to a regular grid for spectral analysis", 
                   ha='center', fontsize=10, style='italic')
    
    # Add main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    return fig


def plot_coherence_analysis(times, true_values, model_predictions, model_names=None,
                          title="Spectral Coherence Analysis", figsize=(12, 10),
                          highlight_milankovitch=True, resample=True):
    """
    Plot spectral coherence between true values and model predictions.
    
    Parameters:
    -----------
    times : array-like
        Time points
    true_values : array-like
        True data values
    model_predictions : list of array-like
        List of model predictions
    model_names : list of str, optional
        List of model names for subplot titles
    title : str, default="Spectral Coherence Analysis"
        Main plot title
    figsize : tuple, default=(12, 10)
        Figure size
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    resample : bool, default=True
        Whether to resample irregular data to regular grid
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_models = len(model_predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharex=True)
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    # Check if time points are regularly spaced
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular and not resample:
        for ax in axes:
            ax.text(0.5, 0.5, 'Cannot compute coherence for irregularly sampled data.\nSet resample=True to enable resampling.',
                   ha='center', va='center', transform=ax.transAxes)
        plt.suptitle(title, fontsize=16)
        return fig
    
    # Handle irregular sampling
    if not is_regular and resample:
        # Resample to regular grid
        regular_times = np.linspace(min(times), max(times), len(times))
        
        # Resample true values
        interp_func = interpolate.interp1d(times, true_values, kind='cubic',
                                         bounds_error=False, fill_value='extrapolate')
        regular_true = interp_func(regular_times)
        
        # Resample model predictions
        regular_preds = []
        for pred in model_predictions:
            interp_func = interpolate.interp1d(times, pred, kind='cubic',
                                             bounds_error=False, fill_value='extrapolate')
            regular_preds.append(interp_func(regular_times))
        
        # Use resampled data
        t_use = regular_times
        v_true = regular_true
        v_preds = regular_preds
    else:
        # Use original data
        t_use = times
        v_true = true_values
        v_preds = model_predictions
    
    # Compute sample spacing
    fs = 1.0 / np.diff(t_use)[0]  # samples per kyr
    
    # Calculate and plot coherence for each model
    for i, pred in enumerate(v_preds):
        label = model_names[i] if model_names is not None else f"Model {i+1}"
        
        # Compute coherence
        f, Cxy = signal.coherence(v_true, pred, fs=fs, nperseg=min(256, len(v_true)//2))
        
        # Convert to period for plotting
        period = 1.0 / f[1:]  # Skip zero frequency
        coh = Cxy[1:]
        
        # Plot coherence
        axes[i].semilogx(period, coh, '-', linewidth=2)
        axes[i].set_title(f"{label} Coherence with True Data")
        
        # Highlight Milankovitch cycles
        if highlight_milankovitch:
            for j, (name, p) in enumerate(zip(['Eccentricity', 'Obliquity', 'Precession'], 
                                             [100, 41, 23])):
                color = ['r', 'g', 'b'][j]
                axes[i].axvline(p, color=color, linestyle='--', 
                              alpha=0.7, label=f'{name} ({p} kyr)')
                
                # Find and mark coherence at this period
                idx = np.argmin(np.abs(period - p))
                axes[i].plot(period[idx], coh[idx], 'o', markersize=8, color=color)
                axes[i].text(period[idx]*1.1, coh[idx], f'{coh[idx]:.2f}', 
                           verticalalignment='center', color=color)
        
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)
        
        # Add legend to first axis
        if i == 0 and highlight_milankovitch:
            axes[i].legend(loc='upper right')
    
    # Set properties for bottom axis
    axes[-1].set_xlabel('Period (kyr)')
    
    # Common y label
    fig.text(0.04, 0.5, 'Coherence', va='center', rotation='vertical')
    
    # Set x-tick labels
    axes[-1].set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
    axes[-1].set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    # Add note about resampling if applicable
    if not is_regular and resample:
        plt.figtext(0.5, 0.01, "Note: Data was resampled to a regular grid for coherence analysis", 
                   ha='center', fontsize=10, style='italic')
    
    # Add main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    return fig


def plot_comprehensive_comparison(times, true_values, model_predictions, model_names=None,
                                metrics_df=None, title="Comprehensive Model Comparison", 
                                figsize=(16, 12), highlight_milankovitch=True, resample=True):
    """
    Create a comprehensive comparison of multiple models.
    
    Parameters:
    -----------
    times : array-like
        Time points
    true_values : array-like
        True data values
    model_predictions : list of array-like
        List of model predictions
    model_names : list of str, optional
        List of model names for legends
    metrics_df : pandas.DataFrame, optional
        DataFrame containing model metrics
    title : str, default="Comprehensive Model Comparison"
        Main plot title
    figsize : tuple, default=(16, 12)
        Figure size
    highlight_milankovitch : bool, default=True
        Whether to highlight Milankovitch cycles
    resample : bool, default=True
        Whether to resample irregular data to regular grid
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Model Reconstructions
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot true values
    ax1.plot(times, true_values, 'k-', linewidth=2, label='True')
    
    # Plot model predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_predictions)))
    
    for i, pred in enumerate(model_predictions):
        label = model_names[i] if model_names is not None else f"Model {i+1}"
        ax1.plot(times, pred, '-', color=colors[i], linewidth=1.5, label=label)
    
    ax1.set_xlabel('Time (kyr)')
    ax1.set_ylabel('Value')
    ax1.set_title('Model Reconstructions')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Metrics Table
    if metrics_df is not None:
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('tight')
        ax2.axis('off')
        
        # Format metrics for table
        table_data = metrics_df.round(3)
        
        # Create table
        table = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                         loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        ax2.set_title('Model Performance Metrics')
    
    # 3. Spectral Analysis
    # Check if time points are regularly spaced
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular and not resample:
        # If data is irregular and no resampling, show message
        ax3 = fig.add_subplot(gs[1, 1:])
        ax3.text(0.5, 0.5, 'Cannot compute spectrum for irregularly sampled data.\nEnable resampling to view spectral analysis.',
               ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Spectral Analysis')
    else:
        # Handle irregular sampling if needed
        if not is_regular and resample:
            # Resample to regular grid
            regular_times = np.linspace(min(times), max(times), len(times))
            
            # Resample true values and model predictions
            interp_func = interpolate.interp1d(times, true_values, kind='cubic',
                                             bounds_error=False, fill_value='extrapolate')
            regular_true = interp_func(regular_times)
            
            regular_preds = []
            for pred in model_predictions:
                interp_func = interpolate.interp1d(times, pred, kind='cubic',
                                                 bounds_error=False, fill_value='extrapolate')
                regular_preds.append(interp_func(regular_times))
            
            # Use resampled data
            t_use = regular_times
            v_true = regular_true
            v_preds = regular_preds
        else:
            # Use original data
            t_use = times
            v_true = true_values
            v_preds = model_predictions
        
        # Compute sample spacing
        fs = 1.0 / np.diff(t_use)[0]  # samples per kyr
        
        # Plot power spectrum
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Compute and plot true data spectrum
        f, Pxx_true = signal.periodogram(v_true, fs=fs, scaling='spectrum')
        period = 1.0 / f[1:]  # Skip zero frequency
        
        ax3.semilogy(period, Pxx_true[1:], 'k-', linewidth=2, label='True')
        
        # Plot model spectra
        for i, pred in enumerate(v_preds):
            f, Pxx = signal.periodogram(pred, fs=fs, scaling='spectrum')
            label = model_names[i] if model_names is not None else f"Model {i+1}"
            ax3.semilogy(period, Pxx[1:], '-', color=colors[i], linewidth=1.5, label=label)
        
        # Highlight Milankovitch cycles
        if highlight_milankovitch:
            for j, (name, p) in enumerate(zip(['Eccentricity', 'Obliquity', 'Precession'], 
                                             [100, 41, 23])):
                color = ['r', 'g', 'b'][j]
                ax3.axvline(p, color=color, linestyle='--', alpha=0.7)
        
        ax3.set_xscale('log')
        ax3.set_xlabel('Period (kyr)')
        ax3.set_ylabel('Power')
        ax3.set_title('Power Spectra')
        ax3.grid(True, which='both', alpha=0.3)
        ax3.legend()
        
        # Set x-ticks
        ax3.set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
        ax3.set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
        
        # Plot coherence
        ax4 = fig.add_subplot(gs[1, 2])
        
        for i, pred in enumerate(v_preds):
            label = model_names[i] if model_names is not None else f"Model {i+1}"
            
            # Compute coherence
            f, Cxy = signal.coherence(v_true, pred, fs=fs, nperseg=min(256, len(v_true)//2))
            period = 1.0 / f[1:]  # Skip zero frequency
            
            ax4.semilogx(period, Cxy[1:], '-', color=colors[i], linewidth=1.5, label=label)
        
        # Highlight Milankovitch cycles
        if highlight_milankovitch:
            for j, p in enumerate([100, 41, 23]):
                color = ['r', 'g', 'b'][j]
                ax4.axvline(p, color=color, linestyle='--', alpha=0.7)
        
        ax4.set_xlabel('Period (kyr)')
        ax4.set_ylabel('Coherence')
        ax4.set_title('Spectral Coherence')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Set x-ticks
        ax4.set_xticks([10, 20, 23, 41, 50, 100, 200, 500])
        ax4.set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
    
    # 4. Residuals
    axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    
    # Plot residuals for up to 3 models
    for i, pred in enumerate(model_predictions[:min(3, len(model_predictions))]):
        residuals = true_values - pred
        label = model_names[i] if model_names is not None else f"Model {i+1}"
        
        axes[i].plot(times, residuals, 'o-', markersize=3, color=colors[i])
        axes[i].axhline(y=0, color='k', linestyle='--')
        axes[i].set_xlabel('Time (kyr)')
        axes[i].set_ylabel('Residual')
        axes[i].set_title(f'{label} Residuals')
        axes[i].grid(True, alpha=0.3)
    
    # If there are more than 3 models, show note
    if len(model_predictions) > 3:
        axes[2].text(0.5, 0.5, f'+ {len(model_predictions) - 3} more models\n(not shown)', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Additional Models')
    
    # Add note about resampling if applicable
    if not is_regular and resample:
        plt.figtext(0.5, 0.01, "Note: Data was resampled to a regular grid for spectral analysis", 
                   ha='center', fontsize=10, style='italic')
    
    # Add main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig