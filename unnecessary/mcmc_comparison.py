"""
MCMC Comparison Script for Bayesian GP State-Space Models

This script compares the traditional GP approach with the Bayesian GP State-Space Model
using proper MCMC for posterior sampling. The comparison demonstrates the improved
uncertainty quantification and more accurate parameter inference.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gpytorch
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Import our model implementations
from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse
from gp_models import MaternGP, RBFKernelGP  # Standard GP implementations for comparison

# Create output directory
output_dir = "data/results/mcmc_comparison"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def create_combined_comparison_figure(
    test_ages: np.ndarray,
    true_sst: np.ndarray,
    proxy_data: Dict,
    standard_preds: Dict,
    bayesian_preds: Dict
):
    """
    Create a combined comparison figure showing standard GP vs Bayesian GP with MCMC.
    
    Args:
        test_ages: Ages for predictions
        true_sst: True SST values
        proxy_data: Dictionary of proxy data
        standard_preds: Predictions from standard GP models
        bayesian_preds: Predictions from Bayesian GP with MCMC
    
    Returns:
        matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    
    # Plot raw data and true SST
    ax = axes[0]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    # Plot proxy data
    markers = ['o', 's']
    colors = ['blue', 'green']
    
    for i, (proxy_type, data) in enumerate(proxy_data.items()):
        if proxy_type == 'd18O':
            scaling = -4.54545  # inverse_slope
            intercept = 3.0
        else:  # UK37
            scaling = 30.303   # inverse_slope
            intercept = 0.044
        
        # Convert to SST
        sst_values = (data['value'] - intercept) * scaling
        
        ax.scatter(data['age'], sst_values, color=colors[i], marker=markers[i], s=40, alpha=0.7,
                  label=f'{proxy_type} derived SST')
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Raw Data and True SST', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Plot standard GP models
    ax = axes[1]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    for model_name, preds in standard_preds.items():
        if model_name == 'rbf':
            color = 'blue'
            label = 'RBF Kernel GP'
        else:  # matern
            color = 'green'
            label = 'Matérn Kernel GP'
        
        ax.plot(test_ages, preds['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(test_ages, preds['lower'], preds['upper'], color=color, alpha=0.2)
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Standard GP Models', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Plot Bayesian GP model with MCMC
    ax = axes[2]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    # Plot main prediction
    ax.plot(test_ages, bayesian_preds['mean'], 'r-', linewidth=2, label='Bayesian GP with MCMC')
    ax.fill_between(test_ages, bayesian_preds['lower'], bayesian_preds['upper'], color='r', alpha=0.2, label='95% CI')
    
    # Plot posterior samples if available
    if 'samples' in bayesian_preds:
        # Plot a subset of samples for visual clarity
        samples = bayesian_preds['samples']
        n_samples_to_plot = min(10, samples.shape[0])
        for i in range(n_samples_to_plot):
            ax.plot(test_ages, samples[i], 'r-', linewidth=0.5, alpha=0.15)
    
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Age (kyr)')
    ax.set_title('Bayesian GP with MCMC', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set common x-axis limits (reverse for geological convention)
    for ax in axes:
        ax.set_xlim(max(test_ages), min(test_ages))
    
    plt.tight_layout()
    return fig

def create_parameter_posterior_figure(bayesian_model):
    """
    Create a figure showing the posterior distributions of model parameters.
    
    Args:
        bayesian_model: Fitted BayesianGPStateSpaceModel
    
    Returns:
        matplotlib figure
    """
    # Check if posterior samples are available
    if bayesian_model.posterior_samples is None:
        raise ValueError("No posterior samples available")
    
    # Get samples
    samples = bayesian_model.posterior_samples
    
    # Determine parameter types based on kernel
    if bayesian_model.kernel_type == 'combined':
        param_names = [
            'mean', 'noise',
            'rbf_lengthscale', 'rbf_outputscale',
            'periodic_lengthscale', 'periodic_period', 'periodic_outputscale'
        ]
    else:
        param_names = ['mean', 'noise', 'lengthscale', 'outputscale']
    
    # Only plot parameters that exist in samples
    param_names = [p for p in param_names if p in samples]
    
    # Create figure
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params))
    
    # Handle single parameter case
    if n_params == 1:
        axes = [axes]
    
    # Plot posterior for each parameter
    for i, param_name in enumerate(param_names):
        param_samples = samples[param_name]
        
        # Create histogram
        axes[i].hist(param_samples, bins=30, alpha=0.7, density=True)
        
        # Add mean and credible intervals
        mean = np.mean(param_samples)
        q025, q975 = np.percentile(param_samples, [2.5, 97.5])
        
        axes[i].axvline(mean, color='r', linestyle='-', linewidth=2)
        axes[i].axvline(q025, color='r', linestyle='--')
        axes[i].axvline(q975, color='r', linestyle='--')
        
        # Add text description
        axes[i].text(0.02, 0.95, f"{param_name}:\nMean: {mean:.4f}\n95% CI: [{q025:.4f}, {q975:.4f}]",
                   transform=axes[i].transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[i].set_title(f"Posterior for {param_name}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_transition_detection_figure(
    test_ages: np.ndarray,
    true_sst: np.ndarray,
    bayesian_preds: Dict,
    detected_transitions: List,
    true_transitions: List
):
    """
    Create a figure showing transition detection capabilities.
    
    Args:
        test_ages: Ages for predictions
        true_sst: True SST values
        bayesian_preds: Predictions from Bayesian GP
        detected_transitions: Detected transition points
        true_transitions: True transition points
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot reconstructed SST with transitions
    ax1.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    ax1.plot(test_ages, bayesian_preds['mean'], 'b-', linewidth=2, label='Bayesian GP Reconstruction')
    ax1.fill_between(test_ages, bayesian_preds['lower'], bayesian_preds['upper'], color='b', alpha=0.2)
    
    # Plot true transition points
    y_range = ax1.get_ylim()
    for trans in true_transitions:
        ax1.axvline(trans['age'], color='r', linestyle='-', linewidth=2)
        ax1.text(trans['age'], y_range[1] - 0.05 * (y_range[1] - y_range[0]),
               f"True: {trans['age']} kyr",
               ha='right', va='top', rotation=90, color='r',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Plot detected transition points
    for trans in detected_transitions:
        ax1.axvline(trans, color='g', linestyle='--', linewidth=2)
        ax1.text(trans, y_range[0] + 0.05 * (y_range[1] - y_range[0]),
               f"Detected: {trans:.1f} kyr",
               ha='right', va='bottom', rotation=90, color='g',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Reconstruction with Transition Detection', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Compute and plot the rate of change
    dx = np.diff(test_ages)
    dy = np.diff(bayesian_preds['mean'])
    rate_of_change = dy / dx
    rate_ages = [(test_ages[i] + test_ages[i+1])/2 for i in range(len(test_ages)-1)]
    
    ax2.plot(rate_ages, rate_of_change, 'k-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Mark true transition points
    for trans in true_transitions:
        ax2.axvline(trans['age'], color='r', linestyle='-', linewidth=2)
    
    # Mark detected transition points
    for trans in detected_transitions:
        ax2.axvline(trans, color='g', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Age (kyr)')
    ax2.set_ylabel('Rate of Change (°C/kyr)')
    ax2.set_title('Rate of Temperature Change', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Set common x-axis limits (reverse for geological convention)
    ax1.set_xlim(max(test_ages), min(test_ages))
    ax2.set_xlim(max(test_ages), min(test_ages))
    
    plt.tight_layout()
    return fig

def compare_metrics_table(standard_metrics, bayesian_metrics):
    """
    Create a DataFrame comparing metrics between standard and Bayesian GP models.
    
    Args:
        standard_metrics: Metrics from standard GP models
        bayesian_metrics: Metrics from Bayesian GP with MCMC
    
    Returns:
        pandas DataFrame
    """
    all_metrics = {}
    
    # Add standard GP metrics
    for model, metrics in standard_metrics.items():
        col_name = f"Standard GP ({model})"
        all_metrics[col_name] = metrics
    
    # Add Bayesian GP metrics
    all_metrics["Bayesian GP (MCMC)"] = bayesian_metrics
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics).T
    
    # Reorder metrics for clarity
    metric_order = ['rmse', 'mae', 'r2', 'coverage', 'crps', 'ci_width']
    df = df[metric_order]
    
    # Set column names
    df.columns = ['RMSE (°C)', 'MAE (°C)', 'R²', 'Coverage', 'CRPS', 'CI Width (°C)']
    
    return df

def run_mcmc_comparison():
    """
    Run a comparison between standard GP and Bayesian GP with MCMC sampling.
    """
    print("Running MCMC Comparison Analysis...")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=60,
        age_min=0,
        age_max=300,
        irregularity=0.7,
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    
    # Set up test ages and ground truth
    test_ages = np.linspace(0, 300, 300)
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    
    # Initialize results dictionaries
    standard_preds = {}
    standard_metrics = {}
    
    # Run standard GP models
    print("Running standard GP models...")
    for kernel_type in ['rbf', 'matern']:
        # Combine proxy data into a single dataset
        # First extract ages and proxy-derived SST from each proxy
        combined_ages = []
        combined_sst = []
        
        for proxy_type, data in proxy_data.items():
            ages = data['age']
            values = data['value']
            
            # Convert to SST
            if proxy_type == 'd18O':
                sst = (values - 3.0) * -4.54545
            else:  # UK37
                sst = (values - 0.044) * 30.303
            
            combined_ages.extend(ages)
            combined_sst.extend(sst)
        
        # Convert to numpy arrays
        combined_ages = np.array(combined_ages)
        combined_sst = np.array(combined_sst)
        
        # Sort by age
        sort_idx = np.argsort(combined_ages)
        combined_ages = combined_ages[sort_idx]
        combined_sst = combined_sst[sort_idx]
        
        # Train standard GP model
        if kernel_type == 'rbf':
            model = RBFKernelGP(combined_ages, combined_sst)
        else:
            model = MaternGP(combined_ages, combined_sst)
        
        model.fit(iterations=500)
        
        # Make predictions
        mean, lower, upper = model.predict(test_ages)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(np.mean((mean - test_true_sst)**2)),
            'mae': np.mean(np.abs(mean - test_true_sst)),
            'r2': 1 - np.sum((test_true_sst - mean)**2) / np.sum((test_true_sst - np.mean(test_true_sst))**2),
            'coverage': np.mean((lower <= test_true_sst) & (test_true_sst <= upper)),
            'ci_width': np.mean(upper - lower),
            'crps': np.mean((mean - test_true_sst)**2 / (upper - lower))
        }
        
        # Store results
        standard_preds[kernel_type] = {'mean': mean, 'lower': lower, 'upper': upper}
        standard_metrics[kernel_type] = metrics
    
    # Run Bayesian GP with MCMC
    print("Running Bayesian GP with MCMC...")
    bayesian_model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37'],
        kernel_type='combined',
        n_mcmc_samples=500,
        random_state=42
    )
    
    # Fit model
    bayesian_model.fit(proxy_data, training_iterations=500)
    
    # Make predictions
    mean, lower, upper, samples = bayesian_model.predict(
        test_ages, return_samples=True, n_samples=50
    )
    
    # Detect transitions
    true_transitions = [{'age': 125, 'magnitude': -3.0}, {'age': 330, 'magnitude': 2.0}]
    detected_transitions = bayesian_model.detect_abrupt_transitions(test_ages)
    
    # Calculate metrics
    bayesian_metrics = bayesian_model.evaluate(test_ages, test_true_sst)
    
    # Store results
    bayesian_preds = {
        'mean': mean,
        'lower': lower,
        'upper': upper,
        'samples': samples
    }
    
    # Create comparison figure
    print("Creating comparison figures...")
    fig_comparison = create_combined_comparison_figure(
        test_ages, test_true_sst, proxy_data, standard_preds, bayesian_preds
    )
    fig_comparison.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create parameter posterior figure
    fig_posterior = create_parameter_posterior_figure(bayesian_model)
    fig_posterior.savefig(f"{output_dir}/parameter_posteriors.png", dpi=300, bbox_inches='tight')
    
    # Create transition detection figure (only for points within range)
    in_range_transitions = [t for t in detected_transitions if t >= min(test_ages) and t <= max(test_ages)]
    fig_transitions = create_transition_detection_figure(
        test_ages, test_true_sst, bayesian_preds, in_range_transitions,
        [t for t in true_transitions if t['age'] >= min(test_ages) and t['age'] <= max(test_ages)]
    )
    fig_transitions.savefig(f"{output_dir}/transition_detection.png", dpi=300, bbox_inches='tight')
    
    # Create metrics table
    metrics_df = compare_metrics_table(standard_metrics, bayesian_metrics)
    metrics_df.to_csv(f"{output_dir}/model_comparison_metrics.csv")
    
    # Create metrics visualization
    fig_metrics, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', y=['RMSE (°C)', 'MAE (°C)', 'CI Width (°C)'], ax=ax)
    ax.set_title('Error and Uncertainty Metrics Comparison')
    ax.set_ylabel('Value (°C)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_metrics.savefig(f"{output_dir}/error_metrics.png", dpi=300, bbox_inches='tight')
    
    # Create coverage and R² visualization
    fig_coverage, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', y=['R²', 'Coverage'], ax=ax)
    ax.set_title('R² and Credible Interval Coverage Comparison')
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_coverage.savefig(f"{output_dir}/coverage_metrics.png", dpi=300, bbox_inches='tight')
    
    print(f"All results saved to {output_dir}")
    print("\nMetrics comparison:")
    print(metrics_df)
    
    return {
        'bayesian_model': bayesian_model,
        'synthetic_data': synthetic_data,
        'standard_preds': standard_preds,
        'bayesian_preds': bayesian_preds,
        'standard_metrics': standard_metrics,
        'bayesian_metrics': bayesian_metrics,
        'metrics_df': metrics_df
    }

if __name__ == "__main__":
    results = run_mcmc_comparison()