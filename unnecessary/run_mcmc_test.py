"""
MCMC Test Script for Bayesian GP State-Space Models

This script tests the fixed MCMC implementation on sparse synthetic data,
compares it with standard approaches, and generates plots to demonstrate
the improvements in uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from datetime import datetime

# Import model classes
from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse
from gp_models import MaternGP, RBFKernelGP
from fixed_mcmc import FixedMCMC

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"data/results/mcmc_test_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Configure paths
figure_dir = f"{output_dir}/figures"
os.makedirs(figure_dir, exist_ok=True)

def test_on_sparse_data(n_points=30, irregularity=0.8):
    """
    Test the MCMC implementations on sparse data.
    
    Args:
        n_points: Number of data points to generate
        irregularity: Level of sampling irregularity (0-1)
        
    Returns:
        Results dictionary
    """
    print(f"Generating sparse synthetic data with {n_points} points and {irregularity} irregularity...")
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data_sparse(
        n_points=n_points,
        age_min=0,
        age_max=200,
        irregularity=irregularity,
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    
    # Create test points for predictions
    test_ages = np.linspace(0, 200, 300)
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    
    # Train models and make predictions
    results = {}
    
    # Standard GP model (no MCMC)
    print("\n=== Training standard GP model ===")
    standard_gp = train_standard_gp(synthetic_data)
    std_mean, std_lower, std_upper = standard_gp.predict(test_ages)
    results['standard_gp'] = {
        'model': standard_gp,
        'mean': std_mean,
        'lower': std_lower,
        'upper': std_upper
    }
    
    # Original BayesianGP with naive MCMC
    print("\n=== Training Bayesian GP with naive MCMC ===")
    original_model = train_bayesian_gp(synthetic_data, 
                                     use_improved_mcmc=False, 
                                     use_fixed_mcmc=False)
    orig_mean, orig_lower, orig_upper, orig_samples = original_model.predict(
        test_ages, return_samples=True, n_samples=50
    )
    results['original_model'] = {
        'model': original_model,
        'mean': orig_mean,
        'lower': orig_lower,
        'upper': orig_upper,
        'samples': orig_samples
    }
    
    # Fixed BayesianGP with robust MCMC
    print("\n=== Training Bayesian GP with fixed MCMC ===")
    fixed_model = train_bayesian_gp(synthetic_data, 
                                  use_improved_mcmc=False, 
                                  use_fixed_mcmc=True)
    fixed_mean, fixed_lower, fixed_upper, fixed_samples = fixed_model.predict(
        test_ages, return_samples=True, n_samples=50
    )
    results['fixed_model'] = {
        'model': fixed_model,
        'mean': fixed_mean,
        'lower': fixed_lower,
        'upper': fixed_upper,
        'samples': fixed_samples
    }
    
    # Calculate metrics
    results['metrics'] = calculate_metrics(results, test_true_sst)
    
    # Create plots
    create_comparison_plots(results, synthetic_data, test_ages, test_true_sst, figure_dir)
    
    return results, synthetic_data

def train_standard_gp(synthetic_data):
    """Train a standard GP model on the synthetic data."""
    # Combine proxy data for standard GP
    combined_ages = []
    combined_sst = []
    
    for proxy_type, data in synthetic_data['proxy_data'].items():
        ages = data['age']
        values = data['value']
        
        # Convert to SST
        if proxy_type == 'd18O':
            sst = (values - 3.0) * -4.54545
        else:  # UK37
            sst = (values - 0.044) * 30.303
        
        combined_ages.extend(ages)
        combined_sst.extend(sst)
    
    # Sort by age
    sort_idx = np.argsort(combined_ages)
    combined_ages = np.array(combined_ages)[sort_idx]
    combined_sst = np.array(combined_sst)[sort_idx]
    
    # Create and fit model
    model = MaternGP(combined_ages, combined_sst)
    model.fit(iterations=300)
    
    return model

def train_bayesian_gp(synthetic_data, use_improved_mcmc=False, use_fixed_mcmc=False):
    """Train a Bayesian GP model with specified MCMC method."""
    # Create and fit model
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37'],
        kernel_type='combined',
        n_mcmc_samples=500,
        random_state=42
    )
    
    model.fit(synthetic_data['proxy_data'], 
             training_iterations=300,
             use_improved_mcmc=use_improved_mcmc,
             use_fixed_mcmc=use_fixed_mcmc)
    
    return model

def calculate_metrics(results, true_sst):
    """Calculate performance metrics for all models."""
    metrics = {}
    
    for model_name, model_results in results.items():
        if model_name == 'metrics':
            continue
            
        mean = model_results['mean']
        lower = model_results['lower']
        upper = model_results['upper']
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((mean - true_sst)**2))
        mae = np.mean(np.abs(mean - true_sst))
        coverage = np.mean((lower <= true_sst) & (true_sst <= upper))
        ci_width = np.mean(upper - lower)
        
        metrics[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'ci_width': ci_width
        }
    
    return metrics

def create_comparison_plots(results, synthetic_data, test_ages, true_sst, figure_dir):
    """Create plots comparing the different methods."""
    # 1. Main comparison plot
    create_main_comparison_plot(results, synthetic_data, test_ages, true_sst, 
                              f"{figure_dir}/model_comparison.png")
    
    # 2. Metrics comparison
    create_metrics_plot(results['metrics'], f"{figure_dir}/metrics_comparison.png")
    
    # 3. MCMC uncertainty comparison
    create_uncertainty_comparison(results, test_ages, true_sst, 
                                f"{figure_dir}/uncertainty_comparison.png")

def create_main_comparison_plot(results, synthetic_data, test_ages, true_sst, filename):
    """Create main comparison plot with all models."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot data and true SST
    ax = axes[0]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    # Plot proxy data
    markers = ['o', 's']
    colors = ['blue', 'green']
    
    for i, (proxy_type, data) in enumerate(synthetic_data['proxy_data'].items()):
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
    
    ax.set_ylabel('SST (°C)')
    ax.set_title('Proxy Data and True SST')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot standard GP
    ax = axes[1]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    std_results = results['standard_gp']
    ax.plot(test_ages, std_results['mean'], 'b-', linewidth=2, label='Standard GP')
    ax.fill_between(test_ages, std_results['lower'], std_results['upper'], 
                   color='b', alpha=0.2, label='95% CI')
    
    ax.set_ylabel('SST (°C)')
    ax.set_title('Standard GP (No MCMC)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot Bayesian GP with fixed MCMC
    ax = axes[2]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    fixed_results = results['fixed_model']
    ax.plot(test_ages, fixed_results['mean'], 'r-', linewidth=2, label='Bayesian GP with Fixed MCMC')
    ax.fill_between(test_ages, fixed_results['lower'], fixed_results['upper'], 
                   color='r', alpha=0.2, label='95% CI')
    
    # Plot samples for visualization
    for i in range(min(10, fixed_results['samples'].shape[0])):
        ax.plot(test_ages, fixed_results['samples'][i], 'r-', linewidth=0.5, alpha=0.15)
    
    ax.set_xlabel('Age (kyr)')
    ax.set_ylabel('SST (°C)')
    ax.set_title('Bayesian GP with Fixed MCMC')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Reverse x-axis for geological convention
    for ax in axes:
        ax.set_xlim(max(test_ages), min(test_ages))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_metrics_plot(metrics, filename):
    """Create bar plot comparing metrics across methods."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Extract metrics
    model_names = list(metrics.keys())
    display_names = {
        'standard_gp': 'Standard GP',
        'original_model': 'BayesianGP (Naive MCMC)',
        'fixed_model': 'BayesianGP (Fixed MCMC)'
    }
    
    # Error metrics
    ax = axes[0]
    rmse_values = [metrics[model]['rmse'] for model in model_names]
    mae_values = [metrics[model]['mae'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, rmse_values, width, label='RMSE (°C)')
    ax.bar(x + width/2, mae_values, width, label='MAE (°C)')
    
    ax.set_ylabel('Error (°C)')
    ax.set_title('Prediction Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[name] for name in model_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Uncertainty metrics
    ax = axes[1]
    coverage_values = [metrics[model]['coverage'] for model in model_names]
    width_values = [metrics[model]['ci_width'] for model in model_names]
    
    ax.bar(x - width/2, coverage_values, width, label='Coverage (ideal: 0.95)')
    ax.bar(x + width/2, width_values, width, label='CI Width (°C)')
    
    ax.axhline(0.95, color='r', linestyle='--', alpha=0.7, label='Ideal Coverage (95%)')
    
    ax.set_ylabel('Value')
    ax.set_title('Uncertainty Quantification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[name] for name in model_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_uncertainty_comparison(results, test_ages, true_sst, filename):
    """Create plot comparing uncertainty quantification between methods."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot original model with naive MCMC
    ax = axes[0]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    orig_results = results['original_model']
    ax.plot(test_ages, orig_results['mean'], 'b-', linewidth=2, label='Bayesian GP (Naive MCMC)')
    ax.fill_between(test_ages, orig_results['lower'], orig_results['upper'], 
                   color='b', alpha=0.2, label='95% CI')
    
    # Plot samples
    for i in range(min(10, orig_results['samples'].shape[0])):
        ax.plot(test_ages, orig_results['samples'][i], 'b-', linewidth=0.5, alpha=0.15)
    
    ax.set_ylabel('SST (°C)')
    ax.set_title('Bayesian GP with Naive MCMC')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot fixed model
    ax = axes[1]
    ax.plot(test_ages, true_sst, 'k-', linewidth=2, label='True SST')
    
    fixed_results = results['fixed_model']
    ax.plot(test_ages, fixed_results['mean'], 'r-', linewidth=2, label='Bayesian GP (Fixed MCMC)')
    ax.fill_between(test_ages, fixed_results['lower'], fixed_results['upper'], 
                   color='r', alpha=0.2, label='95% CI')
    
    # Plot samples
    for i in range(min(10, fixed_results['samples'].shape[0])):
        ax.plot(test_ages, fixed_results['samples'][i], 'r-', linewidth=0.5, alpha=0.15)
    
    ax.set_xlabel('Age (kyr)')
    ax.set_ylabel('SST (°C)')
    ax.set_title('Bayesian GP with Fixed MCMC')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Reverse x-axis for geological convention
    for ax in axes:
        ax.set_xlim(max(test_ages), min(test_ages))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main execution function."""
    print("=" * 80)
    print("MCMC Implementation Test for Bayesian GP State-Space Models")
    print(f"Results directory: {output_dir}")
    print("=" * 80)
    
    try:
        # Test with sparse data
        results, synthetic_data = test_on_sparse_data(n_points=30, irregularity=0.8)
        
        # Add additional tests with different sparsity if desired
        test_on_sparse_data(n_points=15, irregularity=0.9)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("-" * 80)
        metrics = results['metrics']
        print("Performance Metrics:")
        for model_name, model_metrics in metrics.items():
            print(f"  {model_name}:")
            for metric_name, value in model_metrics.items():
                print(f"    {metric_name}: {value:.4f}")
        
        print("\nTest completed successfully.")
        print(f"All results saved to {output_dir}")
        print("=" * 80)
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())