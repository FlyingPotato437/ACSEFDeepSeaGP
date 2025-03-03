"""
Test script for the Bayesian Gaussian Process State-Space model.

This script demonstrates the use of the Bayesian GP State-Space model
for paleoclimate reconstruction from multiple proxies with uncertainty 
quantification and abrupt transition detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

from bayesian_gp_state_space import (
    BayesianGPStateSpaceModel, 
    generate_synthetic_data_sparse,
    test_bayesian_gp_state_space_model
)

# Ensure output directory exists
output_dir = "data/results/bayesian_gp"
os.makedirs(output_dir, exist_ok=True)

def run_simple_demo():
    """
    Run a simple demonstration of the Bayesian GP State-Space model on synthetic data.
    """
    print("Running simple demonstration of Bayesian GP State-Space model...")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=60,
        age_min=0,
        age_max=300,
        irregularity=0.6,
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    
    # Initialize model
    print("Initializing Bayesian GP State-Space model...")
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37'],
        kernel_type='combined',
        n_mcmc_samples=500,
        random_state=42
    )
    
    # Fit model
    print("Fitting model to proxy data...")
    model.fit(proxy_data, training_iterations=500)
    
    # Make predictions
    print("Making predictions...")
    test_ages = np.linspace(0, 300, 300)
    mean, lower_ci, upper_ci = model.predict(test_ages)
    
    # Detect abrupt transitions
    print("Detecting abrupt transitions...")
    transitions = model.detect_abrupt_transitions(test_ages)
    print(f"Detected transitions at ages: {transitions}")
    
    # Evaluate model
    print("Evaluating model performance...")
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    metrics = model.evaluate(test_ages, test_true_sst)
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot reconstruction
    print("Plotting reconstruction...")
    fig = model.plot_reconstruction(
        test_ages, 
        proxy_data, 
        test_true_sst,
        detected_transitions=transitions,
        figure_path=f"{output_dir}/simple_demo_reconstruction.png"
    )
    
    # Plot parameter posterior
    print("Plotting parameter posterior distributions...")
    fig = model.plot_parameter_posterior(
        figure_path=f"{output_dir}/simple_demo_parameter_posterior.png"
    )
    
    # Plot spectral analysis
    print("Plotting spectral analysis...")
    fig = model.plot_spectral_analysis(
        test_ages,
        figure_path=f"{output_dir}/simple_demo_spectral_analysis.png"
    )
    
    print(f"All results saved to {output_dir}")
    
    return model, synthetic_data, metrics


def run_multiproxy_comparison():
    """
    Compare the performance of the model with different combinations of proxies.
    """
    print("Running multiproxy comparison...")
    
    # Generate synthetic data with multiple proxies
    print("Generating synthetic data with multiple proxies...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=80,
        age_min=0,
        age_max=500,
        irregularity=0.7,
        proxy_types=['d18O', 'UK37', 'Mg_Ca'],
        random_state=42
    )
    
    # Set higher jitter for numerical stability
    import gpytorch
    gpytorch.settings.cholesky_jitter._global_float_value = 1e-4
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    
    # Common test ages
    test_ages = np.linspace(0, 500, 500)
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    
    # Proxy combinations to test
    proxy_combinations = [
        ['d18O'],
        ['UK37'],
        ['Mg_Ca'],
        ['d18O', 'UK37'],
        ['d18O', 'Mg_Ca'],
        ['UK37', 'Mg_Ca'],
        ['d18O', 'UK37', 'Mg_Ca']
    ]
    
    # Store results
    results = []
    
    # Test each combination
    for proxies in proxy_combinations:
        print(f"\nTesting proxy combination: {proxies}")
        
        # Create a subset of the proxy data
        proxy_subset = {p: proxy_data[p] for p in proxies}
        
        # Initialize model
        model = BayesianGPStateSpaceModel(
            proxy_types=proxies,
            kernel_type='combined',
            n_mcmc_samples=200,
            random_state=42
        )
        
        # Fit model
        model.fit(proxy_subset, training_iterations=300)
        
        # Make predictions
        mean, lower_ci, upper_ci = model.predict(test_ages)
        
        # Evaluate model
        metrics = model.evaluate(test_ages, test_true_sst)
        
        # Store results
        results.append({
            'proxies': proxies,
            'metrics': metrics,
            'model': model,
            'mean': mean,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        })
        
        # Plot reconstruction for this combination
        proxy_name = '_'.join(proxies)
        model.plot_reconstruction(
            test_ages,
            proxy_subset,
            test_true_sst,
            figure_path=f"{output_dir}/multiproxy_{proxy_name}_reconstruction.png"
        )
    
    # Plot comparison of all proxy combinations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # RMSE comparison
    proxy_labels = ['+'.join(r['proxies']) for r in results]
    rmse_values = [r['metrics']['rmse'] for r in results]
    r2_values = [r['metrics']['r2'] for r in results]
    
    # Color each bar based on the number of proxies
    colors = ['lightblue', 'lightgreen', 'orange']
    bar_colors = [colors[len(p.split('+'))-1] for p in proxy_labels]
    
    # RMSE plot
    ax1.bar(proxy_labels, rmse_values, color=bar_colors)
    ax1.set_ylabel('RMSE (°C)')
    ax1.set_title('RMSE by Proxy Combination')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_values):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    # R² plot
    ax2.bar(proxy_labels, r2_values, color=bar_colors)
    ax2.set_ylabel('R²')
    ax2.set_title('R² by Proxy Combination')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(r2_values):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multiproxy_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Multiproxy comparison plot saved to {output_dir}/multiproxy_comparison.png")
    
    # Save metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame([
        {
            'proxies': '+'.join(r['proxies']),
            'rmse': r['metrics']['rmse'],
            'mae': r['metrics']['mae'],
            'r2': r['metrics']['r2'],
            'coverage': r['metrics']['coverage'],
            'ci_width': r['metrics']['ci_width']
        }
        for r in results
    ])
    
    metrics_df.to_csv(f"{output_dir}/multiproxy_comparison_metrics.csv", index=False)
    print(f"Multiproxy comparison metrics saved to {output_dir}/multiproxy_comparison_metrics.csv")
    
    return results


def run_sparse_data_test():
    """
    Test model performance on extremely sparse and irregularly sampled data.
    """
    print("Testing model performance on extremely sparse data...")
    
    # Generate very sparse synthetic data
    print("Generating sparse synthetic data...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=30,  # Very few points
        age_min=0,
        age_max=500,
        irregularity=0.9,  # Very irregular
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    
    # Common test ages
    test_ages = np.linspace(0, 500, 500)
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    
    # Initialize model
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37'],
        kernel_type='combined',
        n_mcmc_samples=500,
        random_state=42
    )
    
    # Fit model
    model.fit(proxy_data, training_iterations=500)
    
    # Make predictions
    mean, lower_ci, upper_ci = model.predict(test_ages)
    
    # Evaluate model
    metrics = model.evaluate(test_ages, test_true_sst)
    print("Performance metrics on sparse data:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot reconstruction
    fig = model.plot_reconstruction(
        test_ages, 
        proxy_data, 
        test_true_sst,
        figure_path=f"{output_dir}/sparse_data_reconstruction.png"
    )
    
    # Custom plot showing data sparsity challenges
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true SST
    ax.plot(test_ages, test_true_sst, 'k-', linewidth=1.5, label='True SST')
    
    # Plot model prediction with uncertainty
    ax.plot(test_ages, mean, 'r-', linewidth=2, label='Bayesian GP Reconstruction')
    ax.fill_between(test_ages, lower_ci, upper_ci, color='r', alpha=0.2, label='95% CI')
    
    # Highlight data gaps
    for proxy_type, data in proxy_data.items():
        # Plot data points
        if proxy_type == 'd18O':
            color = 'blue'
            marker = 'o'
        else:  # UK37
            color = 'green'
            marker = 's'
        
        # Convert proxy to SST
        params = synthetic_data['calibration_params'][proxy_type]
        proxy_sst = (data['value'] - params['intercept']) * params['inverse_slope']
        
        ax.scatter(data['age'], proxy_sst, color=color, marker=marker, s=30, 
                 label=f'{proxy_type} proxy', alpha=0.7)
        
        # Find gaps > 50 kyr
        ages = data['age']
        for i in range(len(ages)-1):
            gap = ages[i+1] - ages[i]
            if gap > 50:
                # Highlight the gap
                gap_center = (ages[i] + ages[i+1]) / 2
                ax.axvspan(ages[i], ages[i+1], color=color, alpha=0.1)
                ax.text(gap_center, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                       f'Gap: {gap:.0f} kyr', ha='center', color=color)
    
    ax.set_xlabel('Age (kyr)')
    ax.set_ylabel('SST (°C)')
    ax.set_title('Bayesian GP State-Space Model Performance on Sparse, Irregular Data')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(test_ages), min(test_ages))  # Reverse x-axis
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sparse_data_challenges.png", dpi=300, bbox_inches='tight')
    print(f"Sparse data challenge plot saved to {output_dir}/sparse_data_challenges.png")
    
    return model, synthetic_data, metrics


if __name__ == "__main__":
    print("Testing Bayesian GP State-Space Model for Paleoclimate Reconstruction\n")
    
    # Run the comprehensive test first
    print("\n=== RUNNING COMPREHENSIVE TEST ===\n")
    test_bayesian_gp_state_space_model()
    
    # Run simple demo
    print("\n=== RUNNING SIMPLE DEMO ===\n")
    model, data, metrics = run_simple_demo()
    
    # Run multiproxy comparison
    print("\n=== RUNNING MULTIPROXY COMPARISON ===\n")
    multiproxy_results = run_multiproxy_comparison()
    
    # Run sparse data test
    print("\n=== RUNNING SPARSE DATA TEST ===\n")
    sparse_model, sparse_data, sparse_metrics = run_sparse_data_test()
    
    print("\nAll tests completed successfully. Results saved to:", output_dir)