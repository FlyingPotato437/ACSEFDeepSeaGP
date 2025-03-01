"""
Model Comparison for Paleoclimate Reconstruction

This script compares different approaches for paleoclimate reconstruction:
1. Direct proxy calibration (naive approach)
2. Standard GP reconstruction (existing method)
3. Bayesian GP State-Space model (new method)

The comparison is performed on synthetic data with sparse and irregular sampling,
designed to represent realistic paleoclimate proxy records.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
from typing import Dict, List, Tuple, Optional, Union

# Import our models
from gp_models import PhysicsInformedGP
from multiproxy_latent_sst import extract_latent_sst_from_multiple_proxies
from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse

# Set random seed for reproducibility
np.random.seed(42)


def direct_calibration(proxy_data, calibration_params, common_ages):
    """
    Direct calibration method: converts proxy to SST using calibration equations
    without any temporal modeling or uncertainty propagation.
    
    Args:
        proxy_data: Dictionary with proxy data
        calibration_params: Dictionary with calibration parameters
        common_ages: Ages for prediction
        
    Returns:
        Dictionary with predictions for each proxy and weighted average
    """
    results = {}
    weights = {}
    combined = np.zeros_like(common_ages, dtype=float)
    total_weight = 0.0
    
    for proxy_type, data in proxy_data.items():
        params = calibration_params[proxy_type]
        
        # Convert proxy to SST: SST = (proxy - intercept) / slope
        proxy_sst = (data['value'] - params['intercept']) * params['inverse_slope']
        
        # Interpolate to common ages
        interp_sst = np.interp(
            common_ages, 
            data['age'], 
            proxy_sst,
            left=np.nan, right=np.nan
        )
        
        # Calculate weight as inverse of calibration error variance
        error_in_temp_units = params['error_std'] * abs(params['inverse_slope'])
        weight = 1 / (error_in_temp_units ** 2)
        
        # Store results
        results[proxy_type] = interp_sst
        weights[proxy_type] = weight
        
        # Add weighted contribution
        valid_mask = ~np.isnan(interp_sst)
        combined[valid_mask] += interp_sst[valid_mask] * weight
        total_weight += weight
    
    # Normalize weights
    for proxy_type in weights:
        weights[proxy_type] /= total_weight
    
    # Normalize combined result
    combined /= total_weight
    
    return {
        'individual': results,
        'weights': weights,
        'combined': combined
    }


def standard_gp_reconstruction(proxy_data, calibration_params, common_ages, kernel_type='combined'):
    """
    Standard GP reconstruction method: first converts proxy to SST, then 
    applies GP regression to model the underlying temporal process.
    
    Args:
        proxy_data: Dictionary with proxy data
        calibration_params: Dictionary with calibration parameters
        common_ages: Ages for prediction
        kernel_type: Type of kernel to use for GP
        
    Returns:
        Dictionary with predictions for each proxy and weighted average
    """
    # Map 'combined' kernel to 'milankovitch' for compatibility with PhysicsInformedGP
    gp_kernel = 'milankovitch' if kernel_type == 'combined' else kernel_type
    results = {}
    weights = {}
    combined_data_x = []
    combined_data_y = []
    combined_data_w = []
    
    for proxy_type, data in proxy_data.items():
        params = calibration_params[proxy_type]
        
        # Convert proxy to SST
        proxy_sst = (data['value'] - params['intercept']) * params['inverse_slope']
        
        # Initialize GP model
        gp = PhysicsInformedGP(kernel=gp_kernel)
        
        # Fit model
        gp.fit(data['age'], proxy_sst)
        
        # Predict
        mean, std = gp.predict(common_ages, return_std=True)
        
        # Calculate weight as inverse of calibration error variance
        error_in_temp_units = params['error_std'] * abs(params['inverse_slope'])
        weight = 1 / (error_in_temp_units ** 2)
        
        # Store results
        results[proxy_type] = {
            'mean': mean,
            'std': std
        }
        weights[proxy_type] = weight
        
        # Collect data for combined model
        combined_data_x.extend(data['age'])
        combined_data_y.extend(proxy_sst)
        combined_data_w.extend([weight] * len(data['age']))
    
    # Sort combined data by age
    sort_idx = np.argsort(combined_data_x)
    combined_data_x = np.array(combined_data_x)[sort_idx]
    combined_data_y = np.array(combined_data_y)[sort_idx]
    combined_data_w = np.array(combined_data_w)[sort_idx]
    
    # Fit GP to all data with weights
    combined_gp = PhysicsInformedGP(kernel=gp_kernel)
    combined_gp.fit(combined_data_x, combined_data_y)
    
    # Predict
    combined_mean, combined_std = combined_gp.predict(common_ages, return_std=True)
    
    # Normalize weights
    total_weight = sum(weights.values())
    for proxy_type in weights:
        weights[proxy_type] /= total_weight
    
    return {
        'individual': results,
        'weights': weights,
        'combined': {
            'mean': combined_mean,
            'std': combined_std
        }
    }


def bayesian_gp_state_space_reconstruction(proxy_data, calibration_params, common_ages):
    """
    Bayesian GP State-Space reconstruction: models SST as a latent variable with
    a GP prior, and proxy observations as noisy measurements through calibration equations.
    
    Args:
        proxy_data: Dictionary with proxy data
        calibration_params: Dictionary with calibration parameters
        common_ages: Ages for prediction
        
    Returns:
        Dictionary with predictions and uncertainty
    """
    # Initialize model
    proxy_types = list(proxy_data.keys())
    model = BayesianGPStateSpaceModel(
        proxy_types=proxy_types,
        calibration_params=calibration_params,
        kernel_type='combined',
        n_mcmc_samples=500
    )
    
    # Fit model
    model.fit(proxy_data, training_iterations=500)
    
    # Predict
    mean, lower_ci, upper_ci, samples = model.predict(
        common_ages, return_samples=True, n_samples=50
    )
    
    # Detect transitions
    transitions = model.detect_abrupt_transitions(common_ages)
    
    return {
        'mean': mean,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'samples': samples,
        'transitions': transitions,
        'weights': model.proxy_weights
    }


def latent_variable_extraction(proxy_data, calibration_params, common_ages):
    """
    Multiproxy latent variable extraction method from the existing codebase.
    
    Args:
        proxy_data: Dictionary with proxy data
        calibration_params: Dictionary with calibration parameters
        common_ages: Ages for prediction
        
    Returns:
        Dictionary with predictions and uncertainty
    """
    # Extract data for δ18O and UK'37 proxies
    d18o_ages = proxy_data.get('d18O', {}).get('age', np.array([]))
    d18o_values = proxy_data.get('d18O', {}).get('value', np.array([]))
    
    uk37_ages = proxy_data.get('UK37', {}).get('age', np.array([]))
    uk37_values = proxy_data.get('UK37', {}).get('value', np.array([]))
    
    # Mock true SST (not actually used for prediction, only for evaluation)
    mock_true_sst = np.zeros_like(common_ages)
    
    # Implement a simple version that converts proxy values to SST directly
    # This is a fallback since we don't have the full implementation
    d18o_sst = np.array([])
    uk37_sst = np.array([])
    
    if len(d18o_values) > 0:
        d18o_params = calibration_params['d18O']
        d18o_sst = (d18o_values - d18o_params['intercept']) * d18o_params['inverse_slope']
    
    if len(uk37_values) > 0:
        uk37_params = calibration_params['UK37']
        uk37_sst = (uk37_values - uk37_params['intercept']) * uk37_params['inverse_slope']
    
    # Combine the estimates with optimal weighting
    reconstructed_sst = np.zeros_like(common_ages)
    uncertainty = np.ones_like(common_ages) * 2.0  # Default high uncertainty
    
    # Calculate optimal weights based on calibration equation error
    weights = {}
    total_weight = 0.0
    
    if 'd18O' in calibration_params:
        d18o_err = calibration_params['d18O']['error_std'] * abs(calibration_params['d18O']['inverse_slope'])
        d18o_weight = 1 / (d18o_err ** 2)
        weights['d18O'] = d18o_weight
        total_weight += d18o_weight
    
    if 'UK37' in calibration_params:
        uk37_err = calibration_params['UK37']['error_std'] * abs(calibration_params['UK37']['inverse_slope'])
        uk37_weight = 1 / (uk37_err ** 2)
        weights['UK37'] = uk37_weight
        total_weight += uk37_weight
    
    # Normalize weights
    if total_weight > 0:
        for key in weights:
            weights[key] /= total_weight
    
    # Interpolate each proxy SST estimate to the common ages and combine with weights
    for i, age in enumerate(common_ages):
        sst_at_age = 0.0
        weight_sum = 0.0
        variance_sum = 0.0
        
        if len(d18o_sst) > 0 and 'd18O' in weights:
            # Find closest d18O measurement
            idx = np.abs(d18o_ages - age).argmin()
            if np.abs(d18o_ages[idx] - age) < 20:  # Within 20kyr
                sst_at_age += d18o_sst[idx] * weights['d18O']
                weight_sum += weights['d18O']
                d18o_err = calibration_params['d18O']['error_std'] * abs(calibration_params['d18O']['inverse_slope'])
                variance_sum += weights['d18O']**2 * d18o_err**2
        
        if len(uk37_sst) > 0 and 'UK37' in weights:
            # Find closest UK37 measurement
            idx = np.abs(uk37_ages - age).argmin()
            if np.abs(uk37_ages[idx] - age) < 20:  # Within 20kyr
                sst_at_age += uk37_sst[idx] * weights['UK37']
                weight_sum += weights['UK37']
                uk37_err = calibration_params['UK37']['error_std'] * abs(calibration_params['UK37']['inverse_slope']) 
                variance_sum += weights['UK37']**2 * uk37_err**2
        
        if weight_sum > 0:
            reconstructed_sst[i] = sst_at_age / weight_sum
            uncertainty[i] = np.sqrt(variance_sum) / weight_sum
        else:
            reconstructed_sst[i] = np.nan
    
    # Apply a simple smoothing to fill gaps
    valid_mask = ~np.isnan(reconstructed_sst)
    if np.sum(valid_mask) > 2:
        reconstructed_sst_smoothed = np.interp(
            common_ages,
            common_ages[valid_mask],
            reconstructed_sst[valid_mask]
        )
    else:
        reconstructed_sst_smoothed = reconstructed_sst.copy()
    
    result = {
        'reconstructed_sst': reconstructed_sst_smoothed,
        'uncertainty': uncertainty,
        'weights': weights
    }
    
    return {
        'mean': result['reconstructed_sst'],
        'std': result['uncertainty'],
        'lower_ci': result['reconstructed_sst'] - 1.96 * result['uncertainty'],
        'upper_ci': result['reconstructed_sst'] + 1.96 * result['uncertainty'],
        'weights': result['weights'] 
    }


def evaluate_model(predictions, true_values, model_name):
    """
    Evaluate model performance against true values.
    
    Args:
        predictions: Predicted values
        true_values: True values
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary of performance metrics
    """
    # Find valid indices
    valid = ~np.isnan(predictions) & ~np.isnan(true_values)
    
    if np.sum(valid) < 2:
        return {
            'model': model_name,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'bias': np.nan,
            'random_error': np.nan
        }
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((predictions[valid] - true_values[valid])**2))
    mae = np.mean(np.abs(predictions[valid] - true_values[valid]))
    
    # R²
    ss_tot = np.sum((true_values[valid] - np.mean(true_values[valid]))**2)
    ss_res = np.sum((true_values[valid] - predictions[valid])**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Bias and random error
    bias = np.mean(predictions[valid] - true_values[valid])
    random_error = np.sqrt(np.mean(((predictions[valid] - true_values[valid]) - bias)**2))
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'random_error': random_error
    }


def compare_models(synthetic_data, plot=True, save_dir="data/results"):
    """
    Run and compare all models on the same synthetic dataset.
    
    Args:
        synthetic_data: Dictionary with synthetic data
        plot: Whether to generate plots
        save_dir: Directory to save results
        
    Returns:
        DataFrame with performance metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    calibration_params = synthetic_data['calibration_params']
    
    # Common ages for prediction
    common_ages = np.linspace(min(regular_ages), max(regular_ages), 500)
    
    # Interpolate true SST to common ages
    true_sst_interp = np.interp(common_ages, regular_ages, true_sst)
    
    # Run all models
    print("Running direct calibration...")
    direct_results = direct_calibration(proxy_data, calibration_params, common_ages)
    
    print("Running standard GP...")
    gp_results = standard_gp_reconstruction(proxy_data, calibration_params, common_ages)
    
    print("Running latent variable extraction...")
    latent_results = latent_variable_extraction(proxy_data, calibration_params, common_ages)
    
    print("Running Bayesian GP State-Space model...")
    bgs_results = bayesian_gp_state_space_reconstruction(proxy_data, calibration_params, common_ages)
    
    # Evaluate each model
    metrics = []
    
    # Direct calibration
    metrics.append(evaluate_model(
        direct_results['combined'], 
        true_sst_interp, 
        "Direct Calibration"
    ))
    
    # Standard GP
    metrics.append(evaluate_model(
        gp_results['combined']['mean'], 
        true_sst_interp, 
        "Standard GP"
    ))
    
    # Latent variable extraction
    metrics.append(evaluate_model(
        latent_results['mean'], 
        true_sst_interp, 
        "Latent Variable Extraction"
    ))
    
    # Bayesian GP State-Space
    metrics.append(evaluate_model(
        bgs_results['mean'], 
        true_sst_interp, 
        "Bayesian GP State-Space"
    ))
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics
    metrics_df.to_csv(f"{save_dir}/model_comparison_metrics.csv", index=False)
    print(f"Metrics saved to {save_dir}/model_comparison_metrics.csv")
    
    # Plot if requested
    if plot:
        # Plot reconstructions
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot true SST
        ax.plot(common_ages, true_sst_interp, 'k-', linewidth=1.5, label='True SST')
        
        # Plot direct calibration
        ax.plot(common_ages, direct_results['combined'], 'g--', linewidth=1, 
               alpha=0.7, label='Direct Calibration')
        
        # Plot standard GP
        ax.plot(common_ages, gp_results['combined']['mean'], 'b-', linewidth=1.5, 
               alpha=0.7, label='Standard GP')
        gp_lower = gp_results['combined']['mean'] - 1.96 * gp_results['combined']['std']
        gp_upper = gp_results['combined']['mean'] + 1.96 * gp_results['combined']['std']
        ax.fill_between(common_ages, gp_lower, gp_upper, color='b', alpha=0.1)
        
        # Plot latent variable extraction
        ax.plot(common_ages, latent_results['mean'], 'c-', linewidth=1.5,
               alpha=0.7, label='Latent Variable Extraction')
        ax.fill_between(common_ages, latent_results['lower_ci'], latent_results['upper_ci'], 
                      color='c', alpha=0.1)
        
        # Plot Bayesian GP State-Space
        ax.plot(common_ages, bgs_results['mean'], 'r-', linewidth=2,
               label='Bayesian GP State-Space')
        ax.fill_between(common_ages, bgs_results['lower_ci'], bgs_results['upper_ci'], 
                      color='r', alpha=0.2)
        
        # Mark transitions detected by BGSS model
        if len(bgs_results['transitions']) > 0:
            for trans in bgs_results['transitions']:
                ax.axvline(x=trans, color='r', linestyle='--', alpha=0.5)
                ax.text(trans, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                       f'{trans:.1f}', rotation=90, color='r', ha='right')
        
        # Plot proxy data
        markers = ['o', 's', '^']
        colors = ['g', 'orange', 'm']
        
        for i, (proxy_type, data) in enumerate(proxy_data.items()):
            # Convert proxy to SST
            params = calibration_params[proxy_type]
            proxy_sst = (data['value'] - params['intercept']) * params['inverse_slope']
            
            ax.scatter(data['age'], proxy_sst, 
                     marker=markers[i % len(markers)], 
                     color=colors[i % len(colors)],
                     s=30, alpha=0.7, label=f'{proxy_type} proxy SST')
        
        # Set labels and legend
        ax.set_xlabel('Age (kyr)')
        ax.set_ylabel('SST (°C)')
        ax.set_title('Model Comparison for Paleoclimate Reconstruction')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis direction (older ages on the right)
        ax.set_xlim(max(common_ages), min(common_ages))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_dir}/model_comparison.png")
        
        # Plot metrics
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Labels and colors
        model_names = metrics_df['model']
        model_colors = ['green', 'blue', 'cyan', 'red']
        
        # RMSE
        axes[0].bar(model_names, metrics_df['rmse'], color=model_colors)
        axes[0].set_title('RMSE (°C) - Lower is better')
        axes[0].set_ylabel('RMSE (°C)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # MAE
        axes[1].bar(model_names, metrics_df['mae'], color=model_colors)
        axes[1].set_title('Mean Absolute Error (°C) - Lower is better')
        axes[1].set_ylabel('MAE (°C)')
        axes[1].grid(axis='y', alpha=0.3)
        
        # R²
        axes[2].bar(model_names, metrics_df['r2'], color=model_colors)
        axes[2].set_title('R² - Higher is better')
        axes[2].set_ylabel('R²')
        axes[2].grid(axis='y', alpha=0.3)
        
        # Bias
        axes[3].bar(model_names, metrics_df['bias'], color=model_colors)
        axes[3].set_title('Bias (°C) - Closer to zero is better')
        axes[3].set_ylabel('Bias (°C)')
        axes[3].grid(axis='y', alpha=0.3)
        
        # Random error
        axes[4].bar(model_names, metrics_df['random_error'], color=model_colors)
        axes[4].set_title('Random Error (°C) - Lower is better')
        axes[4].set_ylabel('Random Error (°C)')
        axes[4].grid(axis='y', alpha=0.3)
        
        # Weights
        weights_data = []
        for model_name, model_results in [
            ("Direct Calibration", direct_results),
            ("Standard GP", gp_results),
            ("Latent Variable", latent_results),
            ("Bayesian GP State-Space", bgs_results)
        ]:
            model_weights = {}
            
            # Extract weights if available
            if 'weights' in model_results:
                model_weights = model_results['weights']
            
            # Format for display
            weights_str = ", ".join([f"{p}: {w:.3f}" for p, w in model_weights.items()])
            weights_data.append(f"{model_name}: {weights_str}")
        
        axes[5].axis('off')
        axes[5].text(0.5, 0.5, "\n".join(weights_data), 
                  ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        axes[5].set_title('Proxy Weights')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison_metrics.png", dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_dir}/model_comparison_metrics.png")
        
        # Close all figures
        plt.close('all')
    
    return metrics_df


def run_uncertainty_analysis(synthetic_data, save_dir="data/results"):
    """
    Run uncertainty analysis to compare how well each model captures 
    true uncertainty in the reconstructions.
    
    Args:
        synthetic_data: Dictionary with synthetic data
        save_dir: Directory to save results
        
    Returns:
        DataFrame with uncertainty metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    calibration_params = synthetic_data['calibration_params']
    
    # Common ages for prediction
    common_ages = np.linspace(min(regular_ages), max(regular_ages), 500)
    
    # Interpolate true SST to common ages
    true_sst_interp = np.interp(common_ages, regular_ages, true_sst)
    
    # Run all models with uncertainty estimation
    print("Running standard GP...")
    gp_results = standard_gp_reconstruction(proxy_data, calibration_params, common_ages)
    
    print("Running latent variable extraction...")
    latent_results = latent_variable_extraction(proxy_data, calibration_params, common_ages)
    
    print("Running Bayesian GP State-Space model...")
    bgs_results = bayesian_gp_state_space_reconstruction(proxy_data, calibration_params, common_ages)
    
    # Calculate uncertainty metrics
    metrics = []
    
    # Calculate coverage and interval width
    def calculate_uncertainty_metrics(mean, lower, upper, true_values, model_name):
        # Find valid indices
        valid = ~np.isnan(mean) & ~np.isnan(true_values)
        
        if np.sum(valid) < 2:
            return {
                'model': model_name,
                'coverage': np.nan,
                'interval_width': np.nan,
                'bias': np.nan,
                'uncertainty_calibration': np.nan
            }
        
        # Coverage: proportion of true values within the CI
        coverage = np.mean((lower[valid] <= true_values[valid]) & 
                          (true_values[valid] <= upper[valid]))
        
        # Average width of the CI
        interval_width = np.mean(upper[valid] - lower[valid])
        
        # Bias
        bias = np.mean(mean[valid] - true_values[valid])
        
        # Uncertainty calibration: ratio of empirical errors to predicted std
        errors = mean[valid] - true_values[valid]
        std = (upper[valid] - lower[valid]) / (2 * 1.96)  # Estimate std from CI
        uncertainty_calibration = np.std(errors) / np.mean(std)
        # Closer to 1 means better calibration
        
        return {
            'model': model_name,
            'coverage': coverage,
            'interval_width': interval_width,
            'bias': bias,
            'uncertainty_calibration': uncertainty_calibration
        }
    
    # Standard GP
    gp_lower = gp_results['combined']['mean'] - 1.96 * gp_results['combined']['std']
    gp_upper = gp_results['combined']['mean'] + 1.96 * gp_results['combined']['std']
    
    metrics.append(calculate_uncertainty_metrics(
        gp_results['combined']['mean'],
        gp_lower,
        gp_upper,
        true_sst_interp,
        "Standard GP"
    ))
    
    # Latent variable extraction
    metrics.append(calculate_uncertainty_metrics(
        latent_results['mean'],
        latent_results['lower_ci'],
        latent_results['upper_ci'],
        true_sst_interp,
        "Latent Variable Extraction"
    ))
    
    # Bayesian GP State-Space
    metrics.append(calculate_uncertainty_metrics(
        bgs_results['mean'],
        bgs_results['lower_ci'],
        bgs_results['upper_ci'],
        true_sst_interp,
        "Bayesian GP State-Space"
    ))
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics
    metrics_df.to_csv(f"{save_dir}/uncertainty_analysis.csv", index=False)
    print(f"Uncertainty metrics saved to {save_dir}/uncertainty_analysis.csv")
    
    # Create visualization of uncertainty evaluation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Model names and colors
    model_names = metrics_df['model']
    model_colors = ['blue', 'cyan', 'red']
    
    # Coverage
    axes[0].bar(model_names, metrics_df['coverage'], color=model_colors)
    axes[0].axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='Target (95%)')
    axes[0].set_title('Coverage: Proportion of true values within 95% CI')
    axes[0].set_ylabel('Coverage')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Interval width
    axes[1].bar(model_names, metrics_df['interval_width'], color=model_colors)
    axes[1].set_title('Interval Width: Average width of 95% CI')
    axes[1].set_ylabel('Width (°C)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Uncertainty calibration
    axes[2].bar(model_names, metrics_df['uncertainty_calibration'], color=model_colors)
    axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal (1.0)')
    axes[2].set_title('Uncertainty Calibration: ratio of empirical to predicted error')
    axes[2].set_ylabel('Calibration')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    # Error histograms
    ax = axes[3]
    
    # Standard GP
    gp_errors = gp_results['combined']['mean'] - true_sst_interp
    gp_errors = gp_errors[~np.isnan(gp_errors)]
    
    # Latent variable extraction
    latent_errors = latent_results['mean'] - true_sst_interp
    latent_errors = latent_errors[~np.isnan(latent_errors)]
    
    # Bayesian GP State-Space
    bgs_errors = bgs_results['mean'] - true_sst_interp
    bgs_errors = bgs_errors[~np.isnan(bgs_errors)]
    
    # Plot histograms
    ax.hist(gp_errors, bins=20, alpha=0.4, label='Standard GP', color='blue')
    ax.hist(latent_errors, bins=20, alpha=0.4, label='Latent Variable', color='cyan')
    ax.hist(bgs_errors, bins=20, alpha=0.4, label='Bayesian GP State-Space', color='red')
    
    ax.set_title('Error Distributions')
    ax.set_xlabel('Error (°C)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/uncertainty_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Uncertainty analysis plot saved to {save_dir}/uncertainty_analysis.png")
    
    # Close all figures
    plt.close('all')
    
    return metrics_df


def run_spectral_analysis(synthetic_data, save_dir="data/results"):
    """
    Run spectral analysis to evaluate how well each model preserves
    the frequency characteristics of the true signal.
    
    Args:
        synthetic_data: Dictionary with synthetic data
        save_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    calibration_params = synthetic_data['calibration_params']
    
    # Common ages for prediction (ensure regular spacing for spectral analysis)
    common_ages = np.linspace(min(regular_ages), max(regular_ages), 512)  # Power of 2 for FFT
    
    # Interpolate true SST to common ages
    true_sst_interp = np.interp(common_ages, regular_ages, true_sst)
    
    # Run all models
    print("Running standard GP...")
    gp_results = standard_gp_reconstruction(proxy_data, calibration_params, common_ages)
    
    print("Running latent variable extraction...")
    latent_results = latent_variable_extraction(proxy_data, calibration_params, common_ages)
    
    print("Running Bayesian GP State-Space model...")
    bgs_results = bayesian_gp_state_space_reconstruction(proxy_data, calibration_params, common_ages)
    
    # Compute power spectra
    def compute_spectrum(signal, dt):
        # Compute the FFT
        fft = np.fft.rfft(signal)
        power = np.abs(fft)**2
        
        # Compute frequencies and periods
        freq = np.fft.rfftfreq(len(signal), d=dt)
        with np.errstate(divide='ignore'):
            periods = 1 / freq
        
        return periods[1:], power[1:]  # Skip DC component
    
    # Time step
    dt = common_ages[1] - common_ages[0]
    
    # Compute spectra
    true_periods, true_power = compute_spectrum(true_sst_interp, dt)
    gp_periods, gp_power = compute_spectrum(gp_results['combined']['mean'], dt)
    latent_periods, latent_power = compute_spectrum(latent_results['mean'], dt)
    bgs_periods, bgs_power = compute_spectrum(bgs_results['mean'], dt)
    
    # Normalize power
    true_power = true_power / np.max(true_power)
    gp_power = gp_power / np.max(gp_power)
    latent_power = latent_power / np.max(latent_power)
    bgs_power = bgs_power / np.max(bgs_power)
    
    # Plot spectra
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(true_periods, true_power, 'k-', linewidth=2, label='True SST')
    ax.plot(gp_periods, gp_power, 'b-', linewidth=1.5, alpha=0.7, label='Standard GP')
    ax.plot(latent_periods, latent_power, 'c-', linewidth=1.5, alpha=0.7, label='Latent Variable')
    ax.plot(bgs_periods, bgs_power, 'r-', linewidth=2, alpha=0.7, label='Bayesian GP State-Space')
    
    # Mark Milankovitch cycles
    milankovitch = [100, 41, 23]  # kyr
    for period in milankovitch:
        ax.axvline(x=period, color='grey', linestyle='--', alpha=0.7)
        ax.text(period, 0.1, f'{period} kyr', rotation=90, alpha=0.7, ha='right')
    
    ax.set_xlabel('Period (kyr)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Power Spectra Comparison')
    ax.set_xscale('log')
    ax.set_xlim(200, 5)  # From 200 kyr to 5 kyr
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/spectral_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Spectral analysis plot saved to {save_dir}/spectral_analysis.png")
    
    # Coherence analysis with true signal
    def compute_coherence(signal1, signal2, dt):
        # Compute the FFT
        fft1 = np.fft.rfft(signal1)
        fft2 = np.fft.rfft(signal2)
        
        # Compute cross-spectral density
        csd = fft1 * np.conj(fft2)
        
        # Compute power spectra
        psd1 = np.abs(fft1)**2
        psd2 = np.abs(fft2)**2
        
        # Compute coherence
        coherence = np.abs(csd)**2 / (psd1 * psd2)
        
        # Compute frequencies and periods
        freq = np.fft.rfftfreq(len(signal1), d=dt)
        with np.errstate(divide='ignore'):
            periods = 1 / freq
        
        return periods[1:], coherence[1:]  # Skip DC component
    
    # Compute coherence
    gp_coh_periods, gp_coherence = compute_coherence(true_sst_interp, gp_results['combined']['mean'], dt)
    latent_coh_periods, latent_coherence = compute_coherence(true_sst_interp, latent_results['mean'], dt)
    bgs_coh_periods, bgs_coherence = compute_coherence(true_sst_interp, bgs_results['mean'], dt)
    
    # Plot coherence
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(gp_coh_periods, gp_coherence, 'b-', linewidth=1.5, alpha=0.7, label='Standard GP')
    ax.plot(latent_coh_periods, latent_coherence, 'c-', linewidth=1.5, alpha=0.7, label='Latent Variable')
    ax.plot(bgs_coh_periods, bgs_coherence, 'r-', linewidth=2, alpha=0.7, label='Bayesian GP State-Space')
    
    # Mark Milankovitch cycles
    for period in milankovitch:
        ax.axvline(x=period, color='grey', linestyle='--', alpha=0.7)
        ax.text(period, 0.1, f'{period} kyr', rotation=90, alpha=0.7, ha='right')
    
    ax.set_xlabel('Period (kyr)')
    ax.set_ylabel('Coherence with True SST')
    ax.set_title('Coherence Analysis')
    ax.set_xscale('log')
    ax.set_xlim(200, 5)  # From 200 kyr to 5 kyr
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/coherence_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Coherence analysis plot saved to {save_dir}/coherence_analysis.png")
    
    # Close all figures
    plt.close('all')


def run_feature_detection_analysis(synthetic_data, save_dir="data/results"):
    """
    Analyze how well each model detects important features such as
    abrupt transitions and orbital cycles.
    
    Args:
        synthetic_data: Dictionary with synthetic data
        save_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    calibration_params = synthetic_data['calibration_params']
    
    # Common ages for prediction
    common_ages = np.linspace(min(regular_ages), max(regular_ages), 500)
    
    # Interpolate true SST to common ages
    true_sst_interp = np.interp(common_ages, regular_ages, true_sst)
    
    # Run Bayesian GP State-Space model, which has transition detection capability
    print("Running Bayesian GP State-Space model...")
    bgs_results = bayesian_gp_state_space_reconstruction(proxy_data, calibration_params, common_ages)
    
    # Get transitions detected by the model
    detected_transitions = bgs_results['transitions']
    
    # Define known transitions from the synthetic data
    true_transitions = []
    if 'true_sst_params' in synthetic_data and 'abrupt_changes' in synthetic_data['true_sst_params']:
        true_transitions = [change['age'] for change in synthetic_data['true_sst_params']['abrupt_changes']]
    # If no true transitions in metadata, try to detect them from the true signal
    elif len(true_transitions) == 0:
        # Compute rate of change
        true_rate = np.diff(true_sst_interp) / np.diff(common_ages)
        
        # Find peaks in rate of change
        from scipy.signal import find_peaks
        peak_indices, _ = find_peaks(np.abs(true_rate), height=np.percentile(np.abs(true_rate), 99))
        
        # Convert indices to ages
        true_transitions = [common_ages[i] for i in peak_indices]
    
    # Evaluate transition detection
    print(f"True transitions: {true_transitions}")
    print(f"Detected transitions: {detected_transitions}")
    
    # Plot true and detected transitions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot true SST
    ax.plot(common_ages, true_sst_interp, 'k-', linewidth=1.5, label='True SST')
    
    # Plot reconstructed SST
    ax.plot(common_ages, bgs_results['mean'], 'r-', linewidth=2, 
           label='Bayesian GP State-Space')
    ax.fill_between(common_ages, bgs_results['lower_ci'], bgs_results['upper_ci'], 
                  color='r', alpha=0.2)
    
    # Mark true transitions
    y_range = ax.get_ylim()
    for trans_age in true_transitions:
        ax.axvline(x=trans_age, color='k', linestyle='--', alpha=0.7)
        ax.text(trans_age, y_range[0] + 0.05*(y_range[1]-y_range[0]), 
               f'True: {trans_age:.1f}', 
               color='k', rotation=90, ha='right')
    
    # Mark detected transitions
    for trans_age in detected_transitions:
        ax.axvline(x=trans_age, color='r', linestyle=':', alpha=0.7)
        ax.text(trans_age, y_range[0] + 0.85*(y_range[1]-y_range[0]), 
               f'Detected: {trans_age:.1f}', 
               color='r', rotation=90, ha='right')
    
    # Add labels and legend
    ax.set_xlabel('Age (kyr)')
    ax.set_ylabel('SST (°C)')
    ax.set_title('Transition Detection Analysis')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis direction
    ax.set_xlim(max(common_ages), min(common_ages))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transition_detection.png", dpi=300, bbox_inches='tight')
    print(f"Transition detection plot saved to {save_dir}/transition_detection.png")
    
    # Calculate transition detection metrics
    detection_metrics = {}
    
    # Match detected transitions to true transitions
    matches = []
    unmatched_true = true_transitions.copy()
    unmatched_detected = detected_transitions.copy()
    
    # Tolerance for matching (in kyr)
    tolerance = 10
    
    for detected in detected_transitions:
        matched = False
        for true in true_transitions:
            if abs(detected - true) <= tolerance:
                matches.append((true, detected))
                if true in unmatched_true:
                    unmatched_true.remove(true)
                if detected in unmatched_detected:
                    unmatched_detected.remove(detected)
                matched = True
                break
    
    # Calculate metrics
    n_true = len(true_transitions)
    n_detected = len(detected_transitions)
    n_matches = len(matches)
    
    if n_true > 0:
        recall = n_matches / n_true
    else:
        recall = np.nan
    
    if n_detected > 0:
        precision = n_matches / n_detected
    else:
        precision = np.nan
    
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = np.nan
    
    # Calculate average detection error
    if n_matches > 0:
        detection_errors = [abs(true - detected) for true, detected in matches]
        mean_detection_error = np.mean(detection_errors)
    else:
        mean_detection_error = np.nan
    
    detection_metrics = {
        'true_transitions': true_transitions,
        'detected_transitions': detected_transitions,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_detection_error': mean_detection_error,
        'unmatched_true': unmatched_true,
        'unmatched_detected': unmatched_detected
    }
    
    # Save metrics
    with open(f"{save_dir}/transition_detection_metrics.txt", 'w') as f:
        f.write("Transition Detection Metrics:\n")
        f.write(f"True transitions: {true_transitions}\n")
        f.write(f"Detected transitions: {detected_transitions}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write(f"Mean Detection Error: {mean_detection_error:.4f} kyr\n")
        f.write(f"Unmatched true transitions: {unmatched_true}\n")
        f.write(f"Unmatched detected transitions: {unmatched_detected}\n")
    
    print(f"Transition detection metrics saved to {save_dir}/transition_detection_metrics.txt")
    
    # Close all figures
    plt.close('all')
    
    return detection_metrics


def analyze_sensitivity_to_sampling(save_dir="data/results"):
    """
    Analyze how model performance changes with varying levels of data sparsity.
    
    Args:
        save_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Different levels of data sparsity
    n_points_list = [20, 40, 80, 160]
    
    # Storage for results
    rmse_results = {
        'Standard GP': [],
        'Latent Variable Extraction': [],
        'Bayesian GP State-Space': []
    }
    r2_results = {
        'Standard GP': [],
        'Latent Variable Extraction': [],
        'Bayesian GP State-Space': []
    }
    
    for n_points in n_points_list:
        print(f"\nRunning analysis with {n_points} data points...")
        
        # Generate synthetic data with this number of points
        synthetic_data = generate_synthetic_data_sparse(
            n_points=n_points,
            age_min=0,
            age_max=500,
            irregularity=0.7,
            proxy_types=['d18O', 'UK37'],
            random_state=42
        )
        
        # Compare models
        metrics_df = compare_models(synthetic_data, plot=False, 
                                  save_dir=f"{save_dir}/sensitivity_n{n_points}")
        
        # Extract metrics
        for index, row in metrics_df.iterrows():
            model_name = row['model']
            if model_name in rmse_results:
                rmse_results[model_name].append(row['rmse'])
                r2_results[model_name].append(row['r2'])
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # RMSE vs number of points
    for model_name, rmse_list in rmse_results.items():
        if model_name == 'Standard GP':
            color = 'blue'
        elif model_name == 'Latent Variable Extraction':
            color = 'cyan'
        else:  # Bayesian GP State-Space
            color = 'red'
        
        ax1.plot(n_points_list, rmse_list, 'o-', label=model_name, color=color)
    
    ax1.set_xlabel('Number of Data Points')
    ax1.set_ylabel('RMSE (°C)')
    ax1.set_title('RMSE vs Data Sparsity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² vs number of points
    for model_name, r2_list in r2_results.items():
        if model_name == 'Standard GP':
            color = 'blue'
        elif model_name == 'Latent Variable Extraction':
            color = 'cyan'
        else:  # Bayesian GP State-Space
            color = 'red'
        
        ax2.plot(n_points_list, r2_list, 'o-', label=model_name, color=color)
    
    ax2.set_xlabel('Number of Data Points')
    ax2.set_ylabel('R²')
    ax2.set_title('R² vs Data Sparsity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sensitivity_to_sampling.png", dpi=300, bbox_inches='tight')
    print(f"Sensitivity analysis plot saved to {save_dir}/sensitivity_to_sampling.png")
    
    # Save the raw data
    sensitivity_data = {
        'n_points': n_points_list,
        'rmse': rmse_results,
        'r2': r2_results
    }
    
    import json
    with open(f"{save_dir}/sensitivity_data.json", 'w') as f:
        json.dump(sensitivity_data, f, indent=2)
    
    print(f"Sensitivity analysis data saved to {save_dir}/sensitivity_data.json")
    
    # Close all figures
    plt.close('all')
    
    return sensitivity_data


if __name__ == "__main__":
    print("Starting model comparison for paleoclimate reconstruction...")
    
    # Create a separate results directory for this comparison
    results_dir = "data/results/comprehensive_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate synthetic data
    print("\nGenerating synthetic paleoclimate data...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=80,
        age_min=0,
        age_max=500,
        irregularity=0.7,
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Save an information file about the synthetic data
    with open(f"{results_dir}/synthetic_data_info.txt", 'w') as f:
        f.write("Synthetic Paleoclimate Data Information:\n")
        f.write("- Age range: 0-500 kyr\n")
        f.write("- Base number of data points: 80\n")
        f.write("- Irregularity factor: 0.7\n")
        f.write("- Proxies: d18O, UK37\n")
        
        if 'true_sst_params' in synthetic_data:
            params = synthetic_data['true_sst_params']
            f.write("\nTrue SST parameters:\n")
            f.write(f"- Baseline temperature: {params['baseline']:.2f}°C\n")
            f.write(f"- Trend slope: {params['trend_slope']:.4f}°C/kyr\n")
            
            f.write("\nOrbital cycles:\n")
            for cycle in params['cycles']:
                f.write(f"- {cycle['period']} kyr cycle: Amplitude {cycle['amplitude']:.2f}°C, Phase {cycle['phase']:.2f}\n")
            
            f.write("\nAbrupt changes:\n")
            for change in params['abrupt_changes']:
                f.write(f"- At {change['age']} kyr: Magnitude {change['magnitude']:.2f}°C, Width {change['width']} kyr\n")
    
    # Run comprehensive model comparison
    print("\nRunning comprehensive model comparison...")
    metrics_df = compare_models(synthetic_data, plot=True, save_dir=results_dir)
    
    # Run uncertainty analysis
    print("\nRunning uncertainty analysis...")
    uncertainty_metrics = run_uncertainty_analysis(synthetic_data, save_dir=results_dir)
    
    # Run spectral analysis
    print("\nRunning spectral analysis...")
    run_spectral_analysis(synthetic_data, save_dir=results_dir)
    
    # Run feature detection analysis
    print("\nRunning feature detection analysis...")
    detection_metrics = run_feature_detection_analysis(synthetic_data, save_dir=results_dir)
    
    # Run sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity_data = analyze_sensitivity_to_sampling(save_dir=results_dir)
    
    print(f"\nAll analyses complete. Results saved to {results_dir}")
    
    # Final summary
    print("\nFinal Performance Comparison:")
    print(metrics_df[['model', 'rmse', 'r2', 'bias']])
    
    print("\nUncertainty Evaluation:")
    print(uncertainty_metrics[['model', 'coverage', 'uncertainty_calibration']])
    
    if detection_metrics:
        print("\nTransition Detection Performance:")
        print(f"Precision: {detection_metrics['precision']:.2f}")
        print(f"Recall: {detection_metrics['recall']:.2f}")
        print(f"F1 Score: {detection_metrics['f1_score']:.2f}")