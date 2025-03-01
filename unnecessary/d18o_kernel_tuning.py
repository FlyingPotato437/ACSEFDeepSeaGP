"""
d18o_kernel_tuning.py - Hyperparameter tuning for δ¹⁸O combined kernel GP model

This script refines hyperparameters for the δ¹⁸O combined kernel GP model to optimize 
latent SST reconstruction performance. It's based on the fixed latent_sst_reconstruction_improved.py 
but focuses only on tuning the combined kernel for δ¹⁸O proxy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from scipy import special

# Import required classes and functions from the improved latent SST reconstruction
from latent_sst_reconstruction_improved import (
    SyntheticPaleoData, 
    LatentGPModel,
    calibrate_proxy_to_sst,
    evaluate_gp_model,
    plot_gp_predictions,
    plot_residuals
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set up directory for results
output_dir = "final_figures"
os.makedirs(output_dir, exist_ok=True)

# Constants from the main script
TIME_MIN = 0      # Start time in kyr BP
TIME_MAX = 500    # End time in kyr BP
N_POINTS = 300    # Number of data points
TEST_SIZE = 0.2   # Proportion of data for testing

# δ18O proxy calibration parameters
D18O_ALPHA = -4.38   # Slope (α1)
D18O_BETA = 16.9     # Intercept (β1)
D18O_SIGMA = 0.5     # Noise standard deviation (σ1)

def train_gp_model_with_custom_params(train_x, train_y, rbf_lengthscale, periodic_lengthscale, 
                                     periodic_period=None, n_iterations=150, lr=0.05):
    """
    Train a GP model with custom hyperparameter initialization.
    
    Parameters:
        train_x (tensor): Training inputs (ages)
        train_y (tensor): Training targets (proxy-derived SST)
        rbf_lengthscale (float): Initial value for RBF kernel lengthscale
        periodic_lengthscale (float): Initial value for Periodic kernel lengthscale
        periodic_period (float, optional): Initial value for Periodic kernel period length
        n_iterations (int): Number of training iterations
        lr (float): Learning rate for optimizer
        
    Returns:
        tuple: (trained model, likelihood, training losses)
    """
    # Initialize likelihood and model with combined kernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LatentGPModel(train_x, train_y, likelihood, kernel_type='combined')
    
    # Set model and likelihood to training mode
    model.train()
    likelihood.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=lr)
    
    # Set custom kernel hyperparameters
    model.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor(rbf_lengthscale)
    model.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor(periodic_lengthscale)
    
    # Set periodic kernel period length if provided
    if periodic_period is not None:
        model.covar_module.kernels[1].base_kernel.period_length = torch.tensor(periodic_period)
    
    # Loss function (negative log marginal likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    losses = []
    for i in range(n_iterations):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Print progress
            if (i + 1) % 30 == 0:
                print(f'Iteration {i+1}/{n_iterations} - Loss: {loss.item():.4f}')
        except Exception as e:
            print(f"Warning: Optimization error at iteration {i+1}: {e}")
            # Use last good loss value
            if losses:
                losses.append(losses[-1])
            else:
                losses.append(100.0)  # Default high loss
    
    return model, likelihood, losses

def plot_parameter_comparison(results_df, original_metrics):
    """
    Plot comparison of hyperparameter combinations.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame with hyperparameter tuning results
        original_metrics (dict): Original model metrics for comparison
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort by RMSE (best first)
    results_df = results_df.sort_values('true_rmse')
    
    # Add original model as reference
    ref_row = pd.DataFrame({
        'rbf_lengthscale': [original_metrics['rbf_lengthscale']],
        'periodic_lengthscale': [original_metrics['periodic_lengthscale']],
        'true_rmse': [original_metrics['true_rmse']],
        'true_r2': [original_metrics['true_r2']],
        'true_mae': [original_metrics['true_mae']],
        'label': ['Original']
    })
    results_df = pd.concat([results_df, ref_row], ignore_index=True)
    
    # Prepare data for plotting
    labels = results_df['label'].tolist()
    rmse_values = results_df['true_rmse'].tolist()
    r2_values = results_df['true_r2'].tolist()
    mae_values = results_df['true_mae'].tolist()
    
    # Create colormap - green for better than original, red for worse
    original_idx = labels.index('Original')
    original_rmse = rmse_values[original_idx]
    
    colors = ['lightgreen' if rmse <= original_rmse else 'salmon' for rmse in rmse_values]
    colors[original_idx] = 'gold'  # Highlight original in gold
    
    # Plot RMSE (lower is better)
    bars = axes[0].bar(labels, rmse_values, color=colors)
    axes[0].set_ylabel('RMSE (°C)')
    axes[0].set_title('Root Mean Square Error (lower is better)')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, value + 0.05,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot R² (higher is better)
    bars = axes[1].bar(labels, r2_values, color=colors)
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score (higher is better)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, r2_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, value + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot MAE (lower is better)
    bars = axes[2].bar(labels, mae_values, color=colors)
    axes[2].set_ylabel('MAE (°C)')
    axes[2].set_title('Mean Absolute Error (lower is better)')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, mae_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, value + 0.05,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add overall title
    improvement = ((original_rmse - min(rmse_values))/original_rmse * 100)
    fig.suptitle(f'δ¹⁸O Combined Kernel Hyperparameter Tuning Results\nBest Configuration: {results_df.iloc[0]["label"]} (RMSE Improvement: {improvement:.2f}%)', 
                fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    return fig

def main():
    """Main function to tune hyperparameters for δ¹⁸O combined kernel GP model."""
    start_time = time.time()
    print("Starting δ¹⁸O Combined Kernel Hyperparameter Tuning...")
    
    # Step I: Generate synthetic paleoclimate data
    print("\nGenerating synthetic paleoclimate data...")
    synth_data = SyntheticPaleoData(n_points=N_POINTS)
    dataset = synth_data.generate_dataset()
    
    # Extract data
    ages = dataset['age']
    true_sst = dataset['true_sst']
    d18o_values = dataset['d18o']
    
    # Calibrate proxy to SST
    d18o_sst = calibrate_proxy_to_sst(d18o_values, 'd18o')
    
    # Split data into training and testing sets for δ18O
    d18o_train_idx, d18o_test_idx = train_test_split(
        np.arange(len(ages)), test_size=TEST_SIZE, random_state=42
    )
    
    # Sort indices by age
    d18o_train_idx = sorted(d18o_train_idx)
    d18o_test_idx = sorted(d18o_test_idx)
    
    # Prepare δ18O training and testing data
    d18o_train_x = torch.tensor(ages[d18o_train_idx], dtype=torch.float32).reshape(-1, 1)
    d18o_train_y = torch.tensor(d18o_sst[d18o_train_idx], dtype=torch.float32)
    d18o_test_x = torch.tensor(ages[d18o_test_idx], dtype=torch.float32).reshape(-1, 1)
    d18o_test_y = torch.tensor(d18o_sst[d18o_test_idx], dtype=torch.float32)
    d18o_true_y = torch.tensor(true_sst[d18o_test_idx], dtype=torch.float32)  # True latent SST
    
    # Tensor for full domain (for smooth plotting)
    full_x = torch.tensor(ages, dtype=torch.float32).reshape(-1, 1)
    full_y = torch.tensor(true_sst, dtype=torch.float32)
    
    # Step II: Get original model performance as baseline
    print("\nTraining baseline model (original hyperparameters)...")
    
    # Train original model using default hyperparameter initialization
    original_model, original_likelihood, _ = train_gp_model_with_custom_params(
        d18o_train_x, d18o_train_y, 
        rbf_lengthscale=5.0,          # Original approximate value
        periodic_lengthscale=3.0,     # Original approximate value
        periodic_period=0.6           # Original approximate value
    )
    
    # Get optimized parameters
    original_params = original_model.get_kernel_params()
    
    # Evaluate original model
    print("\nEvaluating baseline model...")
    original_metrics, original_predictions = evaluate_gp_model(
        original_model, original_likelihood, d18o_test_x, d18o_test_y, d18o_true_y
    )
    
    # Add metrics to params
    original_params.update({
        'true_rmse': original_metrics['true_rmse'],
        'true_mae': original_metrics['true_mae'],
        'true_r2': original_metrics['true_r2']
    })
    
    print(f"\nBaseline RBF lengthscale: {original_params['rbf_lengthscale']:.4f}")
    print(f"Baseline Periodic lengthscale: {original_params['periodic_lengthscale']:.4f}")
    print(f"Baseline Period length: {original_params['periodic_period_length']:.4f}")
    
    # Step III: Test variations of hyperparameters
    print("\nTesting variations of hyperparameters...")
    
    # Create variations for rbf_lengthscale and periodic_lengthscale
    base_rbf_lengthscale = original_params['rbf_lengthscale']
    base_periodic_lengthscale = original_params['periodic_lengthscale']
    
    # Define variations as percentages of original values
    rbf_variations = [0.8, 0.9, 1.0, 1.1, 1.2]  # ±20% variation
    periodic_variations = [0.8, 0.9, 1.0, 1.1, 1.2]  # ±20% variation
    
    # Keep period length fixed as it's usually related to known cycles
    fixed_period_length = original_params['periodic_period_length']
    
    # Store results
    results = []
    
    # Test combinations of hyperparameters
    for rbf_var in rbf_variations:
        for periodic_var in periodic_variations:
            # Skip the original combination
            if rbf_var == 1.0 and periodic_var == 1.0:
                continue
                
            rbf_lengthscale = base_rbf_lengthscale * rbf_var
            periodic_lengthscale = base_periodic_lengthscale * periodic_var
            
            # Create label
            label = f"RBF={rbf_var:.1f}x, Per={periodic_var:.1f}x"
            print(f"\nTesting {label}...")
            print(f"  RBF lengthscale: {rbf_lengthscale:.4f}")
            print(f"  Periodic lengthscale: {periodic_lengthscale:.4f}")
            
            # Train model with custom hyperparameters
            model, likelihood, _ = train_gp_model_with_custom_params(
                d18o_train_x, d18o_train_y, 
                rbf_lengthscale=rbf_lengthscale,
                periodic_lengthscale=periodic_lengthscale,
                periodic_period=fixed_period_length
            )
            
            # Evaluate model
            metrics, predictions = evaluate_gp_model(
                model, likelihood, d18o_test_x, d18o_test_y, d18o_true_y
            )
            
            # Store results
            results.append({
                'label': label,
                'rbf_lengthscale': rbf_lengthscale,
                'periodic_lengthscale': periodic_lengthscale,
                'periodic_period_length': fixed_period_length,
                'true_rmse': metrics['true_rmse'],
                'true_mae': metrics['true_mae'],
                'true_r2': metrics['true_r2'],
                'true_bias': metrics['true_bias'],
                'true_std_err': metrics['true_std_err'],
                'model': model,
                'likelihood': likelihood,
                'predictions': predictions
            })
    
    # Create DataFrame for visualization
    results_df = pd.DataFrame([
        {
            'label': r['label'],
            'rbf_lengthscale': r['rbf_lengthscale'],
            'periodic_lengthscale': r['periodic_lengthscale'],
            'true_rmse': r['true_rmse'],
            'true_mae': r['true_mae'],
            'true_r2': r['true_r2']
        }
        for r in results
    ])
    
    # Step IV: Find best hyperparameter combination and visualize
    best_result = min(results, key=lambda x: x['true_rmse'])
    
    print("\n===== HYPERPARAMETER TUNING RESULTS =====")
    print(f"\nOriginal combined kernel hyperparameters:")
    print(f"  RBF lengthscale: {original_params['rbf_lengthscale']:.4f}")
    print(f"  Periodic lengthscale: {original_params['periodic_lengthscale']:.4f}")
    print(f"  Periodic period length: {original_params['periodic_period_length']:.4f}")
    print(f"  RMSE vs TRUE latent SST: {original_params['true_rmse']:.4f}°C")
    print(f"  MAE vs TRUE latent SST: {original_params['true_mae']:.4f}°C")
    print(f"  R² vs TRUE latent SST: {original_params['true_r2']:.4f}")
    
    print(f"\nBEST tuned kernel hyperparameters ({best_result['label']}):")
    print(f"  RBF lengthscale: {best_result['rbf_lengthscale']:.4f}")
    print(f"  Periodic lengthscale: {best_result['periodic_lengthscale']:.4f}")
    print(f"  Periodic period length: {best_result['periodic_period_length']:.4f}")
    print(f"  RMSE vs TRUE latent SST: {best_result['true_rmse']:.4f}°C")
    print(f"  MAE vs TRUE latent SST: {best_result['true_mae']:.4f}°C")
    print(f"  R² vs TRUE latent SST: {best_result['true_r2']:.4f}")
    
    # Calculate improvement percentage
    rmse_improvement = ((original_params['true_rmse'] - best_result['true_rmse']) / 
                        original_params['true_rmse']) * 100
    r2_improvement = ((best_result['true_r2'] - original_params['true_r2']) / 
                      abs(original_params['true_r2'])) * 100
    
    print(f"\nImprovements:")
    print(f"  RMSE reduction: {rmse_improvement:.2f}%")
    print(f"  R² improvement: {r2_improvement:.2f}%")
    
    # Plot comparison of hyperparameter combinations
    fig_comparison = plot_parameter_comparison(results_df, original_params)
    fig_comparison.savefig(os.path.join(output_dir, "d18o_hyperparameter_tuning.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_comparison)
    
    # Plot predictions for best model
    fig_best = plot_gp_predictions(
        best_result['model'], best_result['likelihood'], 
        d18o_test_x, d18o_test_y, full_x, full_y, proxy_type='d18o'
    )
    fig_best.savefig(os.path.join(output_dir, "d18o_tuned_model_prediction.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_best)
    
    # Plot predictions for original model
    fig_original = plot_gp_predictions(
        original_model, original_likelihood, 
        d18o_test_x, d18o_test_y, full_x, full_y, proxy_type='d18o'
    )
    fig_original.savefig(os.path.join(output_dir, "d18o_baseline_model_prediction.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_original)
    
    # Plot residuals for best model
    fig_resid = plot_residuals(
        best_result['predictions'], d18o_true_y.numpy(), 
        "Tuned Combined", 'd18o'
    )
    fig_resid.savefig(os.path.join(output_dir, "d18o_tuned_model_residuals.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_resid)
    
    # Create side-by-side comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert tensors to numpy
    test_x_np = d18o_test_x.numpy().flatten()
    test_y_np = d18o_test_y.numpy()
    true_x_np = full_x.numpy().flatten()
    true_y_np = full_y.numpy()
    
    # Original model predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_full_orig = original_likelihood(original_model(full_x))
        mean_full_orig = observed_full_orig.mean.numpy()
    
    # Best model predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_full_best = best_result['likelihood'](best_result['model'](full_x))
        mean_full_best = observed_full_best.mean.numpy()
    
    # Plot original model
    axes[0].plot(true_x_np, true_y_np, 'k-', linewidth=1.5, label='True SST (Latent)')
    axes[0].plot(test_x_np, test_y_np, 'bo', markersize=4, alpha=0.6, label='Calibrated δ¹⁸O (Observed)')
    axes[0].plot(true_x_np, mean_full_orig, 'g-', linewidth=2, label=f'GP Reconstructed Latent SST (Baseline)')
    
    metrics_text = (f"GP vs TRUE LATENT SST METRICS\n"
                    f"RMSE: {original_params['true_rmse']:.3f}°C\n"
                    f"MAE: {original_params['true_mae']:.3f}°C\n"
                    f"R²: {original_params['true_r2']:.3f}")
    
    axes[0].text(0.02, 0.95, metrics_text, transform=axes[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0].set_xlabel('Age (kyr BP)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Baseline Model')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Plot best model
    axes[1].plot(true_x_np, true_y_np, 'k-', linewidth=1.5, label='True SST (Latent)')
    axes[1].plot(test_x_np, test_y_np, 'bo', markersize=4, alpha=0.6, label='Calibrated δ¹⁸O (Observed)')
    axes[1].plot(true_x_np, mean_full_best, 'r-', linewidth=2, label=f'GP Reconstructed Latent SST (Tuned)')
    
    metrics_text = (f"GP vs TRUE LATENT SST METRICS\n"
                    f"RMSE: {best_result['true_rmse']:.3f}°C\n"
                    f"MAE: {best_result['true_mae']:.3f}°C\n"
                    f"R²: {best_result['true_r2']:.3f}")
    
    axes[1].text(0.02, 0.95, metrics_text, transform=axes[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1].set_xlabel('Age (kyr BP)')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title(f'Tuned Model ({best_result["label"]})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Add overall title
    fig.suptitle(f'δ¹⁸O Combined Kernel Model Comparison: Baseline vs Tuned\nRMSE Improvement: {rmse_improvement:.2f}%', 
                fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(os.path.join(output_dir, "d18o_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, "d18o_hyperparameter_tuning_results.csv"), index=False)
    
    # Print final message
    print(f"\nResults saved to {output_dir}")
    end_time = time.time()
    print(f"\nδ¹⁸O Combined Kernel Hyperparameter Tuning completed! Time elapsed: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()