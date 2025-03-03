"""
Test Script for Improved MCMC Implementation on Real Data

This version addresses:
  1) Parameter name mismatches (raw_constant vs. constant, etc.)
  2) Array vs. Tensor issues (.cpu() if param is already NumPy)
  3) Some numeric stability improvements (likelihood noise).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gpytorch
import pandas as pd

from bayesian_gp_state_space import BayesianGPStateSpaceModel
# from bayesian_gp_state_space import generate_synthetic_data_sparse  # not used if we do real data

# Create output directory
output_dir = "data/results/mcmc_final"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def run_mcmc_comparison_test():
    """
    Run a comparison test between the traditional and improved MCMC approaches
    using real ODP722 data (columns 7=Age.1, 8=UK37).
    """
    print("Running MCMC Comparison Test (Real Data)...")
    
    # -------------------------------------------------
    # 1) LOAD REAL DATA
    # -------------------------------------------------
    csv_file = "ODP722.csv"  # Provide the correct path if not in same folder
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    if "Age.1" not in df.columns or "UK37" not in df.columns:
        raise ValueError("CSV must have columns 'Age.1' and 'UK37'")

    age_all = df["Age.1"].values
    uk37_all = df["UK37"].values

    # Remove NaNs
    valid = ~np.isnan(age_all) & ~np.isnan(uk37_all)
    age_all = age_all[valid]
    uk37_all = uk37_all[valid]

    # Truncate to [0, 797.2]
    mask = (age_all >= 0) & (age_all <= 797.2)
    age_all = age_all[mask]
    uk37_all = uk37_all[mask]

    # Sort
    idx = np.argsort(age_all)
    age_all = age_all[idx]
    uk37_all = uk37_all[idx]

    # Make proxy_data for BayesianGPStateSpaceModel
    proxy_data = {
        'UK37': {
            'age': age_all,
            'value': uk37_all
        }
    }

    # -------------------------------------------------
    # 2) SET TEST POINTS (For Predictions)
    # -------------------------------------------------
    test_ages = np.linspace(0, 797.2, 300)
    # We have no "true_sst" - create dummy zeros just so code runs
    test_true_sst = np.zeros_like(test_ages)

    # -------------------------------------------------
    # 3) TRADITIONAL MCMC MODEL
    # -------------------------------------------------
    print("\n=== Training model with traditional MCMC approach ===")
    traditional_model = BayesianGPStateSpaceModel(
        proxy_types=['UK37'],
        kernel_type='combined',   # or whatever your model expects
        n_mcmc_samples=300,
        random_state=42
    )

    # Possibly force some higher baseline noise for stability:
    # e.g.:
    # with torch.no_grad():
    #     traditional_model.likelihood.noise = 0.1

    # Adjust if your `fit` method expects `use_improved_mcmc=False`.
    traditional_model.fit(proxy_data, training_iterations=300, use_improved_mcmc=False)

    # -------------------------------------------------
    # 4) IMPROVED MCMC MODEL
    # -------------------------------------------------
    print("\n=== Training model with improved MCMC approach ===")
    improved_model = BayesianGPStateSpaceModel(
        proxy_types=['UK37'],
        kernel_type='combined',
        n_mcmc_samples=300,
        random_state=42
    )
    improved_model.fit(proxy_data, training_iterations=300, use_improved_mcmc=True)

    # -------------------------------------------------
    # 5) MAKE PREDICTIONS
    # -------------------------------------------------
    print("\nMaking predictions with both models...")
    trad_mean, trad_lower, trad_upper, trad_samples = traditional_model.predict(
        test_ages, return_samples=True, n_samples=20
    )
    impr_mean, impr_lower, impr_upper, impr_samples = improved_model.predict(
        test_ages, return_samples=True, n_samples=20
    )

    # -------------------------------------------------
    # 6) PLOT
    # -------------------------------------------------
    print("Creating comparison plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # (A) Raw data
    ax = axes[0]
    ax.plot(age_all, uk37_all, 'ko', markersize=4, alpha=0.7, label='UK37 Data')
    ax.set_ylabel('UK37')
    ax.set_title('ODP722 Real Data (Truncated 0–797.2 ka)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # (B) Traditional MCMC
    ax = axes[1]
    ax.plot(age_all, uk37_all, 'ko', markersize=3, alpha=0.5, label='Data')
    ax.plot(test_ages, trad_mean, 'b-', linewidth=2, label='Traditional MCMC Mean')
    ax.fill_between(test_ages, trad_lower, trad_upper, color='b', alpha=0.2, label='95% CI')

    # Some sample draws
    for i in range(min(10, trad_samples.shape[0])):
        ax.plot(test_ages, trad_samples[i], 'b-', linewidth=0.5, alpha=0.1)

    ax.set_ylabel('Reconstruction?')
    ax.set_title('Traditional MCMC')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # (C) Improved MCMC
    ax = axes[2]
    ax.plot(age_all, uk37_all, 'ko', markersize=3, alpha=0.5, label='Data')
    ax.plot(test_ages, impr_mean, 'r-', linewidth=2, label='Improved MCMC Mean')
    ax.fill_between(test_ages, impr_lower, impr_upper, color='r', alpha=0.2, label='95% CI')

    # Some sample draws
    for i in range(min(10, impr_samples.shape[0])):
        ax.plot(test_ages, impr_samples[i], 'r-', linewidth=0.5, alpha=0.1)

    ax.set_ylabel('Reconstruction?')
    ax.set_xlabel('Age (kyr)')
    ax.set_title('Improved MCMC')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Reverse x-axis
    for ax in axes:
        ax.set_xlim(max(test_ages), min(test_ages))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mcmc_comparison.png", dpi=300)
    print(f"Saved comparison figure to {output_dir}/mcmc_comparison.png")

    # -------------------------------------------------
    # 7) Parameter Posteriors
    # -------------------------------------------------
    compare_parameter_posteriors(
        traditional_model, 
        improved_model, 
        figure_path=f"{output_dir}/parameter_posterior_comparison.png"
    )

    # -------------------------------------------------
    # 8) Compare Metrics (dummy, since we have no real truth)
    # -------------------------------------------------
    compare_metrics(
        traditional_model,
        improved_model,
        test_ages,
        test_true_sst,
        figure_path=f"{output_dir}/mcmc_metrics_comparison.png"
    )

    print("\nAll results saved to", output_dir)


def compare_parameter_posteriors(trad_model, impr_model, figure_path=None):
    """
    Compare posterior distributions of parameters between models,
    fixing array vs. tensor issues, and mapping param names if needed.
    """

    # Raw samples from each model
    trad_samples = trad_model.posterior_samples
    impr_samples = impr_model.posterior_samples

    # If your code names parameters differently, define a map:
    # e.g. "likelihood.noise" -> "likelihood.noise_covar.raw_noise"
    PARAM_NAME_MAP = {
        "likelihood.noise": "likelihood.noise_covar.raw_noise",
        "mean_module.constant": "mean_module.raw_constant",
        "rbf_kernel.outputscale": "rbf_kernel.raw_outputscale",
        "rbf_kernel.base_kernel.lengthscale": "rbf_kernel.base_kernel.raw_lengthscale",
        "periodic_kernel.outputscale": "periodic_kernel.raw_outputscale",
        "periodic_kernel.base_kernel.lengthscale": "periodic_kernel.base_kernel.raw_lengthscale",
        "periodic_kernel.base_kernel.period_length": "periodic_kernel.base_kernel.raw_period_length",
    }
    
    # If your MCMC code stores only the 'raw_*' param names, 
    # you can just iterate over the keys in trad_samples, impr_samples:
    trad_keys = set(trad_samples.keys())
    impr_keys = set(impr_samples.keys())
    common_keys = sorted(list(trad_keys & impr_keys))

    if not common_keys:
        print("No overlapping parameters found in MCMC samples. Skipping posteriors.")
        return

    fig, axes = plt.subplots(len(common_keys), 1, figsize=(8, 3 * len(common_keys)))
    if len(common_keys) == 1:
        axes = [axes]

    for i, param_name in enumerate(common_keys):
        ax = axes[i]
        tvals = trad_samples[param_name]
        ivals = impr_samples[param_name]

        # Check if Tensors or NumPy
        if torch.is_tensor(tvals):
            tvals = tvals.detach().cpu().numpy()
        if torch.is_tensor(ivals):
            ivals = ivals.detach().cpu().numpy()

        # Flatten
        tvals = tvals.ravel()
        ivals = ivals.ravel()

        # Hist
        ax.hist(tvals, bins=30, alpha=0.5, density=True, color='blue', label='Traditional')
        ax.hist(ivals, bins=30, alpha=0.5, density=True, color='red', label='Improved')

        tmean = np.mean(tvals)
        t025, t975 = np.percentile(tvals, [2.5, 97.5])
        ax.axvline(tmean, color='blue', linewidth=2)
        ax.axvline(t025, color='blue', linestyle='--', linewidth=1)
        ax.axvline(t975, color='blue', linestyle='--', linewidth=1)

        imean = np.mean(ivals)
        i025, i975 = np.percentile(ivals, [2.5, 97.5])
        ax.axvline(imean, color='red', linewidth=2)
        ax.axvline(i025, color='red', linestyle='--', linewidth=1)
        ax.axvline(i975, color='red', linestyle='--', linewidth=1)

        ax.set_title(f"Posterior for '{param_name}'")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Text box
        ax.text(0.02, 0.95,
                f"Traditional:\nMean={tmean:.4f}\n95% CI=[{t025:.3f}, {t975:.3f}]",
                transform=ax.transAxes, va='top', color='blue',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.text(0.02, 0.70,
                f"Improved:\nMean={imean:.4f}\n95% CI=[{i025:.3f}, {i975:.3f}]",
                transform=ax.transAxes, va='top', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    if figure_path:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Parameter posterior figure saved to {figure_path}")


def compare_metrics(trad_model, impr_model, test_x, true_y, figure_path=None):
    """
    Compare performance metrics between models (RMSE, MAE, R^2, coverage, ci_width).
    With no real 'true_y', these are dummy stats.
    """
    trad_metrics = trad_model.evaluate(test_x, true_y)
    impr_metrics = impr_model.evaluate(test_x, true_y)

    # Convert coverage to percentage
    trad_metrics['coverage'] *= 100
    impr_metrics['coverage'] *= 100

    metrics_to_compare = ['rmse', 'mae', 'r2', 'coverage', 'ci_width']
    metric_labels = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R²',
        'coverage': 'Coverage (%)',
        'ci_width': 'CI Width'
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(metrics_to_compare))

    trad_vals = [trad_metrics[m] for m in metrics_to_compare]
    impr_vals = [impr_metrics[m] for m in metrics_to_compare]

    bars_trad = ax.bar(x - bar_width / 2, trad_vals, bar_width, color='blue', alpha=0.6, label='Traditional')
    bars_impr = ax.bar(x + bar_width / 2, impr_vals, bar_width, color='red', alpha=0.6, label='Improved')

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_compare])
    ax.set_ylabel("Value")
    ax.set_title("Performance Metric Comparison (No Real Truth)")
    ax.legend()

    # annotate
    def add_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", 
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels(bars_trad)
    add_labels(bars_impr)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if figure_path:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison figure saved to {figure_path}")

    # Save metrics to CSV
    diff_vals = [iv - tv for iv, tv in zip(impr_vals, trad_vals)]
    metrics_df = pd.DataFrame({
        'Metric': [metric_labels[m] for m in metrics_to_compare],
        'Traditional': trad_vals,
        'Improved': impr_vals,
        'Difference': diff_vals
    })
    csv_path = figure_path.replace('.png', '.csv') if figure_path else f"{output_dir}/metrics_comparison.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to {csv_path}")

if __name__ == "__main__":
    run_mcmc_comparison_test()