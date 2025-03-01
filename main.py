"""
main.py - Main script for comparing paleoclimate reconstruction methods

This script performs a comprehensive comparison of different modeling approaches
for reconstructing paleoclimate time series from proxy data, with a focus on
capturing Milankovitch orbital cycles.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import custom modules - simple flat imports
from ar_models import AR1Model, AR2Model
from gp_models import PhysicsInformedGP
from synthetic_data import SyntheticPaleoData
from model_comparison import ModelComparison
from spectral import calculate_power_spectrum
from evaluation import calculate_bic
from calibration import proxy_to_sst

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """
    Main function to run the paleoclimate reconstruction comparison.
    """
    print("Starting Paleoclimate Reconstruction Project")
    print("============================================")
    
    # Create output directory
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate synthetic data with known Milankovitch cycles
    print("\nStep 1: Generating synthetic paleoclimate data...")
    
    # Initialize synthetic data generator
    synth = SyntheticPaleoData(start_time=0, end_time=800, noise_level=0.5, random_seed=42)
    
    # Set Milankovitch cycle parameters
    synth.set_cycle_parameters('eccentricity', period=100.0, amplitude=2.0, phase=0.0)
    synth.set_cycle_parameters('obliquity', period=41.0, amplitude=1.0, phase=np.pi/4)
    synth.set_cycle_parameters('precession', period=23.0, amplitude=0.5, phase=np.pi/3)
    
    # Set trend parameters (slight cooling trend)
    synth.set_trend(slope=-0.002, intercept=18.0)
    
    # Generate dataset with UK'37 proxy
    n_points = 400  # Dense regular sampling for training
    dataset = synth.generate_dataset(n_points=n_points, regular=True, proxies=['UK37'])
    
    # Generate irregular, sparser dataset for testing
    n_test = 80
    test_dataset = synth.generate_dataset(n_points=n_test, regular=False, 
                                         min_spacing=2.0, proxies=['UK37'],
                                         age_error=1.0)  # Add realistic age errors
    
    # Plot the synthetic data
    fig = synth.plot_dataset(dataset)
    fig.savefig(os.path.join(output_dir, "synthetic_data.png"), dpi=300)
    
    # Plot power spectrum of true SST
    fig = synth.plot_power_spectrum(dataset['true_sst'], dataset['time_points'])
    fig.savefig(os.path.join(output_dir, "true_power_spectrum.png"), dpi=300)
    
    print(f"  Generated {n_points} regular time points and {n_test} irregular test points")
    print("  Synthetic data saved to output directory")
    
    # Step 2: Convert proxy to SST
    print("\nStep 2: Calibrating proxy data to SST...")
    
    # Extract UK'37 proxy data
    uk37_values = dataset['proxy_data']['UK37']['values']
    
    # Calibrate proxy to SST 
    uk37_sst = proxy_to_sst(uk37_values, proxy_type='UK37')
    
    # Calculate calibration error statistics
    true_sst = dataset['true_sst']
    calibration_error = true_sst - uk37_sst
    rmse_cal = np.sqrt(np.mean(calibration_error**2))
    r2_cal = 1 - np.sum(calibration_error**2) / np.sum((true_sst - np.mean(true_sst))**2)
    
    print(f"  Proxy calibration RMSE: {rmse_cal:.3f}°C")
    print(f"  Proxy calibration R²: {r2_cal:.3f}")
    
    # Create calibration plot
    plt.figure(figsize=(10, 6))
    plt.scatter(true_sst, uk37_sst, alpha=0.7)
    plt.plot([min(true_sst), max(true_sst)], [min(true_sst), max(true_sst)], 'r--')
    plt.xlabel('True SST (°C)')
    plt.ylabel('UK\'37 Calibrated SST (°C)')
    plt.title('Proxy Calibration Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "proxy_calibration.png"), dpi=300)
    
    # Step 3: Train the models
    print("\nStep 3: Training paleoclimate reconstruction models...")
    
    # Get training data
    train_times = dataset['time_points']
    train_sst = uk37_sst  # Use calibrated proxy values (not true SST)
    
    # Initialize models
    models = []
    
    # AR1 Model
    print("  Training AR1 model...")
    ar1_model = AR1Model(process_noise=0.1, observation_noise=0.1, optimize_params=True)
    ar1_model.fit(train_times, train_sst)
    models.append(ar1_model)
    print(f"  AR1 model parameters: {ar1_model.get_params()}")
    
    # AR2 Model
    print("  Training AR2 model...")
    ar2_model = AR2Model(process_noise=0.1, observation_noise=0.1, optimize_params=True)
    ar2_model.fit(train_times, train_sst)
    models.append(ar2_model)
    print(f"  AR2 model parameters: {ar2_model.get_params()}")
    
    # GP Model with RBF kernel
    print("  Training GP model with RBF kernel...")
    gp_rbf = PhysicsInformedGP(kernel='rbf', normalize=True, optimize_hyperparams=True)
    gp_rbf.fit(train_times, train_sst)
    models.append(gp_rbf)
    
    # GP Model with Milankovitch-informed kernel
    print("  Training GP model with Milankovitch kernel...")
    gp_mil = PhysicsInformedGP(kernel='milankovitch', normalize=True, optimize_hyperparams=True)
    gp_mil.fit(train_times, train_sst)
    models.append(gp_mil)
    
    # Define model names for display
    model_names = ['AR1', 'AR2', 'GP-RBF', 'GP-Milankovitch']
    
    # Step 4: Evaluate models on test data
    print("\nStep 4: Evaluating models on test data...")
    
    # Get test data
    test_times = test_dataset['time_points']
    test_sst = test_dataset['true_sst']  # Use true SST for evaluation
    
    # Create model comparison object
    comparison = ModelComparison(models, model_names)
    
    # Evaluate all models
    metrics = comparison.evaluate(test_times, test_sst)
    print("\nModel evaluation metrics:")
    print(metrics.to_string(index=False))
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating comparative visualizations...")
    
    # Generate regular time points for continuous reconstruction
    continuous_times = np.linspace(0, 800, 1000)
    
    # Plot reconstructions
    fig = comparison.plot_reconstructions(figsize=(12, 6))
    fig.savefig(os.path.join(output_dir, "model_reconstructions.png"), dpi=300)
    
    # Plot residuals 
    fig = comparison.plot_residuals(figsize=(12, 8))
    fig.savefig(os.path.join(output_dir, "model_residuals.png"), dpi=300)
    
    # Plot power spectra
    fig = comparison.plot_power_spectra(figsize=(12, 10))
    fig.savefig(os.path.join(output_dir, "model_power_spectra.png"), dpi=300)
    
    # Plot coherence
    fig = comparison.plot_coherence(figsize=(12, 10))
    fig.savefig(os.path.join(output_dir, "model_coherence.png"), dpi=300)
    
    # Plot metrics comparison
    fig = comparison.plot_metrics_comparison(figsize=(12, 8))
    fig.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300)
    
    # Plot comprehensive comparison
    fig = comparison.plot_comprehensive_comparison(figsize=(16, 12))
    fig.savefig(os.path.join(output_dir, "comprehensive_comparison.png"), dpi=300)
    
    # Step 6: Generate summary
    print("\nStep 6: Generating summary report...")
    
    # Get summary text
    summary = comparison.get_summary()
    print("\n" + summary)
    
    # Save summary to file
    with open(os.path.join(output_dir, "model_comparison_summary.txt"), 'w') as f:
        f.write(summary)
    
    # Save detailed metrics to CSV
    metrics.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
    
    print("\nResults have been saved to:", output_dir)
    print("\nPaleoclimate Reconstruction Project completed successfully!")


if __name__ == "__main__":
    main()