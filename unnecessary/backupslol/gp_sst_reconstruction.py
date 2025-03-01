"""
gp_sst_reconstruction.py - Gaussian Process reconstruction of SST from proxies using GPyTorch

This module implements Gaussian Process regression models to reconstruct Sea Surface
Temperature (SST) from proxy data (δ18O and UK'37) using the GPyTorch library.
It explores different kernel types and evaluates their performance.
"""

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import detrend
from scipy.stats import norm
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set up directory for results
output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)

# Constants for generating synthetic data
TIME_MIN = 0
TIME_MAX = 500
N_POINTS = 300
TEST_SIZE = 0.2

# Proxy calibration parameters
D18O_INTERCEPT = 16.9
D18O_SLOPE = -4.38
UK37_INTERCEPT = 0.044
UK37_SLOPE = 0.033

class SyntheticPaleoData:
    """
    Class for generating realistic synthetic paleoclimate data including SST and proxies.
    """
    
    def __init__(self, time_min=TIME_MIN, time_max=TIME_MAX, n_points=N_POINTS, random_seed=42):
        """
        Initialize the synthetic data generator.
        
        Parameters:
            time_min (float): Minimum time value (kyr BP)
            time_max (float): Maximum time value (kyr BP)
            n_points (int): Number of data points to generate
            random_seed (int): Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.time_min = time_min
        self.time_max = time_max
        self.n_points = n_points
        
    def generate_realistic_age_model(self):
        """
        Generate a realistic age model with irregular time spacing.
        
        Returns:
            array: Age points (kyr BP)
        """
        # Generate a monotonically increasing sequence with variable gaps
        # Typical paleoclimate records have variable sampling resolution
        
        # Start with evenly spaced points as a base
        base_ages = np.linspace(self.time_min, self.time_max, self.n_points)
        
        # Add perturbations to create irregular spacing
        perturbations = np.random.gamma(1, 1, size=self.n_points)
        perturbations = perturbations / perturbations.sum() * (self.time_max - self.time_min) * 0.3
        
        # Ensure monotonically increasing ages with minimum spacing
        ages = np.zeros(self.n_points)
        ages[0] = self.time_min
        for i in range(1, self.n_points):
            min_spacing = (self.time_max - self.time_min) / (self.n_points * 2)
            ages[i] = max(ages[i-1] + min_spacing, base_ages[i] + perturbations[i])
        
        # Include some sections with higher resolution (clustered points)
        # This mimics how real paleoclimate records often have varying resolution
        cluster_centers = np.random.choice(np.arange(10, self.n_points-10), size=3, replace=False)
        for center in cluster_centers:
            window = slice(max(0, center-5), min(self.n_points, center+5))
            ages[window] = np.linspace(ages[max(0, center-5)], ages[min(self.n_points-1, center+5)], 
                                      len(ages[window]))
            
        return ages
    
    def generate_synthetic_sst(self, ages):
        """
        Generate synthetic SST data with realistic climate features.
        
        Parameters:
            ages (array): Age points (kyr BP)
            
        Returns:
            array: Synthetic SST values
        """
        # Base orbital components (mimicking Milankovitch cycles)
        # 100 kyr eccentricity cycle
        eccentricity = 2.5 * np.sin(2 * np.pi * ages / 100)
        
        # 41 kyr obliquity cycle
        obliquity = 1.2 * np.sin(2 * np.pi * ages / 41 + 0.5)
        
        # 23 kyr precession cycle
        precession = 0.8 * np.sin(2 * np.pi * ages / 23 + 0.3)
        
        # Millennial-scale oscillations (Dansgaard-Oeschger events)
        millennial = 1.0 * np.sin(2 * np.pi * ages / 1.5) * np.exp(-((ages % 10) / 2)**2)
        
        # Add some abrupt climate transitions (Heinrich events, terminations)
        # These are sharp transitions mimicking real climate shifts
        abrupt_events = np.zeros_like(ages)
        
        # Define transition points
        transition_points = np.random.choice(ages, size=5, replace=False)
        for point in transition_points:
            # Create a sigmoidal transition at these points
            transition = 1.5 * (1 / (1 + np.exp(-(ages - point) * 0.5)))
            abrupt_events += transition
        
        # Add in a few "spikes" similar to what's seen in real data
        spike_points = np.random.choice(np.arange(len(ages)), size=8, replace=False)
        spikes = np.zeros_like(ages)
        for point in spike_points:
            spikes[point] = np.random.uniform(1.0, 3.0) * np.random.choice([-1, 1])
        
        # Long-term trend
        trend = -0.01 * ages + 20
        
        # Combine all components
        sst_signal = trend + eccentricity + obliquity + precession + millennial + abrupt_events
        
        # Add a bit of fine-scale noise (weather/short-term variability)
        fine_noise = 0.3 * np.random.randn(len(ages))
        
        # Generate autocorrelated noise (red noise)
        red_noise = np.zeros_like(ages)
        red_noise[0] = np.random.randn()
        for i in range(1, len(ages)):
            red_noise[i] = 0.7 * red_noise[i-1] + 0.3 * np.random.randn()
        red_noise = 0.8 * red_noise / np.std(red_noise)
        
        # Combine all components
        sst = sst_signal + fine_noise + red_noise + spikes
        
        # Scale to realistic SST range (10-30°C)
        sst = (sst - np.min(sst)) / (np.max(sst) - np.min(sst)) * 20 + 10
        
        return sst
    
    def generate_d18o_proxy(self, sst, noise_level=0.5):
        """
        Generate δ18O proxy data from SST values using calibration equation.
        
        Parameters:
            sst (array): SST values
            noise_level (float): Level of noise to add (standard deviation)
            
        Returns:
            array: δ18O proxy values
        """
        # Apply the δ18O calibration equation: δ18O = 16.9 - 4.38 * SST
        d18o_values = D18O_INTERCEPT + D18O_SLOPE * sst
        
        # Add heteroscedastic noise (variable uncertainty)
        # Noise is larger for more extreme values (mimicking real proxy behavior)
        base_noise = np.random.normal(0, noise_level, size=len(sst))
        amplitude_effect = 0.3 * np.abs(sst - np.mean(sst)) / np.std(sst)
        heteroscedastic_noise = base_noise * (1 + amplitude_effect)
        
        # Add systematic biases (mimicking preservation/diagenetic effects)
        # These create structured errors in certain parts of the record
        systematic_bias = 0.2 * np.sin(2 * np.pi * np.linspace(0, 3, len(sst)))
        
        # Create some outliers (analytical errors)
        outlier_mask = np.random.random(len(sst)) < 0.05  # 5% outliers
        outliers = np.zeros_like(sst)
        outliers[outlier_mask] = np.random.normal(0, 3 * noise_level, size=np.sum(outlier_mask))
        
        # Final proxy values with all error components
        d18o_with_noise = d18o_values + heteroscedastic_noise + systematic_bias + outliers
        
        return d18o_with_noise
    
    def generate_uk37_proxy(self, sst, noise_level=0.4):
        """
        Generate UK'37 proxy data from SST values using calibration equation.
        
        Parameters:
            sst (array): SST values
            noise_level (float): Level of noise to add (standard deviation)
            
        Returns:
            array: UK'37 proxy values
        """
        # Apply the UK'37 calibration equation: UK'37 = 0.033 * SST + 0.044
        uk37_values = UK37_SLOPE * sst + UK37_INTERCEPT
        
        # Add heteroscedastic noise
        base_noise = np.random.normal(0, noise_level, size=len(sst))
        temp_effect = 0.2 * (1 + np.exp(-(sst - 15)**2 / 50))  # More precise in mid-range
        heteroscedastic_noise = base_noise * temp_effect
        
        # Add systematic biases (production/preservation effects)
        systematic_bias = 0.15 * (np.sin(2 * np.pi * np.linspace(0, 4, len(sst))) + 
                                 np.cos(2 * np.pi * np.linspace(0, 2, len(sst))))
        
        # Create some analytical outliers
        outlier_mask = np.random.random(len(sst)) < 0.03  # 3% outliers
        outliers = np.zeros_like(sst)
        outliers[outlier_mask] = np.random.normal(0, 2.5 * noise_level, size=np.sum(outlier_mask))
        
        # Combine all error components
        uk37_with_noise = uk37_values + heteroscedastic_noise + systematic_bias + outliers
        
        # Ensure values are within realistic UK'37 range (0 to 1)
        uk37_with_noise = np.clip(uk37_with_noise, 0, 1)
        
        return uk37_with_noise
    
    def generate_dataset(self):
        """
        Generate a complete synthetic dataset with age, SST, and proxy values.
        
        Returns:
            dict: Dictionary containing all synthetic data
        """
        # Generate age model
        ages = self.generate_realistic_age_model()
        
        # Generate true SST
        true_sst = self.generate_synthetic_sst(ages)
        
        # Generate proxy data
        d18o_values = self.generate_d18o_proxy(true_sst)
        uk37_values = self.generate_uk37_proxy(true_sst)
        
        # Create dataset
        dataset = {
            'age': ages,
            'true_sst': true_sst,
            'd18o': d18o_values,
            'uk37': uk37_values
        }
        
        return dataset
    
    def plot_dataset(self, dataset, show_calibrated=True):
        """
        Plot the synthetic dataset.
        
        Parameters:
            dataset (dict): Dictionary containing synthetic data
            show_calibrated (bool): Whether to show calibrated proxy SST
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Get data
        ages = dataset['age']
        true_sst = dataset['true_sst']
        d18o_values = dataset['d18o']
        uk37_values = dataset['uk37']
        
        # Calculate calibrated SST from proxies if requested
        if show_calibrated:
            d18o_sst = (D18O_INTERCEPT - d18o_values) / abs(D18O_SLOPE)
            uk37_sst = (uk37_values - UK37_INTERCEPT) / UK37_SLOPE
        
        # Plot true SST
        axes[0].plot(ages, true_sst, 'k-', linewidth=1.5, label='True SST')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].set_title('Synthetic Sea Surface Temperature (SST)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot δ18O proxy
        axes[1].plot(ages, d18o_values, 'bo', markersize=3, alpha=0.7, label='δ¹⁸O Proxy')
        if show_calibrated:
            axes[1].plot(ages, true_sst, 'k-', linewidth=1, alpha=0.3, label='True SST')
            axes_d18o_sst = axes[1].twinx()
            axes_d18o_sst.plot(ages, d18o_sst, 'g-', linewidth=1, alpha=0.5, label='Calibrated SST')
            axes_d18o_sst.set_ylabel('Calibrated SST (°C)', color='g')
            
        axes[1].set_ylabel('δ¹⁸O (‰)')
        axes[1].set_title('δ¹⁸O Proxy Data (Negatively Correlated with SST)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot UK'37 proxy
        axes[2].plot(ages, uk37_values, 'ro', markersize=3, alpha=0.7, label='UK\'37 Proxy')
        if show_calibrated:
            axes[2].plot(ages, true_sst, 'k-', linewidth=1, alpha=0.3, label='True SST')
            axes_uk37_sst = axes[2].twinx()
            axes_uk37_sst.plot(ages, uk37_sst, 'g-', linewidth=1, alpha=0.5, label='Calibrated SST')
            axes_uk37_sst.set_ylabel('Calibrated SST (°C)', color='g')
            
        axes[2].set_xlabel('Age (kyr BP)')
        axes[2].set_ylabel('UK\'37 Index')
        axes[2].set_title('UK\'37 Proxy Data (Positively Correlated with SST)')
        axes[2].grid(True, alpha=0.3)
        
        # Add correlations to titles
        d18o_corr = np.corrcoef(true_sst, d18o_values)[0, 1]
        uk37_corr = np.corrcoef(true_sst, uk37_values)[0, 1]
        
        axes[1].set_title(f'δ¹⁸O Proxy Data (r = {d18o_corr:.2f})')
        axes[2].set_title(f'UK\'37 Proxy Data (r = {uk37_corr:.2f})')
        
        # Add legend to all plots
        for ax in axes:
            ax.legend(loc='upper right')
            
        plt.tight_layout()
        return fig

def calibrate_proxy_to_sst(proxy_values, proxy_type):
    """
    Convert proxy values to SST using calibration equations.
    
    Parameters:
        proxy_values (array): Proxy measurements
        proxy_type (str): Type of proxy ('d18o' or 'uk37')
        
    Returns:
        array: Calibrated SST values
    """
    if proxy_type == 'd18o':
        # δ18O to SST: reverse the equation δ18O = 16.9 - 4.38 * SST
        return (D18O_INTERCEPT - proxy_values) / abs(D18O_SLOPE)
    elif proxy_type == 'uk37':
        # UK'37 to SST: reverse the equation UK'37 = 0.033 * SST + 0.044
        return (proxy_values - UK37_INTERCEPT) / UK37_SLOPE
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}")

# Define the Gaussian Process model using GPyTorch
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Base GPyTorch model for implementing various kernels for SST reconstruction.
    """
    
    def __init__(self, train_x, train_y, likelihood, kernel='rbf', nu=2.5):
        """
        Initialize the GP model with specified kernel.
        
        Parameters:
            train_x (tensor): Training input features
            train_y (tensor): Training target values
            likelihood (gpytorch.likelihoods): The likelihood for the model
            kernel (str): Kernel type ('rbf', 'matern', 'periodic', 'combined')
            nu (float): Parameter for Matern kernel (0.5, 1.5, or 2.5)
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.kernel_type = kernel
        self.nu = nu
        
        # Mean module (constant mean)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Initialize different kernel types
        if kernel == 'rbf':
            # RBF (squared exponential) kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        elif kernel == 'matern':
            # Matern kernel with specified nu parameter
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=nu)
            )
        elif kernel == 'periodic':
            # Periodic kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel()
            )
        elif kernel == 'combined':
            # Combined kernel: RBF + Periodic
            # Good for capturing both smooth trends and cyclic patterns
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ) + gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel()
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
            x (tensor): Input features
            
        Returns:
            gpytorch.distributions.MultivariateNormal: Output distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_kernel_params(self):
        """
        Get the optimized kernel parameters.
        
        Returns:
            dict: Dictionary of kernel parameters
        """
        params = {}
        
        if self.kernel_type == 'rbf':
            params['lengthscale'] = self.covar_module.base_kernel.lengthscale.item()
            params['outputscale'] = self.covar_module.outputscale.item()
        elif self.kernel_type == 'matern':
            params['lengthscale'] = self.covar_module.base_kernel.lengthscale.item()
            params['outputscale'] = self.covar_module.outputscale.item()
            params['nu'] = self.nu
        elif self.kernel_type == 'periodic':
            params['lengthscale'] = self.covar_module.base_kernel.lengthscale.item()
            params['period_length'] = self.covar_module.base_kernel.period_length.item()
            params['outputscale'] = self.covar_module.outputscale.item()
        elif self.kernel_type == 'combined':
            # Extract parameters from the combined kernel components
            params['rbf_lengthscale'] = self.covar_module.kernels[0].base_kernel.lengthscale.item()
            params['rbf_outputscale'] = self.covar_module.kernels[0].outputscale.item()
            params['periodic_lengthscale'] = self.covar_module.kernels[1].base_kernel.lengthscale.item()
            params['periodic_period_length'] = self.covar_module.kernels[1].base_kernel.period_length.item()
            params['periodic_outputscale'] = self.covar_module.kernels[1].outputscale.item()
        
        return params

def train_gp_model(train_x, train_y, kernel_type='rbf', nu=2.5, n_iterations=100, lr=0.1):
    """
    Train a GP model with specified kernel.
    
    Parameters:
        train_x (tensor): Training inputs
        train_y (tensor): Training targets
        kernel_type (str): Kernel type
        nu (float): Parameter for Matern kernel
        n_iterations (int): Number of training iterations
        lr (float): Learning rate for optimizer
        
    Returns:
        tuple: (trained model, likelihood, training losses)
    """
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel=kernel_type, nu=nu)
    
    # Set model and likelihood to training mode
    model.train()
    likelihood.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=lr)
    
    # Loss function (negative log marginal likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    losses = []
    for i in range(n_iterations):
        # Catch numerical issues that might happen during optimization
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f'Iteration {i+1}/{n_iterations} - Loss: {loss.item():.4f}')
        except Exception as e:
            print(f"Warning: Optimization error at iteration {i+1}: {e}")
            # Use last good loss value
            if losses:
                losses.append(losses[-1])
            else:
                losses.append(100.0)  # Default high loss
    
    return model, likelihood, losses

def evaluate_gp_model(model, likelihood, test_x, test_y):
    """
    Evaluate a trained GP model on test data.
    
    Parameters:
        model (ExactGPModel): Trained GP model
        likelihood (gpytorch.likelihoods): Model likelihood
        test_x (tensor): Test inputs
        test_y (tensor): Test targets
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    likelihood.eval()
    
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean_pred = observed_pred.mean.numpy()
        
    # Convert test_y to numpy for metric calculation
    test_y_np = test_y.numpy()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_y_np, mean_pred))
    mae = mean_absolute_error(test_y_np, mean_pred)
    r2 = r2_score(test_y_np, mean_pred)
    
    # Print metrics
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, mean_pred

def plot_gp_predictions(model, likelihood, test_x, test_y, true_x, true_y, proxy_type='d18o'):
    """
    Plot GP model predictions with uncertainty.
    
    Parameters:
        model (ExactGPModel): Trained GP model
        likelihood (gpytorch.likelihoods): Model likelihood
        test_x (tensor): Test inputs
        test_y (tensor): Test targets
        true_x (tensor): Full domain inputs
        true_y (tensor): Full domain true values
        proxy_type (str): Type of proxy used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Set model to evaluation mode
    model.eval()
    likelihood.eval()
    
    # Make predictions on test data
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get predictions with uncertainty
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean.numpy()
        lower, upper = observed_pred.confidence_region()
        lower, upper = lower.numpy(), upper.numpy()
        
        # Predict on full domain for smooth curve
        observed_full = likelihood(model(true_x))
        mean_full = observed_full.mean.numpy()
        lower_full, upper_full = observed_full.confidence_region()
        lower_full, upper_full = lower_full.numpy(), upper_full.numpy()
    
    # Convert tensors to numpy
    test_x_np = test_x.numpy().flatten()  # Flatten to 1D array
    test_y_np = test_y.numpy()
    true_x_np = true_x.numpy().flatten()  # Flatten to 1D array
    true_y_np = true_y.numpy()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_y_np, mean))
    mae = mean_absolute_error(test_y_np, mean)
    r2 = r2_score(test_y_np, mean)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true SST
    ax.plot(true_x_np, true_y_np, 'k-', linewidth=1, label='True SST')
    
    # Plot test data points
    if proxy_type == 'd18o':
        ax.plot(test_x_np, test_y_np, 'bo', markersize=4, alpha=0.6, label=f'Calibrated {proxy_type}')
    else:
        ax.plot(test_x_np, test_y_np, 'ro', markersize=4, alpha=0.6, label=f'Calibrated {proxy_type}')
    
    # Plot GP mean prediction
    ax.plot(true_x_np, mean_full, 'g-', linewidth=2, label=f'GP Prediction ({model.kernel_type})')
    
    # Plot confidence interval
    ax.fill_between(true_x_np, lower_full, upper_full, alpha=0.3, color='g', 
                    label='95% Confidence')
    
    # Add metrics to plot
    metrics_text = f"RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C\nR²: {r2:.3f}"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Age (kyr BP)')
    ax.set_ylabel('Temperature (°C)')
    kernel_name = model.kernel_type
    if kernel_name == 'matern':
        kernel_name = f"Matern(ν={model.nu})"
    ax.set_title(f'GP Reconstruction of SST from {proxy_type.upper()} using {kernel_name} Kernel')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_training_loss(losses, kernel_type, proxy_type):
    """
    Plot training loss over iterations.
    
    Parameters:
        losses (list): List of loss values
        kernel_type (str): Kernel type used
        proxy_type (str): Proxy type used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot loss
    ax.plot(losses, 'b-', linewidth=1.5)
    
    # Add moving average
    window_size = min(20, len(losses) // 5)
    if window_size > 1:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        ax.plot(np.arange(window_size-1, len(losses)), moving_avg, 'r-', linewidth=2, 
                label=f'Moving Avg (n={window_size})')
    
    # Set labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Negative Log Likelihood Loss')
    kernel_name = kernel_type
    if kernel_type == 'matern':
        nu_val = nu if 'nu' in locals() else 2.5
        kernel_name = f"Matern(ν={nu_val})"
    ax.set_title(f'Training Loss for {kernel_name} Kernel on {proxy_type.upper()} Proxy')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    if window_size > 1:
        ax.legend()
    
    # Set y-axis
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def plot_kernel_comparison(metrics_dict, proxy_type):
    """
    Plot comparison of different kernels.
    
    Parameters:
        metrics_dict (dict): Dictionary of metrics for each kernel
        proxy_type (str): Proxy type used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract kernel names and metrics
    kernels = list(metrics_dict.keys())
    rmse_values = [metrics_dict[k]['rmse'] for k in kernels]
    mae_values = [metrics_dict[k]['mae'] for k in kernels]
    r2_values = [metrics_dict[k]['r2'] for k in kernels]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE plot
    axes[0].bar(kernels, rmse_values, color='skyblue')
    axes[0].set_ylabel('RMSE (°C)')
    axes[0].set_title('Root Mean Square Error')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE plot
    axes[1].bar(kernels, mae_values, color='lightgreen')
    axes[1].set_ylabel('MAE (°C)')
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R² plot
    axes[2].bar(kernels, r2_values, color='salmon')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('R² Score')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add values on top of bars
    for ax, values in zip(axes, [rmse_values, mae_values, r2_values]):
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Set common title
    fig.suptitle(f'Kernel Performance Comparison for {proxy_type.upper()} Proxy', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig

def plot_residuals_comparison(predictions_dict, test_y, proxy_type):
    """
    Plot residuals for different kernels.
    
    Parameters:
        predictions_dict (dict): Dictionary of predictions for each kernel
        test_y (tensor): True test values
        proxy_type (str): Proxy type used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert test_y to numpy
    test_y_np = test_y.numpy()
    
    # Create figure
    n_kernels = len(predictions_dict)
    fig, axes = plt.subplots(n_kernels, 1, figsize=(12, 3*n_kernels), sharex=True)
    
    # Plot residuals for each kernel
    for i, (kernel, predictions) in enumerate(predictions_dict.items()):
        # Calculate residuals
        residuals = predictions - test_y_np
        
        # Plot residuals
        axes[i].scatter(range(len(residuals)), residuals, alpha=0.7)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add statistics
        rmse = np.sqrt(np.mean(residuals**2))
        bias = np.mean(residuals)
        std = np.std(residuals)
        
        # Add normal distribution fit
        x = np.linspace(min(residuals), max(residuals), 100)
        y = norm.pdf(x, bias, std)
        y = y / max(y) * len(residuals) * 0.3  # Scale for visibility
        
        # Add histogram to right side
        divider = make_axes_locatable(axes[i])
        ax_hist = divider.append_axes("right", size="20%", pad=0.1)
        ax_hist.hist(residuals, orientation='horizontal', bins=15, alpha=0.7)
        ax_hist.plot(y, x, 'r-', linewidth=2)
        ax_hist.set_xticks([])
        
        # Add statistics to plot
        axes[i].text(0.02, 0.95, f"RMSE: {rmse:.3f}\nBias: {bias:.3f}\nStd: {std:.3f}", 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set labels
        axes[i].set_ylabel('Residual (°C)')
        axes[i].set_title(f'{kernel} Residuals')
        axes[i].grid(True, alpha=0.3)
    
    # Set common x-label
    axes[-1].set_xlabel('Test Sample Index')
    
    # Set common title
    fig.suptitle(f'Residual Comparison for {proxy_type.upper()} Proxy', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    return fig

def main():
    """Main function to run SST reconstruction using GP models."""
    print("Starting Sea Surface Temperature Reconstruction using Gaussian Processes...")
    
    # Step I: Generate synthetic paleoclimate data
    print("\nGenerating synthetic paleoclimate data...")
    synth_data = SyntheticPaleoData(n_points=N_POINTS)
    dataset = synth_data.generate_dataset()
    
    # Plot and save synthetic data
    fig_data = synth_data.plot_dataset(dataset)
    fig_data.savefig(os.path.join(output_dir, "synthetic_proxy_data.png"), dpi=300)
    plt.close(fig_data)
    
    # Step II: Prepare data for GP models
    print("\nPreparing data for GP models...")
    
    # Extract data
    ages = dataset['age']
    true_sst = dataset['true_sst']
    d18o_values = dataset['d18o']
    uk37_values = dataset['uk37']
    
    # Calibrate proxies to SST
    d18o_sst = calibrate_proxy_to_sst(d18o_values, 'd18o')
    uk37_sst = calibrate_proxy_to_sst(uk37_values, 'uk37')
    
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
    d18o_test_y = torch.tensor(true_sst[d18o_test_idx], dtype=torch.float32)  # True SST for testing
    
    # Tensor for full domain (for smooth plotting)
    full_x = torch.tensor(ages, dtype=torch.float32).reshape(-1, 1)
    full_y = torch.tensor(true_sst, dtype=torch.float32)
    
    # Split data for UK'37
    uk37_train_idx, uk37_test_idx = train_test_split(
        np.arange(len(ages)), test_size=TEST_SIZE, random_state=84
    )
    
    # Sort indices by age
    uk37_train_idx = sorted(uk37_train_idx)
    uk37_test_idx = sorted(uk37_test_idx)
    
    # Prepare UK'37 training and testing data
    uk37_train_x = torch.tensor(ages[uk37_train_idx], dtype=torch.float32).reshape(-1, 1)
    uk37_train_y = torch.tensor(uk37_sst[uk37_train_idx], dtype=torch.float32)
    uk37_test_x = torch.tensor(ages[uk37_test_idx], dtype=torch.float32).reshape(-1, 1)
    uk37_test_y = torch.tensor(true_sst[uk37_test_idx], dtype=torch.float32)  # True SST for testing
    
    # Step III: Train and evaluate GP models with different kernels
    # Define kernels to test - use global variable if defined, otherwise use default
    global kernels
    if 'kernels' not in globals():
        kernels = [
            ('rbf', None),          # RBF kernel
            ('matern', 0.5),        # Matern kernel with ν=0.5
            ('matern', 1.5),        # Matern kernel with ν=1.5
            ('matern', 2.5),        # Matern kernel with ν=2.5
            ('periodic', None),     # Periodic kernel
            ('combined', None)      # Combined RBF + Periodic kernel
        ]
    
    # Dictionary to store results
    d18o_results = {}
    d18o_predictions = {}
    uk37_results = {}
    uk37_predictions = {}
    
    # Train models for δ18O proxy
    print("\n===== Training GP Models for δ18O Proxy =====")
    
    for kernel_type, nu in kernels:
        # Create kernel name
        if kernel_type == 'matern':
            kernel_name = f"{kernel_type}_{nu}"
        else:
            kernel_name = kernel_type
        
        print(f"\nTraining {kernel_name} kernel model...")
        
        # Train model
        model, likelihood, losses = train_gp_model(
            d18o_train_x, d18o_train_y, 
            kernel_type=kernel_type, 
            nu=nu if nu else 2.5
        )
        
        # Evaluate model
        print(f"\nEvaluating {kernel_name} kernel model...")
        metrics, predictions = evaluate_gp_model(model, likelihood, d18o_test_x, d18o_test_y)
        
        # Store results
        d18o_results[kernel_name] = {
            'model': model,
            'likelihood': likelihood,
            'losses': losses,
            'metrics': metrics,
            'kernel_params': model.get_kernel_params()
        }
        
        d18o_predictions[kernel_name] = predictions
        
        # Plot and save predictions
        fig_pred = plot_gp_predictions(
            model, likelihood, d18o_test_x, d18o_test_y, full_x, full_y, proxy_type='d18o'
        )
        fig_pred.savefig(os.path.join(output_dir, f"d18o_{kernel_name}_predictions.png"), dpi=300)
        plt.close(fig_pred)
        
        # Plot and save training loss
        fig_loss = plot_training_loss(losses, kernel_name, 'd18o')
        fig_loss.savefig(os.path.join(output_dir, f"d18o_{kernel_name}_loss.png"), dpi=300)
        plt.close(fig_loss)
    
    # Train models for UK'37 proxy
    print("\n===== Training GP Models for UK'37 Proxy =====")
    
    for kernel_type, nu in kernels:
        # Create kernel name
        if kernel_type == 'matern':
            kernel_name = f"{kernel_type}_{nu}"
        else:
            kernel_name = kernel_type
        
        print(f"\nTraining {kernel_name} kernel model...")
        
        # Train model
        model, likelihood, losses = train_gp_model(
            uk37_train_x, uk37_train_y, 
            kernel_type=kernel_type, 
            nu=nu if nu else 2.5
        )
        
        # Evaluate model
        print(f"\nEvaluating {kernel_name} kernel model...")
        metrics, predictions = evaluate_gp_model(model, likelihood, uk37_test_x, uk37_test_y)
        
        # Store results
        uk37_results[kernel_name] = {
            'model': model,
            'likelihood': likelihood,
            'losses': losses,
            'metrics': metrics,
            'kernel_params': model.get_kernel_params()
        }
        
        uk37_predictions[kernel_name] = predictions
        
        # Plot and save predictions
        fig_pred = plot_gp_predictions(
            model, likelihood, uk37_test_x, uk37_test_y, full_x, full_y, proxy_type='uk37'
        )
        fig_pred.savefig(os.path.join(output_dir, f"uk37_{kernel_name}_predictions.png"), dpi=300)
        plt.close(fig_pred)
        
        # Plot and save training loss
        fig_loss = plot_training_loss(losses, kernel_name, 'uk37')
        fig_loss.savefig(os.path.join(output_dir, f"uk37_{kernel_name}_loss.png"), dpi=300)
        plt.close(fig_loss)
    
    # Step IV: Compare kernel performance
    print("\n===== Kernel Performance Comparison =====")
    
    # Extract metrics for comparison
    d18o_metrics = {k: v['metrics'] for k, v in d18o_results.items()}
    uk37_metrics = {k: v['metrics'] for k, v in uk37_results.items()}
    
    # Plot kernel comparison
    fig_d18o_comp = plot_kernel_comparison(d18o_metrics, 'd18o')
    fig_d18o_comp.savefig(os.path.join(output_dir, "d18o_kernel_comparison.png"), dpi=300)
    plt.close(fig_d18o_comp)
    
    fig_uk37_comp = plot_kernel_comparison(uk37_metrics, 'uk37')
    fig_uk37_comp.savefig(os.path.join(output_dir, "uk37_kernel_comparison.png"), dpi=300)
    plt.close(fig_uk37_comp)
    
    # Print best kernel for each proxy based on RMSE
    d18o_best_kernel = min(d18o_metrics.items(), key=lambda x: x[1]['rmse'])[0]
    uk37_best_kernel = min(uk37_metrics.items(), key=lambda x: x[1]['rmse'])[0]
    
    print(f"\nBest kernel for δ18O proxy: {d18o_best_kernel}")
    print(f"RMSE: {d18o_metrics[d18o_best_kernel]['rmse']:.4f}")
    print(f"MAE: {d18o_metrics[d18o_best_kernel]['mae']:.4f}")
    print(f"R²: {d18o_metrics[d18o_best_kernel]['r2']:.4f}")
    
    print(f"\nBest kernel for UK'37 proxy: {uk37_best_kernel}")
    print(f"RMSE: {uk37_metrics[uk37_best_kernel]['rmse']:.4f}")
    print(f"MAE: {uk37_metrics[uk37_best_kernel]['mae']:.4f}")
    print(f"R²: {uk37_metrics[uk37_best_kernel]['r2']:.4f}")
    
    # Save all metrics to CSV
    d18o_metrics_df = pd.DataFrame([
        {'Kernel': k, 'RMSE': v['rmse'], 'MAE': v['mae'], 'R2': v['r2']} 
        for k, v in d18o_metrics.items()
    ])
    d18o_metrics_df.to_csv(os.path.join(output_dir, "d18o_metrics.csv"), index=False)
    
    uk37_metrics_df = pd.DataFrame([
        {'Kernel': k, 'RMSE': v['rmse'], 'MAE': v['mae'], 'R2': v['r2']} 
        for k, v in uk37_metrics.items()
    ])
    uk37_metrics_df.to_csv(os.path.join(output_dir, "uk37_metrics.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}")
    print("\nGP-based SST reconstruction completed successfully!")

# Define a simplified version of the residuals plot without the histogram
def plot_residuals_comparison(predictions_dict, test_y, proxy_type):
    """
    Plot residuals for different kernels (simplified version without histograms)
    
    Parameters:
        predictions_dict (dict): Dictionary of predictions for each kernel
        test_y (tensor): True test values
        proxy_type (str): Proxy type used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert test_y to numpy
    test_y_np = test_y.numpy()
    
    # Create figure
    n_kernels = len(predictions_dict)
    fig, axes = plt.subplots(n_kernels, 1, figsize=(12, 3*n_kernels), sharex=True)
    
    # Make sure axes is always a list
    if n_kernels == 1:
        axes = [axes]
    
    # Plot residuals for each kernel
    for i, (kernel, predictions) in enumerate(predictions_dict.items()):
        # Calculate residuals
        residuals = predictions - test_y_np
        
        # Plot residuals
        axes[i].scatter(range(len(residuals)), residuals, alpha=0.7)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add statistics
        rmse = np.sqrt(np.mean(residuals**2))
        bias = np.mean(residuals)
        std = np.std(residuals)
        
        # Add statistics to plot
        axes[i].text(0.02, 0.95, f"RMSE: {rmse:.3f}\nBias: {bias:.3f}\nStd: {std:.3f}", 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set labels
        axes[i].set_ylabel('Residual (°C)')
        axes[i].set_title(f'{kernel} Residuals')
        axes[i].grid(True, alpha=0.3)
    
    # Set common x-label
    axes[-1].set_xlabel('Test Sample Index')
    
    # Set common title
    fig.suptitle(f'Residual Comparison for {proxy_type.upper()} Proxy', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    return fig

if __name__ == "__main__":
    # Reduce the number of kernels and iterations for faster execution
    # Update the main function to use fewer kernels
    global kernels
    kernels = [
        ('rbf', None),          # RBF kernel
        ('matern', 2.5),        # Matern kernel with ν=2.5
        ('combined', None)      # Combined RBF + Periodic kernel
    ]
    
    # Run the main function
    main()