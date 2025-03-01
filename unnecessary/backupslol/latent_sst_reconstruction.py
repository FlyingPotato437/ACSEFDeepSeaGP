"""
latent_sst_reconstruction.py - Latent Process GP reconstruction of SST from proxies using GPyTorch

This module implements Gaussian Process regression models to reconstruct the latent (hidden)
Sea Surface Temperature (SST) from proxy data (δ18O and UK'37) using the GPyTorch library.
It explicitly models SST as a hidden variable that drives the observed proxy values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set up directory for results
output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)

# Constants for generating synthetic data
TIME_MIN = 0      # Start time in kyr BP
TIME_MAX = 500    # End time in kyr BP
N_POINTS = 300    # Number of data points
TEST_SIZE = 0.2   # Proportion of data for testing

# Proxy calibration parameters (from established paleoclimate equations)
# δ18O = α1 * SST + β1 + ε1, where ε1 ~ N(0, σ1²)
D18O_ALPHA = -4.38   # Slope (α1)
D18O_BETA = 16.9     # Intercept (β1)
D18O_SIGMA = 0.5     # Noise standard deviation (σ1)

# UK'37 = α2 * SST + β2 + ε2, where ε2 ~ N(0, σ2²)
UK37_ALPHA = 0.033   # Slope (α2)
UK37_BETA = 0.044    # Intercept (β2)
UK37_SIGMA = 0.4     # Noise standard deviation (σ2)


class SyntheticPaleoData:
    """
    Class for generating realistic synthetic paleoclimate data including SST and proxies.
    The proxies are explicitly derived from the latent SST according to established
    paleoclimate equations, with realistic noise patterns.
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
        
        # Define transition points for abrupt events (e.g., glacial terminations)
        transition_points = np.sort(np.random.choice(np.linspace(50, 450, 20), size=5, replace=False))
        for point in transition_points:
            # Create a sigmoidal transition at these points (sharp changes)
            # Higher multiplier (e.g., 3 instead of 0.5) makes transitions sharper
            transition = 1.5 * (1 / (1 + np.exp(-(ages - point) * 3)))
            abrupt_events += transition
        
        # Add in a few short-term "spikes" similar to what's seen in real data
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
    
    def generate_d18o_proxy(self, sst):
        """
        Generate δ18O proxy data from SST values using the established calibration equation:
        δ18O = α1 * SST + β1 + ε1, where ε1 ~ N(0, σ1²)
        
        Parameters:
            sst (array): SST values
            
        Returns:
            array: δ18O proxy values
        """
        # Apply the δ18O calibration equation
        d18o_values = D18O_ALPHA * sst + D18O_BETA
        
        # Add heteroscedastic noise (variable uncertainty)
        # Noise is larger for more extreme values (mimicking real proxy behavior)
        base_noise = np.random.normal(0, D18O_SIGMA, size=len(sst))
        amplitude_effect = 0.3 * np.abs(sst - np.mean(sst)) / np.std(sst)
        heteroscedastic_noise = base_noise * (1 + amplitude_effect)
        
        # Add systematic biases (mimicking preservation/diagenetic effects)
        # These create structured errors in certain parts of the record
        systematic_bias = 0.2 * np.sin(2 * np.pi * np.linspace(0, 3, len(sst)))
        
        # Create some outliers (analytical errors)
        outlier_mask = np.random.random(len(sst)) < 0.05  # 5% outliers
        outliers = np.zeros_like(sst)
        outliers[outlier_mask] = np.random.normal(0, 3 * D18O_SIGMA, size=np.sum(outlier_mask))
        
        # Final proxy values with all error components
        d18o_with_noise = d18o_values + heteroscedastic_noise + systematic_bias + outliers
        
        return d18o_with_noise
    
    def generate_uk37_proxy(self, sst):
        """
        Generate UK'37 proxy data from SST values using the established calibration equation:
        UK'37 = α2 * SST + β2 + ε2, where ε2 ~ N(0, σ2²)
        
        Parameters:
            sst (array): SST values
            
        Returns:
            array: UK'37 proxy values
        """
        # Apply the UK'37 calibration equation
        uk37_values = UK37_ALPHA * sst + UK37_BETA
        
        # Add heteroscedastic noise (variable uncertainty)
        base_noise = np.random.normal(0, UK37_SIGMA, size=len(sst))
        temp_effect = 0.2 * (1 + np.exp(-(sst - 15)**2 / 50))  # More precise in mid-range
        heteroscedastic_noise = base_noise * temp_effect
        
        # Add systematic biases (production/preservation effects)
        systematic_bias = 0.15 * (np.sin(2 * np.pi * np.linspace(0, 4, len(sst))) + 
                                 np.cos(2 * np.pi * np.linspace(0, 2, len(sst))))
        
        # Create some analytical outliers
        outlier_mask = np.random.random(len(sst)) < 0.03  # 3% outliers
        outliers = np.zeros_like(sst)
        outliers[outlier_mask] = np.random.normal(0, 2.5 * UK37_SIGMA, size=np.sum(outlier_mask))
        
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
        
        # Generate true SST (this is the latent variable)
        true_sst = self.generate_synthetic_sst(ages)
        
        # Generate proxy data from SST using established equations
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
        Plot the synthetic dataset showing the latent SST and observed proxies.
        
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
            d18o_sst = (d18o_values - D18O_BETA) / D18O_ALPHA
            uk37_sst = (uk37_values - UK37_BETA) / UK37_ALPHA
        
        # Plot true SST (the latent variable)
        axes[0].plot(ages, true_sst, 'k-', linewidth=1.5, label='True SST (Latent)')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].set_title('Synthetic Sea Surface Temperature (Latent SST)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot δ18O proxy (observed variable)
        axes[1].plot(ages, d18o_values, 'bo', markersize=3, alpha=0.7, label='δ¹⁸O Proxy (Observed)')
        if show_calibrated:
            axes[1].plot(ages, true_sst, 'k-', linewidth=1, alpha=0.3, label='True SST')
            axes_d18o_sst = axes[1].twinx()
            axes_d18o_sst.plot(ages, d18o_sst, 'g-', linewidth=1, alpha=0.5, label='Simple Calibration')
            axes_d18o_sst.set_ylabel('Calibrated SST (°C)', color='g')
            
        axes[1].set_ylabel('δ¹⁸O (‰)')
        axes[1].set_title('δ¹⁸O Proxy Data')
        axes[1].grid(True, alpha=0.3)
        
        # Plot UK'37 proxy (observed variable)
        axes[2].plot(ages, uk37_values, 'ro', markersize=3, alpha=0.7, label='UK\'37 Proxy (Observed)')
        if show_calibrated:
            axes[2].plot(ages, true_sst, 'k-', linewidth=1, alpha=0.3, label='True SST')
            axes_uk37_sst = axes[2].twinx()
            axes_uk37_sst.plot(ages, uk37_sst, 'g-', linewidth=1, alpha=0.5, label='Simple Calibration')
            axes_uk37_sst.set_ylabel('Calibrated SST (°C)', color='g')
            
        axes[2].set_xlabel('Age (kyr BP)')
        axes[2].set_ylabel('UK\'37 Index')
        axes[2].set_title('UK\'37 Proxy Data')
        axes[2].grid(True, alpha=0.3)
        
        # Add correlations and equations to titles
        d18o_corr = np.corrcoef(true_sst, d18o_values)[0, 1]
        uk37_corr = np.corrcoef(true_sst, uk37_values)[0, 1]
        
        axes[1].set_title(f'δ¹⁸O Proxy Data (r = {d18o_corr:.2f})\nδ¹⁸O = {D18O_ALPHA:.2f} × SST + {D18O_BETA:.2f} + ε')
        axes[2].set_title(f'UK\'37 Proxy Data (r = {uk37_corr:.2f})\nUK\'37 = {UK37_ALPHA:.3f} × SST + {UK37_BETA:.3f} + ε')
        
        # Add legend to all plots
        for ax in axes:
            ax.legend(loc='upper right')
            
        plt.tight_layout()
        return fig


# Define the Gaussian Process model for latent variable reconstruction using GPyTorch
class LatentGPModel(gpytorch.models.ExactGP):
    """
    GPyTorch model for implementing latent SST reconstruction from proxy data.
    This model explicitly treats SST as a latent (hidden) variable that generates
    the observed proxy values according to established paleoclimate equations.
    """
    
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf', nu=2.5):
        """
        Initialize the GP model with specified kernel.
        
        Parameters:
            train_x (tensor): Training input features (ages)
            train_y (tensor): Training target values (proxy-derived SST)
            likelihood (gpytorch.likelihoods): The likelihood for the model
            kernel_type (str): Kernel type ('rbf', 'matern', or 'combined')
            nu (float): Parameter for Matern kernel
        """
        super(LatentGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.kernel_type = kernel_type
        self.nu = nu
        
        # Mean module (constant mean)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Initialize different kernel types for modeling the latent SST process
        if kernel_type == 'rbf':
            # RBF (squared exponential) kernel - good for smooth processes
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        elif kernel_type == 'matern':
            # Matern kernel - allows for less smooth processes
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=nu)
            )
        elif kernel_type == 'combined':
            # Combined kernel: RBF + Periodic
            # Good for capturing both smooth trends and cyclic patterns
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ) + gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel()
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
            x (tensor): Input features (ages)
            
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
        elif self.kernel_type == 'combined':
            # Extract parameters from the combined kernel components
            params['rbf_lengthscale'] = self.covar_module.kernels[0].base_kernel.lengthscale.item()
            params['rbf_outputscale'] = self.covar_module.kernels[0].outputscale.item()
            params['periodic_lengthscale'] = self.covar_module.kernels[1].base_kernel.lengthscale.item()
            params['periodic_period_length'] = self.covar_module.kernels[1].base_kernel.period_length.item()
            params['periodic_outputscale'] = self.covar_module.kernels[1].outputscale.item()
        
        return params


def calibrate_proxy_to_sst(proxy_values, proxy_type):
    """
    Convert proxy values to SST using established paleoclimate equations.
    This is a simple inversion of the calibration equations.
    
    Parameters:
        proxy_values (array): Proxy measurements
        proxy_type (str): Type of proxy ('d18o' or 'uk37')
        
    Returns:
        array: Calibrated SST values
    """
    if proxy_type == 'd18o':
        # δ18O to SST: invert the equation δ18O = D18O_ALPHA * SST + D18O_BETA
        return (proxy_values - D18O_BETA) / D18O_ALPHA
    elif proxy_type == 'uk37':
        # UK'37 to SST: invert the equation UK'37 = UK37_ALPHA * SST + UK37_BETA
        return (proxy_values - UK37_BETA) / UK37_ALPHA
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}")


def train_gp_model(train_x, train_y, kernel_type='rbf', nu=2.5, n_iterations=100, lr=0.1):
    """
    Train a GP model to reconstruct the latent SST from proxy-derived values.
    
    Parameters:
        train_x (tensor): Training inputs (ages)
        train_y (tensor): Training targets (proxy-derived SST)
        kernel_type (str): Kernel type
        nu (float): Parameter for Matern kernel
        n_iterations (int): Number of training iterations
        lr (float): Learning rate for optimizer
        
    Returns:
        tuple: (trained model, likelihood, training losses)
    """
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LatentGPModel(train_x, train_y, likelihood, kernel_type=kernel_type, nu=nu)
    
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


def evaluate_gp_model(model, likelihood, test_x, test_y, true_y=None):
    """
    Evaluate a trained GP model on test data.
    
    Parameters:
        model (LatentGPModel): Trained GP model
        likelihood (gpytorch.likelihoods): Model likelihood
        test_x (tensor): Test inputs (ages)
        test_y (tensor): Test targets (proxy-derived SST)
        true_y (tensor, optional): True latent SST values if available
        
    Returns:
        tuple: (metrics, predictions)
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
    
    # Calculate metrics against proxy-derived SST
    rmse = np.sqrt(mean_squared_error(test_y_np, mean_pred))
    mae = mean_absolute_error(test_y_np, mean_pred)
    r2 = r2_score(test_y_np, mean_pred)
    
    # Print metrics
    print(f"Metrics against proxy-derived SST:")
    print(f"RMSE: {rmse:.4f}°C")
    print(f"MAE: {mae:.4f}°C")
    print(f"R²: {r2:.4f}")
    
    # If true latent SST is provided, calculate metrics against it as well
    if true_y is not None:
        true_y_np = true_y.numpy()
        true_rmse = np.sqrt(mean_squared_error(true_y_np, mean_pred))
        true_mae = mean_absolute_error(true_y_np, mean_pred)
        true_r2 = r2_score(true_y_np, mean_pred)
        
        print(f"\nMetrics against TRUE latent SST:")
        print(f"RMSE: {true_rmse:.4f}°C")
        print(f"MAE: {true_mae:.4f}°C")
        print(f"R²: {true_r2:.4f}")
        
        metrics = {
            'proxy_rmse': rmse,
            'proxy_mae': mae,
            'proxy_r2': r2,
            'true_rmse': true_rmse,
            'true_mae': true_mae,
            'true_r2': true_r2
        }
    else:
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    return metrics, mean_pred


def plot_gp_predictions(model, likelihood, test_x, test_y, true_x, true_y, proxy_type):
    """
    Plot GP model predictions with uncertainty intervals.
    
    Parameters:
        model (LatentGPModel): Trained GP model
        likelihood (gpytorch.likelihoods): Model likelihood
        test_x (tensor): Test inputs (ages)
        test_y (tensor): Test targets (proxy-derived SST)
        true_x (tensor): Full domain inputs (ages)
        true_y (tensor): Full domain true latent SST values
        proxy_type (str): Type of proxy used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Set model to evaluation mode
    model.eval()
    likelihood.eval()
    
    # Make predictions on test data
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get predictions with uncertainty for test points
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
    
    # Calculate metrics against true latent SST
    true_rmse = np.sqrt(mean_squared_error(true_y_np, mean_full))
    true_mae = mean_absolute_error(true_y_np, mean_full)
    true_r2 = r2_score(true_y_np, mean_full)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true latent SST
    ax.plot(true_x_np, true_y_np, 'k-', linewidth=1.5, label='True SST (Latent)')
    
    # Plot test data points (proxy-derived SST)
    marker_style = 'bo' if proxy_type == 'd18o' else 'ro'
    ax.plot(test_x_np, test_y_np, marker_style, markersize=4, alpha=0.6, 
            label=f'Calibrated {proxy_type} (Observed)')
    
    # Plot GP mean prediction of latent SST
    ax.plot(true_x_np, mean_full, 'g-', linewidth=2, 
            label=f'GP Reconstructed SST ({model.kernel_type})')
    
    # Plot uncertainty interval (95% confidence)
    ax.fill_between(true_x_np, lower_full, upper_full, alpha=0.3, color='g', 
                   label='95% Confidence Interval')
    
    # Add metrics to plot
    metrics_text = f"RMSE: {true_rmse:.3f}°C\nMAE: {true_mae:.3f}°C\nR²: {true_r2:.3f}"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Age (kyr BP)')
    ax.set_ylabel('Temperature (°C)')
    kernel_name = model.kernel_type
    if kernel_name == 'matern':
        nu_val = model.nu if hasattr(model, 'nu') else 2.5
        kernel_name = f"Matern(ν={nu_val})"
        
    ax.set_title(f'Latent SST Reconstruction from {proxy_type.upper()} using {kernel_name} Kernel')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_training_loss(losses, kernel_type, proxy_type):
    """
    Plot training loss over iterations to verify convergence.
    
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
        kernel_name = f"Matern(ν=2.5)"
    ax.set_title(f'Training Loss for {kernel_name} Kernel on {proxy_type.upper()} Proxy')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    if window_size > 1:
        ax.legend()
    
    # Set y-axis
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_kernel_comparison(metrics_dict, proxy_type, metric_type='true'):
    """
    Plot comparison of different kernels for latent SST reconstruction.
    
    Parameters:
        metrics_dict (dict): Dictionary of metrics for each kernel
        proxy_type (str): Proxy type used
        metric_type (str): Type of metric to use ('true' or 'proxy')
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract kernel names and metrics
    kernels = list(metrics_dict.keys())
    
    # Use the appropriate metrics based on metric_type
    prefix = f"{metric_type}_" if f"{metric_type}_rmse" in metrics_dict[kernels[0]] else ""
    
    rmse_values = [metrics_dict[k][f'{prefix}rmse'] for k in kernels]
    mae_values = [metrics_dict[k][f'{prefix}mae'] for k in kernels]
    r2_values = [metrics_dict[k][f'{prefix}r2'] for k in kernels]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE plot (lower is better)
    axes[0].bar(kernels, rmse_values, color='skyblue')
    axes[0].set_ylabel('RMSE (°C)')
    axes[0].set_title('Root Mean Square Error')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE plot (lower is better)
    axes[1].bar(kernels, mae_values, color='lightgreen')
    axes[1].set_ylabel('MAE (°C)')
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R² plot (higher is better)
    axes[2].bar(kernels, r2_values, color='salmon')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('R² Score')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add values on top of bars
    for ax, values in zip(axes, [rmse_values, mae_values, r2_values]):
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Set common title
    metric_label = "TRUE Latent SST" if metric_type == 'true' else "Proxy-Derived SST"
    fig.suptitle(f'Kernel Performance Comparison for {proxy_type.upper()} Proxy (vs {metric_label})', 
                fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig


def plot_residuals(predictions, true_values, kernel_type, proxy_type):
    """
    Plot residuals to diagnose if reconstructions accurately infer latent SST.
    
    Parameters:
        predictions (array): Predicted SST values
        true_values (array): True latent SST values
        kernel_type (str): Kernel type used
        proxy_type (str): Proxy type used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Calculate residuals
    residuals = predictions - true_values
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot residuals against index
    axes[0].scatter(range(len(residuals)), residuals, alpha=0.7)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add statistics
    rmse = np.sqrt(np.mean(residuals**2))
    bias = np.mean(residuals)
    std = np.std(residuals)
    
    # Add statistics to plot
    axes[0].text(0.02, 0.95, f"RMSE: {rmse:.3f}°C\nBias: {bias:.3f}°C\nStd: {std:.3f}°C", 
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Residual (°C)')
    axes[0].set_title(f'Residuals (Predicted - True)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot residuals histogram
    axes[1].hist(residuals, bins=20, alpha=0.7, density=True)
    
    # Add normal distribution fit
    x = np.linspace(min(residuals), max(residuals), 100)
    y = norm.pdf(x, bias, std)
    axes[1].plot(x, y, 'r-', linewidth=2)
    
    axes[1].set_xlabel('Residual (°C)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Residual Analysis for {kernel_type} Kernel on {proxy_type.upper()} Proxy', 
                fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    return fig


def plot_edge_detection(true_sst, predicted_sst, ages, kernel_type, proxy_type):
    """
    Plot how well the GP captures abrupt changes in SST.
    
    Parameters:
        true_sst (array): True latent SST values
        predicted_sst (array): Predicted SST values
        ages (array): Age points
        kernel_type (str): Kernel type used
        proxy_type (str): Proxy type used
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Calculate first derivative (rate of change) for both true and predicted SST
    true_diff = np.diff(true_sst) / np.diff(ages)
    pred_diff = np.diff(predicted_sst) / np.diff(ages)
    
    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot true and predicted SST
    axes[0].plot(ages, true_sst, 'k-', linewidth=1.5, label='True SST')
    axes[0].plot(ages, predicted_sst, 'g-', linewidth=1.5, label=f'Predicted SST ({kernel_type})')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('SST Reconstruction')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot the derivatives (highlighting sharp transitions)
    axes[1].plot(ages[1:], true_diff, 'k-', linewidth=1.5, label='True SST Rate of Change')
    axes[1].plot(ages[1:], pred_diff, 'g-', linewidth=1.5, label='Predicted SST Rate of Change')
    axes[1].set_xlabel('Age (kyr BP)')
    axes[1].set_ylabel('Rate of Change (°C/kyr)')
    axes[1].set_title('SST Rate of Change (First Derivative)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Add overall title
    fig.suptitle(f'Edge Detection Analysis for {kernel_type} Kernel on {proxy_type.upper()} Proxy', 
                fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    return fig


def main():
    """Main function to reconstruct latent SST from proxy data using GP models."""
    start_time = time.time()
    print("Starting Latent SST Reconstruction using Gaussian Processes...")
    
    # Step I: Generate synthetic paleoclimate data
    print("\nGenerating synthetic paleoclimate data...")
    synth_data = SyntheticPaleoData(n_points=N_POINTS)
    dataset = synth_data.generate_dataset()
    
    # Plot and save synthetic data
    fig_data = synth_data.plot_dataset(dataset)
    fig_data.savefig(os.path.join(output_dir, "latent_synthetic_data.png"), dpi=300)
    plt.close(fig_data)
    
    # Step II: Prepare data for GP models
    print("\nPreparing data for latent variable GP models...")
    
    # Extract data
    ages = dataset['age']
    true_sst = dataset['true_sst']
    d18o_values = dataset['d18o']
    uk37_values = dataset['uk37']
    
    # Calibrate proxies to SST (simple inversion of the calibration equations)
    # This provides the observed, noisy estimate of the latent SST
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
    d18o_test_y = torch.tensor(d18o_sst[d18o_test_idx], dtype=torch.float32)
    d18o_true_y = torch.tensor(true_sst[d18o_test_idx], dtype=torch.float32)  # True latent SST
    
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
    uk37_test_y = torch.tensor(uk37_sst[uk37_test_idx], dtype=torch.float32)
    uk37_true_y = torch.tensor(true_sst[uk37_test_idx], dtype=torch.float32)  # True latent SST
    
    # Step III: Train and evaluate GP models with different kernels
    # Define kernels to test: RBF, Matern 2.5, Combined (RBF + Periodic)
    kernels = [
        ('rbf', None),          # RBF kernel
        ('matern', 2.5),        # Matern kernel with ν=2.5
        ('combined', None)      # Combined RBF + Periodic kernel
    ]
    
    # Dictionary to store results
    d18o_results = {}
    d18o_predictions = {}
    uk37_results = {}
    uk37_predictions = {}
    
    # Train models for δ18O proxy
    print("\n===== Training GP Models for δ18O Proxy =====")
    print("These models reconstruct the latent SST from observed δ18O values")
    
    for kernel_type, nu in kernels:
        # Create kernel name
        if kernel_type == 'matern':
            kernel_name = f"{kernel_type}"
        else:
            kernel_name = kernel_type
        
        print(f"\nTraining {kernel_name} kernel model...")
        
        # Train model
        model, likelihood, losses = train_gp_model(
            d18o_train_x, d18o_train_y, 
            kernel_type=kernel_type, 
            nu=nu if nu else 2.5
        )
        
        # Evaluate model against both proxy-derived SST and true latent SST
        print(f"\nEvaluating {kernel_name} kernel model...")
        metrics, predictions = evaluate_gp_model(
            model, likelihood, d18o_test_x, d18o_test_y, d18o_true_y
        )
        
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
        fig_pred.savefig(os.path.join(output_dir, f"latent_d18o_{kernel_name}_predictions.png"), dpi=300)
        plt.close(fig_pred)
        
        # Plot and save training loss
        fig_loss = plot_training_loss(losses, kernel_name, 'd18o')
        fig_loss.savefig(os.path.join(output_dir, f"latent_d18o_{kernel_name}_loss.png"), dpi=300)
        plt.close(fig_loss)
        
        # Plot residuals
        fig_resid = plot_residuals(
            predictions, d18o_true_y.numpy(), kernel_name, 'd18o'
        )
        fig_resid.savefig(os.path.join(output_dir, f"latent_d18o_{kernel_name}_residuals.png"), dpi=300)
        plt.close(fig_resid)
        
        # Plot edge detection analysis
        fig_edge = plot_edge_detection(
            true_sst[d18o_test_idx], predictions, ages[d18o_test_idx], kernel_name, 'd18o'
        )
        fig_edge.savefig(os.path.join(output_dir, f"latent_d18o_{kernel_name}_edges.png"), dpi=300)
        plt.close(fig_edge)
    
    # Train models for UK'37 proxy
    print("\n===== Training GP Models for UK'37 Proxy =====")
    print("These models reconstruct the latent SST from observed UK'37 values")
    
    for kernel_type, nu in kernels:
        # Create kernel name
        if kernel_type == 'matern':
            kernel_name = f"{kernel_type}"
        else:
            kernel_name = kernel_type
        
        print(f"\nTraining {kernel_name} kernel model...")
        
        # Train model
        model, likelihood, losses = train_gp_model(
            uk37_train_x, uk37_train_y, 
            kernel_type=kernel_type, 
            nu=nu if nu else 2.5
        )
        
        # Evaluate model against both proxy-derived SST and true latent SST
        print(f"\nEvaluating {kernel_name} kernel model...")
        metrics, predictions = evaluate_gp_model(
            model, likelihood, uk37_test_x, uk37_test_y, uk37_true_y
        )
        
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
        fig_pred.savefig(os.path.join(output_dir, f"latent_uk37_{kernel_name}_predictions.png"), dpi=300)
        plt.close(fig_pred)
        
        # Plot and save training loss
        fig_loss = plot_training_loss(losses, kernel_name, 'uk37')
        fig_loss.savefig(os.path.join(output_dir, f"latent_uk37_{kernel_name}_loss.png"), dpi=300)
        plt.close(fig_loss)
        
        # Plot residuals
        fig_resid = plot_residuals(
            predictions, uk37_true_y.numpy(), kernel_name, 'uk37'
        )
        fig_resid.savefig(os.path.join(output_dir, f"latent_uk37_{kernel_name}_residuals.png"), dpi=300)
        plt.close(fig_resid)
        
        # Plot edge detection analysis
        fig_edge = plot_edge_detection(
            true_sst[uk37_test_idx], predictions, ages[uk37_test_idx], kernel_name, 'uk37'
        )
        fig_edge.savefig(os.path.join(output_dir, f"latent_uk37_{kernel_name}_edges.png"), dpi=300)
        plt.close(fig_edge)
    
    # Step IV: Compare kernel performance
    print("\n===== Kernel Performance Comparison =====")
    
    # Extract metrics for comparison
    d18o_metrics = {k: v['metrics'] for k, v in d18o_results.items()}
    uk37_metrics = {k: v['metrics'] for k, v in uk37_results.items()}
    
    # Plot kernel comparison for true latent SST recovery
    fig_d18o_comp = plot_kernel_comparison(d18o_metrics, 'd18o', metric_type='true')
    fig_d18o_comp.savefig(os.path.join(output_dir, "latent_d18o_kernel_comparison.png"), dpi=300)
    plt.close(fig_d18o_comp)
    
    fig_uk37_comp = plot_kernel_comparison(uk37_metrics, 'uk37', metric_type='true')
    fig_uk37_comp.savefig(os.path.join(output_dir, "latent_uk37_kernel_comparison.png"), dpi=300)
    plt.close(fig_uk37_comp)
    
    # Find the best kernel for each proxy based on RMSE against true latent SST
    d18o_best_kernel = min(d18o_metrics.items(), key=lambda x: x[1]['true_rmse'])[0]
    uk37_best_kernel = min(uk37_metrics.items(), key=lambda x: x[1]['true_rmse'])[0]
    
    # Save metrics to CSV
    d18o_metrics_df = pd.DataFrame([
        {
            'Kernel': k, 
            'Proxy_RMSE': v['proxy_rmse'], 
            'Proxy_MAE': v['proxy_mae'], 
            'Proxy_R2': v['proxy_r2'],
            'True_RMSE': v['true_rmse'],
            'True_MAE': v['true_mae'],
            'True_R2': v['true_r2']
        } 
        for k, v in d18o_metrics.items()
    ])
    d18o_metrics_df.to_csv(os.path.join(output_dir, "latent_d18o_metrics.csv"), index=False)
    
    uk37_metrics_df = pd.DataFrame([
        {
            'Kernel': k, 
            'Proxy_RMSE': v['proxy_rmse'], 
            'Proxy_MAE': v['proxy_mae'], 
            'Proxy_R2': v['proxy_r2'],
            'True_RMSE': v['true_rmse'],
            'True_MAE': v['true_mae'],
            'True_R2': v['true_r2']
        } 
        for k, v in uk37_metrics.items()
    ])
    uk37_metrics_df.to_csv(os.path.join(output_dir, "latent_uk37_metrics.csv"), index=False)
    
    # Print summary of results with metrics
    print("\n===== SUMMARY OF RESULTS =====")
    print(f"\nBest kernel for δ18O proxy latent SST reconstruction: {d18o_best_kernel}")
    print(f"RMSE against true SST: {d18o_metrics[d18o_best_kernel]['true_rmse']:.4f}°C")
    print(f"MAE against true SST: {d18o_metrics[d18o_best_kernel]['true_mae']:.4f}°C")
    print(f"R² against true SST: {d18o_metrics[d18o_best_kernel]['true_r2']:.4f}")
    
    print(f"\nBest kernel for UK'37 proxy latent SST reconstruction: {uk37_best_kernel}")
    print(f"RMSE against true SST: {uk37_metrics[uk37_best_kernel]['true_rmse']:.4f}°C")
    print(f"MAE against true SST: {uk37_metrics[uk37_best_kernel]['true_mae']:.4f}°C")
    print(f"R² against true SST: {uk37_metrics[uk37_best_kernel]['true_r2']:.4f}")
    
    # Diagnostic information about capturing abrupt changes
    print("\n===== MODEL DIAGNOSTIC SUMMARY =====")
    
    # For δ18O proxy
    d18o_best_model_params = d18o_results[d18o_best_kernel]['kernel_params']
    print(f"\nδ18O best model ({d18o_best_kernel}) kernel parameters:")
    for param, value in d18o_best_model_params.items():
        print(f"  {param}: {value:.4f}")
    
    # Analyze how well the model captures abrupt changes
    true_diffs = np.abs(np.diff(true_sst[d18o_test_idx]))
    pred_diffs = np.abs(np.diff(d18o_predictions[d18o_best_kernel]))
    edge_correlation = np.corrcoef(true_diffs, pred_diffs)[0,1]
    print(f"\nEdge detection performance for δ18O {d18o_best_kernel} model:")
    print(f"  Correlation of rate-of-change: {edge_correlation:.4f}")
    print(f"  Ability to capture abrupt changes: {'Good' if edge_correlation > 0.6 else 'Moderate' if edge_correlation > 0.3 else 'Poor'}")
    
    # For UK'37 proxy
    uk37_best_model_params = uk37_results[uk37_best_kernel]['kernel_params']
    print(f"\nUK'37 best model ({uk37_best_kernel}) kernel parameters:")
    for param, value in uk37_best_model_params.items():
        print(f"  {param}: {value:.4f}")
    
    # Analyze how well the model captures abrupt changes
    true_diffs = np.abs(np.diff(true_sst[uk37_test_idx]))
    pred_diffs = np.abs(np.diff(uk37_predictions[uk37_best_kernel]))
    edge_correlation = np.corrcoef(true_diffs, pred_diffs)[0,1]
    print(f"\nEdge detection performance for UK'37 {uk37_best_kernel} model:")
    print(f"  Correlation of rate-of-change: {edge_correlation:.4f}")
    print(f"  Ability to capture abrupt changes: {'Good' if edge_correlation > 0.6 else 'Moderate' if edge_correlation > 0.3 else 'Poor'}")
    
    # Provide final recommendation
    print("\n===== FINAL RECOMMENDATION =====")
    if d18o_metrics[d18o_best_kernel]['true_r2'] > uk37_metrics[uk37_best_kernel]['true_r2']:
        print(f"The δ18O proxy with {d18o_best_kernel} kernel provides better latent SST reconstruction")
        print(f"R² = {d18o_metrics[d18o_best_kernel]['true_r2']:.4f} vs {uk37_metrics[uk37_best_kernel]['true_r2']:.4f}")
        
        if d18o_best_kernel == 'combined':
            print("\nThe combined kernel (RBF + Periodic) effectively captures both long-term trends")
            print("and cyclical patterns in the data, resulting in better SST reconstruction.")
    else:
        print(f"The UK'37 proxy with {uk37_best_kernel} kernel provides better latent SST reconstruction")
        print(f"R² = {uk37_metrics[uk37_best_kernel]['true_r2']:.4f} vs {d18o_metrics[d18o_best_kernel]['true_r2']:.4f}")
        
        if uk37_best_kernel == 'combined':
            print("\nThe combined kernel (RBF + Periodic) effectively captures both long-term trends")
            print("and cyclical patterns in the data, resulting in better SST reconstruction.")
    
    # Print final message
    print(f"\nResults saved to {output_dir}")
    end_time = time.time()
    print(f"\nLatent SST reconstruction completed successfully! Time elapsed: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()