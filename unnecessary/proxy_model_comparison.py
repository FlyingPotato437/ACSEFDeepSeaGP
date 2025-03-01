"""
proxy_model_comparison.py - Compare AR and GP models for SST reconstruction from proxies

This module builds and compares AR(1), AR(2), and Gaussian Process models for
reconstructing sea surface temperature from proxy data (δ18O and UK'37).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Import synthetic data generation functions
from synthetic_proxy_data import (
    generate_synthetic_sst,
    generate_d18o_proxy,
    generate_uk37_proxy,
    calculate_power_spectrum
)

# Set random seed for reproducibility
np.random.seed(42)


class AR1Model:
    """
    Autoregressive model of order 1 (AR1) for time series modeling.
    y_t = c + φ * y_(t-1) + ε_t
    """
    
    def __init__(self):
        """Initialize AR1 model parameters."""
        self.phi = None  # AR coefficient
        self.c = None    # Constant term
        self.sigma = None  # Noise standard deviation
        
    def fit(self, y):
        """
        Fit AR1 model to the given time series.
        
        Parameters:
            y (array): Time series data
        """
        # Calculate AR(1) coefficient using lag-1 autocorrelation
        n = len(y)
        y_lag1 = y[:-1]
        y_current = y[1:]
        
        # Fit linear regression: y_t = c + φ * y_(t-1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_lag1, y_current)
        
        self.phi = slope
        self.c = intercept
        
        # Calculate residuals and their standard deviation
        residuals = y_current - (intercept + slope * y_lag1)
        self.sigma = np.std(residuals)
        
        # Print model summary
        print(f"AR(1) model: y_t = {self.c:.4f} + {self.phi:.4f} * y_(t-1)")
        print(f"Residual std: {self.sigma:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        
    def predict(self, y_init, n_steps):
        """
        Generate predictions for n_steps ahead using the AR1 model.
        
        Parameters:
            y_init (float): Initial value for prediction
            n_steps (int): Number of steps to predict
            
        Returns:
            array: Predicted values
        """
        predictions = np.zeros(n_steps)
        predictions[0] = y_init
        
        for t in range(1, n_steps):
            # Predict next value: y_t = c + φ * y_(t-1)
            predictions[t] = self.c + self.phi * predictions[t-1]
            
        return predictions
    
    def simulate(self, y_init, n_steps, include_noise=True):
        """
        Simulate time series from AR1 model including noise.
        
        Parameters:
            y_init (float): Initial value for simulation
            n_steps (int): Number of steps to simulate
            include_noise (bool): Whether to include noise in simulation
            
        Returns:
            array: Simulated time series
        """
        simulated = np.zeros(n_steps)
        simulated[0] = y_init
        
        for t in range(1, n_steps):
            # Simulate next value: y_t = c + φ * y_(t-1) + ε_t
            if include_noise:
                noise = np.random.normal(0, self.sigma)
            else:
                noise = 0
                
            simulated[t] = self.c + self.phi * simulated[t-1] + noise
            
        return simulated


class AR2Model:
    """
    Autoregressive model of order 2 (AR2) for time series modeling.
    y_t = c + φ1 * y_(t-1) + φ2 * y_(t-2) + ε_t
    """
    
    def __init__(self):
        """Initialize AR2 model parameters."""
        self.phi1 = None  # First AR coefficient
        self.phi2 = None  # Second AR coefficient
        self.c = None     # Constant term
        self.sigma = None  # Noise standard deviation
        
    def fit(self, y):
        """
        Fit AR2 model to the given time series.
        
        Parameters:
            y (array): Time series data
        """
        # Prepare data for regression
        n = len(y)
        y_lag1 = y[1:-1]  # y_(t-1)
        y_lag2 = y[:-2]   # y_(t-2)
        y_current = y[2:] # y_t
        
        # Create design matrix for multiple regression
        X = np.column_stack((np.ones(len(y_lag1)), y_lag1, y_lag2))
        
        # Fit regression: y_t = c + φ1 * y_(t-1) + φ2 * y_(t-2)
        coeffs, residuals, _, _ = np.linalg.lstsq(X, y_current, rcond=None)
        
        self.c = coeffs[0]
        self.phi1 = coeffs[1]
        self.phi2 = coeffs[2]
        
        # Calculate residuals and their standard deviation
        y_pred = X @ coeffs
        residuals = y_current - y_pred
        self.sigma = np.std(residuals)
        
        # Calculate R-squared
        ss_total = np.sum((y_current - np.mean(y_current))**2)
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Print model summary
        print(f"AR(2) model: y_t = {self.c:.4f} + {self.phi1:.4f} * y_(t-1) + {self.phi2:.4f} * y_(t-2)")
        print(f"Residual std: {self.sigma:.4f}")
        print(f"R-squared: {r_squared:.4f}")
        
    def predict(self, y_init_1, y_init_2, n_steps):
        """
        Generate predictions for n_steps ahead using the AR2 model.
        
        Parameters:
            y_init_1 (float): Initial value at t-1
            y_init_2 (float): Initial value at t-2
            n_steps (int): Number of steps to predict
            
        Returns:
            array: Predicted values
        """
        predictions = np.zeros(n_steps)
        
        # Set initial values
        if n_steps > 0:
            predictions[0] = y_init_1
        if n_steps > 1:
            predictions[1] = y_init_2
            
        for t in range(2, n_steps):
            # Predict next value: y_t = c + φ1 * y_(t-1) + φ2 * y_(t-2)
            predictions[t] = self.c + self.phi1 * predictions[t-1] + self.phi2 * predictions[t-2]
            
        return predictions
    
    def simulate(self, y_init_1, y_init_2, n_steps, include_noise=True):
        """
        Simulate time series from AR2 model including noise.
        
        Parameters:
            y_init_1 (float): Initial value at t-1
            y_init_2 (float): Initial value at t-2
            n_steps (int): Number of steps to simulate
            include_noise (bool): Whether to include noise in simulation
            
        Returns:
            array: Simulated time series
        """
        simulated = np.zeros(n_steps)
        
        # Set initial values
        if n_steps > 0:
            simulated[0] = y_init_1
        if n_steps > 1:
            simulated[1] = y_init_2
            
        for t in range(2, n_steps):
            # Simulate next value: y_t = c + φ1 * y_(t-1) + φ2 * y_(t-2) + ε_t
            if include_noise:
                noise = np.random.normal(0, self.sigma)
            else:
                noise = 0
                
            simulated[t] = self.c + self.phi1 * simulated[t-1] + self.phi2 * simulated[t-2] + noise
            
        return simulated


class GaussianProcessModel:
    """
    Gaussian Process regression model for time series modeling with various kernels.
    """
    
    def __init__(self, kernel_type='rbf'):
        """
        Initialize GP model with specified kernel type.
        
        Parameters:
            kernel_type (str): Type of kernel to use ('rbf', 'periodic', or 'combined')
        """
        self.kernel_type = kernel_type
        
        # Define different types of kernels
        if kernel_type == 'rbf':
            # RBF kernel with white noise - good for smooth trends
            self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        elif kernel_type == 'periodic':
            # Periodic kernel - good for seasonal patterns
            self.kernel = ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0) + WhiteKernel(noise_level=0.1)
        elif kernel_type == 'combined':
            # Combined kernel - good for both trends and seasonality
            rbf_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            periodic_kernel = ConstantKernel(0.5) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
            self.kernel = rbf_kernel + periodic_kernel + WhiteKernel(noise_level=0.1)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Initialize the GP model
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            alpha=1e-10,  # Numerical stability
            normalize_y=True
        )
        
    def fit(self, X, y):
        """
        Fit GP model to the given time series.
        
        Parameters:
            X (array): Input features (typically time points)
            y (array): Target values (time series data)
        """
        # Reshape X to 2D array if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Fit the model
        self.model.fit(X, y)
        
        # Print model summary
        print(f"GP model with {self.kernel_type} kernel:")
        print(f"Optimized kernel: {self.model.kernel_}")
        print(f"Log-likelihood: {self.model.log_marginal_likelihood(self.model.kernel_.theta):.4f}")
        
    def predict(self, X, return_std=False):
        """
        Generate predictions using the GP model.
        
        Parameters:
            X (array): Input features for prediction
            return_std (bool): Whether to return standard deviation
            
        Returns:
            array or tuple: Predicted values and optionally standard deviations
        """
        # Reshape X to 2D array if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return self.model.predict(X, return_std=return_std)


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
        return (16.9 - proxy_values) / 4.38
    elif proxy_type == 'uk37':
        # UK'37 to SST: reverse the equation UK'37 = 0.033 * SST + 0.044
        return (proxy_values - 0.044) / 0.033
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}")


def evaluate_models(true_sst, predicted_sst_dict):
    """
    Evaluate models by calculating performance metrics.
    
    Parameters:
        true_sst (array): True SST values
        predicted_sst_dict (dict): Dictionary of model predictions
        
    Returns:
        pd.DataFrame: DataFrame with evaluation metrics
    """
    metrics = []
    
    for model_name, predicted_sst in predicted_sst_dict.items():
        rmse = np.sqrt(mean_squared_error(true_sst, predicted_sst))
        r2 = r2_score(true_sst, predicted_sst)
        mae = np.mean(np.abs(true_sst - predicted_sst))
        bias = np.mean(predicted_sst - true_sst)
        corr = np.corrcoef(true_sst, predicted_sst)[0, 1]
        
        metrics.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Bias': bias,
            'Correlation': corr
        })
    
    return pd.DataFrame(metrics)


def plot_model_reconstructions(time_points, true_sst, predicted_sst_dict, proxy_type):
    """
    Plot model reconstructions against true SST.
    
    Parameters:
        time_points (array): Time points for x-axis
        true_sst (array): True SST values
        predicted_sst_dict (dict): Dictionary of model predictions
        proxy_type (str): Type of proxy used for reconstruction
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true SST
    ax.plot(time_points, true_sst, 'k-', linewidth=2, label='True SST')
    
    # Plot model reconstructions
    colors = ['b', 'g', 'r', 'm', 'c']
    for i, (model_name, predicted_sst) in enumerate(predicted_sst_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(time_points, predicted_sst, color=color, linestyle='-', linewidth=1.5, 
                label=f'{model_name} Reconstruction')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'SST Reconstructions from {proxy_type} Proxy')
    ax.grid(True)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig


def plot_model_residuals(time_points, true_sst, predicted_sst_dict, proxy_type):
    """
    Plot residuals for each model.
    
    Parameters:
        time_points (array): Time points for x-axis
        true_sst (array): True SST values
        predicted_sst_dict (dict): Dictionary of model predictions
        proxy_type (str): Type of proxy used for reconstruction
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    n_models = len(predicted_sst_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3*n_models), sharex=True)
    
    if n_models == 1:
        axes = [axes]  # Make axes iterable if there's only one model
    
    colors = ['b', 'g', 'r', 'm', 'c']
    for i, (model_name, predicted_sst) in enumerate(predicted_sst_dict.items()):
        color = colors[i % len(colors)]
        
        # Calculate residuals
        residuals = predicted_sst - true_sst
        
        # Plot residuals
        axes[i].plot(time_points, residuals, color=color, linestyle='-', linewidth=1.5)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[i].set_ylabel('Residual (°C)')
        axes[i].set_title(f'{model_name} Residuals')
        axes[i].grid(True)
        
        # Add RMSE and mean bias to the plot
        rmse = np.sqrt(np.mean(residuals**2))
        bias = np.mean(residuals)
        axes[i].text(0.02, 0.95, f'RMSE: {rmse:.2f}°C\nBias: {bias:.2f}°C', 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[-1].set_xlabel('Time (years)')
    
    plt.tight_layout()
    return fig


def plot_power_spectra_comparison(time_points, true_sst, predicted_sst_dict, proxy_type):
    """
    Plot power spectra for true SST and model reconstructions.
    
    Parameters:
        time_points (array): Time points
        true_sst (array): True SST values
        predicted_sst_dict (dict): Dictionary of model predictions
        proxy_type (str): Type of proxy used for reconstruction
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Calculate time step
    dt = time_points[1] - time_points[0]
    
    # Calculate power spectrum for true SST
    freq_true, power_true = calculate_power_spectrum(true_sst, dt)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true SST spectrum
    ax.loglog(freq_true, power_true, 'k-', linewidth=2, label='True SST')
    
    # Plot model spectra
    colors = ['b', 'g', 'r', 'm', 'c']
    for i, (model_name, predicted_sst) in enumerate(predicted_sst_dict.items()):
        color = colors[i % len(colors)]
        
        # Calculate power spectrum
        freq_model, power_model = calculate_power_spectrum(predicted_sst, dt)
        
        # Plot model spectrum
        ax.loglog(freq_model, power_model, color=color, linestyle='-', linewidth=1.5, 
                 label=f'{model_name}')
    
    # Add vertical lines at key frequencies (if applicable)
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.3, label='Annual')
    
    ax.set_xlabel('Frequency (1/year)')
    ax.set_ylabel('Power')
    ax.set_title(f'Power Spectra Comparison for {proxy_type} Proxy Reconstructions')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig


def main():
    """Main function to compare models for proxy-based SST reconstruction."""
    print("Starting Proxy-Based SST Reconstruction Model Comparison...")
    
    # Create output directory if it doesn't exist
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    print("\nGenerating synthetic SST and proxy data...")
    time_points, true_sst = generate_synthetic_sst(n_points=200)
    d18o_values = generate_d18o_proxy(true_sst)
    uk37_values = generate_uk37_proxy(true_sst)
    
    # Split data into training and testing sets
    train_indices, test_indices = train_test_split(
        np.arange(len(time_points)), test_size=0.3, random_state=42
    )
    
    # Training data
    train_time = time_points[train_indices]
    train_sst = true_sst[train_indices]
    train_d18o = d18o_values[train_indices]
    train_uk37 = uk37_values[train_indices]
    
    # Testing data
    test_time = time_points[test_indices]
    test_sst = true_sst[test_indices]
    test_d18o = d18o_values[test_indices]
    test_uk37 = uk37_values[test_indices]
    
    # Sort testing data by time for proper visualization
    sort_idx = np.argsort(test_time)
    test_time = test_time[sort_idx]
    test_sst = test_sst[sort_idx]
    test_d18o = test_d18o[sort_idx]
    test_uk37 = test_uk37[sort_idx]
    
    # Run model comparison for δ18O proxy
    print("\n===== Model Comparison for δ18O Proxy =====")
    
    # Convert δ18O to SST
    print("\nCalibrating δ18O proxy to SST...")
    train_d18o_sst = calibrate_proxy_to_sst(train_d18o, 'd18o')
    
    # Train AR1 model
    print("\nTraining AR(1) model...")
    ar1_d18o = AR1Model()
    ar1_d18o.fit(train_d18o_sst)
    
    # Train AR2 model
    print("\nTraining AR(2) model...")
    ar2_d18o = AR2Model()
    ar2_d18o.fit(train_d18o_sst)
    
    # Train GP models
    print("\nTraining GP models...")
    gp_rbf_d18o = GaussianProcessModel(kernel_type='rbf')
    gp_rbf_d18o.fit(train_time.reshape(-1, 1), train_d18o_sst)
    
    gp_comb_d18o = GaussianProcessModel(kernel_type='combined')
    gp_comb_d18o.fit(train_time.reshape(-1, 1), train_d18o_sst)
    
    # Make predictions on test data for δ18O
    calibrated_test_d18o = calibrate_proxy_to_sst(test_d18o, 'd18o')
    
    # AR1 predictions
    ar1_preds_d18o = ar1_d18o.predict(calibrated_test_d18o[0], len(test_time))
    
    # AR2 predictions
    ar2_preds_d18o = ar2_d18o.predict(
        calibrated_test_d18o[0], 
        calibrated_test_d18o[1] if len(calibrated_test_d18o) > 1 else calibrated_test_d18o[0],
        len(test_time)
    )
    
    # GP predictions
    gp_rbf_preds_d18o = gp_rbf_d18o.predict(test_time.reshape(-1, 1))
    gp_comb_preds_d18o = gp_comb_d18o.predict(test_time.reshape(-1, 1))
    
    # Collect predictions
    d18o_predictions = {
        'AR(1)': ar1_preds_d18o,
        'AR(2)': ar2_preds_d18o,
        'GP-RBF': gp_rbf_preds_d18o,
        'GP-Combined': gp_comb_preds_d18o,
        'Raw Proxy': calibrated_test_d18o  # Include raw calibrated proxy for comparison
    }
    
    # Evaluate models for δ18O
    print("\nEvaluating models for δ18O...")
    metrics_d18o = evaluate_models(test_sst, d18o_predictions)
    print("\nδ18O Proxy Reconstruction Metrics:")
    print(metrics_d18o.to_string(index=False))
    
    # Create plots for δ18O
    print("\nCreating plots for δ18O reconstructions...")
    fig_recon_d18o = plot_model_reconstructions(test_time, test_sst, d18o_predictions, 'δ18O')
    fig_recon_d18o.savefig(os.path.join(output_dir, "d18o_reconstructions.png"), dpi=300)
    
    fig_resid_d18o = plot_model_residuals(test_time, test_sst, d18o_predictions, 'δ18O')
    fig_resid_d18o.savefig(os.path.join(output_dir, "d18o_residuals.png"), dpi=300)
    
    fig_spectra_d18o = plot_power_spectra_comparison(test_time, test_sst, d18o_predictions, 'δ18O')
    fig_spectra_d18o.savefig(os.path.join(output_dir, "d18o_spectra.png"), dpi=300)
    
    # Run model comparison for UK'37 proxy
    print("\n===== Model Comparison for UK'37 Proxy =====")
    
    # Convert UK'37 to SST
    print("\nCalibrating UK'37 proxy to SST...")
    train_uk37_sst = calibrate_proxy_to_sst(train_uk37, 'uk37')
    
    # Train AR1 model
    print("\nTraining AR(1) model...")
    ar1_uk37 = AR1Model()
    ar1_uk37.fit(train_uk37_sst)
    
    # Train AR2 model
    print("\nTraining AR(2) model...")
    ar2_uk37 = AR2Model()
    ar2_uk37.fit(train_uk37_sst)
    
    # Train GP models
    print("\nTraining GP models...")
    gp_rbf_uk37 = GaussianProcessModel(kernel_type='rbf')
    gp_rbf_uk37.fit(train_time.reshape(-1, 1), train_uk37_sst)
    
    gp_comb_uk37 = GaussianProcessModel(kernel_type='combined')
    gp_comb_uk37.fit(train_time.reshape(-1, 1), train_uk37_sst)
    
    # Make predictions on test data for UK'37
    calibrated_test_uk37 = calibrate_proxy_to_sst(test_uk37, 'uk37')
    
    # AR1 predictions
    ar1_preds_uk37 = ar1_uk37.predict(calibrated_test_uk37[0], len(test_time))
    
    # AR2 predictions
    ar2_preds_uk37 = ar2_uk37.predict(
        calibrated_test_uk37[0], 
        calibrated_test_uk37[1] if len(calibrated_test_uk37) > 1 else calibrated_test_uk37[0],
        len(test_time)
    )
    
    # GP predictions
    gp_rbf_preds_uk37 = gp_rbf_uk37.predict(test_time.reshape(-1, 1))
    gp_comb_preds_uk37 = gp_comb_uk37.predict(test_time.reshape(-1, 1))
    
    # Collect predictions
    uk37_predictions = {
        'AR(1)': ar1_preds_uk37,
        'AR(2)': ar2_preds_uk37,
        'GP-RBF': gp_rbf_preds_uk37,
        'GP-Combined': gp_comb_preds_uk37,
        'Raw Proxy': calibrated_test_uk37  # Include raw calibrated proxy for comparison
    }
    
    # Evaluate models for UK'37
    print("\nEvaluating models for UK'37...")
    metrics_uk37 = evaluate_models(test_sst, uk37_predictions)
    print("\nUK'37 Proxy Reconstruction Metrics:")
    print(metrics_uk37.to_string(index=False))
    
    # Create plots for UK'37
    print("\nCreating plots for UK'37 reconstructions...")
    fig_recon_uk37 = plot_model_reconstructions(test_time, test_sst, uk37_predictions, 'UK\'37')
    fig_recon_uk37.savefig(os.path.join(output_dir, "uk37_reconstructions.png"), dpi=300)
    
    fig_resid_uk37 = plot_model_residuals(test_time, test_sst, uk37_predictions, 'UK\'37')
    fig_resid_uk37.savefig(os.path.join(output_dir, "uk37_residuals.png"), dpi=300)
    
    fig_spectra_uk37 = plot_power_spectra_comparison(test_time, test_sst, uk37_predictions, 'UK\'37')
    fig_spectra_uk37.savefig(os.path.join(output_dir, "uk37_spectra.png"), dpi=300)
    
    # Save metrics to CSV
    metrics_d18o.to_csv(os.path.join(output_dir, "d18o_metrics.csv"), index=False)
    metrics_uk37.to_csv(os.path.join(output_dir, "uk37_metrics.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}")
    print("\nModel comparison completed successfully!")
    
    # Return figures for display
    return fig_recon_d18o, fig_recon_uk37


if __name__ == "__main__":
    main()
    plt.show()  # Show all figures