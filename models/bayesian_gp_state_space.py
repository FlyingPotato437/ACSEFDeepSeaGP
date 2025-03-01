"""
Enhanced Bayesian Gaussian Process State-Space Model for Paleoclimate Reconstruction

This module implements a sophisticated Bayesian GP State-Space model for reconstructing
Sea Surface Temperature (SST) from multiple paleoclimate proxies, with specialized
components for handling:

1. Adaptive kernel lengthscales that adjust to capture abrupt climate transitions
2. Balanced multi-proxy weighting that prevents any single proxy from dominating
3. Multi-scale periodic components for capturing orbital (Milankovitch) cycles
4. Heteroscedastic noise modeling for observation-specific uncertainty
5. Robust MCMC sampling for full Bayesian uncertainty quantification
"""

import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Callable

# Import components from other modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.adaptive_kernel import AdaptiveKernel, RateEstimator, MultiScalePeriodicKernel
from utils.proxy_calibration import (
    proxy_to_sst, sst_to_proxy, combine_proxy_data, 
    calculate_proxy_weights, HeteroscedasticNoiseModel,
    DEFAULT_CALIBRATION_PARAMS
)
from mcmc.sampler import MCMCSampler, HeteroscedasticMCMCSampler


# Configure GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdaptiveKernelGPModel(ExactGP):
    """
    GP model with adaptive kernel for capturing abrupt paleoclimate transitions.
    
    This model combines adaptive lengthscale kernels with multi-scale periodic
    components to accurately model climate dynamics across different timescales.
    """
    
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        likelihood: GaussianLikelihood,
        kernel_config: Dict
    ):
        """
        Initialize the adaptive kernel GP model.
        
        Args:
            x: Training input points (ages)
            y: Training output values (SST)
            likelihood: Gaussian likelihood
            kernel_config: Kernel configuration dictionary with parameters
        """
        super(AdaptiveKernelGPModel, self).__init__(x, y, likelihood)
        
        # Initialize mean function
        self.mean_module = ConstantMean()
        
        # Create adaptive kernel component
        self.adaptive_kernel = AdaptiveKernel(
            base_kernel_type=kernel_config.get('base_kernel_type', 'matern'),
            min_lengthscale=kernel_config.get('min_lengthscale', 2.0),
            max_lengthscale=kernel_config.get('max_lengthscale', 10.0),
            base_lengthscale=kernel_config.get('base_lengthscale', 5.0),
            adaptation_strength=kernel_config.get('adaptation_strength', 1.0),
            lengthscale_regularization=kernel_config.get('lengthscale_regularization', 0.1)
        )
        self.scaled_adaptive_kernel = ScaleKernel(self.adaptive_kernel)
        
        # Initialize multi-scale periodic kernel for Milankovitch cycles if requested
        if kernel_config.get('include_periodic', True):
            # Commenting out the MultiScalePeriodicKernel for now due to compatibility issues
            # Use a simpler alternative with a single periodic kernel
            self.periodic_kernel = gpytorch.kernels.PeriodicKernel(
                period_length=kernel_config.get('periods', [100.0, 41.0, 23.0])[0],  # Use first period
            )
            self.periodic_kernel = ScaleKernel(self.periodic_kernel)
            
            # Combine kernels
            self.covar_module = self.scaled_adaptive_kernel + self.periodic_kernel
        else:
            # Just use the adaptive kernel
            self.covar_module = self.scaled_adaptive_kernel
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass for GP prediction.
        
        Args:
            x: Input points
            
        Returns:
            MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class BayesianGPStateSpaceModel:
    """
    Enhanced Bayesian Gaussian Process State-Space Model for paleoclimate reconstruction.
    
    This model combines multiple innovations:
    1. Adaptive kernels that vary lengthscale based on climate rate of change
    2. Multi-scale periodic kernels for orbital cycles
    3. Balanced proxy weighting to prevent any proxy from dominating
    4. Heteroscedastic noise modeling for observation-specific uncertainty
    5. Full MCMC sampling for robust uncertainty quantification
    
    Attributes:
        proxy_types: List of proxy types used in the model
        weighting_method: Method used for weighting proxies
        kernel_config: Configuration for the GP kernel
        mcmc_config: Configuration for MCMC sampling
        calibration_params: Dictionary with calibration equations parameters
    """
    
    def __init__(
        self, 
        proxy_types: List[str],
        weighting_method: str = 'balanced',
        kernel_config: Optional[Dict] = None,
        mcmc_config: Optional[Dict] = None,
        calibration_params: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the Bayesian GP State-Space model.
        
        Args:
            proxy_types: List of proxies to be used (e.g. ['d18O', 'UK37'])
            weighting_method: Method for weighting proxies ('balanced', 'error', 'snr', 'equal')
            kernel_config: Configuration for the GP kernel
            mcmc_config: Configuration for MCMC sampling
            calibration_params: Dictionary with calibration parameters.
                              If None, default parameters will be used.
            random_state: Random seed for reproducibility
        """
        self.proxy_types = proxy_types
        self.weighting_method = weighting_method
        self.random_state = random_state
        
        # Default kernel configuration
        if kernel_config is None:
            self.kernel_config = {
                'base_kernel_type': 'matern',
                'min_lengthscale': 2.0,     # Minimum physically meaningful lengthscale
                'max_lengthscale': 10.0,    # Maximum lengthscale
                'base_lengthscale': 5.0,    # Base lengthscale
                'adaptation_strength': 1.0,  # How strongly to adapt to rate of change
                'lengthscale_regularization': 0.1,  # Regularization for lengthscale changes
                'include_periodic': True,    # Include periodic components
                'periods': [100.0, 41.0, 23.0],  # Milankovitch cycles (kyr)
                'outputscales': [2.0, 1.0, 0.5]  # Initial weights for periodic components
            }
        else:
            self.kernel_config = kernel_config
        
        # Default MCMC configuration
        if mcmc_config is None:
            self.mcmc_config = {
                'n_samples': 1000,
                'burn_in': 200,
                'thinning': 2,
                'step_size': 0.1,
                'target_acceptance': 0.6,
                'adaptation_steps': 100
            }
        else:
            self.mcmc_config = mcmc_config
            
        # Set calibration parameters
        self.calibration_params = calibration_params or DEFAULT_CALIBRATION_PARAMS
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # GP model components (initialized during fitting)
        self.model = None
        self.likelihood = None
        self.mll = None
        self.rate_estimator = None
        self.noise_model = None
        self.mcmc_sampler = None
        
        # State tracking
        self.is_fitted = False
        self.train_x = None
        self.train_y = None
        self.proxy_weights = None
        self.rate_points = None
        self.rate_values = None
        
        # Normalization info (if needed)
        self._normalized_y = False
        self._y_mean = 0.0
        self._y_std = 1.0
    
    def _preprocess_data(self, proxy_data_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess proxy data for model fitting.
        
        This function:
        1. Combines proxy data using the specified weighting method
        2. Handles normalization if needed
        3. Creates rate of change estimates for adaptive kernels
        
        Args:
            proxy_data_dict: Dictionary with proxy data
            
        Returns:
            train_x: Training input points (ages)
            train_y: Training output values (SST)
        """
        # Combine proxy data using the specified weighting method
        print(f"Combining proxy data using {self.weighting_method} weighting...")
        combined_ages, combined_sst, weights = combine_proxy_data(
            self.proxy_types,
            proxy_data_dict,
            weighting_method=self.weighting_method,
            calibration_params=self.calibration_params
        )
        
        # Store proxy weights
        self.proxy_weights = weights
        print(f"Proxy weights: {self.proxy_weights}")
        
        # Remove NaN values
        valid_mask = ~np.isnan(combined_sst)
        train_x = combined_ages[valid_mask]
        train_y = combined_sst[valid_mask]
        
        # Sort by age (important for rate estimation)
        sort_idx = np.argsort(train_x)
        train_x = train_x[sort_idx]
        train_y = train_y[sort_idx]
        
        if len(train_x) < 5:
            raise ValueError("Not enough valid data points after combining proxies")
        
        # Initialize rate estimator for adaptive kernel
        self.rate_estimator = RateEstimator(
            smoothing_method='gaussian', 
            gaussian_sigma=3.0,
            use_central_diff=True,
            normalize_method='robust'
        )
        
        # Estimate rate of change for adaptive kernel
        self.rate_points, self.rate_values = self.rate_estimator.estimate_rate(train_x, train_y)
        
        # Initialize heteroscedastic noise model
        self.noise_model = HeteroscedasticNoiseModel(
            proxy_types=self.proxy_types,
            calibration_params=self.calibration_params,
            base_noise_level=0.5,
            transition_scaling=2.0,
            age_dependent_scaling=True
        )
        
        # Normalize data if multiple proxy types (for numerical stability)
        if len(self.proxy_types) > 1:
            print("Normalizing target data for numerical stability...")
            y_mean = np.mean(train_y)
            y_std = np.std(train_y)
            
            if y_std > 0:
                train_y = (train_y - y_mean) / y_std
                
            # Set flag to denormalize predictions later
            self._normalized_y = True
            self._y_mean = y_mean
            self._y_std = y_std
        else:
            self._normalized_y = False
        
        return train_x, train_y
    
    def _init_model(self, train_x: np.ndarray, train_y: np.ndarray) -> Tuple[ExactGP, GaussianLikelihood]:
        """
        Initialize the GP model with training data.
        
        Args:
            train_x: Training input points
            train_y: Training output values
            
        Returns:
            model: GP model
            likelihood: Gaussian likelihood
        """
        # Initialize likelihood
        likelihood = GaussianLikelihood().to(device)
        
        # Convert numpy arrays to torch tensors
        x_tensor = torch.tensor(train_x, dtype=torch.float32).reshape(-1, 1).to(device)
        y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
        
        # Initialize model
        model = AdaptiveKernelGPModel(
            x_tensor, y_tensor, likelihood, self.kernel_config
        ).to(device)
        
        # Set adaptive kernel rate of change
        model.adaptive_kernel.update_rate_of_change(self.rate_points, self.rate_values)
        
        return model, likelihood
    
    def fit(self, proxy_data_dict: Dict, training_iterations: int = 1000, run_mcmc: bool = True):
        """
        Fit the Bayesian GP State-Space model to the proxy data.
        
        Args:
            proxy_data_dict: Dictionary with proxy data. Each key is a proxy type,
                           and each value is a dict with 'age' and 'value' arrays.
            training_iterations: Number of iterations for optimizer
            run_mcmc: Whether to run MCMC sampling after optimization
            
        Returns:
            self: The fitted model
        """
        # Preprocess data
        train_x, train_y = self._preprocess_data(proxy_data_dict)
        
        # Store training data
        self.train_x = train_x
        self.train_y = train_y
        
        # Initialize model
        self.model, self.likelihood = self._init_model(train_x, train_y)
        
        print("Model initialized. Training...")
        # Set model to training mode
        self.model.train()
        self.likelihood.train()
        
        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}
        ], lr=0.05)
        
        # "Loss" for GP is the negative log marginal likelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Initialize loss history
        losses = []
        
        # Training loop
        with gpytorch.settings.cholesky_jitter(1e-4):
            # Use try-except for numerical stability
            try:
                for i in range(training_iterations):
                    optimizer.zero_grad()
                    output = self.model(self.model.train_inputs[0])
                    
                    # Catch and handle numerical issues in loss calculation
                    try:
                        loss = -self.mll(output, self.model.train_targets)
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    except RuntimeError as e:
                        if "singular" in str(e).lower() or "cholesky" in str(e).lower() or "positive definite" in str(e).lower():
                            print(f"Numerical issue at iteration {i+1}: {e}")
                            print("Adjusting optimization parameters and continuing...")
                            # Add more jitter to the model
                            with torch.no_grad():
                                if hasattr(self.likelihood, 'noise'):
                                    self.likelihood.noise = self.likelihood.noise * 1.05
                        else:
                            raise e
                    
                    if (i+1) % 100 == 0:
                        print(f'Iteration {i+1}/{training_iterations} - Loss: {losses[-1] if losses else "N/A"}')
            
            except Exception as e:
                print(f"Training halted early due to: {str(e)}")
                print(f"Completed {len(losses)} iterations before error")
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Run MCMC sampling if requested
        if run_mcmc:
            print("Running MCMC sampling for uncertainty quantification...")
            self._run_mcmc_sampling()
        
        self.is_fitted = True
        return self
    
    def _run_mcmc_sampling(self):
        """Run MCMC sampling for uncertainty quantification."""
        # Determine if we should use heteroscedastic MCMC
        use_heteroscedastic = hasattr(self, 'noise_model') and self.noise_model is not None
        
        if use_heteroscedastic:
            # Calculate observation-specific noise levels
            proxy_types = [self.proxy_types[0]] * len(self.train_x)  # Simplified for now
            noise_levels = self.noise_model.get_noise_level(
                self.train_x, 
                proxy_types,
                rate_of_change=self.rate_values
            )
            
            # Convert to tensor
            noise_tensor = torch.tensor(noise_levels, dtype=torch.float32).to(device)
            
            # Create heteroscedastic sampler
            self.mcmc_sampler = HeteroscedasticMCMCSampler(
                model=self.model,
                likelihood=self.likelihood,
                noise_levels=noise_tensor,
                n_samples=self.mcmc_config.get('n_samples', 1000),
                burn_in=self.mcmc_config.get('burn_in', 200),
                thinning=self.mcmc_config.get('thinning', 2),
                step_size=self.mcmc_config.get('step_size', 0.1),
                target_acceptance=self.mcmc_config.get('target_acceptance', 0.6),
                adaptation_steps=self.mcmc_config.get('adaptation_steps', 100),
                random_state=self.random_state
            )
        else:
            # Create standard sampler
            self.mcmc_sampler = MCMCSampler(
                model=self.model,
                likelihood=self.likelihood,
                n_samples=self.mcmc_config.get('n_samples', 1000),
                burn_in=self.mcmc_config.get('burn_in', 200),
                thinning=self.mcmc_config.get('thinning', 2),
                step_size=self.mcmc_config.get('step_size', 0.1),
                target_acceptance=self.mcmc_config.get('target_acceptance', 0.6),
                adaptation_steps=self.mcmc_config.get('adaptation_steps', 100),
                random_state=self.random_state
            )
        
        # Run HMC sampling
        self.mcmc_sampler.run_hmc(progress_bar=True)
    
    def predict(self, test_x: np.ndarray, return_samples: bool = False, n_samples: int = 100):
        """
        Make predictions at test points.
        
        Args:
            test_x: Ages at which to predict SST
            return_samples: If True, return posterior predictive samples
            n_samples: Number of samples to return if return_samples is True
            
        Returns:
            mean, lower_ci, upper_ci, (optional: samples)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        test_x = np.asarray(test_x).flatten()
        
        # If we have MCMC samples, use them for prediction
        if hasattr(self, 'mcmc_sampler') and self.mcmc_sampler is not None:
            # Get prediction samples from MCMC
            pred_dict = self.mcmc_sampler.get_samples(
                test_x, 
                return_numpy=True,
                num_pred_samples=n_samples
            )
            
            # Extract statistics
            mean = pred_dict['mean']
            lower_ci = pred_dict['lower_ci']
            upper_ci = pred_dict['upper_ci']
            samples = pred_dict['samples']
        else:
            # Use standard GP prediction
            # Convert to tensor
            x_tensor = torch.tensor(test_x, dtype=torch.float32).reshape(-1, 1).to(device)
            
            # Make predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                # Get distribution
                posterior = self.likelihood(self.model(x_tensor))
                
                # Get mean and variance
                mean = posterior.mean.cpu().numpy()
                variance = posterior.variance.cpu().numpy()
                
                # Calculate 95% credible intervals
                lower_ci = mean - 1.96 * np.sqrt(variance)
                upper_ci = mean + 1.96 * np.sqrt(variance)
                
                if return_samples:
                    # Draw samples from the posterior predictive distribution
                    samples = posterior.sample(sample_shape=torch.Size([n_samples])).cpu().numpy()
                
        # Denormalize if needed
        if hasattr(self, '_normalized_y') and self._normalized_y:
            mean = mean * self._y_std + self._y_mean
            lower_ci = lower_ci * self._y_std + self._y_mean
            upper_ci = upper_ci * self._y_std + self._y_mean
            
            if return_samples:
                samples = samples * self._y_std + self._y_mean
        
        if return_samples:
            return mean, lower_ci, upper_ci, samples
        else:
            return mean, lower_ci, upper_ci
    
    def detect_abrupt_transitions(self, test_x: np.ndarray, threshold_percentile: int = 95, min_separation: int = 5):
        """
        Detect abrupt transitions in the reconstructed SST.
        
        This method uses the rate of change and the adaptive kernel lengthscales
        to identify regions of rapid climate change.
        
        Args:
            test_x: Ages at which to evaluate transitions
            threshold_percentile: Percentile to use for threshold (default: 95)
            min_separation: Minimum separation between detected transitions
            
        Returns:
            List of ages where abrupt transitions occur
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before detecting transitions")
        
        # Get model predictions
        mean, _, _, samples = self.predict(test_x, return_samples=True, n_samples=100)
        
        # Estimate rate of change from the mean prediction
        if hasattr(self.model, 'adaptive_kernel'):
            # Use the rate estimator from model fitting
            rate_points, rate_values = self.rate_estimator.estimate_rate(test_x, mean)
            
            # Use adaptive model's lengthscales as additional information
            x_tensor = torch.tensor(test_x, dtype=torch.float32).reshape(-1, 1).to(device)
            with torch.no_grad():
                lengthscales = self.model.adaptive_kernel._get_lengthscale(x_tensor).cpu().numpy()
                
            # Combine rate and inverse lengthscale information (both indicate transitions)
            # Normalize lengthscales to 0-1 range inverted (smaller = higher rate)
            norm_lengthscales = 1.0 - (lengthscales - np.min(lengthscales)) / (np.max(lengthscales) - np.min(lengthscales) + 1e-8)
            
            # Combine signals (mean of both indicators)
            combined_indicator = 0.5 * rate_values + 0.5 * norm_lengthscales
        else:
            # Fall back to calculating rate of change directly
            dx = np.diff(test_x)
            dy = np.diff(mean)
            rate_of_change = dy / dx
            
            # Calculate uncertainty in rate of change from samples
            sample_rates = np.diff(samples, axis=1) / dx
            rate_std = np.std(sample_rates, axis=0)
            
            # Normalize rate of change by its uncertainty
            normalized_rate = np.abs(rate_of_change) / (rate_std + 1e-8)
            
            rate_points = test_x[:-1] + dx/2  # Midpoints
            combined_indicator = np.abs(normalized_rate)
        
        # Find peaks above threshold
        threshold = np.percentile(combined_indicator, threshold_percentile)
        peak_indices = np.where(combined_indicator > threshold)[0]
        
        # Group peaks by proximity
        if len(peak_indices) == 0:
            return []
        
        grouped_peaks = [[peak_indices[0]]]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] > min_separation:
                grouped_peaks.append([peak_indices[i]])
            else:
                grouped_peaks[-1].append(peak_indices[i])
        
        # Find maximum peak in each group
        transition_indices = []
        for group in grouped_peaks:
            max_idx = group[np.argmax(combined_indicator[group])]
            transition_indices.append(max_idx)
        
        # Convert indices to ages
        transition_ages = [rate_points[i] for i in transition_indices]
        
        return transition_ages
    
    def evaluate(self, test_x: np.ndarray, true_sst: np.ndarray):
        """
        Evaluate model performance against true SST values.
        
        Args:
            test_x: Test ages
            true_sst: True SST values
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Make predictions
        mean, lower_ci, upper_ci = self.predict(test_x)
        
        # Find common indices where both predicted and true values exist
        valid_indices = ~np.isnan(true_sst) & ~np.isnan(mean)
        
        if np.sum(valid_indices) < 2:
            print("Warning: Not enough valid points for evaluation")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'coverage': np.nan,
                'ci_width': np.nan
            }
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_sst[valid_indices], mean[valid_indices]))
        mae = mean_absolute_error(true_sst[valid_indices], mean[valid_indices])
        r2 = r2_score(true_sst[valid_indices], mean[valid_indices])
        
        # Coverage: proportion of true values within the 95% CI
        coverage = np.mean((lower_ci[valid_indices] <= true_sst[valid_indices]) & 
                          (true_sst[valid_indices] <= upper_ci[valid_indices]))
        
        # Average width of confidence interval
        ci_width = np.mean(upper_ci[valid_indices] - lower_ci[valid_indices])
        
        # Calculate RMSE in top 10% rate of change regions (transitions)
        if hasattr(self, 'rate_estimator') and self.rate_values is not None:
            # Interpolate rate values to test points
            interp_rates = np.interp(
                test_x, 
                self.rate_points, 
                self.rate_values,
                left=self.rate_values[0],
                right=self.rate_values[-1]
            )
            
            # Define transition regions as top 10% of rate values
            threshold = np.percentile(interp_rates[valid_indices], 90)
            transition_mask = (interp_rates > threshold) & valid_indices
            
            if np.sum(transition_mask) > 1:
                transition_rmse = np.sqrt(mean_squared_error(
                    true_sst[transition_mask], 
                    mean[transition_mask]
                ))
            else:
                transition_rmse = np.nan
        else:
            transition_rmse = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coverage': coverage,
            'ci_width': ci_width,
            'transition_rmse': transition_rmse
        }
    
    def plot_reconstruction(self, test_x: np.ndarray, proxy_data_dict: Optional[Dict] = None, 
                          true_sst: Optional[np.ndarray] = None, 
                          detected_transitions: Optional[List] = None, 
                          figure_path: Optional[str] = None):
        """
        Plot the reconstructed SST with uncertainty intervals.
        
        Args:
            test_x: Ages at which to plot reconstruction
            proxy_data_dict: Dictionary with proxy data to plot
            true_sst: True SST values if available
            detected_transitions: List of ages where transitions are detected
            figure_path: Path to save the figure
            
        Returns:
            matplotlib figure
        """
        # Make predictions
        mean, lower_ci, upper_ci = self.predict(test_x)
        
        # Get adaptive lengthscales if available
        if hasattr(self.model, 'adaptive_kernel'):
            x_tensor = torch.tensor(test_x, dtype=torch.float32).reshape(-1, 1).to(device)
            with torch.no_grad():
                lengthscales = self.model.adaptive_kernel._get_lengthscale(x_tensor).cpu().numpy()
                
            # Interpolate rate values to test points
            interp_rates = np.interp(
                test_x,
                self.rate_points,
                self.rate_values,
                left=self.rate_values[0],
                right=self.rate_values[-1]
            )
            
            show_adaptation = True
        else:
            show_adaptation = False
            
        # Create figure
        if show_adaptation:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot the reconstruction with uncertainty
        ax1.plot(test_x, mean, 'b-', linewidth=2, label='GP Reconstruction')
        ax1.fill_between(test_x, lower_ci, upper_ci, color='b', alpha=0.2, label='95% CI')
        
        # Plot proxies if provided
        if proxy_data_dict is not None:
            markers = ['o', 's', '^', 'D']
            colors = ['green', 'orange', 'red', 'purple']
            
            for i, proxy_type in enumerate(self.proxy_types):
                if proxy_type in proxy_data_dict:
                    proxy_data = proxy_data_dict[proxy_type]
                    proxy_ages = proxy_data['age']
                    proxy_values = proxy_data['value']
                    
                    # Convert proxy to SST
                    proxy_sst = proxy_to_sst(proxy_values, proxy_type, self.calibration_params)
                    
                    ax1.scatter(proxy_ages, proxy_sst, 
                              marker=markers[i % len(markers)],
                              color=colors[i % len(colors)], s=30, alpha=0.7,
                              label=f'{proxy_type} derived SST')
        
        # Plot true SST if provided
        if true_sst is not None:
            ax1.plot(test_x, true_sst, 'k--', linewidth=1.5, label='True SST')
        
        # Mark detected transitions if provided
        if detected_transitions is not None and len(detected_transitions) > 0:
            y_range = ax1.get_ylim()
            for trans_age in detected_transitions:
                ax1.axvline(x=trans_age, color='r', linestyle='--', alpha=0.7)
                ax1.text(trans_age, y_range[0] + 0.95*(y_range[1]-y_range[0]), 
                       f'{trans_age:.1f}', 
                       color='r', rotation=90, ha='right')
        
        # Add labels and legend
        ax1.set_xlabel('Age (kyr)')
        ax1.set_ylabel('Sea Surface Temperature (°C)')
        ax1.set_title('Bayesian GP State-Space SST Reconstruction')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis direction to be reversed (older ages on the right)
        ax1.set_xlim(max(test_x), min(test_x))
        
        # Plot adaptation information if available
        if show_adaptation:
            # Plot lengthscale and rate in second subplot
            ax2.plot(test_x, lengthscales, 'b-', linewidth=2, label='Adaptive Lengthscale')
            ax2.set_ylabel('Lengthscale (kyr)', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.set_xlabel('Age (kyr)')
            
            # Add rate of change on secondary y-axis
            ax3 = ax2.twinx()
            ax3.plot(test_x, interp_rates, 'r-', linewidth=1.5, label='Rate of Change')
            ax3.set_ylabel('Normalized Rate of Change', color='r')
            ax3.tick_params(axis='y', labelcolor='r')
            
            # Add legend for both curves
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Mark detected transitions if provided
            if detected_transitions is not None and len(detected_transitions) > 0:
                for trans_age in detected_transitions:
                    ax2.axvline(x=trans_age, color='r', linestyle='--', alpha=0.5)
            
            # Reverse x-axis to match top plot
            ax2.set_xlim(max(test_x), min(test_x))
        
        plt.tight_layout()
        
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {figure_path}")
            
        return fig
    
    def plot_parameter_posterior(self, figure_path: Optional[str] = None):
        """
        Plot posterior distributions of model parameters from MCMC samples.
        
        Args:
            figure_path: Path to save the figure
            
        Returns:
            matplotlib figure
        """
        if not hasattr(self, 'mcmc_sampler') or self.mcmc_sampler is None:
            raise RuntimeError("MCMC sampling must be performed first")
        
        # Get relevant parameter names from MCMC sampler
        param_names = self.mcmc_sampler.param_names
        
        # Organize parameters by category
        kernel_params = [p for p in param_names if 'kernel' in p or 'lengthscale' in p]
        mean_params = [p for p in param_names if 'mean' in p]
        noise_params = [p for p in param_names if 'noise' in p or 'likelihood' in p]
        
        # Calculate number of parameters to plot
        n_params = len(kernel_params) + len(mean_params) + len(noise_params)
        
        # Create figure grid based on number of parameters
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
        
        # Flatten axes for easier indexing
        if n_params > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot posteriors for each parameter category
        param_idx = 0
        
        # Plot mean parameters
        for name in mean_params:
            if param_idx < len(axes):
                self.mcmc_sampler.plot_samples(name, figsize=(5, 10))
                plt.savefig("temp.png")  # Save to temp file to close figure
                plt.close()
                
                # Plot histogram in our grid
                samples = self.mcmc_sampler.samples[name].flatten()
                axes[param_idx].hist(samples, bins=30, density=True)
                axes[param_idx].axvline(np.mean(samples), color='r', linestyle='--', 
                                      label=f'Mean: {np.mean(samples):.4f}')
                axes[param_idx].axvline(np.median(samples), color='g', linestyle=':', 
                                      label=f'Median: {np.median(samples):.4f}')
                axes[param_idx].set_title(f'Posterior: {name}')
                axes[param_idx].legend()
                param_idx += 1
        
        # Plot kernel parameters
        for name in kernel_params:
            if param_idx < len(axes):
                samples = self.mcmc_sampler.samples[name].flatten()
                axes[param_idx].hist(samples, bins=30, density=True)
                axes[param_idx].axvline(np.mean(samples), color='r', linestyle='--', 
                                      label=f'Mean: {np.mean(samples):.4f}')
                axes[param_idx].axvline(np.median(samples), color='g', linestyle=':', 
                                      label=f'Median: {np.median(samples):.4f}')
                axes[param_idx].set_title(f'Posterior: {name}')
                axes[param_idx].legend()
                param_idx += 1
        
        # Plot noise parameters
        for name in noise_params:
            if param_idx < len(axes):
                samples = self.mcmc_sampler.samples[name].flatten()
                axes[param_idx].hist(samples, bins=30, density=True)
                axes[param_idx].axvline(np.mean(samples), color='r', linestyle='--', 
                                      label=f'Mean: {np.mean(samples):.4f}')
                axes[param_idx].axvline(np.median(samples), color='g', linestyle=':', 
                                      label=f'Median: {np.median(samples):.4f}')
                axes[param_idx].set_title(f'Posterior: {name}')
                axes[param_idx].legend()
                param_idx += 1
        
        # Hide unused subplots
        for i in range(param_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {figure_path}")
            
        # Clean up temp file if created
        if os.path.exists("temp.png"):
            os.remove("temp.png")
            
        return fig


def generate_synthetic_multiproxy_data(
    n_points: int = 200,
    age_min: float = 0,
    age_max: float = 500,
    proxy_types: List[str] = ['d18O', 'UK37', 'Mg_Ca'],
    n_transitions: int = 2,
    transition_magnitude: float = 3.0,
    include_orbital_cycles: bool = True,
    smoothness: float = 1.0,
    proxy_noise_scale: float = 1.0,
    random_state: int = 42
) -> Dict:
    """
    Generate synthetic multi-proxy paleoclimate data with realistic features.
    
    Args:
        n_points: Base number of data points (per proxy)
        age_min, age_max: Age range in kyr
        proxy_types: List of proxy types to generate
        n_transitions: Number of abrupt transitions to include
        transition_magnitude: Magnitude of transitions in °C
        include_orbital_cycles: Whether to include Milankovitch cycles
        smoothness: Controls overall smoothness (higher = smoother)
        proxy_noise_scale: Scale factor for proxy noise (higher = noisier)
        random_state: Random seed
        
    Returns:
        Dictionary with synthetic data
    """
    np.random.seed(random_state)
    
    # Generate regular time grid for true temperature
    ages_full = np.linspace(age_min, age_max, 1000)
    
    # Initialize with long-term cooling trend
    true_sst = 15.0 - 0.01 * ages_full
    
    # Add Milankovitch cycles if requested
    if include_orbital_cycles:
        # Eccentricity (100 kyr)
        true_sst += 2.0 * np.sin(2 * np.pi * ages_full / 100)
        
        # Obliquity (41 kyr)
        true_sst += 1.0 * np.sin(2 * np.pi * ages_full / 41 + 0.2)
        
        # Precession (23 kyr)
        true_sst += 0.5 * np.sin(2 * np.pi * ages_full / 23 + 0.8)
    
    # Add abrupt transitions at random locations
    transition_ages = np.random.uniform(age_min + 50, age_max - 50, n_transitions)
    transition_ages.sort()
    
    for age in transition_ages:
        # Random direction (warming or cooling)
        direction = np.random.choice([-1, 1])
        magnitude = direction * transition_magnitude
        
        # Width of transition (smaller = more abrupt)
        width = np.random.uniform(3, 10) / smoothness
        
        # Add sigmoid-shaped transition
        true_sst += magnitude / (1 + np.exp((ages_full - age) / (width / 5)))
    
    # Add some background variability
    variability_scale = 0.5 / smoothness
    true_sst += np.random.normal(0, variability_scale, size=len(ages_full))
    
    # Apply slight smoothing for physical realism
    from scipy.ndimage import gaussian_filter1d
    true_sst = gaussian_filter1d(true_sst, sigma=2.0)
    
    # Generate proxy data for each proxy type
    proxy_data = {}
    calibration_params = DEFAULT_CALIBRATION_PARAMS
    
    for proxy_type in proxy_types:
        # Get calibration parameters
        params = calibration_params[proxy_type]
        
        # Adjust number of samples for this proxy
        if proxy_type == 'd18O':
            proxy_n_points = int(n_points * 1.2)  # More d18O samples
        elif proxy_type == 'UK37':
            proxy_n_points = int(n_points * 0.8)  # Fewer UK37 samples
        else:
            proxy_n_points = n_points
        
        # Generate random age points for this proxy
        proxy_ages = np.sort(np.random.uniform(age_min, age_max, proxy_n_points))
        
        # Interpolate true SST to proxy ages
        proxy_true_sst = np.interp(proxy_ages, ages_full, true_sst)
        
        # Convert SST to proxy units using calibration equation
        proxy_values = sst_to_proxy(proxy_true_sst, proxy_type, calibration_params)
        
        # Add proxy-specific noise
        noise_scale = params['error_std'] * proxy_noise_scale
        
        # Add heteroscedastic noise (higher in transition regions)
        # Find closest transition for each point
        point_noise = np.ones_like(proxy_ages) * noise_scale
        for trans_age in transition_ages:
            # Calculate distance to transition (normalized)
            dist = np.abs(proxy_ages - trans_age) / 20  # Scale distance
            
            # Increase noise near transitions
            noise_factor = 1.0 + 1.0 * np.exp(-dist**2)  # Gaussian bump
            point_noise = np.maximum(point_noise, noise_scale * noise_factor)
        
        # Add noise
        proxy_values += np.random.normal(0, point_noise)
        
        # Store proxy data
        proxy_data[proxy_type] = {
            'age': proxy_ages,
            'value': proxy_values,
            'true_sst': proxy_true_sst
        }
    
    return {
        'ages_full': ages_full,
        'true_sst': true_sst,
        'proxy_data': proxy_data,
        'calibration_params': calibration_params,
        'transition_ages': transition_ages
    }


def demo_model():
    """
    Run a demonstration of the enhanced Bayesian GP State-Space model.
    """
    print("Generating synthetic multi-proxy data...")
    data = generate_synthetic_multiproxy_data(
        n_points=80,
        age_min=0,
        age_max=500,
        proxy_types=['d18O', 'UK37', 'Mg_Ca'],
        n_transitions=3,
        transition_magnitude=3.0,
        include_orbital_cycles=True,
        smoothness=1.0,
        proxy_noise_scale=1.0,
        random_state=42
    )
    
    # Create output directory
    output_dir = "data/results/enhanced_bayesian_gp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with enhanced features
    kernel_config = {
        'base_kernel_type': 'matern',
        'min_lengthscale': 2.0,
        'max_lengthscale': 10.0,
        'base_lengthscale': 5.0,
        'adaptation_strength': 1.5,
        'lengthscale_regularization': 0.1,
        'include_periodic': True,
        'periods': [100.0, 41.0, 23.0],
        'outputscales': [2.0, 1.0, 0.5]
    }
    
    mcmc_config = {
        'n_samples': 500,  # Reduced for demo
        'burn_in': 100,
        'thinning': 2,
        'step_size': 0.05,
        'target_acceptance': 0.6,
        'adaptation_steps': 50
    }
    
    print("Initializing model...")
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37', 'Mg_Ca'],
        weighting_method='balanced',
        kernel_config=kernel_config,
        mcmc_config=mcmc_config,
        calibration_params=data['calibration_params'],
        random_state=42
    )
    
    # Fit model
    print("Fitting model...")
    model.fit(
        data['proxy_data'],
        training_iterations=300,
        run_mcmc=True
    )
    
    # Evaluate on test points
    print("Evaluating model...")
    test_ages = np.linspace(0, 500, 500)
    metrics = model.evaluate(test_ages, data['true_sst'])
    
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Detect transitions
    print("Detecting transitions...")
    transitions = model.detect_abrupt_transitions(test_ages)
    print(f"Detected transitions at ages: {transitions}")
    print(f"True transitions: {data['transition_ages']}")
    
    # Plot reconstruction
    print("Generating visualizations...")
    fig = model.plot_reconstruction(
        test_ages,
        proxy_data_dict=data['proxy_data'],
        true_sst=data['true_sst'],
        detected_transitions=transitions,
        figure_path=f"{output_dir}/reconstruction.png"
    )
    
    # Plot parameter posteriors
    if hasattr(model, 'mcmc_sampler') and model.mcmc_sampler is not None:
        fig = model.plot_parameter_posterior(
            figure_path=f"{output_dir}/parameter_posteriors.png"
        )
    
    print(f"All results saved to {output_dir}")
    return model, data


if __name__ == "__main__":
    demo_model()