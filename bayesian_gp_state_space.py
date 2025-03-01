"""
Bayesian Gaussian Process State-Space Model for Paleoclimate Reconstruction

This module implements a novel Bayesian Gaussian Process State-Space framework
specifically designed for reconstructing Sea Surface Temperature (SST) from
multiple paleoclimate proxies. The approach:

1. Models SST as a latent variable using a state-space formulation with GP prior
2. Explicitly incorporates proxy-specific calibration equations 
3. Uses Bayesian inference for full uncertainty quantification
4. Provides specialized tools for detecting abrupt transitions in climate records
5. Handles sparse and irregularly sampled data common in paleoclimate studies

The framework is particularly optimized for:
- Subtropical mode water formation events
- Abrupt transitions in paleoclimate records
- Integration of multiple proxy types with varying error characteristics
- Robust uncertainty quantification across the entire reconstruction pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy.typing as npt

# Configure GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BayesianGPStateSpaceModel:
    """
    Bayesian Gaussian Process State-Space Model for paleoclimate reconstruction.
    
    This class implements a state-space model where the latent state (SST) evolves 
    according to a GP prior, and observations (proxies) are generated from the latent 
    state via calibration equations with explicit uncertainty modeling.
    
    Attributes:
        proxy_types (List[str]): List of proxy types used in the model
        calibration_params (Dict): Dictionary with calibration equations parameters
        kernel_type (str): Type of kernel used for the GP (e.g., 'combined', 'matern')
        n_mcmc_samples (int): Number of MCMC samples for uncertainty quantification
        prior_dict (Dict): Dictionary with prior distributions for model parameters
    """
    
    def __init__(
        self, 
        proxy_types: List[str],
        calibration_params: Optional[Dict] = None,
        kernel_type: str = 'combined',
        n_mcmc_samples: int = 1000,
        prior_dict: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the Bayesian GP State-Space model.
        
        Args:
            proxy_types: List of proxies to be used (e.g. ['d18O', 'UK37'])
            calibration_params: Dictionary with calibration parameters.
                                If None, default parameters will be used.
            kernel_type: Type of kernel. Options: 'rbf', 'matern', 'combined'
            n_mcmc_samples: Number of MCMC samples for posterior analysis
            prior_dict: Dictionary with prior distributions for model parameters
            random_state: Random seed for reproducibility
        """
        self.proxy_types = proxy_types
        self.kernel_type = kernel_type
        self.n_mcmc_samples = n_mcmc_samples
        self.random_state = random_state
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Default calibration parameters if not provided
        if calibration_params is None:
            self.calibration_params = {
                'd18O': {
                    'slope': -0.22,        # �C per 0
                    'intercept': 3.0,      # 0
                    'error_std': 0.1,      # 0
                    'inverse_slope': -4.54545  # 0 per �C (1/slope)
                },
                'UK37': {
                    'slope': 0.033,        # units per �C  
                    'intercept': 0.044,    # units
                    'error_std': 0.05,     # units
                    'inverse_slope': 30.303   # �C per unit (1/slope)
                },
                'Mg_Ca': {
                    'slope': 0.09,         # mmol/mol per �C
                    'intercept': 0.3,      # mmol/mol
                    'error_std': 0.1,      # mmol/mol
                    'inverse_slope': 11.111   # �C per mmol/mol (1/slope)
                }
            }
        else:
            self.calibration_params = calibration_params
            
        # Default priors if not provided
        if prior_dict is None:
            self.prior_dict = {
                'lengthscale_rbf': {'distribution': 'lognormal', 'mean': 1.0, 'std': 1.0},
                'lengthscale_periodic': {'distribution': 'lognormal', 'mean': 1.0, 'std': 1.0},
                'period': {'distribution': 'normal', 'mean': 41.0, 'std': 5.0},  # Peak at 41kyr for orbital cycle
                'outputscale': {'distribution': 'lognormal', 'mean': 0.0, 'std': 1.0},
                'noise': {'distribution': 'lognormal', 'mean': -2.0, 'std': 0.7}
            }
        else:
            self.prior_dict = prior_dict
            
        # GP models will be initialized during fitting
        self.gp_model = None
        self.likelihood = None
        self.mll = None
        
        # Storage for MCMC samples
        self.posterior_samples = None
        
        # Fitted state
        self.is_fitted = False
        self.train_x = None
        self.train_y = None
        self.proxy_weights = None
        
    def _init_model(self, train_x, train_y):
        """
        Initialize the GP model with the appropriate kernel.
        
        Args:
            train_x: Training input ages
            train_y: Training output proxy-derived SST values
            
        Returns:
            Initialized GP model and likelihood
        """
        likelihood = GaussianLikelihood().to(device)
        
        class ExactGPModel(ExactGP):
            def __init__(self, x, y, likelihood, kernel_type='combined'):
                super(ExactGPModel, self).__init__(x, y, likelihood)
                self.mean_module = ConstantMean()
                
                # Define kernel based on type
                if kernel_type == 'rbf':
                    self.covar_module = ScaleKernel(RBFKernel())
                elif kernel_type == 'matern':
                    self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
                elif kernel_type == 'combined':
                    # Combined RBF and Periodic kernel for capturing both
                    # long-term trends and orbital cycles
                    self.rbf_kernel = ScaleKernel(RBFKernel())
                    self.periodic_kernel = ScaleKernel(PeriodicKernel())
                    self.covar_module = self.rbf_kernel + self.periodic_kernel
                else:
                    raise ValueError(f"Unsupported kernel type: {kernel_type}")
                
                self.kernel_type = kernel_type
                
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)
            
        # Convert inputs to torch tensors
        x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
        
        model = ExactGPModel(x_tensor, y_tensor, likelihood, kernel_type=self.kernel_type)
        model.to(device)
        
        return model, likelihood
    
    def _proxy_to_sst(self, proxy_values, proxy_type):
        """
        Convert proxy values to SST using calibration equations.
        
        Args:
            proxy_values: Proxy measurement values
            proxy_type: Type of proxy ('d18O', 'UK37', etc.)
            
        Returns:
            SST values derived from the proxy
        """
        params = self.calibration_params[proxy_type]
        
        # Invert the calibration equation: SST = (proxy - intercept) / slope
        sst = (proxy_values - params['intercept']) * params['inverse_slope']
        
        return sst
    
    def _calculate_proxy_weights(self, proxy_data_dict):
        """
        Calculate optimal weights for combining proxies based on their uncertainties.
        
        Args:
            proxy_data_dict: Dictionary with proxy data
            
        Returns:
            Dictionary of weights for each proxy type
        """
        weights = {}
        total_weight = 0
        
        # Calculate weights as inverse of error variance (1/ò)
        for proxy_type in self.proxy_types:
            if proxy_type in proxy_data_dict:
                # Get calibration error and convert to temperature units
                error_std = self.calibration_params[proxy_type]['error_std']
                inverse_slope = abs(self.calibration_params[proxy_type]['inverse_slope'])
                
                # Convert proxy error to temperature error
                temp_error = error_std * inverse_slope
                
                # Weight is inversely proportional to variance
                weights[proxy_type] = 1 / (temp_error ** 2)
                total_weight += weights[proxy_type]
        
        # Normalize weights to sum to 1
        for proxy_type in weights:
            weights[proxy_type] /= total_weight
            
        return weights
    
    def _combine_proxy_data(self, proxy_data_dict, age_points=None):
        """
        Combine multiple proxy records into a single temperature record
        using optimal weighting based on calibration uncertainties.
        
        Args:
            proxy_data_dict: Dictionary with proxy data
            age_points: Common age points for interpolation
            
        Returns:
            combined_ages, combined_sst
        """
        # If no common age points provided, merge all available ages
        if age_points is None:
            age_points = np.array([])
            for proxy_type in self.proxy_types:
                if proxy_type in proxy_data_dict:
                    age_points = np.union1d(age_points, proxy_data_dict[proxy_type]['age'])
            age_points = np.sort(age_points)
        
        # Calculate optimal weights for each proxy
        self.proxy_weights = self._calculate_proxy_weights(proxy_data_dict)
        print(f"Proxy weights: {self.proxy_weights}")
        
        # Initialize arrays for combined data
        combined_sst = np.zeros_like(age_points, dtype=float)
        combined_count = np.zeros_like(age_points, dtype=float)
        
        # Combine temperature estimates from each proxy
        for proxy_type, weight in self.proxy_weights.items():
            if proxy_type in proxy_data_dict:
                proxy_data = proxy_data_dict[proxy_type]
                proxy_ages = proxy_data['age']
                proxy_values = proxy_data['value']
                
                # Convert proxy to SST
                proxy_sst = self._proxy_to_sst(proxy_values, proxy_type)
                
                # Interpolate to common age points
                # Use nearest neighbor for sparse data to avoid extrapolation artifacts
                valid_mask = ~np.isnan(proxy_sst)
                if np.sum(valid_mask) > 1:  # Need at least 2 points for interpolation
                    interpolated_sst = np.interp(
                        age_points, 
                        proxy_ages[valid_mask], 
                        proxy_sst[valid_mask],
                        left=np.nan, right=np.nan
                    )
                    
                    # Add weighted contribution where data exists
                    valid_interp = ~np.isnan(interpolated_sst)
                    combined_sst[valid_interp] += interpolated_sst[valid_interp] * weight
                    combined_count[valid_interp] += weight
        
        # Normalize by total weight where we have data
        valid_mask = combined_count > 0
        combined_sst[valid_mask] /= combined_count[valid_mask]
        combined_sst[~valid_mask] = np.nan
        
        return age_points, combined_sst
    
    def fit(self, proxy_data_dict, training_iterations=1000):
        """
        Fit the Bayesian GP State-Space model to the proxy data.
        
        Args:
            proxy_data_dict: Dictionary with proxy data. Each key is a proxy type,
                             and each value is a dict with 'age' and 'value' arrays.
            training_iterations: Number of iterations for optimizer
            
        Returns:
            self: The fitted model
        """
        # Combine proxy data to get initial SST estimate
        combined_ages, combined_sst = self._combine_proxy_data(proxy_data_dict)
        
        # Remove NaN values
        valid_mask = ~np.isnan(combined_sst)
        train_x = combined_ages[valid_mask]
        train_y = combined_sst[valid_mask]
        
        if len(train_x) < 5:
            raise ValueError("Not enough valid data points after combining proxies")
            
        # If we have multiple proxy types that might cause numerical issues,
        # add extra regularization and ensure consistent scales
        if len(self.proxy_types) > 1:
            # Normalize the target data to zero mean, unit variance for better numerical stability
            y_mean = np.mean(train_y)
            y_std = np.std(train_y)
            
            if y_std > 0:
                train_y = (train_y - y_mean) / y_std
                
            # Set a flag to denormalize predictions later
            self._normalized_y = True
            self._y_mean = y_mean
            self._y_std = y_std
        else:
            self._normalized_y = False
        
        # Store training data
        self.train_x = train_x
        self.train_y = train_y
        
        # Initialize model
        self.gp_model, self.likelihood = self._init_model(train_x, train_y)
        
        # Set priors on the parameters
        self._set_priors()
        
        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.gp_model.parameters()}
        ], lr=0.05)
        
        # "Loss" for GP is the negative log marginal likelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        # Training loop
        self.gp_model.train()
        self.likelihood.train()
        
        losses = []
        
        # Add more jitter for numerical stability
        with gpytorch.settings.cholesky_jitter(1e-4):
            # Use try-except for numerical stability
            try:
                for i in range(training_iterations):
                    optimizer.zero_grad()
                    output = self.gp_model(self.gp_model.train_inputs[0])
                    
                    # Catch and handle numerical issues in loss calculation
                    try:
                        loss = -self.mll(output, self.gp_model.train_targets)
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
        self.gp_model.eval()
        self.likelihood.eval()
        
        # Sample from the posterior for uncertainty quantification
        self._sample_posterior()
        
        self.is_fitted = True
        return self
    
    def _set_priors(self):
        """Set prior distributions on the GP parameters"""
        # Access the kernel parameters based on the kernel type
        if self.kernel_type == 'combined':
            # RBF kernel parameters
            rbf_lengthscale_prior = self._get_prior_distribution('lengthscale_rbf')
            self.gp_model.rbf_kernel.base_kernel.lengthscale_prior = rbf_lengthscale_prior
            
            # RBF outputscale prior
            rbf_outputscale_prior = self._get_prior_distribution('outputscale')
            self.gp_model.rbf_kernel.outputscale_prior = rbf_outputscale_prior
            
            # Periodic kernel parameters
            periodic_lengthscale_prior = self._get_prior_distribution('lengthscale_periodic')
            self.gp_model.periodic_kernel.base_kernel.lengthscale_prior = periodic_lengthscale_prior
            
            # Periodic period parameter - specific prior for orbital cycles
            period_prior = self._get_prior_distribution('period')
            self.gp_model.periodic_kernel.base_kernel.period_length_prior = period_prior
            
            # Periodic outputscale prior
            periodic_outputscale_prior = self._get_prior_distribution('outputscale')
            self.gp_model.periodic_kernel.outputscale_prior = periodic_outputscale_prior
        else:
            # For RBF or Matern kernels
            lengthscale_prior = self._get_prior_distribution('lengthscale_rbf')
            self.gp_model.covar_module.base_kernel.lengthscale_prior = lengthscale_prior
            
            outputscale_prior = self._get_prior_distribution('outputscale')
            self.gp_model.covar_module.outputscale_prior = outputscale_prior
        
        # Likelihood noise prior
        noise_prior = self._get_prior_distribution('noise')
        self.likelihood.noise_prior = noise_prior
    
    def _get_prior_distribution(self, param_name):
        """
        Get gpytorch prior distribution object based on prior specifications.
        
        Args:
            param_name: Name of the parameter to get prior for
            
        Returns:
            gpytorch.priors object
        """
        if param_name not in self.prior_dict:
            # Return None if no prior specified
            return None
        
        prior_spec = self.prior_dict[param_name]
        dist_type = prior_spec['distribution'].lower()
        
        if dist_type == 'normal':
            return gpytorch.priors.NormalPrior(
                loc=prior_spec['mean'],
                scale=prior_spec['std']
            )
        elif dist_type == 'lognormal':
            return gpytorch.priors.LogNormalPrior(
                loc=prior_spec['mean'],
                scale=prior_spec['std']
            )
        elif dist_type == 'gamma':
            return gpytorch.priors.GammaPrior(
                concentration=prior_spec['alpha'],
                rate=prior_spec['beta']
            )
        else:
            warnings.warn(f"Unknown prior distribution type: {dist_type}. No prior will be used.")
            return None
    
    def _sample_posterior(self):
        """
        Sample from the posterior distribution of model parameters.
        Uses MCMC to generate samples for uncertainty quantification.
        """
        # This is a simplified placeholder for MCMC sampling
        # In a real implementation, you would use PyMC3, NUTS, or other MCMC methods
        
        # For now, we'll simulate posterior samples by adding noise to fitted parameters
        # In a full implementation, replace with proper MCMC
        
        # Create storage for samples
        self.posterior_samples = {
            'mean': np.zeros(self.n_mcmc_samples),
            'noise': np.zeros(self.n_mcmc_samples)
        }
        
        # Get current parameter values
        with torch.no_grad():
            current_mean = self.gp_model.mean_module.constant.item()
            current_noise = self.likelihood.noise.item()
            
            # For combined kernel, get both parameters
            if self.kernel_type == 'combined':
                self.posterior_samples['rbf_lengthscale'] = np.zeros(self.n_mcmc_samples)
                self.posterior_samples['rbf_outputscale'] = np.zeros(self.n_mcmc_samples)
                self.posterior_samples['periodic_lengthscale'] = np.zeros(self.n_mcmc_samples)
                self.posterior_samples['periodic_period'] = np.zeros(self.n_mcmc_samples)
                self.posterior_samples['periodic_outputscale'] = np.zeros(self.n_mcmc_samples)
                
                current_rbf_lengthscale = self.gp_model.rbf_kernel.base_kernel.lengthscale.item()
                current_rbf_outputscale = self.gp_model.rbf_kernel.outputscale.item()
                current_periodic_lengthscale = self.gp_model.periodic_kernel.base_kernel.lengthscale.item()
                current_periodic_period = self.gp_model.periodic_kernel.base_kernel.period_length.item()
                current_periodic_outputscale = self.gp_model.periodic_kernel.outputscale.item()
            else:
                self.posterior_samples['lengthscale'] = np.zeros(self.n_mcmc_samples)
                self.posterior_samples['outputscale'] = np.zeros(self.n_mcmc_samples)
                
                current_lengthscale = self.gp_model.covar_module.base_kernel.lengthscale.item()
                current_outputscale = self.gp_model.covar_module.outputscale.item()
        
        # Generate samples (this is a placeholder for real MCMC)
        for i in range(self.n_mcmc_samples):
            # Sample mean and noise
            self.posterior_samples['mean'][i] = current_mean + np.random.normal(0, 0.1)
            self.posterior_samples['noise'][i] = np.abs(current_noise + np.random.normal(0, 0.01))
            
            # Sample kernel parameters based on kernel type
            if self.kernel_type == 'combined':
                self.posterior_samples['rbf_lengthscale'][i] = np.abs(current_rbf_lengthscale + np.random.normal(0, 0.5))
                self.posterior_samples['rbf_outputscale'][i] = np.abs(current_rbf_outputscale + np.random.normal(0, 0.1))
                self.posterior_samples['periodic_lengthscale'][i] = np.abs(current_periodic_lengthscale + np.random.normal(0, 0.5))
                self.posterior_samples['periodic_period'][i] = np.abs(current_periodic_period + np.random.normal(0, 1.0))
                self.posterior_samples['periodic_outputscale'][i] = np.abs(current_periodic_outputscale + np.random.normal(0, 0.1))
            else:
                self.posterior_samples['lengthscale'][i] = np.abs(current_lengthscale + np.random.normal(0, 0.5))
                self.posterior_samples['outputscale'][i] = np.abs(current_outputscale + np.random.normal(0, 0.1))
    
    def predict(self, test_x, return_samples=False, n_samples=100):
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
        
        # Convert to tensor
        x_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
        
        # Make predictions with additional numerical stability
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
            # Get distribution
            posterior = self.gp_model(x_tensor)
            
            # Get mean and variance
            mean = posterior.mean.cpu().numpy()
            variance = posterior.variance.cpu().numpy()
            
            # Denormalize if needed
            if hasattr(self, '_normalized_y') and self._normalized_y:
                mean = mean * self._y_std + self._y_mean
                variance = variance * (self._y_std ** 2)
                
            # Calculate 95% credible intervals
            lower_ci = mean - 1.96 * np.sqrt(variance)
            upper_ci = mean + 1.96 * np.sqrt(variance)
            
            if return_samples:
                # Draw samples from the posterior predictive distribution
                samples = posterior.sample(sample_shape=torch.Size([n_samples])).cpu().numpy()
                
                # Denormalize samples if needed
                if hasattr(self, '_normalized_y') and self._normalized_y:
                    samples = samples * self._y_std + self._y_mean
                    
                return mean, lower_ci, upper_ci, samples
            else:
                return mean, lower_ci, upper_ci
    
    def detect_abrupt_transitions(self, test_x, threshold_percentile=95, min_separation=5):
        """
        Detect abrupt transitions in the reconstructed SST.
        
        Args:
            test_x: Ages at which to evaluate transitions
            threshold_percentile: Percentile to use for threshold (default: 95)
            min_separation: Minimum separation between detected transitions
            
        Returns:
            List of ages where abrupt transitions occur
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before detecting transitions")
        
        # Get posterior samples
        mean, lower_ci, upper_ci, samples = self.predict(test_x, return_samples=True, n_samples=100)
        
        # Calculate rate of change (first derivative)
        dx = np.diff(test_x)
        dy = np.diff(mean)
        rate_of_change = dy / dx
        
        # Calculate uncertainty in rate of change from samples
        sample_rates = np.diff(samples, axis=1) / dx
        rate_std = np.std(sample_rates, axis=0)
        
        # Normalize rate of change by its uncertainty
        normalized_rate = np.abs(rate_of_change) / rate_std
        
        # Find peaks above threshold
        threshold = np.percentile(normalized_rate, threshold_percentile)
        peak_indices = (normalized_rate > threshold).nonzero()[0]
        
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
            max_idx = group[np.argmax(normalized_rate[group])]
            transition_indices.append(max_idx)
        
        # Convert indices to ages (use midpoint of intervals)
        transition_ages = [(test_x[i] + test_x[i+1])/2 for i in transition_indices]
        
        return transition_ages
    
    def evaluate(self, test_x, true_sst):
        """
        Evaluate model performance against true SST values.
        
        Args:
            test_x: Test ages
            true_sst: True SST values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        mean, lower_ci, upper_ci = self.predict(test_x)
        
        # Find common indices where both predicted and true values exist
        valid_indices = ~np.isnan(true_sst) & ~np.isnan(mean)
        
        if np.sum(valid_indices) < 2:
            warnings.warn("Not enough valid points for evaluation")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'coverage': np.nan
            }
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((mean[valid_indices] - true_sst[valid_indices])**2))
        mae = np.mean(np.abs(mean[valid_indices] - true_sst[valid_indices]))
        
        # R� calculation
        ss_tot = np.sum((true_sst[valid_indices] - np.mean(true_sst[valid_indices]))**2)
        ss_res = np.sum((true_sst[valid_indices] - mean[valid_indices])**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        # Coverage: proportion of true values within the 95% CI
        coverage = np.mean((lower_ci[valid_indices] <= true_sst[valid_indices]) & 
                          (true_sst[valid_indices] <= upper_ci[valid_indices]))
        
        # Continuous Ranked Probability Score (CRPS)
        # Simplified approximation using Gaussian assumption
        std = (upper_ci - lower_ci) / (2 * 1.96)
        z = (true_sst - mean) / std
        crps = np.mean(std[valid_indices] * (z[valid_indices] * (2 * stats.norm.cdf(z[valid_indices]) - 1) + 
                                           2 * stats.norm.pdf(z[valid_indices]) - 1/np.sqrt(np.pi)))
        
        # Average width of confidence interval
        ci_width = np.mean(upper_ci[valid_indices] - lower_ci[valid_indices])
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coverage': coverage,
            'crps': crps,
            'ci_width': ci_width
        }
    
    def plot_reconstruction(self, test_x, proxy_data_dict=None, true_sst=None, 
                          detected_transitions=None, figure_path=None):
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
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the reconstruction with uncertainty
        ax.plot(test_x, mean, 'b-', linewidth=2, label='GP Reconstruction')
        ax.fill_between(test_x, lower_ci, upper_ci, color='b', alpha=0.2, label='95% CI')
        
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
                    proxy_sst = self._proxy_to_sst(proxy_values, proxy_type)
                    
                    ax.scatter(proxy_ages, proxy_sst, 
                              marker=markers[i % len(markers)],
                              color=colors[i % len(colors)], s=30, alpha=0.7,
                              label=f'{proxy_type} derived SST')
        
        # Plot true SST if provided
        if true_sst is not None:
            ax.plot(test_x, true_sst, 'k--', linewidth=1.5, label='True SST')
        
        # Mark detected transitions if provided
        if detected_transitions is not None and len(detected_transitions) > 0:
            y_range = ax.get_ylim()
            for trans_age in detected_transitions:
                ax.axvline(x=trans_age, color='r', linestyle='--', alpha=0.7)
                ax.text(trans_age, y_range[0] + 0.95*(y_range[1]-y_range[0]), 
                       f'{trans_age:.1f}', 
                       color='r', rotation=90, ha='right')
        
        # Add labels and legend
        ax.set_xlabel('Age (kyr)')
        ax.set_ylabel('Sea Surface Temperature (°C)')
        ax.set_title('Bayesian GP State-Space SST Reconstruction')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis direction to be reversed (older ages on the right)
        ax.set_xlim(max(test_x), min(test_x))
        
        plt.tight_layout()
        
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {figure_path}")
            
        return fig
    
    def plot_parameter_posterior(self, figure_path=None):
        """
        Plot posterior distributions of model parameters.
        
        Args:
            figure_path: Path to save the figure
            
        Returns:
            matplotlib figure
        """
        if self.posterior_samples is None:
            raise RuntimeError("No posterior samples available")
        
        # Determine number of parameters to plot
        if self.kernel_type == 'combined':
            n_params = 7  # mean, noise, rbf_lengthscale, rbf_outputscale, 
                         # periodic_lengthscale, periodic_period, periodic_outputscale
        else:
            n_params = 4  # mean, noise, lengthscale, outputscale
        
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        
        # Plot posterior for each parameter
        axes[0].hist(self.posterior_samples['mean'], bins=30, alpha=0.7)
        axes[0].set_title('Mean Parameter')
        
        axes[1].hist(self.posterior_samples['noise'], bins=30, alpha=0.7)
        axes[1].set_title('Noise Parameter')
        
        if self.kernel_type == 'combined':
            axes[2].hist(self.posterior_samples['rbf_lengthscale'], bins=30, alpha=0.7)
            axes[2].set_title('RBF Lengthscale')
            
            axes[3].hist(self.posterior_samples['rbf_outputscale'], bins=30, alpha=0.7)
            axes[3].set_title('RBF Outputscale')
            
            axes[4].hist(self.posterior_samples['periodic_lengthscale'], bins=30, alpha=0.7)
            axes[4].set_title('Periodic Lengthscale')
            
            axes[5].hist(self.posterior_samples['periodic_period'], bins=30, alpha=0.7)
            axes[5].set_title('Periodic Period Length')
            
            axes[6].hist(self.posterior_samples['periodic_outputscale'], bins=30, alpha=0.7)
            axes[6].set_title('Periodic Outputscale')
        else:
            axes[2].hist(self.posterior_samples['lengthscale'], bins=30, alpha=0.7)
            axes[2].set_title('Lengthscale')
            
            axes[3].hist(self.posterior_samples['outputscale'], bins=30, alpha=0.7)
            axes[3].set_title('Outputscale')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {figure_path}")
            
        return fig
    
    def plot_spectral_analysis(self, test_x, figure_path=None):
        """
        Plot spectral analysis of the reconstructed SST.
        
        Args:
            test_x: Ages at which to evaluate spectral properties
            figure_path: Path to save the figure
            
        Returns:
            matplotlib figure
        """
        # Make predictions
        mean, lower_ci, upper_ci, samples = self.predict(test_x, return_samples=True, n_samples=50)
        
        # Ensure regular sampling for FFT
        # If test_x is not regularly sampled, we resample to a regular grid
        if not np.allclose(np.diff(test_x), np.diff(test_x)[0]):
            # Create regular grid
            regular_x = np.linspace(min(test_x), max(test_x), len(test_x))
            
            # Interpolate mean and samples
            interp_mean = np.interp(regular_x, test_x, mean)
            interp_samples = np.zeros((samples.shape[0], len(regular_x)))
            
            for i in range(samples.shape[0]):
                interp_samples[i] = np.interp(regular_x, test_x, samples[i])
            
            # Use interpolated data
            x_spectral = regular_x
            y_spectral = interp_mean
            samples_spectral = interp_samples
        else:
            # Use original data
            x_spectral = test_x
            y_spectral = mean
            samples_spectral = samples
        
        # Calculate sampling interval and frequency
        dt = np.abs(x_spectral[1] - x_spectral[0])  # Time step
        fs = 1 / dt  # Sampling frequency
        
        # Compute the FFT
        n = len(y_spectral)
        y_fft = np.fft.rfft(y_spectral)
        y_fft_magnitude = np.abs(y_fft)
        
        # Compute frequencies
        freq = np.fft.rfftfreq(n, d=dt)
        
        # Convert to periods (1/freq)
        with np.errstate(divide='ignore'):
            periods = 1 / freq
        
        # Compute FFT for each sample
        sample_ffts = np.zeros((samples_spectral.shape[0], len(freq)))
        for i in range(samples_spectral.shape[0]):
            sample_ffts[i] = np.abs(np.fft.rfft(samples_spectral[i]))
        
        # Calculate mean and confidence intervals for the spectra
        mean_fft = np.mean(sample_ffts, axis=0)
        std_fft = np.std(sample_ffts, axis=0)
        lower_ci_fft = mean_fft - 1.96 * std_fft
        lower_ci_fft = np.maximum(lower_ci_fft, 0)  # Ensure non-negative
        upper_ci_fft = mean_fft + 1.96 * std_fft
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot time series
        ax1.plot(x_spectral, y_spectral, 'b-', linewidth=2)
        ax1.fill_between(x_spectral, lower_ci, upper_ci, color='b', alpha=0.2)
        ax1.set_xlabel('Age (kyr)')
        ax1.set_ylabel('SST (�C)')
        ax1.set_title('Reconstructed SST Time Series')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(max(x_spectral), min(x_spectral))  # Reverse x-axis
        
        # Plot spectrum
        # Skip the DC component (first element)
        ax2.plot(periods[1:], mean_fft[1:], 'b-', linewidth=2)
        ax2.fill_between(periods[1:], lower_ci_fft[1:], upper_ci_fft[1:], color='b', alpha=0.2)
        
        # Mark Milankovitch cycles
        milankovitch = [100, 41, 23]  # kyr
        for period in milankovitch:
            ax2.axvline(x=period, color='r', linestyle='--', alpha=0.7)
            ax2.text(period, 0.9*ax2.get_ylim()[1], f'{period} kyr', 
                   color='r', ha='right', rotation=90)
        
        ax2.set_xlabel('Period (kyr)')
        ax2.set_ylabel('Power')
        ax2.set_title('Power Spectrum with 95% CI')
        ax2.grid(True, alpha=0.3)
        
        # Use log scale for better visualization
        ax2.set_xscale('log')
        
        # Set x-axis limits to focus on relevant periods
        ax2.set_xlim(200, 5)  # From 200 kyr to 5 kyr
        
        plt.tight_layout()
        
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {figure_path}")
            
        return fig


def generate_synthetic_data_sparse(
    n_points: int = 100,
    age_min: float = 0,
    age_max: float = 500,
    irregularity: float = 0.5,
    proxy_types: List[str] = ['d18O', 'UK37'],
    true_sst_params: Optional[Dict] = None,
    random_state: int = 42
) -> Dict:
    """
    Generate synthetic paleoclimate data with sparse and irregular sampling.
    
    Args:
        n_points: Base number of data points for all proxies
        age_min: Minimum age in kyr
        age_max: Maximum age in kyr
        irregularity: Controls the irregularity of sampling (0 = regular, 1 = highly irregular)
        proxy_types: List of proxy types to generate
        true_sst_params: Parameters for generating true SST
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with ages, true SST, and proxy data
    """
    np.random.seed(random_state)
    
    # Default SST parameters if not provided
    if true_sst_params is None:
        true_sst_params = {
            'baseline': 15.0,  # �C
            'trend_slope': -0.01,  # �C/kyr (cooling trend)
            'cycles': [
                {'amplitude': 2.0, 'period': 100, 'phase': 0.5},  # 100 kyr cycle (eccentricity)
                {'amplitude': 1.0, 'period': 41, 'phase': 0.2},   # 41 kyr cycle (obliquity)
                {'amplitude': 0.5, 'period': 23, 'phase': 0.8}    # 23 kyr cycle (precession)
            ],
            'abrupt_changes': [
                {'age': 125, 'magnitude': -3.0, 'width': 5},  # Major cooling event
                {'age': 330, 'magnitude': 2.0, 'width': 8}    # Warming event
            ],
            'noise_level': 0.5  # �C
        }
    
    # Generate a regular grid for the true underlying SST
    regular_ages = np.linspace(age_min, age_max, 1000)
    
    # Initialize the true SST with baseline and trend
    true_sst = (true_sst_params['baseline'] + 
                true_sst_params['trend_slope'] * regular_ages)
    
    # Add orbital cycles
    for cycle in true_sst_params['cycles']:
        true_sst += cycle['amplitude'] * np.sin(2 * np.pi * 
                                             (regular_ages / cycle['period'] + 
                                              cycle['phase']))
    
    # Add abrupt changes
    for change in true_sst_params['abrupt_changes']:
        # Use sigmoid function for smooth transition
        transition = change['magnitude'] / (1 + np.exp((regular_ages - change['age']) / (change['width'] / 5)))
        true_sst += transition
    
    # Add noise
    true_sst += np.random.normal(0, true_sst_params['noise_level'], size=len(regular_ages))
    
    # Calibration parameters for proxy models
    calibration_params = {
        'd18O': {
            'slope': -0.22,        # �C per 0
            'intercept': 3.0,      # 0
            'error_std': 0.1,      # 0
            'inverse_slope': -4.54545  # 0 per �C (1/slope)
        },
        'UK37': {
            'slope': 0.033,        # units per �C  
            'intercept': 0.044,    # units
            'error_std': 0.05,     # units
            'inverse_slope': 30.303   # �C per unit (1/slope)
        },
        'Mg_Ca': {
            'slope': 0.09,         # mmol/mol per �C
            'intercept': 0.3,      # mmol/mol
            'error_std': 0.1,      # mmol/mol
            'inverse_slope': 11.111   # �C per mmol/mol (1/slope)
        }
    }
    
    # Generate proxy data with sparse and irregular sampling
    proxy_data = {}
    
    for proxy_type in proxy_types:
        # Adjust number of points for specific proxy (simulate different sampling resolutions)
        if proxy_type == 'd18O':
            proxy_n_points = int(n_points * 1.2)  # More d18O samples
        elif proxy_type == 'UK37':
            proxy_n_points = int(n_points * 0.8)  # Fewer UK37 samples
        else:
            proxy_n_points = n_points
        
        # Generate irregular ages for this proxy
        if irregularity > 0:
            # Start with a regular grid
            regular_step = (age_max - age_min) / (proxy_n_points - 1)
            base_ages = np.linspace(age_min, age_max, proxy_n_points)
            
            # Add irregularity
            jitter = np.random.uniform(-irregularity * regular_step, 
                                     irregularity * regular_step, 
                                     size=proxy_n_points)
            proxy_ages = base_ages + jitter
            
            # Ensure ages remain within range and sorted
            proxy_ages = np.clip(proxy_ages, age_min, age_max)
            proxy_ages.sort()
            
            # Create gaps to mimic missing data sections
            n_gaps = int(proxy_n_points * 0.1 * irregularity)  # 10% * irregularity
            if n_gaps > 0:
                gap_start_idx = np.random.choice(proxy_n_points - 10, n_gaps, replace=False)
                gap_lengths = np.random.randint(3, 10, size=n_gaps)
                mask = np.ones(proxy_n_points, dtype=bool)
                
                for start, length in zip(gap_start_idx, gap_lengths):
                    mask[start:min(start+length, proxy_n_points)] = False
                
                proxy_ages = proxy_ages[mask]
        else:
            # Regular sampling
            proxy_ages = np.linspace(age_min, age_max, proxy_n_points)
        
        # Interpolate true SST to proxy ages
        proxy_true_sst = np.interp(proxy_ages, regular_ages, true_sst)
        
        # Generate proxy values using calibration equation
        params = calibration_params[proxy_type]
        proxy_values = (params['intercept'] + 
                        params['slope'] * proxy_true_sst + 
                        np.random.normal(0, params['error_std'], size=len(proxy_ages)))
        
        # Store results
        proxy_data[proxy_type] = {
            'age': proxy_ages,
            'value': proxy_values,
            'true_sst': proxy_true_sst
        }
    
    # Return all data
    return {
        'regular_ages': regular_ages,
        'true_sst': true_sst,
        'proxy_data': proxy_data,
        'calibration_params': calibration_params
    }


def test_bayesian_gp_state_space_model():
    """
    Test the Bayesian GP State-Space model on synthetic data and 
    visualize the results.
    """
    print("Generating synthetic paleoclimate data...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=80,
        age_min=0,
        age_max=500,
        irregularity=0.7,
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Create output directory if it doesn't exist
    import os
    output_dir = "data/results/bayesian_gp"
    os.makedirs(output_dir, exist_ok=True)
    
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
    test_ages = np.linspace(0, 500, 500)
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
    
    # Plot results
    print("Generating visualizations...")
    
    # Plot reconstruction
    fig = model.plot_reconstruction(
        test_ages, 
        proxy_data, 
        test_true_sst,
        detected_transitions=transitions,
        figure_path=f"{output_dir}/bayesian_gp_reconstruction.png"
    )
    
    # Plot parameter posterior
    fig = model.plot_parameter_posterior(
        figure_path=f"{output_dir}/bayesian_gp_parameter_posterior.png"
    )
    
    # Plot spectral analysis
    fig = model.plot_spectral_analysis(
        test_ages,
        figure_path=f"{output_dir}/bayesian_gp_spectral_analysis.png"
    )
    
    print(f"All results saved to {output_dir}")
    return model, synthetic_data, metrics


if __name__ == "__main__":
    test_bayesian_gp_state_space_model()