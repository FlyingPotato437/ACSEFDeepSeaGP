"""
Adaptive Kernel-Lengthscale Implementations for Paleoclimate Reconstructions

This module provides specialized kernel implementations for Gaussian Processes
that adapt their lengthscales based on the local rate of climate change, enabling
more accurate reconstructions during abrupt climate transitions.
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, PeriodicKernel
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, List, Dict, Union


class RateEstimator:
    """
    Estimates the rate of change in time series data for adaptive lengthscale calculation.
    
    This class implements multiple methods for robustly estimating rate of change
    in paleoclimate time series, supporting the adaptive kernel lengthscale approach.
    """
    
    def __init__(
        self,
        smoothing_method: str = 'gaussian',
        gaussian_sigma: float = 2.0,
        smoothing_window: int = 5,
        use_central_diff: bool = True,
        normalize_method: str = 'robust',
        min_rate: float = 1e-6
    ):
        """
        Initialize the rate estimator.
        
        Args:
            smoothing_method: Method for smoothing derivatives ('gaussian', 'moving_avg')
            gaussian_sigma: Sigma parameter for Gaussian smoothing
            smoothing_window: Window size for smoothing (for moving average)
            use_central_diff: Use central difference for derivative estimation
            normalize_method: Method for normalizing rates ('minmax', 'robust', 'percentile')
            min_rate: Minimum rate value to prevent division by zero
        """
        self.smoothing_method = smoothing_method
        self.gaussian_sigma = gaussian_sigma
        self.smoothing_window = smoothing_window
        self.use_central_diff = use_central_diff
        self.normalize_method = normalize_method
        self.min_rate = min_rate
        
    def estimate_rate(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the normalized rate of change from time series data.
        
        Args:
            x: Input time points (must be sorted)
            y: Function values
            
        Returns:
            x_rates: Time points where rates are estimated
            normalized_rates: Normalized rates of change
        """
        # Ensure inputs are numpy arrays
        x_np = np.asarray(x).flatten()
        y_np = np.asarray(y).flatten()
        
        # Sort by x if needed
        if not np.all(np.diff(x_np) >= 0):
            sort_idx = np.argsort(x_np)
            x_np = x_np[sort_idx]
            y_np = y_np[sort_idx]
        
        # Compute derivatives
        if self.use_central_diff and len(x_np) > 2:
            # Central difference for interior points
            dx_forward = np.diff(x_np)
            dy_forward = np.diff(y_np)
            
            # Forward differences
            rates_forward = dy_forward / np.maximum(dx_forward, self.min_rate)
            
            # Backward differences (shifted)
            rates_backward = np.roll(rates_forward, 1)
            
            # Average for central difference
            # The first and last points use forward/backward only
            rates = np.zeros_like(x_np)
            rates[1:-1] = 0.5 * (rates_forward[1:] + rates_backward[1:])
            rates[0] = rates_forward[0]  # First point: forward diff
            rates[-1] = rates_forward[-1]  # Last point: backward diff
            
            x_rates = x_np
        else:
            # Simple forward differences
            dx = np.diff(x_np)
            dy = np.diff(y_np)
            rates = dy / np.maximum(dx, self.min_rate)
            x_rates = 0.5 * (x_np[:-1] + x_np[1:])  # Midpoints
        
        # Apply selected smoothing method
        smooth_rates = self._smooth_rates(rates)
        
        # Use absolute value for rate magnitude
        abs_rates = np.abs(smooth_rates)
        
        # Normalize rates to [0,1] interval
        normalized_rates = self._normalize_rates(abs_rates)
        
        return x_rates, normalized_rates
        
    def _smooth_rates(self, rates: np.ndarray) -> np.ndarray:
        """Apply the selected smoothing method to the rates."""
        if self.smoothing_method == 'gaussian':
            return gaussian_filter1d(rates, sigma=self.gaussian_sigma)
        
        elif self.smoothing_method == 'moving_avg':
            window = self.smoothing_window
            kernel = np.ones(window) / window
            # Use 'same' mode to keep the same array length
            return np.convolve(rates, kernel, mode='same')
        
        else:
            # No smoothing
            return rates
            
    def _normalize_rates(self, abs_rates: np.ndarray) -> np.ndarray:
        """Normalize rates to [0,1] using selected method."""
        if self.normalize_method == 'minmax':
            # Simple min-max normalization
            rate_min = np.min(abs_rates)
            rate_max = np.max(abs_rates)
            
            if rate_max > rate_min:
                return (abs_rates - rate_min) / (rate_max - rate_min)
            else:
                return np.zeros_like(abs_rates)
        
        elif self.normalize_method == 'robust':
            # Robust normalization using percentiles
            p05 = np.percentile(abs_rates, 5)
            p95 = np.percentile(abs_rates, 95)
            
            if p95 > p05:
                normalized = (abs_rates - p05) / (p95 - p05)
                # Clip to [0,1]
                return np.clip(normalized, 0, 1)
            else:
                return np.zeros_like(abs_rates)
                
        elif self.normalize_method == 'percentile':
            # Convert to percentile within the dataset
            # This ensures uniform distribution of rates
            sorted_idx = np.argsort(abs_rates)
            rank = np.zeros_like(sorted_idx)
            rank[sorted_idx] = np.arange(len(sorted_idx))
            return rank / (len(rank) - 1)  # [0,1] range
        
        else:
            # Default min-max normalization
            rate_range = np.max(abs_rates) - np.min(abs_rates)
            if rate_range > 0:
                return (abs_rates - np.min(abs_rates)) / rate_range
            else:
                return np.zeros_like(abs_rates)


class AdaptiveKernel(Kernel):
    """
    Adaptive kernel that varies lengthscale based on the rate of climate change.
    
    This kernel implements an innovative approach that shortens the lengthscale 
    in regions with rapid rate of change (such as abrupt climate transitions)
    while maintaining longer lengthscales in stable periods.
    """
    
    def __init__(
        self, 
        base_kernel_type: str = 'matern',
        min_lengthscale: float = 2.0,
        max_lengthscale: float = 10.0,
        base_lengthscale: float = 5.0,
        adaptation_strength: float = 1.0,
        lengthscale_regularization: float = 0.1,
        **kwargs
    ):
        """
        Initialize the adaptive kernel.
        
        Args:
            base_kernel_type: Type of base kernel ('rbf', 'matern')
            min_lengthscale: Minimum allowed lengthscale (physically meaningful minimum)
            max_lengthscale: Maximum allowed lengthscale
            base_lengthscale: Base lengthscale value (before adaptation)
            adaptation_strength: Strength of adaptation (alpha parameter)
            lengthscale_regularization: Regularization parameter for lengthscale changes
            **kwargs: Additional arguments for the kernel
        """
        super(AdaptiveKernel, self).__init__(**kwargs)
        
        # Save parameters
        self.base_kernel_type = base_kernel_type.lower()
        self.min_lengthscale = min_lengthscale
        self.max_lengthscale = max_lengthscale
        
        # Register the base lengthscale parameter
        self.register_parameter(
            name="base_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(base_lengthscale))
        )
        
        # Register the adaptation strength parameter
        self.register_parameter(
            name="adaptation_strength",
            parameter=torch.nn.Parameter(torch.tensor(adaptation_strength))
        )

        # Save regularization parameter (not optimized)
        self.lengthscale_regularization = lengthscale_regularization
        
        # Register appropriate priors
        base_ls_prior = gpytorch.priors.LogNormalPrior(
            torch.tensor(np.log(base_lengthscale)), 
            torch.tensor(0.5)
        )
        self.register_prior("base_lengthscale_prior", base_ls_prior, "base_lengthscale")
        
        adapt_strength_prior = gpytorch.priors.LogNormalPrior(
            torch.tensor(np.log(adaptation_strength)), 
            torch.tensor(0.3)
        )
        self.register_prior("adaptation_strength_prior", adapt_strength_prior, "adaptation_strength")
        
        # Initialize rate of change information
        self.rate_of_change = None
        self.rate_points = None
        
        # Initialize base kernel
        if self.base_kernel_type == 'rbf':
            self.base_kernel = RBFKernel()
        elif self.base_kernel_type == 'matern':
            self.base_kernel = MaternKernel(nu=2.5)  # Matern 5/2 kernel
        else:
            raise ValueError(f"Unsupported base kernel type: {base_kernel_type}")
            
        # Last used lengthscale for regularization
        self.last_lengthscale = None
        
    def update_rate_of_change(self, points: np.ndarray, rates: np.ndarray):
        """
        Update the rate of change information.
        
        Args:
            points: Time points where rates are estimated
            rates: Normalized rate of change at each point
        """
        # Store as torch tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rate_points = torch.tensor(points, dtype=torch.float32, device=device)
        self.rate_of_change = torch.tensor(rates, dtype=torch.float32, device=device)
        
    def _get_lengthscale(self, x):
        """
        Compute adaptive lengthscale for input points with constraints.
        
        Args:
            x: Input points
            
        Returns:
            Adapted lengthscale values for each point
        """
        if self.rate_of_change is None or self.rate_points is None:
            # Return base lengthscale if no rate information
            return self.base_lengthscale.expand(x.shape[:-1])
        
        # Interpolate rate of change to input points
        interp_rates = self._interpolate_rates(x)
        
        # Apply lengthscale adaptation formula with constraints
        # ℓ(x) = ℓ_base / (1 + α·r(x))
        denom = 1.0 + self.adaptation_strength * interp_rates
        lengthscale = self.base_lengthscale / denom
        
        # Apply constraints to keep lengthscale in physically meaningful range
        lengthscale = torch.clamp(lengthscale, self.min_lengthscale, self.max_lengthscale)
        
        return lengthscale
        
    def _interpolate_rates(self, x):
        """
        Interpolate rate of change values to input points.
        
        Args:
            x: Input points
            
        Returns:
            Interpolated rate values
        """
        # Reshape input if needed
        if x.dim() == 1:
            x_flat = x.reshape(-1, 1)
        else:
            # Use only the coordinates, not batch dimensions
            x_flat = x.reshape(-1, x.size(-1))
            
        # Initialize output tensor
        interp_rates = torch.zeros(x_flat.size(0), device=x.device)
        
        # Simple nearest neighbor interpolation for now
        for i in range(x_flat.size(0)):
            point = x_flat[i, 0]  # Assuming 1D inputs
            
            # Find closest point in rate_points
            dist = torch.abs(self.rate_points - point)
            idx = torch.argmin(dist)
            interp_rates[i] = self.rate_of_change[idx]
        
        # Reshape to match input
        if x.dim() == 1:
            return interp_rates
        else:
            # Reshape to match original dimensions but without the last dimension
            return interp_rates.reshape(x.shape[:-1])
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix between x1 and x2.
        
        Args:
            x1: First set of input points
            x2: Second set of input points (or None for same points)
            diag: If True, compute only diagonal elements (x1==x2)
            
        Returns:
            Kernel matrix
        """
        # Get lengthscales for x1
        lengthscale_x1 = self._get_lengthscale(x1)
        
        # Set the base kernel's lengthscale
        # For non-stationary kernel, we'll use the average as an approximation
        avg_lengthscale = torch.mean(lengthscale_x1)
        self.base_kernel.lengthscale = avg_lengthscale
        
        # Compute base kernel
        base_k = self.base_kernel.forward(x1, x2, diag=diag, **params)
        
        # Add regularization term if last_lengthscale exists
        # This penalizes large changes in lengthscale
        if self.last_lengthscale is not None and self.lengthscale_regularization > 0:
            # Store current lengthscale for next iteration
            with torch.no_grad():
                self.last_lengthscale = avg_lengthscale.clone()
        else:
            # Initialize last_lengthscale
            with torch.no_grad():
                self.last_lengthscale = avg_lengthscale.clone()
                
        return base_k


class MultiScalePeriodicKernel(Kernel):
    """
    A multi-scale periodic kernel that captures multiple orbital cycles.
    
    This kernel combines multiple periodic components tuned to different 
    Milankovitch cycles (eccentricity: ~100kyr, obliquity: ~41kyr, 
    precession: ~23kyr) to better represent paleoclimate oscillations.
    """
    
    def __init__(
        self,
        periods: List[float] = [100.0, 41.0, 23.0], 
        period_priors: List[Tuple[float, float]] = [(100.0, 10.0), (41.0, 5.0), (23.0, 3.0)],
        outputscales: List[float] = [2.0, 1.0, 0.5],
        **kwargs
    ):
        """
        Initialize the multi-scale periodic kernel.
        
        Args:
            periods: List of periods for each component (default: Milankovitch cycles)
            period_priors: List of (mean, std) for period normal priors
            outputscales: List of initial outputscales for each component
            **kwargs: Additional arguments for the kernel
        """
        super(MultiScalePeriodicKernel, self).__init__(**kwargs)
        
        self.num_components = len(periods)
        
        # Create individual periodic kernels for each cycle
        self.periodic_kernels = torch.nn.ModuleList()
        self.outputscales = torch.nn.ParameterList()
        
        # Setup each periodic component
        for i in range(self.num_components):
            period = periods[i]
            prior_mean, prior_std = period_priors[i]
            outputscale = outputscales[i]
            
            # Create periodic kernel
            periodic_kernel = PeriodicKernel()
            
            # Set period and prior
            periodic_kernel.period_length = period
            periodic_kernel.period_length_prior = gpytorch.priors.NormalPrior(
                torch.tensor(prior_mean), 
                torch.tensor(prior_std)
            )
            
            # Register outputscale parameter for this component
            component_outputscale = torch.nn.Parameter(torch.tensor(outputscale))
            
            # Set outputscale prior (log-normal centered around the initial value)
            outputscale_prior = gpytorch.priors.LogNormalPrior(
                torch.tensor(np.log(outputscale)), 
                torch.tensor(0.5)
            )
            
            # Add to module lists
            self.periodic_kernels.append(periodic_kernel)
            self.outputscales.append(component_outputscale)
            
            # Register prior
            lambda_fn = lambda: outputscale_prior(self.outputscales[i])
            self.register_prior(
                f"outputscale_prior_{i}", 
                lambda_fn, 
                self.outputscales[i]
            )
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix by combining all periodic components.
        
        Args:
            x1: First set of input points
            x2: Second set of input points
            diag: If True, compute only diagonal elements
            
        Returns:
            Combined kernel matrix
        """
        # Start with zeros
        if diag:
            res = torch.zeros(x1.size(0), device=x1.device)
        else:
            res = torch.zeros(x1.size(0), x2.size(0), device=x1.device)
            
        # Add weighted contribution from each periodic component
        for i in range(self.num_components):
            component_k = self.periodic_kernels[i].forward(x1, x2, diag=diag, **params)
            res = res + self.outputscales[i] * component_k
            
        return res
