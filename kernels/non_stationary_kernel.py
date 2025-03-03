"""
Non-Stationary Kernels for Paleoclimate Reconstruction

This module implements advanced non-stationary covariance functions specifically
designed for paleoclimate data, including:

1. Temperature-dependent kernels that vary their characteristics based on the absolute
   temperature, accounting for different climate regimes
2. Locally-adaptive kernels with location-dependent parameters beyond just lengthscale
3. Non-stationary spectral mixture kernels for complex cyclic behaviors
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, PeriodicKernel
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable
from functools import partial


class TemperatureDependentKernel(Kernel):
    """
    Temperature-dependent kernel that adapts its properties based on the absolute
    temperature value, enabling different behavior in different climate regimes
    (e.g., glacial vs. interglacial).
    
    The kernel modifies its parameters based on the input temperature value, using
    a monotonic function to transition between different parameter regimes.
    """
    
    def __init__(
        self,
        base_kernel_type: str = 'matern',
        cold_lengthscale: float = 4.0,     # Lengthscale for cold regime (glacial)
        warm_lengthscale: float = 8.0,     # Lengthscale for warm regime (interglacial)
        transition_midpoint: float = 10.0,  # Temperature at transition midpoint (°C)
        transition_width: float = 2.0,     # Width of transition region (°C)
        **kwargs
    ):
        """
        Initialize the temperature-dependent kernel.
        
        Args:
            base_kernel_type: Type of base kernel ('matern', 'rbf')
            cold_lengthscale: Lengthscale for cold regime
            warm_lengthscale: Lengthscale for warm regime
            transition_midpoint: Temperature at transition midpoint
            transition_width: Width of transition region
            **kwargs: Additional arguments for kernel
        """
        super(TemperatureDependentKernel, self).__init__(**kwargs)
        
        # Initialize parameters
        self.transition_midpoint = transition_midpoint
        self.transition_width = transition_width
        
        # Register lengthscale parameters with appropriate constraints
        self.register_parameter(
            name="cold_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(cold_lengthscale))
        )
        self.register_parameter(
            name="warm_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(warm_lengthscale))
        )
        
        # Add positive constraints
        self.register_constraint("cold_lengthscale", gpytorch.constraints.Positive())
        self.register_constraint("warm_lengthscale", gpytorch.constraints.Positive())
        
        # Set up base kernel based on type
        if base_kernel_type.lower() == 'matern':
            self.base_kernel = MaternKernel(nu=2.5)
        elif base_kernel_type.lower() == 'rbf':
            self.base_kernel = RBFKernel()
        else:
            raise ValueError(f"Unsupported base kernel type: {base_kernel_type}")
        
        # Temperature values for the data points
        self.temperature_values = None
    
    def update_temperature_values(self, temp_values: torch.Tensor):
        """
        Update the temperature values for the data points.
        
        Args:
            temp_values: Temperature values corresponding to input points
        """
        self.temperature_values = temp_values
    
    def _get_lengthscale(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate temperature-dependent lengthscale.
        
        Args:
            temperature: Temperature values
            
        Returns:
            Adaptive lengthscale based on temperature
        """
        # Apply sigmoid function to smoothly transition between regimes
        # sigmoid(x) = 1 / (1 + exp(-x))
        transition_factor = 1.0 / (1.0 + torch.exp(
            -(temperature - self.transition_midpoint) / (self.transition_width / 4.0)
        ))
        
        # Interpolate between cold and warm lengthscales
        return self.cold_lengthscale + transition_factor * (
            self.warm_lengthscale - self.cold_lengthscale
        )
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix between inputs.
        
        Args:
            x1: First input (n x d)
            x2: Second input (m x d)
            diag: Return diagonal of kernel matrix
            
        Returns:
            Kernel matrix
        """
        # We need temperature values to compute the kernel
        if self.temperature_values is None:
            raise RuntimeError("Temperature values not set. Call update_temperature_values() first.")
        
        # For non-stationary kernels, we need to build the kernel matrix element by element
        if diag:
            # Only need diagonal elements (x1 == x2)
            if x1.size() != x2.size():
                raise RuntimeError("x1 and x2 must have the same size for diag=True")
            
            n = x1.size(0)
            res = torch.zeros(n, device=x1.device)
            
            # Calculate diagonal elements
            for i in range(n):
                # Get temperatures
                temp_i = self.temperature_values[i]
                
                # Calculate lengthscale
                lengthscale_i = self._get_lengthscale(temp_i)
                
                # Set base kernel lengthscale
                self.base_kernel.lengthscale = lengthscale_i
                
                # Compute kernel element
                res[i] = self.base_kernel.forward(
                    x1[i:i+1], x2[i:i+1], diag=True, **params
                )
            
            return res
        else:
            # Full kernel matrix
            n, m = x1.size(0), x2.size(0)
            res = torch.zeros(n, m, device=x1.device)
            
            # Calculate full matrix
            for i in range(n):
                for j in range(m):
                    # Get temperatures
                    temp_i = self.temperature_values[i]
                    temp_j = self.temperature_values[j] if i != j else temp_i
                    
                    # Calculate average lengthscale
                    lengthscale_ij = 0.5 * (
                        self._get_lengthscale(temp_i) + self._get_lengthscale(temp_j)
                    )
                    
                    # Set base kernel lengthscale
                    self.base_kernel.lengthscale = lengthscale_ij
                    
                    # Compute kernel element
                    res[i, j] = self.base_kernel.forward(
                        x1[i:i+1], x2[j:j+1], diag=False, **params
                    )
            
            return res


class NonStationaryMaternKernel(Kernel):
    """
    Non-stationary Matern kernel with spatially-varying parameters.
    
    This implementation is based on the paciorek & Schervish (2004) formulation,
    where both lengthscale and smoothness can vary with input location.
    """
    
    def __init__(
        self,
        lengthscale_function: Optional[Callable] = None,
        nu_function: Optional[Callable] = None,
        base_lengthscale: float = 5.0,
        base_nu: float = 2.5,
        **kwargs
    ):
        """
        Initialize the non-stationary Matern kernel.
        
        Args:
            lengthscale_function: Function mapping input to lengthscale
            nu_function: Function mapping input to smoothness parameter
            base_lengthscale: Base lengthscale value
            base_nu: Base smoothness parameter
            **kwargs: Additional kernel arguments
        """
        super(NonStationaryMaternKernel, self).__init__(**kwargs)
        
        # Store the parameter functions
        self.lengthscale_function = lengthscale_function or (lambda x: torch.ones_like(x) * base_lengthscale)
        self.nu_function = nu_function or (lambda x: torch.ones_like(x) * base_nu)
        
        # Register base parameters
        self.register_parameter(
            name="base_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(base_lengthscale))
        )
        self.register_parameter(
            name="base_nu",
            parameter=torch.nn.Parameter(torch.tensor(base_nu))
        )
        
        # Add constraints
        self.register_constraint("base_lengthscale", gpytorch.constraints.Positive())
        self.register_constraint("base_nu", gpytorch.constraints.Positive())
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix.
        
        Implementation of the Paciorek & Schervish (2004) non-stationary Matern kernel.
        
        Args:
            x1: First input
            x2: Second input
            diag: Return diagonal of kernel matrix
            
        Returns:
            Kernel matrix
        """
        if diag:
            # Only need diagonal elements (x1 == x2)
            if x1.size() != x2.size():
                raise RuntimeError("x1 and x2 must have the same size for diag=True")
            
            # For diagonal, the formula simplifies
            n = x1.size(0)
            res = torch.ones(n, device=x1.device)
            return res
        else:
            # Full kernel matrix calculation
            n, m = x1.size(0), x2.size(0)
            d = x1.size(1)  # Input dimension
            
            # Get parameter values at each location
            lengthscale_x1 = self.lengthscale_function(x1) * self.base_lengthscale
            lengthscale_x2 = self.lengthscale_function(x2) * self.base_lengthscale
            
            nu_x1 = self.nu_function(x1) * self.base_nu
            nu_x2 = self.nu_function(x2) * self.base_nu
            
            # Initialize result matrix
            res = torch.zeros(n, m, device=x1.device)
            
            # Compute pairwise distances
            for i in range(n):
                for j in range(m):
                    # Get local parameters
                    l_i = lengthscale_x1[i]
                    l_j = lengthscale_x2[j]
                    
                    nu_i = nu_x1[i]
                    nu_j = nu_x2[j]
                    
                    # The effective parameters for this pair
                    l_ij = 0.5 * (l_i**2 + l_j**2)
                    nu_ij = 0.5 * (nu_i + nu_j)
                    
                    # Compute quadratic form
                    dist = torch.sum((x1[i] - x2[j])**2) / l_ij
                    
                    # Compute determinant term
                    det_term = (2 * torch.sqrt(l_i * l_j) / (l_i + l_j))**d
                    
                    # Compute Matern function
                    # We need to approximate the Matern function based on nu
                    if nu_ij <= 0.5:
                        # Exponential covariance
                        matern_term = torch.exp(-torch.sqrt(dist))
                    elif nu_ij <= 1.5:
                        # Matern 1/2
                        scaled_dist = torch.sqrt(2 * nu_ij * dist)
                        matern_term = (1 + scaled_dist) * torch.exp(-scaled_dist)
                    elif nu_ij <= 2.5:
                        # Matern 3/2
                        scaled_dist = torch.sqrt(2 * nu_ij * dist)
                        matern_term = (1 + scaled_dist + scaled_dist**2/3) * torch.exp(-scaled_dist)
                    else:
                        # Matern 5/2
                        scaled_dist = torch.sqrt(2 * nu_ij * dist)
                        matern_term = (1 + scaled_dist + scaled_dist**2/3 + scaled_dist**3/15) * torch.exp(-scaled_dist)
                    
                    # Combine terms
                    res[i, j] = det_term * matern_term
            
            return res


class SpectralMixtureKernelWithTrend(Kernel):
    """
    Extended Spectral Mixture Kernel with trend components for paleoclimate data.
    
    This kernel combines a flexible spectral mixture with trend components to
    capture both cyclical patterns and long-term trends in the data.
    """
    
    def __init__(
        self,
        num_mixtures: int = 3,
        mixture_scales: Optional[List[float]] = None,
        mixture_means: Optional[List[float]] = None,
        mixture_weights: Optional[List[float]] = None,
        trend_degree: int = 1,
        **kwargs
    ):
        """
        Initialize the spectral mixture kernel with trend.
        
        Args:
            num_mixtures: Number of mixtures
            mixture_scales: Initial scales for each mixture
            mixture_means: Initial means for each mixture
            mixture_weights: Initial weights for each mixture
            trend_degree: Degree of polynomial trend (0=constant, 1=linear, etc.)
            **kwargs: Additional kernel arguments
        """
        super(SpectralMixtureKernelWithTrend, self).__init__(**kwargs)
        
        self.num_mixtures = num_mixtures
        self.trend_degree = trend_degree
        
        # Initialize default mixture parameters if not provided
        if mixture_scales is None:
            mixture_scales = [1.0] * num_mixtures
        
        if mixture_means is None:
            # Initialize means to cover different frequencies
            mixture_means = [0.01 * (i + 1) for i in range(num_mixtures)]
        
        if mixture_weights is None:
            mixture_weights = [1.0 / num_mixtures] * num_mixtures
        
        # Register mixture parameters
        self.register_parameter(
            "mixture_weights",
            torch.nn.Parameter(torch.tensor(mixture_weights))
        )
        self.register_parameter(
            "mixture_means",
            torch.nn.Parameter(torch.tensor(mixture_means))
        )
        self.register_parameter(
            "mixture_scales",
            torch.nn.Parameter(torch.tensor(mixture_scales))
        )
        
        # Register constraints
        self.register_constraint("mixture_weights", gpytorch.constraints.Positive())
        self.register_constraint("mixture_scales", gpytorch.constraints.Positive())
        
        # Initialize trend parameters
        if trend_degree > 0:
            self.register_parameter(
                "trend_coeffs",
                torch.nn.Parameter(torch.zeros(trend_degree + 1))
            )
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix.
        
        Args:
            x1: First input
            x2: Second input
            diag: Return diagonal of kernel matrix
            
        Returns:
            Kernel matrix
        """
        if diag:
            # Only diagonal elements (x1 == x2)
            n = x1.size(0)
            res = torch.zeros(n, device=x1.device)
            
            # Spectral mixture component
            for i in range(self.num_mixtures):
                weight = self.mixture_weights[i]
                scale = self.mixture_scales[i]
                mean = self.mixture_means[i]
                
                # For identical inputs, the cosine term is 1
                res = res + weight
            
            # Add trend component if needed
            if self.trend_degree > 0:
                for i in range(n):
                    # Polynomial terms
                    for j in range(self.trend_degree + 1):
                        res[i] = res[i] + self.trend_coeffs[j] * x1[i][0]**j
            
            return res
        else:
            # Full kernel matrix
            n, m = x1.size(0), x2.size(0)
            res = torch.zeros(n, m, device=x1.device)
            
            # Compute spectral mixture component
            for i in range(self.num_mixtures):
                weight = self.mixture_weights[i]
                scale = self.mixture_scales[i]
                mean = self.mixture_means[i]
                
                # Compute distance matrix
                dist = torch.cdist(x1, x2, p=2)
                
                # Compute RBF component
                rbf_component = torch.exp(-0.5 * (dist**2) / scale**2)
                
                # Compute cosine component
                x1_expanded = x1.unsqueeze(1)  # n x 1 x d
                x2_expanded = x2.unsqueeze(0)  # 1 x m x d
                product = x1_expanded * x2_expanded  # n x m x d
                cosine_term = torch.cos(2 * np.pi * mean * product.sum(dim=2))
                
                # Add to result
                res = res + weight * rbf_component * cosine_term
            
            # Add trend component
            if self.trend_degree > 0:
                for i in range(n):
                    for j in range(m):
                        # Polynomial basis
                        x1_term = x1[i][0]
                        x2_term = x2[j][0]
                        
                        # Add polynomial terms
                        for k in range(self.trend_degree + 1):
                            res[i, j] = res[i, j] + self.trend_coeffs[k] * (x1_term * x2_term)**k
            
            return res