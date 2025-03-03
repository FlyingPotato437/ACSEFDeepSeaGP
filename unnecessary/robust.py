"""
Robust likelihood models for paleoclimate data with outliers.

This module implements robust likelihood models specifically designed for 
paleoclimate proxy data, which often contains outliers and non-Gaussian noise.
"""

import torch
import gpytorch
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import base_distributions

class StudentTLikelihood(Likelihood):
    """
    Student-T likelihood for robust handling of outliers.
    
    The Student-T distribution has heavier tails than Gaussian, making it less
    sensitive to outliers, which prevents spikes in the posterior mean. This is
    critical for properly modeling the SST latent process without being overly
    influenced by extreme UK37 measurements.
    """
    def __init__(self, df=4.0, batch_shape=torch.Size([])):
        """
        Initialize the Student-T likelihood.
        
        Args:
            df (float): Degrees of freedom parameter (lower = heavier tails)
            batch_shape (torch.Size): Batch shape for the likelihood
        """
        super().__init__(batch_shape=batch_shape)
        self.df = df
        
        # Register noise scale parameter
        self.register_parameter(
            name="raw_noise", 
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        )
        
        # Apply positive constraint
        self.register_constraint("raw_noise", gpytorch.constraints.Positive())
        
    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)
        
    @noise.setter
    def noise(self, value):
        self._set_noise(value)
        
    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        # Ensure value has correct shape
        if value.numel() == 1 and self.raw_noise.numel() > 1:
            value = value.expand_as(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))
        
    def forward(self, function_samples):
        """
        Define the conditional distribution p(y|f) of observations given latent function
        
        Args:
            function_samples: Samples from latent GP function
            
        Returns:
            A Student-T distribution with location=function_samples and scale=noise
        """
        noise = self.noise.expand_as(function_samples)
        return base_distributions.StudentT(
            df=self.df, 
            loc=function_samples, 
            scale=noise
        )
        
    def log_prob(self, function_samples, target):
        """
        Compute log probability of observations under the Student-T likelihood
        
        Args:
            function_samples: Samples from latent GP function
            target: Observed values
            
        Returns:
            Log probability of observations
        """
        if not isinstance(function_samples, base_distributions.StudentT):
            function_samples = self.forward(function_samples)
        return function_samples.log_prob(target)


class HeteroskedasticStudentTLikelihood(Likelihood):
    """
    Heteroskedastic Student-T likelihood for robust handling of outliers with
    observation-specific noise levels.
    
    This likelihood combines the robustness of Student-T with the flexibility of
    observation-specific noise levels, which is important for paleoclimate data
    where measurement precision can vary across different time periods.
    """
    def __init__(self, df=4.0, noise_values=None, batch_shape=torch.Size([])):
        """
        Initialize the heteroskedastic Student-T likelihood.
        
        Args:
            df (float): Degrees of freedom parameter
            noise_values (torch.Tensor): Initial observation-specific noise values
            batch_shape (torch.Size): Batch shape for the likelihood
        """
        super().__init__(batch_shape=batch_shape)
        self.df = df
        
        # Register observation-specific noise parameters
        if noise_values is not None:
            # Use provided initial values
            self.register_parameter(
                name="raw_noise", 
                parameter=torch.nn.Parameter(noise_values.clone().detach())
            )
        else:
            # Default initialization
            self.register_parameter(
                name="raw_noise", 
                parameter=torch.nn.Parameter(torch.ones(*batch_shape, 1))
            )
        
        # Apply positive constraint
        self.register_constraint("raw_noise", gpytorch.constraints.Positive())
        
    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)
        
    @noise.setter
    def noise(self, value):
        self._set_noise(value)
        
    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        # Handle shape issues carefully
        if value.shape != self.raw_noise.shape:
            if value.numel() == 1:
                value = value.expand_as(self.raw_noise)
            else:
                try:
                    value = value.reshape(self.raw_noise.shape)
                except RuntimeError:
                    raise ValueError(f"Cannot reshape noise values from {value.shape} to {self.raw_noise.shape}")
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))
        
    def forward(self, function_samples):
        """
        Define the conditional distribution p(y|f) with observation-specific noise
        
        Args:
            function_samples: Samples from latent GP function
            
        Returns:
            A Student-T distribution with location=function_samples and observation-specific scale
        """
        # Ensure noise has correct shape
        if function_samples.shape[-1] != self.noise.shape[-1] and self.noise.numel() == function_samples.shape[-1]:
            noise = self.noise.reshape(*function_samples.shape)
        else:
            noise = self.noise.expand(*function_samples.shape)
            
        return base_distributions.StudentT(
            df=self.df, 
            loc=function_samples, 
            scale=noise
        )