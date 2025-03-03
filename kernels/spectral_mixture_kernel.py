"""
Advanced spectral kernels for paleoclimate time series analysis.

This module implements advanced spectral mixture kernels that can capture
complex frequency patterns in paleoclimate data, especially the Milankovitch
cycles and their harmonics in SST reconstructions.
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel
import numpy as np
import math


class SpectralMixtureKernel(Kernel):
    """
    Advanced Spectral Mixture Kernel for climate time series.
    
    This kernel can learn the frequency content directly from the data,
    automatically identifying periodic patterns at multiple scales. It's
    especially suited for paleoclimate time series which contain both
    orbital-scale oscillations and higher-frequency variations.
    
    Based on Wilson & Adams (2013): https://arxiv.org/abs/1302.4245
    """
    
    def __init__(
        self,
        num_mixtures=4,
        mixture_means=None,
        mixture_scales=None,
        mixture_weights=None,
        **kwargs
    ):
        """
        Initialize the spectral mixture kernel.
        
        Args:
            num_mixtures (int): Number of spectral mixtures
            mixture_means (torch.Tensor): Initial mixture means (frequencies)
            mixture_scales (torch.Tensor): Initial mixture scales (bandwidths)
            mixture_weights (torch.Tensor): Initial mixture weights
            **kwargs: Additional kernel arguments
        """
        super(SpectralMixtureKernel, self).__init__(**kwargs)
        
        self.num_mixtures = num_mixtures
        
        # Register initial parameters if not provided
        if mixture_means is None:
            # Initialize with linearly spaced frequencies
            mixture_means = torch.linspace(0.01, 0.5, num_mixtures)
        
        if mixture_scales is None:
            # Initialize with reasonable bandwidths
            mixture_scales = torch.ones(num_mixtures) * 0.1
            
        if mixture_weights is None:
            # Equal weights initially
            mixture_weights = torch.ones(num_mixtures) / num_mixtures
        
        # Register parameters
        self.register_parameter(
            name="raw_mixture_weights",
            parameter=torch.nn.Parameter(mixture_weights)
        )
        self.register_parameter(
            name="raw_mixture_means",
            parameter=torch.nn.Parameter(mixture_means)
        )
        self.register_parameter(
            name="raw_mixture_scales",
            parameter=torch.nn.Parameter(mixture_scales)
        )
        
        # Register constraints
        self.register_constraint("raw_mixture_weights", gpytorch.constraints.Positive())
        self.register_constraint("raw_mixture_means", gpytorch.constraints.Positive())
        self.register_constraint("raw_mixture_scales", gpytorch.constraints.Positive())
        
    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)
    
    @property
    def mixture_means(self):
        return self.raw_mixture_means_constraint.transform(self.raw_mixture_means)
    
    @property
    def mixture_scales(self):
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)
    
    def initialize_from_data(self, x, y):
        """
        Automatically initialize kernel parameters using FFT of the data.
        
        This provides a much better starting point for optimization than
        random initialization, leading to better fits and faster convergence.
        
        Args:
            x (torch.Tensor): Input locations
            y (torch.Tensor): Target values
        """
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            
        # Ensure x is sorted and unique
        if x.dim() > 1:
            x = x.squeeze()
        
        sorted_indices = torch.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Compute median spacing for frequency limit
        dx = torch.median(x_sorted[1:] - x_sorted[:-1])
        
        # Compute FFT (need approximately evenly spaced data)
        n = x.numel()
        # Simple linear interpolation to regular grid
        x_reg = torch.linspace(x_sorted.min(), x_sorted.max(), n)
        y_reg = torch.zeros_like(x_reg)
        
        # Linear interpolation
        for i in range(n):
            # Find bracketing points
            idx = torch.searchsorted(x_sorted, x_reg[i])
            if idx == 0:
                y_reg[i] = y_sorted[0]
            elif idx == n:
                y_reg[i] = y_sorted[-1]
            else:
                # Linear interpolation
                alpha = (x_reg[i] - x_sorted[idx-1]) / (x_sorted[idx] - x_sorted[idx-1])
                y_reg[i] = (1-alpha) * y_sorted[idx-1] + alpha * y_sorted[idx]
        
        # Compute FFT
        y_fft = torch.fft.rfft(y_reg)
        fft_freq = torch.fft.rfftfreq(n, dx)
        
        # Get power spectrum
        power = torch.abs(y_fft)**2
        
        # Find highest peaks in power spectrum
        num_peaks = min(self.num_mixtures * 2, len(power) // 2)  # Look for more peaks than mixtures
        _, peak_indices = torch.topk(power[1:], num_peaks)  # Skip DC component
        peak_indices = peak_indices + 1  # Adjust for DC offset
        
        # Sort by frequency
        peak_indices = torch.sort(peak_indices)[0]
        
        # Group adjacent peaks and take strongest from each group
        peaks = []
        current_group = [peak_indices[0]]
        
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] <= 2:  # Adjacent in frequency
                current_group.append(peak_indices[i])
            else:
                # Find strongest in group
                group_powers = power[current_group]
                strongest = current_group[torch.argmax(group_powers)]
                peaks.append(strongest)
                current_group = [peak_indices[i]]
        
        # Add the last group
        if current_group:
            group_powers = power[current_group]
            strongest = current_group[torch.argmax(group_powers)]
            peaks.append(strongest)
        
        # Take top num_mixtures peaks
        if len(peaks) > self.num_mixtures:
            peak_powers = power[peaks]
            _, top_indices = torch.topk(peak_powers, self.num_mixtures)
            peaks = [peaks[i] for i in top_indices]
        
        # If we have fewer peaks than mixtures, pad with random frequencies
        while len(peaks) < self.num_mixtures:
            # Add random frequency not already in peaks
            new_freq = torch.randint(1, len(power) // 2, (1,))[0]
            if new_freq not in peaks:
                peaks.append(new_freq)
        
        # Extract frequencies and bandwidths
        init_means = fft_freq[peaks]
        
        # Estimate bandwidths from peak width
        init_scales = torch.ones_like(init_means) * 0.1
        for i, peak in enumerate(peaks):
            # Find half-power points
            half_power = power[peak] / 2
            
            # Search to the left
            left_idx = peak
            while left_idx > 0 and power[left_idx] > half_power:
                left_idx -= 1
                
            # Search to the right
            right_idx = peak
            while right_idx < len(power) - 1 and power[right_idx] > half_power:
                right_idx += 1
                
            # Bandwidth is the difference between half-power points
            if right_idx > left_idx:
                bandwidth = fft_freq[right_idx] - fft_freq[left_idx]
                init_scales[i] = max(bandwidth, 0.01)  # Ensure positive
        
        # Weights proportional to peak power
        init_weights = power[peaks]
        init_weights = init_weights / init_weights.sum()
        
        # Set parameters
        self.raw_mixture_means.data = self.raw_mixture_means_constraint.inverse_transform(init_means)
        self.raw_mixture_scales.data = self.raw_mixture_scales_constraint.inverse_transform(init_scales)
        self.raw_mixture_weights.data = self.raw_mixture_weights_constraint.inverse_transform(init_weights)
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix.
        
        Args:
            x1 (torch.Tensor): First set of input points
            x2 (torch.Tensor): Second set of input points
            diag (bool): If True, return only the diagonal of the covariance matrix
            **params: Additional parameters
            
        Returns:
            torch.Tensor: Kernel matrix
        """
        # Get parameters
        mixture_weights = self.mixture_weights
        mixture_means = self.mixture_means
        mixture_scales = self.mixture_scales
        
        # Normalize weights to sum to 1
        mixture_weights = mixture_weights / mixture_weights.sum()
        
        # Compute distance matrix
        if diag:
            # Just zeros for diagonal
            distance = torch.zeros(x1.size(0), device=x1.device, dtype=x1.dtype)
        else:
            # Compute pairwise distance
            if x1.dim() == 1:
                x1 = x1.unsqueeze(1)
            if x2.dim() == 1:
                x2 = x2.unsqueeze(1)
                
            x1_expand = x1.unsqueeze(1)  # [n, 1, d]
            x2_expand = x2.unsqueeze(0)  # [1, m, d]
            distance = (x1_expand - x2_expand).squeeze(-1)  # [n, m]
        
        # Initialize kernel
        if diag:
            # Special case for diagonal
            result = torch.zeros(x1.size(0), device=x1.device, dtype=x1.dtype)
            # Spectral components
            for i in range(self.num_mixtures):
                result = result + mixture_weights[i]
            return result
        else:
            result = torch.zeros_like(distance)
            
            # Spectral components
            for i in range(self.num_mixtures):
                weight = mixture_weights[i]
                mean = mixture_means[i]
                scale = mixture_scales[i]
                
                # Compute spectral component
                cos_term = torch.cos(2 * math.pi * mean * distance)
                exp_term = torch.exp(-2 * (math.pi * scale * distance)**2)
                result = result + weight * cos_term * exp_term
            
            return result


class MilankovitchKernel(Kernel):
    """
    Specialized kernel for Milankovitch cycles in paleoclimate data.
    
    This kernel explicitly models the known astronomical cycles that drive
    climate oscillations, including eccentricity (100 kyr), obliquity (41 kyr),
    and precession (23 kyr and 19 kyr). Each cycle can have its own amplitude
    and phase, allowing the model to capture the complex interplay of these
    cycles in driving climate variability.
    """
    
    def __init__(
        self,
        periods=[100.0, 41.0, 23.0, 19.0],
        bandwidths=None,
        amplitudes=None,
        **kwargs
    ):
        """
        Initialize the Milankovitch kernel.
        
        Args:
            periods (list): Periods of the Milankovitch cycles in kyr
            bandwidths (list): Bandwidths for each cycle
            amplitudes (list): Amplitudes for each cycle
            **kwargs: Additional kernel arguments
        """
        super(MilankovitchKernel, self).__init__(**kwargs)
        
        self.num_cycles = len(periods)
        
        # Set default bandwidths and amplitudes if not provided
        if bandwidths is None:
            # Narrow bandwidths for precise cycles
            bandwidths = [p * 0.05 for p in periods]  # 5% of period
            
        if amplitudes is None:
            # Reasonable default amplitudes with 100 kyr cycle strongest
            amplitudes = [1.0, 0.7, 0.4, 0.3][:len(periods)]
        
        # Register periods (fixed physical constants)
        self.register_buffer("periods", torch.tensor(periods))
        
        # Register bandwidths and amplitudes as parameters
        self.register_parameter(
            name="raw_bandwidths",
            parameter=torch.nn.Parameter(torch.tensor(bandwidths))
        )
        self.register_parameter(
            name="raw_amplitudes",
            parameter=torch.nn.Parameter(torch.tensor(amplitudes))
        )
        
        # Register constraints
        self.register_constraint("raw_bandwidths", gpytorch.constraints.Positive())
        self.register_constraint("raw_amplitudes", gpytorch.constraints.Positive())
    
    @property
    def bandwidths(self):
        return self.raw_bandwidths_constraint.transform(self.raw_bandwidths)
    
    @property
    def amplitudes(self):
        return self.raw_amplitudes_constraint.transform(self.raw_amplitudes)
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix.
        
        Args:
            x1 (torch.Tensor): First set of input points
            x2 (torch.Tensor): Second set of input points
            diag (bool): If True, return only the diagonal of the covariance matrix
            **params: Additional parameters
            
        Returns:
            torch.Tensor: Kernel matrix
        """
        # Convert periods to frequencies
        frequencies = 1.0 / self.periods
        bandwidths = self.bandwidths
        amplitudes = self.amplitudes
        
        # Normalize amplitudes to sum to 1
        amplitudes = amplitudes / amplitudes.sum()
        
        # Compute distance matrix
        if diag:
            # Just zeros for diagonal
            distance = torch.zeros(x1.size(0), device=x1.device, dtype=x1.dtype)
        else:
            # Compute pairwise distance
            if x1.dim() == 1:
                x1 = x1.unsqueeze(1)
            if x2.dim() == 1:
                x2 = x2.unsqueeze(1)
                
            x1_expand = x1.unsqueeze(1)  # [n, 1, d]
            x2_expand = x2.unsqueeze(0)  # [1, m, d]
            distance = (x1_expand - x2_expand).squeeze(-1)  # [n, m]
        
        # Initialize kernel
        if diag:
            # Special case for diagonal
            result = torch.zeros(x1.size(0), device=x1.device, dtype=x1.dtype)
            # Each cycle contributes its amplitude to the diagonal
            for i in range(self.num_cycles):
                result = result + amplitudes[i]
            return result
        else:
            result = torch.zeros_like(distance)
            
            # Add contribution from each cycle
            for i in range(self.num_cycles):
                freq = frequencies[i]
                bandwidth = bandwidths[i]
                amplitude = amplitudes[i]
                
                # Periodic component with Gaussian envelope
                cos_term = torch.cos(2 * math.pi * freq * distance)
                exp_term = torch.exp(-2 * (math.pi * bandwidth * distance)**2)
                result = result + amplitude * cos_term * exp_term
            
            return result