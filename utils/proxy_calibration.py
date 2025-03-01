"""
Proxy Calibration and Weighting Utilities for Paleoclimate Reconstruction

This module provides functions for calibrating different proxy types (δ18O, UK37, Mg/Ca)
to Sea Surface Temperature (SST) and optimizing the weighting of multiple proxies
in multi-proxy reconstructions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import pandas as pd


# Default calibration parameters for common proxy types
DEFAULT_CALIBRATION_PARAMS = {
    'd18O': {
        'slope': -0.22,        # °C per ‰
        'intercept': 3.0,      # ‰
        'error_std': 0.1,      # ‰
        'inverse_slope': -4.54545  # ‰ per °C (1/slope)
    },
    'UK37': {
        'slope': 0.033,        # units per °C  
        'intercept': 0.044,    # units
        'error_std': 0.05,     # units
        'inverse_slope': 30.303   # °C per unit (1/slope)
    },
    'Mg_Ca': {
        'slope': 0.09,         # mmol/mol per °C
        'intercept': 0.3,      # mmol/mol
        'error_std': 0.1,      # mmol/mol
        'inverse_slope': 11.111   # °C per mmol/mol (1/slope)
    },
    'TEX86': {
        'slope': 0.015,        # units per °C
        'intercept': 0.10,     # units
        'error_std': 0.03,     # units
        'inverse_slope': 66.667   # °C per unit (1/slope)
    }
}


def proxy_to_sst(proxy_values: np.ndarray, proxy_type: str, 
                  calibration_params: Optional[Dict] = None) -> np.ndarray:
    """
    Convert proxy values to SST using calibration equations.
    
    Args:
        proxy_values: Proxy measurement values
        proxy_type: Type of proxy ('d18O', 'UK37', etc.)
        calibration_params: Optional custom calibration parameters
        
    Returns:
        SST values derived from the proxy
    """
    if calibration_params is None:
        if proxy_type not in DEFAULT_CALIBRATION_PARAMS:
            raise ValueError(f"Unknown proxy type: {proxy_type}")
        params = DEFAULT_CALIBRATION_PARAMS[proxy_type]
    else:
        params = calibration_params.get(proxy_type, DEFAULT_CALIBRATION_PARAMS.get(proxy_type))
    
    # Invert the calibration equation: SST = (proxy - intercept) / slope
    # or SST = (proxy - intercept) * inverse_slope
    sst = (proxy_values - params['intercept']) * params['inverse_slope']
    
    return sst


def sst_to_proxy(sst_values: np.ndarray, proxy_type: str,
                  calibration_params: Optional[Dict] = None) -> np.ndarray:
    """
    Convert SST values to proxy values using calibration equations.
    
    Args:
        sst_values: Temperature values in °C
        proxy_type: Type of proxy ('d18O', 'UK37', etc.)
        calibration_params: Optional custom calibration parameters
        
    Returns:
        Proxy values corresponding to the given SST
    """
    if calibration_params is None:
        if proxy_type not in DEFAULT_CALIBRATION_PARAMS:
            raise ValueError(f"Unknown proxy type: {proxy_type}")
        params = DEFAULT_CALIBRATION_PARAMS[proxy_type]
    else:
        params = calibration_params.get(proxy_type, DEFAULT_CALIBRATION_PARAMS.get(proxy_type))
    
    # Apply calibration equation: proxy = intercept + slope * SST
    proxy_values = params['intercept'] + params['slope'] * sst_values
    
    return proxy_values


def calculate_proxy_weights(proxy_types: List[str], proxy_data_dict: Dict,
                            calibration_params: Optional[Dict] = None,
                            weighting_method: str = 'balanced') -> Dict[str, float]:
    """
    Calculate optimal weights for combining proxies using enhanced weighting methodologies.
    
    This improved implementation offers multiple weighting approaches:
    - 'balanced': prevents any proxy from dominating by normalizing influence
    - 'error': weights based on calibration errors (inverse variance)
    - 'snr': weights based on signal-to-noise ratio estimates
    - 'equal': equal weighting for all proxies
    
    Args:
        proxy_types: List of proxy types
        proxy_data_dict: Dictionary with proxy data
        calibration_params: Optional custom calibration parameters
        weighting_method: Method for calculating weights
        
    Returns:
        Dictionary of weights for each proxy type
    """
    weights = {}
    
    # If custom calibration params not provided, use defaults
    if calibration_params is None:
        calibration_params = DEFAULT_CALIBRATION_PARAMS
    
    # Equal weighting
    if weighting_method == 'equal':
        n_proxies = len([pt for pt in proxy_types if pt in proxy_data_dict])
        for proxy_type in proxy_types:
            if proxy_type in proxy_data_dict:
                weights[proxy_type] = 1.0 / n_proxies
        
        return weights
    
    # Calculate initial weights based on error variance
    error_weights = {}
    total_error_weight = 0
    
    for proxy_type in proxy_types:
        if proxy_type in proxy_data_dict:
            # Get calibration error and convert to temperature units
            error_std = calibration_params[proxy_type]['error_std']
            inverse_slope = abs(calibration_params[proxy_type]['inverse_slope'])
            
            # Convert proxy error to temperature error
            temp_error = error_std * inverse_slope
            
            # Weight is inversely proportional to variance (1/σ²)
            error_weights[proxy_type] = 1 / (temp_error ** 2)
            total_error_weight += error_weights[proxy_type]
    
    # Balanced weighting: prevent δ18O or any proxy from dominating
    if weighting_method == 'balanced':
        # Normalize error weights
        if total_error_weight > 0:
            for proxy_type in error_weights:
                error_weights[proxy_type] /= total_error_weight
        
        # Apply balanced normalization
        max_weight = max(error_weights.values()) if error_weights else 1.0
        
        for proxy_type in error_weights:
            # Square root transformation to reduce influence of high-weight proxies
            weights[proxy_type] = np.sqrt(error_weights[proxy_type] / max_weight)
        
        # Renormalize
        total_weight = sum(weights.values())
        for proxy_type in weights:
            weights[proxy_type] /= total_weight
    
    # Error-based weighting (original method)
    elif weighting_method == 'error':
        # Normalize error weights
        if total_error_weight > 0:
            for proxy_type in error_weights:
                weights[proxy_type] = error_weights[proxy_type] / total_error_weight
    
    # Signal-to-noise ratio based weighting
    elif weighting_method == 'snr':
        snr_weights = {}
        total_snr_weight = 0
        
        for proxy_type in proxy_types:
            if proxy_type in proxy_data_dict:
                # Get proxy data and convert to SST
                proxy_data = proxy_data_dict[proxy_type]
                proxy_values = proxy_data['value']
                
                # Estimate signal amplitude (variance of the data)
                signal_var = np.var(proxy_values) if len(proxy_values) > 1 else 1.0
                
                # Noise from calibration
                error_std = calibration_params[proxy_type]['error_std']
                noise_var = error_std ** 2
                
                # Calculate SNR
                snr = signal_var / noise_var if noise_var > 0 else 1.0
                
                # Weight based on SNR
                snr_weights[proxy_type] = snr
                total_snr_weight += snr
        
        # Normalize
        if total_snr_weight > 0:
            for proxy_type in snr_weights:
                weights[proxy_type] = snr_weights[proxy_type] / total_snr_weight
    
    return weights


def combine_proxy_data(proxy_types: List[str], proxy_data_dict: Dict,
                     age_points: Optional[np.ndarray] = None,
                     weighting_method: str = 'balanced',
                     calibration_params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Combine multiple proxy records into a single temperature record
    using optimized weighting based on the selected method.
    
    Args:
        proxy_types: List of proxy types
        proxy_data_dict: Dictionary with proxy data
        age_points: Common age points for interpolation (optional)
        weighting_method: Method for calculating weights
        calibration_params: Optional custom calibration parameters
        
    Returns:
        combined_ages: Common age points
        combined_sst: Combined SST estimates
        weights: Dictionary of weights used for each proxy
    """
    # If no common age points provided, merge all available ages
    if age_points is None:
        age_points = np.array([])
        for proxy_type in proxy_types:
            if proxy_type in proxy_data_dict:
                age_points = np.union1d(age_points, proxy_data_dict[proxy_type]['age'])
        age_points = np.sort(age_points)
    
    # Calculate optimal weights for each proxy
    weights = calculate_proxy_weights(
        proxy_types, 
        proxy_data_dict,
        calibration_params=calibration_params,
        weighting_method=weighting_method
    )
    
    # Initialize arrays for combined data
    combined_sst = np.zeros_like(age_points, dtype=float)
    combined_count = np.zeros_like(age_points, dtype=float)
    
    # Combine temperature estimates from each proxy
    for proxy_type, weight in weights.items():
        if proxy_type in proxy_data_dict:
            proxy_data = proxy_data_dict[proxy_type]
            proxy_ages = proxy_data['age']
            proxy_values = proxy_data['value']
            
            # Convert proxy to SST
            proxy_sst = proxy_to_sst(proxy_values, proxy_type, calibration_params)
            
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
    
    return age_points, combined_sst, weights


class HeteroscedasticNoiseModel:
    """
    Heteroscedastic noise model for variable uncertainty in proxy data.
    
    This model allows observation-specific noise levels based on:
    1. Proxy type calibration uncertainty
    2. Age-dependent uncertainty scaling
    3. Measurement-specific error estimates
    """
    
    def __init__(
        self,
        proxy_types: List[str],
        calibration_params: Optional[Dict] = None,
        base_noise_level: float = 0.5,
        transition_scaling: float = 2.0,
        age_dependent_scaling: bool = True
    ):
        """
        Initialize the heteroscedastic noise model.
        
        Args:
            proxy_types: List of proxy types
            calibration_params: Calibration parameters for each proxy
            base_noise_level: Base noise level in °C
            transition_scaling: Scaling factor for noise in transition regions
            age_dependent_scaling: Whether to use age-dependent noise scaling
        """
        self.proxy_types = proxy_types
        self.calibration_params = calibration_params or DEFAULT_CALIBRATION_PARAMS
        self.base_noise_level = base_noise_level
        self.transition_scaling = transition_scaling
        self.age_dependent_scaling = age_dependent_scaling
        
        # Pre-compute proxy-specific base noise levels in temperature units
        self.proxy_base_noise = {}
        for proxy_type in self.proxy_types:
            if proxy_type in self.calibration_params:
                error_std = self.calibration_params[proxy_type]['error_std']
                inverse_slope = abs(self.calibration_params[proxy_type]['inverse_slope'])
                self.proxy_base_noise[proxy_type] = error_std * inverse_slope
    
    def get_noise_level(
        self, 
        ages: np.ndarray, 
        proxy_types: Union[str, List[str]],
        rate_of_change: Optional[np.ndarray] = None,
        measurement_errors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate observation-specific noise levels.
        
        Args:
            ages: Ages of the observations
            proxy_types: Proxy type(s) for each observation
            rate_of_change: Optional rate of change estimates at each point
            measurement_errors: Optional measurement-specific error estimates
            
        Returns:
            Array of noise standard deviations for each observation
        """
        n_points = len(ages)
        
        # Initialize with base noise level
        noise_levels = np.ones(n_points) * self.base_noise_level
        
        # If string, convert to list for consistent processing
        if isinstance(proxy_types, str):
            proxy_types = [proxy_types] * n_points
        
        # Apply proxy-specific base noise
        for i, proxy_type in enumerate(proxy_types):
            if proxy_type in self.proxy_base_noise:
                noise_levels[i] = self.proxy_base_noise[proxy_type]
        
        # Apply age-dependent scaling if enabled
        if self.age_dependent_scaling:
            # Simple model: older ages have higher uncertainty
            # Scale linearly from 1.0 to 1.5 times the base noise over 600kyr
            max_age = np.max(ages)
            if max_age > 0:
                # Scale factor from 1.0 to 1.5 based on age
                age_scaling = 1.0 + 0.5 * (ages / max_age)
                noise_levels *= age_scaling
        
        # Apply rate-of-change scaling for transitions if provided
        if rate_of_change is not None:
            # Increase noise in high rate of change regions
            transition_scaling = 1.0 + (self.transition_scaling - 1.0) * rate_of_change
            noise_levels *= transition_scaling
        
        # Apply measurement-specific errors if provided
        if measurement_errors is not None:
            # Combine in quadrature: σ_total² = σ_base² + σ_measurement²
            noise_levels = np.sqrt(noise_levels**2 + measurement_errors**2)
        
        return noise_levels
