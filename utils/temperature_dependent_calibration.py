"""
Temperature-Dependent Proxy Calibration for Paleoclimate Reconstructions

This module provides functions for handling temperature-dependent calibration of
paleoclimate proxies, which accounts for the fact that proxy-temperature relationships
can vary under different climate regimes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Temperature-dependent calibration functions

def nonlinear_d18o_calibration(
    temperature: np.ndarray,
    a: float = -0.22,
    b: float = 3.0,
    c: float = 0.02,
    threshold: float = 10.0
) -> np.ndarray:
    """
    Nonlinear temperature-dependent δ18O calibration.
    
    Models the δ18O-temperature relationship with different slopes in
    cold vs. warm regimes.
    
    Args:
        temperature: Temperature values in °C
        a: Linear coefficient (cold regime)
        b: Intercept
        c: Nonlinearity coefficient
        threshold: Temperature threshold separating regimes
        
    Returns:
        Calibrated δ18O values
    """
    # Calculate regime indicator (smooth transition between regimes)
    regime_factor = 1 / (1 + np.exp(-(temperature - threshold)))
    
    # Calculate δ18O values with temperature-dependent slope
    slope = a - c * regime_factor  # Slope becomes more negative in warm regime
    d18o = b + slope * temperature
    
    return d18o


def temperature_to_d18o(
    temperature: np.ndarray,
    params: Dict[str, float] = None
) -> np.ndarray:
    """
    Convert temperature to δ18O using temperature-dependent calibration.
    
    Args:
        temperature: Temperature values in °C
        params: Dictionary of calibration parameters
        
    Returns:
        δ18O values
    """
    if params is None:
        params = {
            'a': -0.22,      # Linear coefficient (cold regime)
            'b': 3.0,        # Intercept
            'c': 0.02,       # Nonlinearity coefficient
            'threshold': 10.0  # Temperature threshold
        }
    
    return nonlinear_d18o_calibration(
        temperature,
        a=params.get('a', -0.22),
        b=params.get('b', 3.0),
        c=params.get('c', 0.02),
        threshold=params.get('threshold', 10.0)
    )


def d18o_to_temperature(
    d18o: np.ndarray,
    params: Dict[str, float] = None,
    method: str = 'iterative',
    max_iter: int = 10,
    tolerance: float = 1e-4,
    initial_temp: float = 15.0
) -> np.ndarray:
    """
    Convert δ18O to temperature using temperature-dependent calibration.
    
    Since the calibration is nonlinear and temperature-dependent, this
    function uses either iterative refinement or a lookup table approach.
    
    Args:
        d18o: δ18O values
        params: Dictionary of calibration parameters
        method: Conversion method ('iterative', 'lookup', 'approx')
        max_iter: Maximum iterations for iterative method
        tolerance: Convergence tolerance for iterative method
        initial_temp: Initial temperature guess
        
    Returns:
        Temperature values in °C
    """
    if params is None:
        params = {
            'a': -0.22,      # Linear coefficient (cold regime)
            'b': 3.0,        # Intercept
            'c': 0.02,       # Nonlinearity coefficient
            'threshold': 10.0  # Temperature threshold
        }
    
    a = params.get('a', -0.22)
    b = params.get('b', 3.0)
    c = params.get('c', 0.02)
    threshold = params.get('threshold', 10.0)
    
    # Convert to numpy array if needed
    d18o_np = np.asarray(d18o)
    
    if method == 'iterative':
        # Iterative refinement method
        temperatures = np.ones_like(d18o_np) * initial_temp
        
        for i in range(max_iter):
            # Calculate regime factor for current temperature estimate
            regime_factor = 1 / (1 + np.exp(-(temperatures - threshold)))
            
            # Calculate effective slope for current temperature
            slope = a - c * regime_factor
            
            # Update temperature estimate using current slope
            new_temps = (d18o_np - b) / slope
            
            # Check convergence
            if np.max(np.abs(new_temps - temperatures)) < tolerance:
                break
                
            temperatures = new_temps
        
        return temperatures
    
    elif method == 'lookup':
        # Lookup table approach
        # Generate a lookup table of temperature → δ18O
        temp_range = np.linspace(-5, 30, 1000)
        d18o_lookup = nonlinear_d18o_calibration(temp_range, a, b, c, threshold)
        
        # Interpolate to find temperature for each δ18O value
        temperatures = np.interp(d18o_np, d18o_lookup[::-1], temp_range[::-1])
        
        return temperatures
    
    elif method == 'approx':
        # Approximate method using piecewise linear approximation
        # Cold regime (standard linear equation)
        cold_temps = (d18o_np - b) / a
        
        # Warm regime (adjusted slope)
        warm_slope = a - c
        warm_temps = (d18o_np - b) / warm_slope
        
        # Smooth transition between regimes
        weights = 1 / (1 + np.exp(-(cold_temps - threshold) * 0.5))
        temperatures = (1 - weights) * cold_temps + weights * warm_temps
        
        return temperatures
    
    else:
        raise ValueError(f"Unsupported method: {method}")


# Temperature-dependent UK'37 calibration

def nonlinear_uk37_calibration(
    temperature: np.ndarray,
    a: float = 0.033,
    b: float = 0.044,
    c: float = 0.0008,
    threshold: float = 24.0
) -> np.ndarray:
    """
    Nonlinear temperature-dependent UK'37 calibration.
    
    Models the UK'37-temperature relationship with different slopes in
    cold vs. warm regimes.
    
    Args:
        temperature: Temperature values in °C
        a: Linear coefficient (cold regime)
        b: Intercept
        c: Nonlinearity coefficient
        threshold: Temperature threshold separating regimes
        
    Returns:
        Calibrated UK'37 values
    """
    # Calculate regime indicator (smooth transition between regimes)
    regime_factor = 1 / (1 + np.exp(-(temperature - threshold)))
    
    # Calculate UK'37 values with temperature-dependent slope
    slope = a + c * regime_factor  # Slope increases in warm regime
    uk37 = b + slope * temperature
    
    # Apply saturation effect at high temperatures
    saturation = 1.0 - np.exp(-(temperature / 40.0)**2)
    uk37 = uk37 * saturation
    
    # Ensure values are within physical limits [0, 1]
    uk37 = np.clip(uk37, 0.0, 1.0)
    
    return uk37


def temperature_to_uk37(
    temperature: np.ndarray,
    params: Dict[str, float] = None
) -> np.ndarray:
    """
    Convert temperature to UK'37 using temperature-dependent calibration.
    
    Args:
        temperature: Temperature values in °C
        params: Dictionary of calibration parameters
        
    Returns:
        UK'37 values
    """
    if params is None:
        params = {
            'a': 0.033,      # Linear coefficient (cold regime)
            'b': 0.044,      # Intercept
            'c': 0.0008,     # Nonlinearity coefficient
            'threshold': 24.0  # Temperature threshold
        }
    
    return nonlinear_uk37_calibration(
        temperature,
        a=params.get('a', 0.033),
        b=params.get('b', 0.044),
        c=params.get('c', 0.0008),
        threshold=params.get('threshold', 24.0)
    )


def uk37_to_temperature(
    uk37: np.ndarray,
    params: Dict[str, float] = None,
    method: str = 'lookup'
) -> np.ndarray:
    """
    Convert UK'37 to temperature using temperature-dependent calibration.
    
    Args:
        uk37: UK'37 values
        params: Dictionary of calibration parameters
        method: Conversion method ('lookup', 'approx', 'iterative')
        
    Returns:
        Temperature values in °C
    """
    if params is None:
        params = {
            'a': 0.033,      # Linear coefficient (cold regime)
            'b': 0.044,      # Intercept
            'c': 0.0008,     # Nonlinearity coefficient
            'threshold': 24.0  # Temperature threshold
        }
    
    a = params.get('a', 0.033)
    b = params.get('b', 0.044)
    c = params.get('c', 0.0008)
    threshold = params.get('threshold', 24.0)
    
    # Convert to numpy array if needed
    uk37_np = np.asarray(uk37)
    
    # Ensure values are within physical limits [0, 1]
    uk37_np = np.clip(uk37_np, 0.0, 1.0)
    
    if method == 'lookup':
        # Lookup table approach
        # Generate a lookup table of temperature → UK'37
        temp_range = np.linspace(0, 40, 1000)
        uk37_lookup = nonlinear_uk37_calibration(temp_range, a, b, c, threshold)
        
        # Interpolate to find temperature for each UK'37 value
        temperatures = np.interp(uk37_np, uk37_lookup, temp_range)
        
        return temperatures
    
    elif method == 'approx':
        # Approximate method using linear approximation
        # This is less accurate but faster
        base_temps = (uk37_np - b) / a
        
        # Apply correction for nonlinearity and saturation
        correction = np.zeros_like(base_temps)
        warm_mask = base_temps > threshold
        
        if np.any(warm_mask):
            # Correction for warm regime
            correction[warm_mask] = c * (base_temps[warm_mask] - threshold)**2 / (2*a)
        
        temperatures = base_temps - correction
        
        return temperatures
    
    elif method == 'iterative':
        # Iterative refinement method (similar to d18o_to_temperature)
        # Implementation omitted for brevity, follows same pattern
        raise NotImplementedError("Iterative method not implemented for UK'37")
    
    else:
        raise ValueError(f"Unsupported method: {method}")


# Temperature-dependent Mg/Ca calibration

def exponential_mgca_calibration(
    temperature: np.ndarray,
    a: float = 0.3,
    b: float = 0.09,
    c: float = 0.01,
    threshold: float = 18.0
) -> np.ndarray:
    """
    Exponential temperature-dependent Mg/Ca calibration.
    
    Models the Mg/Ca-temperature relationship with exponential form
    and parameters that vary based on temperature regime.
    
    Args:
        temperature: Temperature values in °C
        a: Pre-exponential factor
        b: Exponential coefficient (cold regime)
        c: Additional coefficient for warm regime
        threshold: Temperature threshold separating regimes
        
    Returns:
        Calibrated Mg/Ca values
    """
    # Calculate regime indicator (smooth transition between regimes)
    regime_factor = 1 / (1 + np.exp(-(temperature - threshold)))
    
    # Calculate effective exponential coefficient
    exp_coef = b + c * regime_factor
    
    # Calculate Mg/Ca with temperature-dependent exponential
    mgca = a * np.exp(exp_coef * temperature)
    
    return mgca
"""
d18O Calibration Functions for Ice Volume Estimation

This module provides specialized functions for calibrating δ18O records to global
ice volume estimates, including uncertainty quantification and nonlinear effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


def d18o_to_ice_volume(d18o_values: np.ndarray, 
                     params: Optional[Dict] = None,
                     method: str = 'linear') -> np.ndarray:
    """
    Convert δ18O values to global ice volume estimates.
    
    Args:
        d18o_values: δ18O values in ‰
        params: Calibration parameters dictionary
        method: Calibration method ('linear', 'nonlinear', 'shackleton')
        
    Returns:
        Global ice volume estimates (%)
    """
    # Default parameters if not provided
    if params is None:
        params = {
            'modern_value': 3.2,    # Modern δ18O value
            'glacial_value': 5.0,   # Last Glacial Maximum δ18O value
            'ice_effect': 0.011,    # ‰ δ18O per meter sea level change
            'temperature_effect': 0.25,  # ‰ δ18O per °C
            'full_glacial_sealevel': -120.0  # meters
        }
    
    # Extract parameters
    modern = params.get('modern_value', 3.2)
    glacial = params.get('glacial_value', 5.0)
    
    if method == 'linear':
        # Simple linear scaling: 0% at modern value, 100% at glacial maximum
        ice_volume = (d18o_values - modern) / (glacial - modern) * 100
        
    elif method == 'nonlinear':
        # Nonlinear calibration accounting for temperature effects
        # Waelbroeck et al. (2002) style calibration
        ice_effect = params.get('ice_effect', 0.011)
        temp_effect = params.get('temperature_effect', 0.25)
        
        # First estimate the sea level component
        # Convert to sea level change in meters
        full_glacial_sealevel = params.get('full_glacial_sealevel', -120.0)
        
        # Estimate sea level, accounting for temperature effects
        sea_level = ((d18o_values - modern) - temp_effect) / ice_effect
        
        # Convert sea level to ice volume percentage
        ice_volume = sea_level / full_glacial_sealevel * 100
        
    elif method == 'shackleton':
        # Shackleton (2000) calibration
        # Special nonlinear calibration for deep ocean δ18O
        glacial_adj = 1.0  # Adjust based on the record
        
        # Shackleton's equation 1 (simplified)
        # Ice volume = a * (δ18O - b)^2 + c * (δ18O - b)
        a = 30.0
        b = modern
        c = 70.0
        
        ice_volume = a * np.power(d18o_values - b, 2) + c * (d18o_values - b)
        
        # Scale to 0-100%
        ice_volume = ice_volume * (100.0 / ice_volume.max())
        
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return ice_volume


def get_ice_volume_uncertainty(d18o_values: np.ndarray, 
                             measurement_error: Union[float, np.ndarray] = 0.1,
                             params: Optional[Dict] = None,
                             method: str = 'linear') -> np.ndarray:
    """
    Calculate uncertainty in ice volume estimates based on δ18O measurement errors.
    
    Args:
        d18o_values: δ18O values in ‰
        measurement_error: Measurement error(s) for δ18O values
        params: Calibration parameters dictionary
        method: Calibration method ('linear', 'nonlinear', 'shackleton')
        
    Returns:
        Uncertainty in ice volume estimates (%)
    """
    # Default parameters if not provided
    if params is None:
        params = {
            'modern_value': 3.2,    # Modern δ18O value
            'glacial_value': 5.0,   # Last Glacial Maximum δ18O value
            'calibration_error': 0.2  # Additional calibration uncertainty
        }
    
    # Extract parameters
    modern = params.get('modern_value', 3.2)
    glacial = params.get('glacial_value', 5.0)
    calibration_error = params.get('calibration_error', 0.2)
    
    # Convert to array if scalar
    if np.isscalar(measurement_error):
        measurement_error = np.ones_like(d18o_values) * measurement_error
    
    if method == 'linear':
        # Simple propagation of uncertainty for linear calibration
        # σ_ice = 100 * σ_d18o / (glacial - modern)
        ice_uncertainty = 100.0 * measurement_error / (glacial - modern)
        
        # Add calibration uncertainty in quadrature
        ice_uncertainty = np.sqrt(ice_uncertainty**2 + calibration_error**2)
        
    elif method == 'nonlinear':
        # For nonlinear calibration, we use numerical approximation
        # Calculate ice volume at d18o +/- error
        d18o_plus = d18o_values + measurement_error
        d18o_minus = d18o_values - measurement_error
        
        ice_plus = d18o_to_ice_volume(d18o_plus, params, 'nonlinear')
        ice_minus = d18o_to_ice_volume(d18o_minus, params, 'nonlinear')
        
        # Take half the range as the uncertainty
        ice_uncertainty = np.abs(ice_plus - ice_minus) / 2.0
        
        # Add calibration uncertainty in quadrature
        ice_uncertainty = np.sqrt(ice_uncertainty**2 + calibration_error**2)
        
    elif method == 'shackleton':
        # For Shackleton calibration, use numerical approximation similarly
        d18o_plus = d18o_values + measurement_error
        d18o_minus = d18o_values - measurement_error
        
        ice_plus = d18o_to_ice_volume(d18o_plus, params, 'shackleton')
        ice_minus = d18o_to_ice_volume(d18o_minus, params, 'shackleton')
        
        # Take half the range as the uncertainty
        ice_uncertainty = np.abs(ice_plus - ice_minus) / 2.0
        
        # Add calibration uncertainty in quadrature
        ice_uncertainty = np.sqrt(ice_uncertainty**2 + calibration_error**2)
        
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return ice_uncertainty


def ice_volume_to_d18o(ice_volume: np.ndarray,
                     params: Optional[Dict] = None,
                     method: str = 'linear') -> np.ndarray:
    """
    Convert ice volume estimates back to δ18O values.
    
    Args:
        ice_volume: Global ice volume estimates (%)
        params: Calibration parameters dictionary
        method: Calibration method ('linear', 'nonlinear', 'shackleton')
        
    Returns:
        Estimated δ18O values in ‰
    """
    # Default parameters if not provided
    if params is None:
        params = {
            'modern_value': 3.2,    # Modern δ18O value
            'glacial_value': 5.0,   # Last Glacial Maximum δ18O value
            'ice_effect': 0.011,    # ‰ δ18O per meter sea level change
            'temperature_effect': 0.25,  # ‰ δ18O per °C
            'full_glacial_sealevel': -120.0  # meters
        }
    
    # Extract parameters
    modern = params.get('modern_value', 3.2)
    glacial = params.get('glacial_value', 5.0)
    
    if method == 'linear':
        # Simple linear scaling
        d18o = modern + (ice_volume / 100.0) * (glacial - modern)
        
    elif method == 'nonlinear':
        # Nonlinear calibration, inverse of d18o_to_ice_volume
        ice_effect = params.get('ice_effect', 0.011)
        temp_effect = params.get('temperature_effect', 0.25)
        full_glacial_sealevel = params.get('full_glacial_sealevel', -120.0)
        
        # Convert ice volume to sea level
        sea_level = (ice_volume / 100.0) * full_glacial_sealevel
        
        # Calculate δ18O from sea level and temperature components
        d18o = modern + (sea_level * ice_effect) + temp_effect
        
    elif method == 'shackleton':
        # Shackleton calibration is more complex to invert
        # For simplicity, use a lookup table approach
        
        # Generate a lookup table
        test_d18o = np.linspace(modern, glacial, 1000)
        test_ice = d18o_to_ice_volume(test_d18o, params, 'shackleton')
        
        # Interpolate to find δ18O for each ice volume value
        d18o = np.interp(ice_volume, test_ice, test_d18o)
        
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return d18o


def correct_d18o_for_temperature(d18o_values: np.ndarray,
                               temperature_effect: np.ndarray,
                               params: Optional[Dict] = None) -> np.ndarray:
    """
    Correct δ18O values for temperature effects to isolate the ice volume signal.
    
    Args:
        d18o_values: Raw δ18O values in ‰
        temperature_effect: Temperature contribution to δ18O, derived from
                         independent temperature proxies (Mg/Ca, UK37, etc.)
        params: Calibration parameters
        
    Returns:
        Ice volume component of δ18O
    """
    # Default parameters if not provided
    if params is None:
        params = {
            'modern_value': 3.2,        # Modern δ18O value
            'temp_sensitivity': 0.25    # ‰ δ18O per °C
        }
    
    # Extract parameters
    modern = params.get('modern_value', 3.2)
    temp_sensitivity = params.get('temp_sensitivity', 0.25)
    
    # Calculate correction
    # Subtract the temperature component to isolate ice volume signal
    d18o_ice_volume = d18o_values - (temperature_effect * temp_sensitivity)
    
    return d18o_ice_volume

def temperature_to_mgca(
    temperature: np.ndarray,
    params: Dict[str, float] = None
) -> np.ndarray:
    """
    Convert temperature to Mg/Ca using temperature-dependent calibration.
    
    Args:
        temperature: Temperature values in °C
        params: Dictionary of calibration parameters
        
    Returns:
        Mg/Ca values
    """
    if params is None:
        params = {
            'a': 0.3,        # Pre-exponential factor
            'b': 0.09,       # Exponential coefficient (cold regime)
            'c': 0.01,       # Additional coefficient for warm regime
            'threshold': 18.0  # Temperature threshold
        }
    
    return exponential_mgca_calibration(
        temperature,
        a=params.get('a', 0.3),
        b=params.get('b', 0.09),
        c=params.get('c', 0.01),
        threshold=params.get('threshold', 18.0)
    )


def mgca_to_temperature(
    mgca: np.ndarray,
    params: Dict[str, float] = None,
    method: str = 'analytic'
) -> np.ndarray:
    """
    Convert Mg/Ca to temperature using temperature-dependent calibration.
    
    Args:
        mgca: Mg/Ca values
        params: Dictionary of calibration parameters
        method: Conversion method ('analytic', 'lookup', 'iterative')
        
    Returns:
        Temperature values in °C
    """
    if params is None:
        params = {
            'a': 0.3,        # Pre-exponential factor
            'b': 0.09,       # Exponential coefficient (cold regime)
            'c': 0.01,       # Additional coefficient for warm regime
            'threshold': 18.0  # Temperature threshold
        }
    
    a = params.get('a', 0.3)
    b = params.get('b', 0.09)
    c = params.get('c', 0.01)
    threshold = params.get('threshold', 18.0)
    
    # Convert to numpy array if needed
    mgca_np = np.asarray(mgca)
    
    if method == 'analytic':
        # For simple exponential, we can solve analytically
        # For the cold regime: Mg/Ca = a * exp(b * T)
        # So T = ln(Mg/Ca / a) / b
        temperatures = np.log(mgca_np / a) / b
        
        # Apply correction for warm regime
        # This is an approximation
        warm_mask = temperatures > threshold
        if np.any(warm_mask):
            # Iteratively refine warm temperatures
            warm_temps = temperatures[warm_mask]
            for _ in range(3):  # Few iterations are usually enough
                regime_factor = 1 / (1 + np.exp(-(warm_temps - threshold)))
                exp_coef = b + c * regime_factor
                warm_temps = np.log(mgca_np[warm_mask] / a) / exp_coef
            temperatures[warm_mask] = warm_temps
        
        return temperatures
    
    elif method == 'lookup':
        # Lookup table approach
        # Generate a lookup table of temperature → Mg/Ca
        temp_range = np.linspace(0, 40, 1000)
        mgca_lookup = exponential_mgca_calibration(temp_range, a, b, c, threshold)
        
        # Interpolate to find temperature for each Mg/Ca value
        temperatures = np.interp(mgca_np, mgca_lookup, temp_range)
        
        return temperatures
    
    elif method == 'iterative':
        # Iterative method could be implemented similar to other proxies
        raise NotImplementedError("Iterative method not implemented for Mg/Ca")
    
    else:
        raise ValueError(f"Unsupported method: {method}")


# Utility functions

def fit_nonlinear_calibration(
    temperature: np.ndarray,
    proxy_values: np.ndarray,
    proxy_type: str,
    initial_params: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Fit temperature-dependent calibration model to laboratory data.
    
    Args:
        temperature: Measured temperature values
        proxy_values: Measured proxy values
        proxy_type: Type of proxy ('d18O', 'UK37', 'Mg_Ca')
        initial_params: Initial parameter estimates
        
    Returns:
        Dictionary of fitted parameters
    """
    # Select calibration function based on proxy type
    if proxy_type.lower() == 'd18o':
        if initial_params is None:
            initial_params = {'a': -0.22, 'b': 3.0, 'c': 0.02, 'threshold': 10.0}
        
        def fit_func(t, a, b, c, threshold):
            return nonlinear_d18o_calibration(t, a, b, c, threshold)
        
        p0 = [initial_params.get('a', -0.22),
              initial_params.get('b', 3.0),
              initial_params.get('c', 0.02),
              initial_params.get('threshold', 10.0)]
    
    elif proxy_type.lower() in ['uk37', 'uk\'37']:
        if initial_params is None:
            initial_params = {'a': 0.033, 'b': 0.044, 'c': 0.0008, 'threshold': 24.0}
        
        def fit_func(t, a, b, c, threshold):
            return nonlinear_uk37_calibration(t, a, b, c, threshold)
        
        p0 = [initial_params.get('a', 0.033),
              initial_params.get('b', 0.044),
              initial_params.get('c', 0.0008),
              initial_params.get('threshold', 24.0)]
    
    elif proxy_type.lower() in ['mgca', 'mg_ca', 'mg/ca']:
        if initial_params is None:
            initial_params = {'a': 0.3, 'b': 0.09, 'c': 0.01, 'threshold': 18.0}
        
        def fit_func(t, a, b, c, threshold):
            return exponential_mgca_calibration(t, a, b, c, threshold)
        
        p0 = [initial_params.get('a', 0.3),
              initial_params.get('b', 0.09),
              initial_params.get('c', 0.01),
              initial_params.get('threshold', 18.0)]
    
    else:
        raise ValueError(f"Unsupported proxy type: {proxy_type}")
    
    try:
        # Fit the model
        popt, _ = curve_fit(fit_func, temperature, proxy_values, p0=p0)
        
        # Return fitted parameters
        if proxy_type.lower() == 'd18o':
            return {'a': popt[0], 'b': popt[1], 'c': popt[2], 'threshold': popt[3]}
        elif proxy_type.lower() in ['uk37', 'uk\'37']:
            return {'a': popt[0], 'b': popt[1], 'c': popt[2], 'threshold': popt[3]}
        else:  # Mg/Ca
            return {'a': popt[0], 'b': popt[1], 'c': popt[2], 'threshold': popt[3]}
    
    except RuntimeError:
        # If fitting fails, return initial parameters
        print(f"Warning: Calibration fitting failed for {proxy_type}. Using initial parameters.")
        return initial_params


def plot_calibration_curve(
    proxy_type: str,
    params: Dict[str, float],
    temperature_range: Optional[Tuple[float, float]] = None,
    comparison_data: Optional[Dict[str, np.ndarray]] = None,
    figure_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve for a given proxy type and parameters.
    
    Args:
        proxy_type: Type of proxy ('d18O', 'UK37', 'Mg_Ca')
        params: Dictionary of calibration parameters
        temperature_range: Optional temperature range to plot
        comparison_data: Optional data for comparison
        figure_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Determine temperature range if not provided
    if temperature_range is None:
        if proxy_type.lower() == 'd18o':
            temperature_range = (-5, 30)
        elif proxy_type.lower() in ['uk37', 'uk\'37']:
            temperature_range = (0, 35)
        else:  # Mg/Ca
            temperature_range = (0, 30)
    
    # Generate temperature range for plotting
    temperatures = np.linspace(temperature_range[0], temperature_range[1], 500)
    
    # Calculate proxy values using appropriate function
    if proxy_type.lower() == 'd18o':
        proxy_values = temperature_to_d18o(temperatures, params)
    elif proxy_type.lower() in ['uk37', 'uk\'37']:
        proxy_values = temperature_to_uk37(temperatures, params)
    else:  # Mg/Ca
        proxy_values = temperature_to_mgca(temperatures, params)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot calibration curve
    ax.plot(temperatures, proxy_values, 'b-', linewidth=2, label='Temperature-Dependent Model')
    
    # Plot standard calibration if parameters available
    if proxy_type.lower() == 'd18o':
        # Standard linear calibration for comparison
        standard_d18o = params['b'] + params['a'] * temperatures
        ax.plot(temperatures, standard_d18o, 'k--', linewidth=1.5, label='Standard Linear Model')
    
    elif proxy_type.lower() in ['uk37', 'uk\'37']:
        # Standard linear calibration for comparison
        standard_uk37 = params['b'] + params['a'] * temperatures
        standard_uk37 = np.clip(standard_uk37, 0.0, 1.0)
        ax.plot(temperatures, standard_uk37, 'k--', linewidth=1.5, label='Standard Linear Model')
    
    elif proxy_type.lower() in ['mgca', 'mg_ca', 'mg/ca']:
        # Standard exponential calibration for comparison
        standard_mgca = params['a'] * np.exp(params['b'] * temperatures)
        ax.plot(temperatures, standard_mgca, 'k--', linewidth=1.5, label='Standard Exponential Model')
    
    # Plot comparison data if provided
    if comparison_data is not None and 'temperature' in comparison_data and 'proxy' in comparison_data:
        ax.scatter(comparison_data['temperature'], comparison_data['proxy'], 
                  c='r', s=50, alpha=0.7, label='Calibration Data')
    
    # Add vertical line at threshold
    if 'threshold' in params:
        ax.axvline(params['threshold'], color='gray', linestyle=':', alpha=0.7)
        ax.text(params['threshold'], ax.get_ylim()[0], f' Threshold: {params["threshold"]}°C',
               ha='left', va='bottom', color='gray')
    
    # Add labels and legend
    ax.set_xlabel('Temperature (°C)')
    
    if proxy_type.lower() == 'd18o':
        ax.set_ylabel('δ¹⁸O (‰)')
        ax.set_title('Temperature-Dependent δ¹⁸O Calibration')
    elif proxy_type.lower() in ['uk37', 'uk\'37']:
        ax.set_ylabel('UK\'37')
        ax.set_title('Temperature-Dependent UK\'37 Calibration')
    else:  # Mg/Ca
        ax.set_ylabel('Mg/Ca (mmol/mol)')
        ax.set_title('Temperature-Dependent Mg/Ca Calibration')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotate parameters
    param_text = f"Parameters:\n"
    for key, value in params.items():
        param_text += f"{key} = {value:.4f}\n"
    
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Save figure if path provided
    if figure_path:
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    
    return fig