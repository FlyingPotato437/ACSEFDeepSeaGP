import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# No changes needed for this file as it only imports standard libraries

def uk37_to_sst(uk37_values, method='muller1998'):
    """
    Convert UK'37 proxy values to sea surface temperature (SST).
    
    Parameters:
    -----------
    uk37_values : array-like
        UK'37 proxy values
    method : str, default='muller1998'
        Calibration method. Options:
        - 'muller1998': Global core-top calibration (Müller et al., 1998)
        - 'conte2006': Updated global calibration (Conte et al., 2006)
        - 'tierney2012': Bayesian calibration (Tierney & Tingley, 2012)
        
    Returns:
    --------
    sst : array-like
        Reconstructed SST in °C
    """
    # Convert to numpy array
    uk37_values = np.asarray(uk37_values)
    
    if method == 'muller1998':
        # Müller et al. (1998) calibration: UK'37 = 0.033 * SST + 0.044
        # Inverting for SST: SST = (UK'37 - 0.044) / 0.033
        sst = (uk37_values - 0.044) / 0.033
    
    elif method == 'conte2006':
        # Conte et al. (2006) calibration: UK'37 = 0.0331 * SST + 0.0439
        # Inverting for SST: SST = (UK'37 - 0.0439) / 0.0331
        sst = (uk37_values - 0.0439) / 0.0331
    
    elif method == 'tierney2012':
        # Approximate BAYSPLINE calibration (simplified)
        # Based on Tierney & Tingley (2012)
        # For a more accurate implementation, the full BAYSPLINE R code should be used
        
        # This is a simplified version that adds non-linearity at temperature extremes
        sst = np.zeros_like(uk37_values)
        
        # Split into temperature ranges
        low_idx = uk37_values < 0.3
        high_idx = uk37_values > 0.9
        mid_idx = ~(low_idx | high_idx)
        
        # Middle range: approximately linear
        sst[mid_idx] = (uk37_values[mid_idx] - 0.044) / 0.033
        
        # Low range: reduced sensitivity
        sst[low_idx] = (uk37_values[low_idx] - 0.05) / 0.025
        
        # High range: reduced sensitivity
        sst[high_idx] = (uk37_values[high_idx] - 0.084) / 0.031
    
    else:
        raise ValueError(f"Unknown calibration method: {method}. Expected one of: 'muller1998', 'conte2006', 'tierney2012'")
    
    return sst


def d18o_to_sst(d18o_values, d18o_water=0.0, method='shackleton1974'):
    """
    Convert δ18O proxy values to sea surface temperature (SST).
    
    Parameters:
    -----------
    d18o_values : array-like
        δ18O proxy values (‰)
    d18o_water : float or array-like, default=0.0
        δ18O of seawater (‰)
    method : str, default='shackleton1974'
        Calibration method. Options:
        - 'shackleton1974': Shackleton (1974) calibration
        - 'erez1983': Erez & Luz (1983) calibration
        - 'bemis1998': Bemis et al. (1998) calibration
        
    Returns:
    --------
    sst : array-like
        Reconstructed SST in °C
    """
    # Convert to numpy array
    d18o_values = np.asarray(d18o_values)
    
    if method == 'shackleton1974':
        # Shackleton (1974) calibration
        # T = 16.9 - 4.38 * (δ18O_calcite - δ18O_water) + 0.1 * (δ18O_calcite - δ18O_water)²
        delta = d18o_values - d18o_water
        sst = 16.9 - 4.38 * delta + 0.1 * delta**2
    
    elif method == 'erez1983':
        # Erez & Luz (1983) calibration
        # T = 17.0 - 4.52 * (δ18O_calcite - δ18O_water) + 0.03 * (δ18O_calcite - δ18O_water)²
        delta = d18o_values - d18o_water
        sst = 17.0 - 4.52 * delta + 0.03 * delta**2
    
    elif method == 'bemis1998':
        # Bemis et al. (1998) calibration - low light equation
        # T = 16.1 - 4.64 * (δ18O_calcite - δ18O_water)
        delta = d18o_values - d18o_water
        sst = 16.1 - 4.64 * delta
    
    else:
        raise ValueError(f"Unknown calibration method: {method}. Expected one of: 'shackleton1974', 'erez1983', 'bemis1998'")
    
    return sst


def mg_ca_to_sst(mg_ca_values, species='globigerinoides_ruber', method='anand2003'):
    """
    Convert Mg/Ca proxy values to sea surface temperature (SST).
    
    Parameters:
    -----------
    mg_ca_values : array-like
        Mg/Ca proxy values (mmol/mol)
    species : str, default='globigerinoides_ruber'
        Foraminiferal species
    method : str, default='anand2003'
        Calibration method. Options:
        - 'anand2003': Anand et al. (2003) calibration
        - 'elderfield2002': Elderfield & Ganssen (2000) calibration
        - 'dekens2002': Dekens et al. (2002) calibration
        
    Returns:
    --------
    sst : array-like
        Reconstructed SST in °C
    """
    # Convert to numpy array
    mg_ca_values = np.asarray(mg_ca_values)
    
    # Constants for different species and calibrations
    constants = {
        'anand2003': {
            'globigerinoides_ruber': {'A': 0.38, 'B': 0.09},
            'globigerina_bulloides': {'A': 0.794, 'B': 0.1},
            'neogloboquadrina_pachyderma': {'A': 0.406, 'B': 0.074},
        },
        'elderfield2002': {
            'globigerinoides_ruber': {'A': 0.3, 'B': 0.1},
            'globigerina_bulloides': {'A': 0.8, 'B': 0.1},
        },
        'dekens2002': {
            'globigerinoides_ruber': {'A': 0.38, 'B': 0.09, 'depth_correction': True},
        }
    }
    
    if method not in constants:
        raise ValueError(f"Unknown calibration method: {method}. Expected one of: {list(constants.keys())}")
    
    if species not in constants[method]:
        raise ValueError(f"Unknown species for method {method}: {species}. Expected one of: {list(constants[method].keys())}")
    
    # Get calibration constants
    cal = constants[method][species]
    
    # General form: T = (1/A) * ln(Mg/Ca / B)
    sst = (1.0 / cal['A']) * np.log(mg_ca_values / cal['B'])
    
    return sst


def proxy_to_sst(proxy_values, proxy_type='UK37', **kwargs):
    """
    Convert proxy values to sea surface temperature (SST).
    
    Parameters:
    -----------
    proxy_values : array-like
        Proxy values
    proxy_type : str, default='UK37'
        Type of proxy. Options: 'UK37', 'd18O', 'Mg_Ca'
    **kwargs :
        Additional parameters passed to the specific calibration function
        
    Returns:
    --------
    sst : array-like
        Reconstructed SST in °C
    """
    if proxy_type == 'UK37':
        return uk37_to_sst(proxy_values, **kwargs)
    elif proxy_type == 'd18O':
        return d18o_to_sst(proxy_values, **kwargs)
    elif proxy_type == 'Mg_Ca':
        return mg_ca_to_sst(proxy_values, **kwargs)
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}. Expected one of: 'UK37', 'd18O', 'Mg_Ca'")


def sst_to_proxy(sst_values, proxy_type='UK37', **kwargs):
    """
    Convert sea surface temperature (SST) to proxy values.
    
    Parameters:
    -----------
    sst_values : array-like
        SST values in °C
    proxy_type : str, default='UK37'
        Type of proxy. Options: 'UK37', 'd18O', 'Mg_Ca'
    **kwargs :
        Additional parameters passed to the specific calibration function
        
    Returns:
    --------
    proxy_values : array-like
        Proxy values
    """
    # Convert to numpy array
    sst_values = np.asarray(sst_values)
    
    if proxy_type == 'UK37':
        method = kwargs.get('method', 'muller1998')
        
        if method == 'muller1998':
            # Müller et al. (1998) calibration: UK'37 = 0.033 * SST + 0.044
            proxy_values = 0.033 * sst_values + 0.044
        elif method == 'conte2006':
            # Conte et al. (2006) calibration: UK'37 = 0.0331 * SST + 0.0439
            proxy_values = 0.0331 * sst_values + 0.0439
        else:
            raise ValueError(f"Unknown calibration method for {proxy_type}: {method}")
    
    elif proxy_type == 'd18O':
        method = kwargs.get('method', 'shackleton1974')
        d18o_water = kwargs.get('d18o_water', 0.0)
        
        if method == 'shackleton1974':
            # Simplified inverse of Shackleton (1974)
            # Original: T = 16.9 - 4.38 * (δ18O_calcite - δ18O_water) + 0.1 * (δ18O_calcite - δ18O_water)²
            # Simplified inverse: δ18O_calcite = δ18O_water - (T - 16.9) / 4.38
            proxy_values = d18o_water - (sst_values - 16.9) / 4.38
        elif method == 'bemis1998':
            # Bemis et al. (1998) - low light
            # T = 16.1 - 4.64 * (δ18O_calcite - δ18O_water)
            # Inverse: δ18O_calcite = δ18O_water - (T - 16.1) / 4.64
            proxy_values = d18o_water - (sst_values - 16.1) / 4.64
        else:
            raise ValueError(f"Unknown calibration method for {proxy_type}: {method}")
    
    elif proxy_type == 'Mg_Ca':
        method = kwargs.get('method', 'anand2003')
        species = kwargs.get('species', 'globigerinoides_ruber')
        
        # Constants for different species and calibrations
        constants = {
            'anand2003': {
                'globigerinoides_ruber': {'A': 0.38, 'B': 0.09},
                'globigerina_bulloides': {'A': 0.794, 'B': 0.1},
            },
        }
        
        if method not in constants:
            raise ValueError(f"Unknown calibration method: {method}")
        
        if species not in constants[method]:
            raise ValueError(f"Unknown species for method {method}: {species}")
        
        # Get calibration constants
        cal = constants[method][species]
        
        # General form: Mg/Ca = B * exp(A * T)
        proxy_values = cal['B'] * np.exp(cal['A'] * sst_values)
    
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}. Expected one of: 'UK37', 'd18O', 'Mg_Ca'")
    
    return proxy_values


def propagate_calibration_uncertainty(proxy_values, proxy_type='UK37', n_samples=1000, **kwargs):
    """
    Propagate calibration uncertainty through the proxy-to-SST conversion.
    
    Parameters:
    -----------
    proxy_values : array-like
        Proxy values
    proxy_type : str, default='UK37'
        Type of proxy. Options: 'UK37', 'd18O', 'Mg_Ca'
    n_samples : int, default=1000
        Number of Monte Carlo samples to generate
    **kwargs :
        Additional parameters passed to the specific calibration function
        
    Returns:
    --------
    sst_mean : array-like
        Mean SST values
    sst_std : array-like
        Standard deviation of SST values
    sst_samples : array-like, optional
        Monte Carlo samples if return_samples=True
    """
    # Convert to numpy array
    proxy_values = np.asarray(proxy_values)
    return_samples = kwargs.pop('return_samples', False)
    
    # Define calibration uncertainty for each proxy type
    if proxy_type == 'UK37':
        # Default calibration uncertainty for UK'37: σ of slope and intercept
        method = kwargs.get('method', 'muller1998')
        
        if method == 'muller1998':
            # Müller et al. (1998) calibration: UK'37 = 0.033 * SST + 0.044
            # With uncertainties: σ_slope = 0.001, σ_intercept = 0.016
            slope_mean, slope_std = 0.033, 0.001
            intercept_mean, intercept_std = 0.044, 0.016
            
            # Generate Monte Carlo samples of slope and intercept
            slope_samples = np.random.normal(slope_mean, slope_std, n_samples)
            intercept_samples = np.random.normal(intercept_mean, intercept_std, n_samples)
            
            # Generate SST samples
            sst_samples = np.zeros((len(proxy_values), n_samples))
            for i in range(n_samples):
                sst_samples[:, i] = (proxy_values - intercept_samples[i]) / slope_samples[i]
        
        else:
            raise ValueError(f"Uncertainty propagation not implemented for method: {method}")
    
    elif proxy_type == 'd18O':
        # Default calibration uncertainty for δ18O
        method = kwargs.get('method', 'shackleton1974')
        d18o_water = kwargs.get('d18o_water', 0.0)
        d18o_water_std = kwargs.get('d18o_water_std', 0.1)  # Uncertainty in seawater δ18O
        
        if method == 'shackleton1974':
            # Shackleton (1974) calibration with uncertainties
            a_mean, a_std = 16.9, 0.2
            b_mean, b_std = 4.38, 0.1
            c_mean, c_std = 0.1, 0.02
            
            # Generate Monte Carlo samples
            sst_samples = np.zeros((len(proxy_values), n_samples))
            
            for i in range(n_samples):
                a = np.random.normal(a_mean, a_std)
                b = np.random.normal(b_mean, b_std)
                c = np.random.normal(c_mean, c_std)
                d18o_w = np.random.normal(d18o_water, d18o_water_std)
                
                delta = proxy_values - d18o_w
                sst_samples[:, i] = a - b * delta + c * delta**2
        
        else:
            raise ValueError(f"Uncertainty propagation not implemented for method: {method}")
    
    elif proxy_type == 'Mg_Ca':
        # Default calibration uncertainty for Mg/Ca
        method = kwargs.get('method', 'anand2003')
        species = kwargs.get('species', 'globigerinoides_ruber')
        
        if method == 'anand2003' and species == 'globigerinoides_ruber':
            # Anand et al. (2003) calibration with uncertainties
            A_mean, A_std = 0.38, 0.02  # Slope
            B_mean, B_std = 0.09, 0.005  # Pre-exponential factor
            
            # Generate Monte Carlo samples
            sst_samples = np.zeros((len(proxy_values), n_samples))
            
            for i in range(n_samples):
                A = np.random.normal(A_mean, A_std)
                B = np.random.normal(B_mean, B_std)
                
                sst_samples[:, i] = (1.0 / A) * np.log(proxy_values / B)
        
        else:
            raise ValueError(f"Uncertainty propagation not implemented for method: {method} and species: {species}")
    
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}. Expected one of: 'UK37', 'd18O', 'Mg_Ca'")
    
    # Calculate mean and standard deviation
    sst_mean = np.mean(sst_samples, axis=1)
    sst_std = np.std(sst_samples, axis=1)
    
    if return_samples:
        return sst_mean, sst_std, sst_samples
    else:
        return sst_mean, sst_std


def plot_calibration_curve(proxy_type='UK37', sst_range=(0, 30), method=None, **kwargs):
    """
    Plot the calibration curve for a specific proxy.
    
    Parameters:
    -----------
    proxy_type : str, default='UK37'
        Type of proxy. Options: 'UK37', 'd18O', 'Mg_Ca'
    sst_range : tuple, default=(0, 30)
        Range of SST values to plot
    method : str, optional
        Calibration method. If None, use the default for each proxy type.
    **kwargs :
        Additional parameters passed to the specific calibration function
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Generate SST values
    sst_values = np.linspace(sst_range[0], sst_range[1], 100)
    
    # Set default method if not provided
    if method is None:
        if proxy_type == 'UK37':
            method = 'muller1998'
        elif proxy_type == 'd18O':
            method = 'shackleton1974'
        elif proxy_type == 'Mg_Ca':
            method = 'anand2003'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert SST to proxy
    proxy_values = sst_to_proxy(sst_values, proxy_type=proxy_type, method=method, **kwargs)
    
    # Plot calibration curve
    ax.plot(sst_values, proxy_values, 'b-', linewidth=2)
    
    # Add uncertainty if available
    try:
        # Generate proxy values with uncertainty
        _, proxy_std = propagate_calibration_uncertainty(
            sst_values, proxy_type=proxy_type, method=method, **kwargs
        )
        
        # Plot uncertainty envelope
        ax.fill_between(
            sst_values, 
            proxy_values - 2 * proxy_std, 
            proxy_values + 2 * proxy_std, 
            alpha=0.2, color='b'
        )
    except:
        pass
    
    # Add labels
    ax.set_xlabel('SST (°C)')
    
    if proxy_type == 'UK37':
        ax.set_ylabel('UK\'37')
        ax.set_title(f'UK\'37 Calibration Curve ({method})')
    elif proxy_type == 'd18O':
        ax.set_ylabel('δ18O (‰)')
        ax.set_title(f'δ18O Calibration Curve ({method})')
    elif proxy_type == 'Mg_Ca':
        ax.set_ylabel('Mg/Ca (mmol/mol)')
        ax.set_title(f'Mg/Ca Calibration Curve ({method})')
    
    ax.grid(True)
    
    return fig