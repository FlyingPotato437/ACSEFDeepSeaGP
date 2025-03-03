import numpy as np
import matplotlib.pyplot as plt

# No changes needed for this file as it only imports standard libraries

class SyntheticPaleoData:
    """
    Generate synthetic paleoclimate data with Milankovitch orbital cycles.
    
    This class creates synthetic sea surface temperature (SST) time series
    with known orbital components and optional proxy calibrations.
    """
    
    def __init__(self, start_time=0, end_time=800, noise_level=0.5, random_seed=None):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        start_time : float, default=0
            Start time (kyr)
        end_time : float, default=800
            End time (kyr)
        noise_level : float, default=0.5
            Standard deviation of Gaussian noise added to the signal
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.start_time = start_time
        self.end_time = end_time
        self.noise_level = noise_level
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Default cycle parameters
        self.cycles = {
            'eccentricity': {'period': 100.0, 'amplitude': 2.0, 'phase': 0.0},
            'obliquity': {'period': 41.0, 'amplitude': 1.0, 'phase': np.pi/4},
            'precession': {'period': 23.0, 'amplitude': 0.5, 'phase': np.pi/3}
        }
        
        # Default trend parameters
        self.trend = {'slope': 0.0, 'intercept': 15.0}  # deg C
        
        # Default proxy calibration
        self.proxy_calibrations = {
            'UK37': {'slope': 0.033, 'intercept': 0.044, 'error': 0.5},  # Müller et al., 1998
            'Mg_Ca': {'slope': 0.38, 'intercept': 0.0, 'error': 0.7},
            'd18O': {'slope': -0.228, 'intercept': 0.0, 'error': 0.3}     # Shackleton, 1974
        }
    
    def set_cycle_parameters(self, cycle_name, period=None, amplitude=None, phase=None):
        """
        Set parameters for a specific orbital cycle.
        
        Parameters:
        -----------
        cycle_name : str
            Name of the cycle ('eccentricity', 'obliquity', or 'precession')
        period : float, optional
            Cycle period (kyr)
        amplitude : float, optional
            Cycle amplitude (°C)
        phase : float, optional
            Cycle phase (radians)
            
        Returns:
        --------
        self : object
            Returns self
        """
        if cycle_name not in self.cycles:
            raise ValueError(f"Unknown cycle: {cycle_name}. Expected one of: {list(self.cycles.keys())}")
        
        if period is not None:
            self.cycles[cycle_name]['period'] = period
        if amplitude is not None:
            self.cycles[cycle_name]['amplitude'] = amplitude
        if phase is not None:
            self.cycles[cycle_name]['phase'] = phase
        
        return self
    
    def set_trend(self, slope=None, intercept=None):
        """
        Set parameters for the linear trend.
        
        Parameters:
        -----------
        slope : float, optional
            Slope of the linear trend (°C/kyr)
        intercept : float, optional
            Intercept of the linear trend (°C)
            
        Returns:
        --------
        self : object
            Returns self
        """
        if slope is not None:
            self.trend['slope'] = slope
        if intercept is not None:
            self.trend['intercept'] = intercept
        
        return self
    
    def set_proxy_calibration(self, proxy_name, slope=None, intercept=None, error=None):
        """
        Set parameters for a proxy calibration.
        
        Parameters:
        -----------
        proxy_name : str
            Name of the proxy ('UK37', 'Mg_Ca', or 'd18O')
        slope : float, optional
            Slope of the calibration
        intercept : float, optional
            Intercept of the calibration
        error : float, optional
            Proxy calibration error
            
        Returns:
        --------
        self : object
            Returns self
        """
        if proxy_name not in self.proxy_calibrations:
            raise ValueError(f"Unknown proxy: {proxy_name}. Expected one of: {list(self.proxy_calibrations.keys())}")
        
        if slope is not None:
            self.proxy_calibrations[proxy_name]['slope'] = slope
        if intercept is not None:
            self.proxy_calibrations[proxy_name]['intercept'] = intercept
        if error is not None:
            self.proxy_calibrations[proxy_name]['error'] = error
        
        return self
    
    def generate_time_points(self, n_points=100, regular=True, min_spacing=1.0):
        """
        Generate time points for the synthetic data.
        
        Parameters:
        -----------
        n_points : int, default=100
            Number of time points
        regular : bool, default=True
            Whether to generate regularly spaced points
        min_spacing : float, default=1.0
            Minimum spacing between time points (kyr) for irregular sampling
            
        Returns:
        --------
        time_points : array-like
            Generated time points
        """
        if regular:
            # Regular sampling
            return np.linspace(self.start_time, self.end_time, n_points)
        else:
            # Irregular sampling
            time_range = self.end_time - self.start_time
            max_points = int(time_range / min_spacing)
            
            if n_points > max_points:
                raise ValueError(f"Too many points ({n_points}) for the given time range and minimum spacing. Maximum allowed: {max_points}")
            
            # Generate random time points
            time_points = self.start_time + time_range * np.random.random(n_points)
            
            # Sort time points
            time_points.sort()
            
            # Ensure minimum spacing
            for i in range(1, len(time_points)):
                if time_points[i] - time_points[i-1] < min_spacing:
                    time_points[i] = time_points[i-1] + min_spacing
            
            return time_points
    
    def generate_orbital_component(self, time_points, cycle_name):
        """
        Generate a single orbital cycle component.
        
        Parameters:
        -----------
        time_points : array-like
            Time points (kyr)
        cycle_name : str
            Name of the cycle ('eccentricity', 'obliquity', or 'precession')
            
        Returns:
        --------
        component : array-like
            Orbital cycle component
        """
        if cycle_name not in self.cycles:
            raise ValueError(f"Unknown cycle: {cycle_name}. Expected one of: {list(self.cycles.keys())}")
        
        cycle = self.cycles[cycle_name]
        period = cycle['period']
        amplitude = cycle['amplitude']
        phase = cycle['phase']
        
        # Generate cycle component
        frequency = 1.0 / period
        return amplitude * np.sin(2 * np.pi * frequency * time_points + phase)
    
    def generate_true_sst(self, time_points):
        """
        Generate true sea surface temperature (SST) with known orbital components.
        
        Parameters:
        -----------
        time_points : array-like
            Time points (kyr)
            
        Returns:
        --------
        sst : array-like
            Sea surface temperature (°C)
        """
        # Linear trend
        trend = self.trend['slope'] * time_points + self.trend['intercept']
        
        # Sum of orbital components
        orbital = np.zeros_like(time_points)
        for cycle_name in self.cycles:
            orbital += self.generate_orbital_component(time_points, cycle_name)
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, len(time_points))
        
        # Combine components
        sst = trend + orbital + noise
        
        return sst
    
    def generate_proxy(self, time_points, proxy_name, age_error=0.0):
        """
        Generate proxy measurements with calibration errors.
        
        Parameters:
        -----------
        time_points : array-like
            Time points (kyr)
        proxy_name : str
            Name of the proxy ('UK37', 'Mg_Ca', or 'd18O')
        age_error : float, default=0.0
            Standard deviation of Gaussian noise added to time points (kyr)
            
        Returns:
        --------
        proxy_times : array-like
            Time points with age errors (kyr)
        proxy_values : array-like
            Proxy measurements
        """
        if proxy_name not in self.proxy_calibrations:
            raise ValueError(f"Unknown proxy: {proxy_name}. Expected one of: {list(self.proxy_calibrations.keys())}")
        
        # Get true SST
        sst = self.generate_true_sst(time_points)
        
        # Apply proxy calibration
        calibration = self.proxy_calibrations[proxy_name]
        slope = calibration['slope']
        intercept = calibration['intercept']
        error = calibration['error']
        
        # Apply calibration
        if proxy_name == 'd18O':
            # For d18O, higher SST means lower d18O (negative slope)
            proxy_values = intercept + slope * sst
        else:
            # For UK37 and Mg/Ca, higher SST means higher proxy value
            proxy_values = intercept + slope * sst
        
        # Add calibration error
        proxy_values += np.random.normal(0, error, len(time_points))
        
        # Add age error if requested
        if age_error > 0:
            proxy_times = time_points + np.random.normal(0, age_error, len(time_points))
            # Ensure times remain in order
            idx = np.argsort(proxy_times)
            proxy_times = proxy_times[idx]
            proxy_values = proxy_values[idx]
        else:
            proxy_times = time_points
        
        return proxy_times, proxy_values
    
    def generate_dataset(self, n_points=100, regular=True, min_spacing=1.0, 
                         proxies=None, age_error=0.0):
        """
        Generate a complete synthetic dataset with true SST and proxy measurements.
        
        Parameters:
        -----------
        n_points : int, default=100
            Number of time points
        regular : bool, default=True
            Whether to generate regularly spaced points
        min_spacing : float, default=1.0
            Minimum spacing between time points (kyr) for irregular sampling
        proxies : list, optional
            List of proxies to generate. Default: ['UK37']
        age_error : float, default=0.0
            Standard deviation of Gaussian noise added to time points (kyr)
            
        Returns:
        --------
        dataset : dict
            Dictionary containing time points, true SST, and proxy measurements
        """
        # Set default proxies
        if proxies is None:
            proxies = ['UK37']
        
        # Generate time points
        time_points = self.generate_time_points(n_points, regular, min_spacing)
        
        # Generate true SST
        true_sst = self.generate_true_sst(time_points)
        
        # Generate proxy measurements
        proxy_data = {}
        for proxy in proxies:
            proxy_times, proxy_values = self.generate_proxy(time_points, proxy, age_error)
            proxy_data[proxy] = {'times': proxy_times, 'values': proxy_values}
        
        # Create dataset
        dataset = {
            'time_points': time_points,
            'true_sst': true_sst,
            'proxy_data': proxy_data
        }
        
        return dataset
    
    def plot_dataset(self, dataset, figsize=(12, 10)):
        """
        Plot the synthetic dataset.
        
        Parameters:
        -----------
        dataset : dict
            Dataset generated by generate_dataset
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Get data
        time_points = dataset['time_points']
        true_sst = dataset['true_sst']
        proxy_data = dataset['proxy_data']
        
        # Count the number of subplots needed
        n_plots = 1 + len(proxy_data)
        
        # Create figure
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Ensure axes is a list even if there's only one subplot
        if n_plots == 1:
            axes = [axes]
        
        # Plot true SST
        axes[0].plot(time_points, true_sst, 'k-', label='True SST')
        axes[0].set_ylabel('SST (°C)')
        axes[0].set_title('Synthetic Paleoclimate Data')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot proxy data
        for i, (proxy_name, data) in enumerate(proxy_data.items(), 1):
            axes[i].plot(data['times'], data['values'], 'o-', label=f'{proxy_name}')
            axes[i].set_ylabel(f'{proxy_name}')
            axes[i].legend()
            axes[i].grid(True)
        
        # Set common x-axis label
        axes[-1].set_xlabel('Time (kyr)')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def extract_cycles(self, sst, time_points):
        """
        Extract Milankovitch cycles from the SST data using FFT.
        
        Parameters:
        -----------
        sst : array-like
            Sea surface temperature (°C)
        time_points : array-like
            Time points (kyr)
            
        Returns:
        --------
        frequencies : array-like
            Frequencies (cycles/kyr)
        power : array-like
            Power spectral density
        """
        # Check if time points are regularly spaced
        dt = np.diff(time_points)
        if not np.allclose(dt, dt[0], rtol=1e-3):
            raise ValueError("Time points must be regularly spaced for FFT")
        
        # Compute sample spacing
        sampling_rate = 1.0 / dt[0]  # samples per kyr
        
        # Compute FFT
        n = len(sst)
        fft = np.fft.rfft(sst)
        
        # Compute power spectral density
        power = np.abs(fft) ** 2
        
        # Compute frequencies
        frequencies = np.fft.rfftfreq(n, d=1.0/sampling_rate)
        
        return frequencies, power
    
    def plot_power_spectrum(self, sst, time_points, figsize=(12, 6)):
        """
        Plot the power spectrum of the SST data.
        
        Parameters:
        -----------
        sst : array-like
            Sea surface temperature (°C)
        time_points : array-like
            Time points (kyr)
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Extract cycles
        frequencies, power = self.extract_cycles(sst, time_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot power spectrum
        ax.semilogy(frequencies, power, 'b-')
        
        # Add Milankovitch cycle markers
        for cycle_name, cycle in self.cycles.items():
            frequency = 1.0 / cycle['period']
            ax.axvline(frequency, color='r', linestyle='--', 
                      label=f"{cycle_name.capitalize()} ({cycle['period']} kyr)")
        
        # Add labels and legend
        ax.set_xlabel('Frequency (cycles/kyr)')
        ax.set_ylabel('Power')
        ax.set_title('Power Spectrum of SST Data')
        ax.legend()
        ax.grid(True)
        
        # Convert x-axis to period
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([1.0/p for p in [10, 20, 23, 41, 50, 100, 200, 500]])
        ax2.set_xticklabels(['10', '20', '23', '41', '50', '100', '200', '500'])
        ax2.set_xlabel('Period (kyr)')
        
        return fig