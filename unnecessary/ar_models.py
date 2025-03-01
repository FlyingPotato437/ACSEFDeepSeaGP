import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# No changes needed for this file as it only imports standard libraries

class BaseARModel:
    """
    Base class for autoregressive models with Kalman filter.
    
    This serves as a parent class for AR1 and AR2 models, providing common
    functionality and interfaces.
    """
    def __init__(self, process_noise=0.1, observation_noise=0.1, optimize_params=True):
        """
        Initialize the AR model.
        
        Parameters:
        -----------
        process_noise : float, default=0.1
            Standard deviation of the process noise (σ_x)
        observation_noise : float, default=0.1
            Standard deviation of the observation noise (σ_y)
        optimize_params : bool, default=True
            Whether to optimize model parameters
        """
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.optimize_params = optimize_params
        self.params = None
        self.is_fitted = False
    
    def _init_params(self):
        """
        Initialize model parameters.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Subclasses must implement _init_params")
    
    def _log_likelihood(self, params, times, observations):
        """
        Compute the log-likelihood of the model.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Subclasses must implement _log_likelihood")
    
    def _neg_log_likelihood(self, params, times, observations):
        """Negative log-likelihood for optimization."""
        return -self._log_likelihood(params, times, observations)
    
    def _optimize_params(self, times, observations):
        """
        Optimize model parameters using L-BFGS-B.
        
        Parameters:
        -----------
        times : array-like of shape (n_samples,)
            Time points
        observations : array-like of shape (n_samples,)
            Observed values
        """
        # Define bounds for parameters
        bounds = self._get_param_bounds()
        
        # Run optimization
        result = optimize.minimize(
            lambda p: self._neg_log_likelihood(p, times, observations),
            self.params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update parameters
        self.params = result.x
    
    def _get_param_bounds(self):
        """
        Get bounds for parameter optimization.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Subclasses must implement _get_param_bounds")
    
    def _kalman_filter(self, times, observations):
        """
        Apply Kalman filter to estimate the state.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Subclasses must implement _kalman_filter")
    
    def fit(self, times, observations):
        """
        Fit the AR model to the data.
        
        Parameters:
        -----------
        times : array-like of shape (n_samples,)
            Time points
        observations : array-like of shape (n_samples,)
            Observed values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert inputs to numpy arrays
        times = np.asarray(times).ravel()
        observations = np.asarray(observations).ravel()
        
        # Sort by time if necessary
        if not np.all(np.diff(times) > 0):
            idx = np.argsort(times)
            times = times[idx]
            observations = observations[idx]
        
        # Initialize parameters
        self._init_params()
        
        # Optimize parameters if requested
        if self.optimize_params:
            self._optimize_params(times, observations)
        
        # Store the data
        self.times = times
        self.observations = observations
        
        # Calculate state estimates
        self._states, self._state_covs = self._kalman_filter(times, observations)
        
        self.is_fitted = True
        return self
    
    def predict(self, times, return_std=False):
        """
        Predict using the AR model.
        
        Parameters:
        -----------
        times : array-like of shape (n_samples,)
            Time points to predict
        return_std : bool, default=False
            If True, return the standard deviation of the prediction
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        y_std : array-like of shape (n_samples,), optional
            Standard deviation of the prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert times to numpy array
        times = np.asarray(times).ravel()
        
        # Interpolate state estimates
        y_pred = np.interp(times, self.times, self._states[:, 0])
        
        if return_std:
            # Interpolate state covariances
            if len(self._states.shape) > 1:
                # For multi-dimensional states (AR2), use the first state
                y_var = np.interp(times, self.times, self._state_covs[:, 0, 0])
            else:
                # For scalar states (AR1)
                y_var = np.interp(times, self.times, self._state_covs[:, 0, 0])
            
            # Add observation noise
            y_var += self.observation_noise**2
            
            return y_pred, np.sqrt(y_var)
        
        return y_pred
    
    def score(self, times, observations):
        """
        Compute R² score.
        
        Parameters:
        -----------
        times : array-like of shape (n_samples,)
            Time points
        observations : array-like of shape (n_samples,)
            True values
            
        Returns:
        --------
        r2 : float
            R² score
        """
        y_pred = self.predict(times)
        u = ((observations - y_pred) ** 2).sum()
        v = ((observations - observations.mean()) ** 2).sum()
        return 1 - u / v
    
    def plot(self, times=None, observations=None, figsize=(12, 6)):
        """
        Plot the AR model prediction with uncertainty.
        
        Parameters:
        -----------
        times : array-like of shape (n_samples,), optional
            Time points to predict (default: use training times)
        observations : array-like of shape (n_samples,), optional
            True values (default: use training observations)
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before plotting")
        
        # Use training data if not provided
        if times is None:
            times = self.times
        if observations is None:
            observations = self.observations
        
        # Ensure arrays are 1D
        times = np.asarray(times).ravel()
        
        # Sort points for plotting
        sorted_idx = np.argsort(times)
        times_sorted = times[sorted_idx]
        
        # Generate prediction times
        pred_times = np.linspace(np.min(times), np.max(times), 1000)
        
        # Predict with uncertainty
        y_pred, y_std = self.predict(pred_times, return_std=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data
        ax.scatter(self.times, self.observations, c='k', label='Training data')
        
        # Plot test data if provided
        if observations is not None and times is not self.times:
            ax.scatter(times, observations, c='r', label='Test data')
        
        # Plot prediction
        ax.plot(pred_times, y_pred, 'b-', label='Prediction')
        
        # Plot uncertainty
        ax.fill_between(pred_times, 
                       y_pred - 1.96 * y_std, 
                       y_pred + 1.96 * y_std, 
                       alpha=0.2, color='b', label='95% confidence interval')
        
        # Add labels and legend
        ax.set_xlabel('Time (kyr)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{self.__class__.__name__} Paleoclimate Reconstruction')
        ax.legend()
        ax.grid(True)
        
        return ax


class AR1Model(BaseARModel):
    """
    First-order autoregressive model with Kalman filter for paleoclimate reconstruction.
    
    Mathematical form:
        x_t = α * x_{t-1} + w_t, where w_t ~ N(0, σ_x²)
        y_t = x_t + v_t, where v_t ~ N(0, σ_y²)
    """
    
    def _init_params(self):
        """Initialize model parameters."""
        # Format: [alpha, log_process_noise, log_observation_noise]
        self.params = np.array([0.9, np.log(self.process_noise), np.log(self.observation_noise)])
    
    def _get_param_bounds(self):
        """Get bounds for parameter optimization."""
        # Bounds for [alpha, log_process_noise, log_observation_noise]
        return [
            (-0.99, 0.99),      # alpha (stability requirement: |alpha| < 1)
            (-6.0, 2.0),        # log_process_noise
            (-6.0, 2.0)         # log_observation_noise
        ]
    
    def _log_likelihood(self, params, times, observations):
        """
        Compute the log-likelihood of the AR1 model.
        
        Parameters:
        -----------
        params : array-like
            Model parameters: [alpha, log_process_noise, log_observation_noise]
        times : array-like
            Time points
        observations : array-like
            Observed values
            
        Returns:
        --------
        ll : float
            Log-likelihood
        """
        alpha, log_sigma_x, log_sigma_y = params
        
        # Get noise standard deviations
        sigma_x = np.exp(log_sigma_x)
        sigma_y = np.exp(log_sigma_y)
        
        # Apply Kalman filter
        states, state_covs = self._kalman_filter_with_params(times, observations, alpha, sigma_x, sigma_y)
        
        # Compute log-likelihood
        n = len(observations)
        ll = 0.0
        
        for t in range(n):
            # Innovation (prediction error)
            v_t = observations[t] - states[t, 0]
            
            # Innovation variance
            S_t = state_covs[t, 0, 0] + sigma_y**2
            
            # Log-likelihood contribution
            ll += -0.5 * (np.log(2 * np.pi * S_t) + v_t**2 / S_t)
        
        return ll
    
    def _kalman_filter(self, times, observations):
        """
        Apply Kalman filter to estimate the state.
        
        Parameters:
        -----------
        times : array-like
            Time points
        observations : array-like
            Observed values
            
        Returns:
        --------
        states : array-like
            Estimated states
        state_covs : array-like
            State covariances
        """
        alpha = self.params[0]
        sigma_x = np.exp(self.params[1])
        sigma_y = np.exp(self.params[2])
        
        return self._kalman_filter_with_params(times, observations, alpha, sigma_x, sigma_y)
    
    def _kalman_filter_with_params(self, times, observations, alpha, sigma_x, sigma_y):
        """
        Apply Kalman filter with specified parameters.
        
        Parameters:
        -----------
        times : array-like
            Time points
        observations : array-like
            Observed values
        alpha : float
            AR coefficient
        sigma_x : float
            Process noise standard deviation
        sigma_y : float
            Observation noise standard deviation
            
        Returns:
        --------
        states : array-like
            Estimated states
        state_covs : array-like
            State covariances
        """
        n = len(observations)
        
        # Compute time differences for irregular sampling
        dt = np.diff(times)
        dt = np.insert(dt, 0, 1.0)  # Assume unit time step for first point
        
        # Initialize state and covariance matrices
        states = np.zeros((n, 1))
        state_covs = np.zeros((n, 1, 1))
        
        # Initial state estimate (use first observation)
        states[0, 0] = observations[0]
        state_covs[0, 0, 0] = sigma_y**2
        
        # Run Kalman filter
        for t in range(1, n):
            # Time update (prediction)
            # For irregularly sampled data, adjust the process model
            a_t = alpha ** dt[t]
            q_t = (1 - a_t**2) * sigma_x**2  # Adjusted process noise
            
            # Predicted state
            x_pred = a_t * states[t-1, 0]
            
            # Predicted covariance
            P_pred = a_t**2 * state_covs[t-1, 0, 0] + q_t
            
            # Measurement update (correction)
            # Kalman gain
            K = P_pred / (P_pred + sigma_y**2)
            
            # Updated state
            states[t, 0] = x_pred + K * (observations[t] - x_pred)
            
            # Updated covariance
            state_covs[t, 0, 0] = (1 - K) * P_pred
        
        return states, state_covs
    
    @property
    def alpha(self):
        """Get the AR coefficient."""
        return self.params[0]
    
    @property
    def process_noise_std(self):
        """Get the process noise standard deviation."""
        return np.exp(self.params[1])
    
    @property
    def observation_noise_std(self):
        """Get the observation noise standard deviation."""
        return np.exp(self.params[2])
    
    def get_params(self):
        """Get model parameters in a readable format."""
        return {
            'alpha': self.alpha,
            'process_noise_std': self.process_noise_std,
            'observation_noise_std': self.observation_noise_std
        }
    
    def compute_spectral_density(self, frequencies):
        """
        Compute the spectral density of the AR1 process.
        
        Parameters:
        -----------
        frequencies : array-like
            Frequencies to compute the spectral density at (cycles/kyr)
            
        Returns:
        --------
        S : array-like
            Spectral density at the requested frequencies
        """
        # AR1 spectral density: S(f) = σ_x^2 / (1 + α^2 - 2α*cos(2πf))
        alpha = self.alpha
        sigma_x = self.process_noise_std
        
        S = sigma_x**2 / (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * frequencies))
        
        return S
    
    def plot_spectral_density(self, f_min=0.001, f_max=0.1, n_points=1000, figsize=(12, 6)):
        """
        Plot the spectral density of the AR1 process.
        
        Parameters:
        -----------
        f_min : float, default=0.001
            Minimum frequency (cycles/kyr)
        f_max : float, default=0.1
            Maximum frequency (cycles/kyr)
        n_points : int, default=1000
            Number of points to compute
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before plotting spectral density")
        
        # Generate frequencies
        frequencies = np.linspace(f_min, f_max, n_points)
        
        # Compute spectral density
        S = self.compute_spectral_density(frequencies)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectral density
        ax.plot(frequencies, S, 'b-')
        
        # Add Milankovitch cycle markers
        ax.axvline(1.0/100.0, color='r', linestyle='--', label='Eccentricity (100 kyr)')
        ax.axvline(1.0/41.0, color='g', linestyle='--', label='Obliquity (41 kyr)')
        ax.axvline(1.0/23.0, color='m', linestyle='--', label='Precession (23 kyr)')
        
        # Add labels and legend
        ax.set_xlabel('Frequency (cycles/kyr)')
        ax.set_ylabel('Spectral density')
        ax.set_title('Spectral Density of AR1 Process')
        ax.legend()
        ax.grid(True)
        
        # Convert x-axis to period
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        period_ticks = [10, 20, 23, 41, 50, 100, 200, 500]
        period_ticks = [p for p in period_ticks if 1.0/p >= f_min and 1.0/p <= f_max]
        ax2.set_xticks([1.0/p for p in period_ticks])
        ax2.set_xticklabels([str(p) for p in period_ticks])
        ax2.set_xlabel('Period (kyr)')
        
        return ax


class AR2Model(BaseARModel):
    """
    Second-order autoregressive model with Kalman filter for paleoclimate reconstruction.
    
    Mathematical form:
        x_t = α * x_{t-1} + β * x_{t-2} + w_t, where w_t ~ N(0, σ_x²)
        y_t = x_t + v_t, where v_t ~ N(0, σ_y²)
    """
    
    def _init_params(self):
        """Initialize model parameters."""
        # Format: [alpha, beta, log_process_noise, log_observation_noise]
        self.params = np.array([0.9, 0.0, np.log(self.process_noise), np.log(self.observation_noise)])
    
    def _get_param_bounds(self):
        """Get bounds for parameter optimization."""
        # Bounds for [alpha, beta, log_process_noise, log_observation_noise]
        # Stability conditions for AR2: |alpha| < 1, |beta| < 1, alpha + beta < 1, beta - alpha < 1
        return [
            (-0.99, 0.99),      # alpha
            (-0.99, 0.99),      # beta
            (-6.0, 2.0),        # log_process_noise
            (-6.0, 2.0)         # log_observation_noise
        ]
    
    def _check_stability(self, alpha, beta):
        """
        Check if AR2 parameters satisfy stability conditions.
        
        Parameters:
        -----------
        alpha : float
            First AR coefficient
        beta : float
            Second AR coefficient
            
        Returns:
        --------
        stable : bool
            True if parameters satisfy stability conditions
        """
        return (np.abs(alpha) < 1 and
                np.abs(beta) < 1 and
                alpha + beta < 1 and
                beta - alpha < 1)
    
    def _log_likelihood(self, params, times, observations):
        """
        Compute the log-likelihood of the AR2 model.
        
        Parameters:
        -----------
        params : array-like
            Model parameters: [alpha, beta, log_process_noise, log_observation_noise]
        times : array-like
            Time points
        observations : array-like
            Observed values
            
        Returns:
        --------
        ll : float
            Log-likelihood
        """
        alpha, beta, log_sigma_x, log_sigma_y = params
        
        # Check stability conditions
        if not self._check_stability(alpha, beta):
            return -np.inf
        
        # Get noise standard deviations
        sigma_x = np.exp(log_sigma_x)
        sigma_y = np.exp(log_sigma_y)
        
        # Apply Kalman filter
        states, state_covs = self._kalman_filter_with_params(times, observations, alpha, beta, sigma_x, sigma_y)
        
        # Compute log-likelihood
        n = len(observations)
        ll = 0.0
        
        for t in range(n):
            # Innovation (prediction error)
            v_t = observations[t] - states[t, 0]
            
            # Innovation variance
            S_t = state_covs[t, 0, 0] + sigma_y**2
            
            # Log-likelihood contribution
            ll += -0.5 * (np.log(2 * np.pi * S_t) + v_t**2 / S_t)
        
        return ll
    
    def _kalman_filter(self, times, observations):
        """
        Apply Kalman filter to estimate the state.
        
        Parameters:
        -----------
        times : array-like
            Time points
        observations : array-like
            Observed values
            
        Returns:
        --------
        states : array-like
            Estimated states
        state_covs : array-like
            State covariances
        """
        alpha = self.params[0]
        beta = self.params[1]
        sigma_x = np.exp(self.params[2])
        sigma_y = np.exp(self.params[3])
        
        return self._kalman_filter_with_params(times, observations, alpha, beta, sigma_x, sigma_y)
    
    def _kalman_filter_with_params(self, times, observations, alpha, beta, sigma_x, sigma_y):
        """
        Apply Kalman filter with specified parameters.
        
        Parameters:
        -----------
        times : array-like
            Time points
        observations : array-like
            Observed values
        alpha : float
            First AR coefficient
        beta : float
            Second AR coefficient
        sigma_x : float
            Process noise standard deviation
        sigma_y : float
            Observation noise standard deviation
            
        Returns:
        --------
        states : array-like
            Estimated states
        state_covs : array-like
            State covariances
        """
        n = len(observations)
        
        # Compute time differences for irregular sampling
        dt = np.diff(times)
        dt = np.insert(dt, 0, 1.0)  # Assume unit time step for first point
        
        # Initialize state and covariance matrices
        # State vector: [x_t, x_{t-1}]
        states = np.zeros((n, 2))
        state_covs = np.zeros((n, 2, 2))
        
        # Initialize state transition matrix
        A = np.array([[alpha, beta],
                      [1.0, 0.0]])
        
        # Initialize process noise covariance
        Q = np.array([[sigma_x**2, 0.0],
                      [0.0, 0.0]])
        
        # Initialize observation matrix
        H = np.array([1.0, 0.0])
        
        # Initial state estimate (use first observation for both states)
        states[0, 0] = observations[0]
        states[0, 1] = observations[0]
        
        # Initial state covariance (high uncertainty for x_{t-1})
        state_covs[0, 0, 0] = sigma_y**2
        state_covs[0, 1, 1] = 2 * sigma_y**2
        
        # Run Kalman filter
        for t in range(1, n):
            # Time update (prediction)
            # For irregularly sampled data, we need to adjust the process model
            # This is a simplification for AR2; a more complex approach would
            # be to use a continuous-time state-space model
            
            # Predicted state
            x_pred = A @ states[t-1]
            
            # Predicted covariance
            P_pred = A @ state_covs[t-1] @ A.T + Q
            
            # Measurement update (correction)
            # Innovation (prediction error)
            v_t = observations[t] - H @ x_pred
            
            # Innovation covariance
            S_t = H @ P_pred @ H.T + sigma_y**2
            
            # Kalman gain
            K = P_pred @ H.T / S_t
            
            # Updated state
            states[t] = x_pred + K * v_t
            
            # Updated covariance
            state_covs[t] = P_pred - np.outer(K, H @ P_pred)
        
        return states, state_covs
    
    @property
    def alpha(self):
        """Get the first AR coefficient."""
        return self.params[0]
    
    @property
    def beta(self):
        """Get the second AR coefficient."""
        return self.params[1]
    
    @property
    def process_noise_std(self):
        """Get the process noise standard deviation."""
        return np.exp(self.params[2])
    
    @property
    def observation_noise_std(self):
        """Get the observation noise standard deviation."""
        return np.exp(self.params[3])
    
    def get_params(self):
        """Get model parameters in a readable format."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'process_noise_std': self.process_noise_std,
            'observation_noise_std': self.observation_noise_std
        }
    
    def compute_spectral_density(self, frequencies):
        """
        Compute the spectral density of the AR2 process.
        
        Parameters:
        -----------
        frequencies : array-like
            Frequencies to compute the spectral density at (cycles/kyr)
            
        Returns:
        --------
        S : array-like
            Spectral density at the requested frequencies
        """
        # AR2 spectral density: 
        # S(f) = σ_x^2 / |1 - α*exp(-i2πf) - β*exp(-i2π2f)|^2
        alpha = self.alpha
        beta = self.beta
        sigma_x = self.process_noise_std
        
        # Compute frequency response
        omega = 2 * np.pi * frequencies
        z = np.exp(-1j * omega)
        H = 1.0 / (1.0 - alpha * z - beta * z**2)
        
        # Compute spectral density
        S = sigma_x**2 * np.abs(H)**2
        
        return S
    
    def plot_spectral_density(self, f_min=0.001, f_max=0.1, n_points=1000, figsize=(12, 6)):
        """
        Plot the spectral density of the AR2 process.
        
        Parameters:
        -----------
        f_min : float, default=0.001
            Minimum frequency (cycles/kyr)
        f_max : float, default=0.1
            Maximum frequency (cycles/kyr)
        n_points : int, default=1000
            Number of points to compute
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before plotting spectral density")
        
        # Generate frequencies
        frequencies = np.linspace(f_min, f_max, n_points)
        
        # Compute spectral density
        S = self.compute_spectral_density(frequencies)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectral density
        ax.plot(frequencies, S, 'b-')
        
        # Add Milankovitch cycle markers
        ax.axvline(1.0/100.0, color='r', linestyle='--', label='Eccentricity (100 kyr)')
        ax.axvline(1.0/41.0, color='g', linestyle='--', label='Obliquity (41 kyr)')
        ax.axvline(1.0/23.0, color='m', linestyle='--', label='Precession (23 kyr)')
        
        # Add labels and legend
        ax.set_xlabel('Frequency (cycles/kyr)')
        ax.set_ylabel('Spectral density')
        ax.set_title('Spectral Density of AR2 Process')
        ax.legend()
        ax.grid(True)
        
        # Convert x-axis to period
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        period_ticks = [10, 20, 23, 41, 50, 100, 200, 500]
        period_ticks = [p for p in period_ticks if 1.0/p >= f_min and 1.0/p <= f_max]
        ax2.set_xticks([1.0/p for p in period_ticks])
        ax2.set_xticklabels([str(p) for p in period_ticks])
        ax2.set_xlabel('Period (kyr)')
        
        return ax