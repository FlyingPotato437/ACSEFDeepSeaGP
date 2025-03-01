"""
gp_models.py - Gaussian Process models for paleoclimate reconstruction

This module implements physics-informed Gaussian Process models that can
explicitly model Milankovitch cycles (100kyr, 41kyr, and 23kyr periods)
using specialized kernels.
"""

import numpy as np
import scipy.optimize as optimize
from scipy.linalg import cholesky, cho_solve
import matplotlib.pyplot as plt

class PhysicsInformedGP:
    """
    Physics-informed Gaussian Process for paleoclimate reconstruction.
    
    This model extends standard GPs with kernels that explicitly model
    Milankovitch orbital cycles (100kyr, 41kyr, and 23kyr periods).
    """
    
    def __init__(self, kernel='milankovitch', normalize=True, optimize_hyperparams=True):
        """
        Initialize the GP model.
        
        Parameters:
        -----------
        kernel : str, default='milankovitch'
            Kernel type to use. Options:
            - 'milankovitch': Composite kernel with orbital components
            - 'rbf': Standard RBF kernel 
            - 'matern52': Matérn 5/2 kernel
        normalize : bool, default=True
            Whether to normalize the input data
        optimize_hyperparams : bool, default=True
            Whether to optimize kernel hyperparameters
        """
        self.kernel_type = kernel
        self.normalize = normalize
        self.optimize_hyperparams = optimize_hyperparams
        self.hyperparams = None
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.K = None
        self.L = None
        self.alpha = None
        
        # Initialize default hyperparameters
        self._init_hyperparams()
        
    def _init_hyperparams(self):
        """Initialize default hyperparameters based on kernel type."""
        if self.kernel_type == 'milankovitch':
            # Format: [amplitude, length_scale, noise, 
            #          eccentricity_amp, obliquity_amp, precession_amp]
            self.hyperparams = np.array([1.0, 50.0, 0.1, 0.5, 0.3, 0.2])
        elif self.kernel_type == 'rbf':
            # Format: [amplitude, length_scale, noise]
            self.hyperparams = np.array([1.0, 50.0, 0.1])
        elif self.kernel_type == 'matern52':
            # Format: [amplitude, length_scale, noise]
            self.hyperparams = np.array([1.0, 50.0, 0.1])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _kernel(self, X1, X2, hyperparams):
        """
        Compute the kernel matrix between X1 and X2.
        
        Parameters:
        -----------
        X1, X2 : array-like
            Input time points
        hyperparams : array-like
            Kernel hyperparameters
            
        Returns:
        --------
        K : array-like
            Kernel matrix
        """
        if self.kernel_type == 'milankovitch':
            return self._milankovitch_kernel(X1, X2, hyperparams)
        elif self.kernel_type == 'rbf':
            return self._rbf_kernel(X1, X2, hyperparams)
        elif self.kernel_type == 'matern52':
            return self._matern52_kernel(X1, X2, hyperparams)
    
    def _milankovitch_kernel(self, X1, X2, hyperparams):
        """
        Milankovitch physics-informed kernel.
        
        This kernel combines:
        1. A background RBF kernel for overall trends
        2. Periodic components for the 3 Milankovitch cycles:
           - Eccentricity (100 kyr)
           - Obliquity (41 kyr)
           - Precession (23 kyr)
        
        Parameters:
        -----------
        X1, X2 : array-like
            Input time points (kyr)
        hyperparams : array-like
            [amplitude, length_scale, noise, eccentricity_amp, obliquity_amp, precession_amp]
            
        Returns:
        --------
        K : array-like
            Kernel matrix
        """
        amplitude, length_scale, _, eccentricity_amp, obliquity_amp, precession_amp = hyperparams
        
        # Convert 2D arrays to 1D if necessary
        X1 = X1.ravel()
        X2 = X2.ravel()
        
        # Compute distances between points
        X1_col = X1[:, np.newaxis]
        X2_row = X2[np.newaxis, :]
        
        # Background RBF kernel
        sq_dist = np.square(X1_col - X2_row)
        rbf = amplitude * np.exp(-0.5 * sq_dist / length_scale**2)
        
        # Eccentricity cycle (100 kyr)
        period_e = 100.0
        periodic_e = np.exp(-2 * np.sin(np.pi * np.abs(X1_col - X2_row) / period_e)**2 / length_scale**2)
        
        # Obliquity cycle (41 kyr)
        period_o = 41.0
        periodic_o = np.exp(-2 * np.sin(np.pi * np.abs(X1_col - X2_row) / period_o)**2 / length_scale**2)
        
        # Precession cycle (23 kyr)
        period_p = 23.0
        periodic_p = np.exp(-2 * np.sin(np.pi * np.abs(X1_col - X2_row) / period_p)**2 / length_scale**2)
        
        # Combine all components
        K = rbf + eccentricity_amp * periodic_e + obliquity_amp * periodic_o + precession_amp * periodic_p
        
        return K
    
    def _rbf_kernel(self, X1, X2, hyperparams):
        """Standard RBF kernel."""
        amplitude, length_scale, _ = hyperparams
        
        X1 = X1.ravel()
        X2 = X2.ravel()
        
        X1_col = X1[:, np.newaxis]
        X2_row = X2[np.newaxis, :]
        
        sq_dist = np.square(X1_col - X2_row)
        K = amplitude * np.exp(-0.5 * sq_dist / length_scale**2)
        
        return K
    
    def _matern52_kernel(self, X1, X2, hyperparams):
        """Matérn 5/2 kernel."""
        amplitude, length_scale, _ = hyperparams
        
        X1 = X1.ravel()
        X2 = X2.ravel()
        
        X1_col = X1[:, np.newaxis]
        X2_row = X2[np.newaxis, :]
        
        d = np.abs(X1_col - X2_row)
        sqrt5_d = np.sqrt(5) * d / length_scale
        K = amplitude * (1 + sqrt5_d + 5/3 * sqrt5_d**2) * np.exp(-sqrt5_d)
        
        return K
    
    def _log_marginal_likelihood(self, hyperparams):
        """
        Compute the log marginal likelihood.
        
        Parameters:
        -----------
        hyperparams : array-like
            Kernel hyperparameters
            
        Returns:
        --------
        lml : float
            Log marginal likelihood
        """
        n_samples = self.X_train.shape[0]
        
        # Get kernel matrix with noise term
        K = self._kernel(self.X_train, self.X_train, hyperparams)
        K += hyperparams[2]**2 * np.eye(n_samples)  # Add noise
        
        try:
            # Cholesky decomposition
            L = cholesky(K, lower=True)
            
            # Compute alpha = K^-1 * y
            alpha = cho_solve((L, True), self.y_train)
            
            # Compute log marginal likelihood
            lml = -0.5 * np.dot(self.y_train, alpha)
            lml -= np.sum(np.log(np.diag(L)))
            lml -= n_samples / 2 * np.log(2 * np.pi)
            
            return lml
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails (non-PD matrix), return large negative value
            return -1e10
    
    def _neg_log_marginal_likelihood(self, hyperparams):
        """Negative log marginal likelihood for optimization."""
        return -self._log_marginal_likelihood(hyperparams)
    
    def _optimize_hyperparams(self):
        """Optimize kernel hyperparameters using L-BFGS-B."""
        # Define bounds for hyperparameters
        if self.kernel_type == 'milankovitch':
            bounds = [(1e-3, 10.0),   # amplitude
                      (1.0, 500.0),   # length_scale
                      (1e-3, 2.0),    # noise
                      (1e-3, 5.0),    # eccentricity_amp
                      (1e-3, 5.0),    # obliquity_amp
                      (1e-3, 5.0)]    # precession_amp
        else:
            bounds = [(1e-3, 10.0),   # amplitude
                      (1.0, 500.0),   # length_scale
                      (1e-3, 2.0)]    # noise
        
        # Run optimization
        result = optimize.minimize(
            self._neg_log_marginal_likelihood,
            self.hyperparams,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update hyperparameters
        self.hyperparams = result.x
    
    def fit(self, X, y):
        """
        Fit the GP model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Time points (in kyr)
        y : array-like of shape (n_samples,)
            Temperature values (in °C)
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Ensure arrays are 2D
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).ravel()
        
        # Store data for prediction
        if self.normalize:
            # Normalize inputs
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            
            # Store normalized data
            self.X_train = (X - self.X_mean) / self.X_std
            self.y_train = (y - self.y_mean) / self.y_std
        else:
            self.X_train = X
            self.y_train = y
        
        # Optimize hyperparameters if requested
        if self.optimize_hyperparams:
            self._optimize_hyperparams()
        
        # Compute kernel matrix with optimized hyperparameters
        n_samples = X.shape[0]
        K = self._kernel(self.X_train, self.X_train, self.hyperparams)
        K += self.hyperparams[2]**2 * np.eye(n_samples)  # Add noise
        
        # Compute Cholesky decomposition
        self.L = cholesky(K, lower=True)
        
        # Compute alpha = K^-1 * y
        self.alpha = cho_solve((self.L, True), self.y_train)
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Predict using the GP model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Time points to predict (in kyr)
        return_std : bool, default=False
            If True, return the standard deviation of the prediction
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted temperature values
        y_std : array-like of shape (n_samples,), optional
            Standard deviation of the prediction
        """
        # Ensure array is 2D
        X = np.asarray(X).reshape(-1, 1)
        
        # Normalize if necessary
        if self.normalize:
            X_norm = (X - self.X_mean) / self.X_std
        else:
            X_norm = X
        
        # Compute kernel between test and training points
        K_s = self._kernel(X_norm, self.X_train, self.hyperparams)
        
        # Compute predictive mean
        y_pred = K_s @ self.alpha
        
        # Denormalize if necessary
        if self.normalize:
            y_pred = y_pred * self.y_std + self.y_mean
        
        if return_std:
            # Compute kernel for test points
            K_ss = self._kernel(X_norm, X_norm, self.hyperparams)
            
            # Compute predictive variance
            v = cho_solve((self.L, True), K_s.T)
            y_var = np.diag(K_ss) - np.sum(K_s * v.T, axis=1)
            y_var = np.maximum(y_var, 1e-6)  # Ensure positive variance
            
            # Denormalize if necessary
            if self.normalize:
                y_std = np.sqrt(y_var) * self.y_std
            else:
                y_std = np.sqrt(y_var)
            
            return y_pred, y_std
        
        return y_pred
    
    def sample_posterior(self, X, n_samples=10):
        """
        Sample from the posterior distribution.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Time points to sample at (in kyr)
        n_samples : int, default=10
            Number of samples to draw
            
        Returns:
        --------
        samples : array-like of shape (n_samples, n_time_points)
            Samples from the posterior distribution
        """
        # Ensure array is 2D
        X = np.asarray(X).reshape(-1, 1)
        
        # Normalize if necessary
        if self.normalize:
            X_norm = (X - self.X_mean) / self.X_std
        else:
            X_norm = X
        
        # Compute kernel between test and training points
        K_s = self._kernel(X_norm, self.X_train, self.hyperparams)
        
        # Compute predictive mean
        mu = K_s @ self.alpha
        
        # Compute kernel for test points
        K_ss = self._kernel(X_norm, X_norm, self.hyperparams)
        
        # Compute predictive covariance
        v = cho_solve((self.L, True), K_s.T)
        cov = K_ss - K_s @ v
        
        # Draw samples from multivariate normal
        samples = np.random.multivariate_normal(mu, cov, n_samples)
        
        # Denormalize if necessary
        if self.normalize:
            samples = samples * self.y_std + self.y_mean
        
        return samples
    
    def score(self, X, y):
        """
        Compute R² score.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Time points
        y : array-like of shape (n_samples,)
            True temperature values
            
        Returns:
        --------
        r2 : float
            R² score
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
    
    def plot(self, X, y=None, n_samples=0, figsize=(12, 6)):
        """
        Plot the GP prediction with uncertainty.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Time points to predict
        y : array-like of shape (n_samples,), optional
            True temperature values
        n_samples : int, default=0
            Number of posterior samples to draw
        figsize : tuple, default=(12, 6)
            Figure size
        """
        # Ensure array is 2D
        X = np.asarray(X).reshape(-1, 1)
        
        # Sort points for plotting
        sorted_idx = np.argsort(X.ravel())
        X_sorted = X[sorted_idx]
        
        # Predict with uncertainty
        y_pred, y_std = self.predict(X_sorted, return_std=True)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot training data if available
        if hasattr(self, 'X_train') and not self.normalize:
            plt.scatter(self.X_train, self.y_train, c='k', label='Training data')
        elif hasattr(self, 'X_train'):
            plt.scatter(self.X_train * self.X_std + self.X_mean, 
                        self.y_train * self.y_std + self.y_mean, 
                        c='k', label='Training data')
        
        # Plot test data if available
        if y is not None:
            plt.scatter(X, y, c='r', label='Test data')
        
        # Plot prediction
        plt.plot(X_sorted, y_pred, 'b-', label='Prediction')
        
        # Plot uncertainty
        plt.fill_between(X_sorted.ravel(), 
                         y_pred - 1.96 * y_std, 
                         y_pred + 1.96 * y_std, 
                         alpha=0.2, color='b', label='95% confidence interval')
        
        # Plot posterior samples if requested
        if n_samples > 0:
            samples = self.sample_posterior(X_sorted, n_samples=n_samples)
            for i in range(n_samples):
                plt.plot(X_sorted, samples[i], 'b-', alpha=0.1)
        
        # Add labels and legend
        plt.xlabel('Time (kyr)')
        plt.ylabel('Temperature (°C)')
        plt.title('Gaussian Process Paleoclimate Reconstruction')
        plt.legend()
        plt.grid(True)
        
        return plt.gca()
    
    def compute_spectral_density(self, frequencies):
        """
        Compute the spectral density of the GP.
        
        Parameters:
        -----------
        frequencies : array-like
            Frequencies to compute the spectral density at (cycles/kyr)
            
        Returns:
        --------
        S : array-like
            Spectral density at the requested frequencies
        """
        S = np.zeros_like(frequencies)
        
        if self.kernel_type == 'milankovitch':
            amplitude, length_scale, _, e_amp, o_amp, p_amp = self.hyperparams
            
            # Background RBF component
            S += amplitude * np.sqrt(2 * np.pi) * length_scale * np.exp(-2 * (np.pi * frequencies * length_scale) ** 2)
            
            # Eccentricity component (100 kyr)
            period_e = 100.0
            freq_e = 1.0 / period_e
            S += e_amp * np.exp(-(frequencies - freq_e) ** 2 / (2 * 0.01 ** 2)) / np.sqrt(2 * np.pi * 0.01 ** 2)
            S += e_amp * np.exp(-(frequencies + freq_e) ** 2 / (2 * 0.01 ** 2)) / np.sqrt(2 * np.pi * 0.01 ** 2)
            
            # Obliquity component (41 kyr)
            period_o = 41.0
            freq_o = 1.0 / period_o
            S += o_amp * np.exp(-(frequencies - freq_o) ** 2 / (2 * 0.02 ** 2)) / np.sqrt(2 * np.pi * 0.02 ** 2)
            S += o_amp * np.exp(-(frequencies + freq_o) ** 2 / (2 * 0.02 ** 2)) / np.sqrt(2 * np.pi * 0.02 ** 2)
            
            # Precession component (23 kyr)
            period_p = 23.0
            freq_p = 1.0 / period_p
            S += p_amp * np.exp(-(frequencies - freq_p) ** 2 / (2 * 0.03 ** 2)) / np.sqrt(2 * np.pi * 0.03 ** 2)
            S += p_amp * np.exp(-(frequencies + freq_p) ** 2 / (2 * 0.03 ** 2)) / np.sqrt(2 * np.pi * 0.03 ** 2)
        
        elif self.kernel_type == 'rbf':
            amplitude, length_scale, _ = self.hyperparams
            S = amplitude * np.sqrt(2 * np.pi) * length_scale * np.exp(-2 * (np.pi * frequencies * length_scale) ** 2)
        
        elif self.kernel_type == 'matern52':
            amplitude, length_scale, _ = self.hyperparams
            # Approximate spectral density for Matern 5/2
            S = amplitude * 8 * np.sqrt(5) * np.pi * length_scale / \
                (3 * (1 + (2 * np.pi * frequencies * length_scale) ** 2) ** 3)
        
        return S
    
    def plot_spectral_density(self, f_min=0.001, f_max=0.1, n_points=1000, figsize=(12, 6)):
        """
        Plot the spectral density of the GP.
        
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
        # Generate frequencies
        frequencies = np.linspace(f_min, f_max, n_points)
        
        # Compute spectral density
        S = self.compute_spectral_density(frequencies)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot spectral density
        plt.plot(frequencies, S, 'b-')
        
        # Add Milankovitch cycle markers
        plt.axvline(1.0/100.0, color='r', linestyle='--', label='Eccentricity (100 kyr)')
        plt.axvline(1.0/41.0, color='g', linestyle='--', label='Obliquity (41 kyr)')
        plt.axvline(1.0/23.0, color='m', linestyle='--', label='Precession (23 kyr)')
        
        # Add labels and legend
        plt.xlabel('Frequency (cycles/kyr)')
        plt.ylabel('Spectral density')
        plt.title('Spectral Density of GP Kernel')
        plt.legend()
        plt.grid(True)
        
        # Convert x-axis to period
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        period_ticks = [10, 20, 23, 41, 50, 100, 200, 500]
        period_ticks = [p for p in period_ticks if 1.0/p >= f_min and 1.0/p <= f_max]
        ax2.set_xticks([1.0/p for p in period_ticks])
        ax2.set_xticklabels([str(p) for p in period_ticks])
        ax2.set_xlabel('Period (kyr)')
        
        return plt.gca()