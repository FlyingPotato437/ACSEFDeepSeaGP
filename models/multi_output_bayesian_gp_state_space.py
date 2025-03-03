"""
Multi-Output Bayesian GP Model for Paleoclimate Reconstruction
Inspired by the matplotlib approach with less smoothing
"""

import numpy as np
from scipy import optimize, signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class MultiOutputBayesianGP:
    """
    Multi-Output Bayesian GP model for paleoclimate reconstruction
    """
    def __init__(self, random_state=42):
        # Set random seed
        np.random.seed(random_state)
        
        # Default calibration parameters
        self.uk37_params = {
            'slope': 0.033,
            'intercept': 0.044,
        }
        
        self.d18o_params = {
            'modern_value': 3.2,
            'glacial_value': 5.0,
        }
        
        # Initialize kernel parameters with SMALLER LENGTHSCALES to avoid oversmoothing
        self.sst_kernel = {
            'lengthscale': 10.0,     # Smaller lengthscale to capture more detail
            'amplitude': 1.5,        # Slightly higher amplitude for more variation
            'noise': 0.3             # Lower noise to avoid oversmoothing
        }
        
        self.ice_kernel = {
            'lengthscale': 15.0,     # Smaller lengthscale for ice volume too
            'amplitude': 2.0,        # Higher amplitude for more variation
            'noise': 0.5             # Lower noise
        }
        
        # Initialize data attributes
        self.uk37_data = None
        self.d18o_data = None
        self.sst_posterior = None
        self.ice_posterior = None
        self.transitions = {'SST': [], 'Ice Volume': []}
        
    def load_data(self, csv_file, max_age=800):
        """
        Load and process data from CSV file
        
        Parameters:
        -----------
        csv_file : str
            Path to CSV file
        max_age : float
            Maximum age to consider (in kyr)
        """
        import pandas as pd
        
        print(f"Loading data from {csv_file}...")
        try:
            # Load CSV data
            df = pd.read_csv(csv_file)
            print(f"CSV loaded with {len(df)} rows")
            
            # Extract UK37 data with age filter
            uk37_data = df[['Age.1', 'UK37']].dropna().rename(columns={'Age.1': 'age', 'UK37': 'value'})
            uk37_data = uk37_data[uk37_data['age'] <= max_age]
            
            # Convert to SST
            uk37_data['sst'] = (uk37_data['value'] - self.uk37_params['intercept']) / self.uk37_params['slope']
            
            self.uk37_data = {
                'age': uk37_data['age'].values,
                'value': uk37_data['value'].values,
                'sst': uk37_data['sst'].values
            }
            
            print(f"Processed UK37 data: {len(self.uk37_data['age'])} measurements")
            print(f"  - Age range: {min(self.uk37_data['age']):.1f} to {max(self.uk37_data['age']):.1f} ka")
            print(f"  - UK37 range: {min(self.uk37_data['value']):.3f} to {max(self.uk37_data['value']):.3f}")
            print(f"  - SST range: {min(self.uk37_data['sst']):.1f} to {max(self.uk37_data['sst']):.1f} °C")
            
            # Extract d18O data with age filter
            d18o_data = df[['Age', 'd18O']].dropna().rename(columns={'Age': 'age', 'd18O': 'value'})
            d18o_data = d18o_data[d18o_data['age'] <= max_age]
            
            # Convert to ice volume (0-100%)
            d18o_range = self.d18o_params['glacial_value'] - self.d18o_params['modern_value']
            d18o_data['ice_volume'] = 100 * (d18o_data['value'] - self.d18o_params['modern_value']) / d18o_range
            
            self.d18o_data = {
                'age': d18o_data['age'].values,
                'value': d18o_data['value'].values,
                'ice_volume': d18o_data['ice_volume'].values
            }
            
            print(f"Processed d18O data: {len(self.d18o_data['age'])} measurements")
            print(f"  - Age range: {min(self.d18o_data['age']):.1f} to {max(self.d18o_data['age']):.1f} ka")
            print(f"  - d18O range: {min(self.d18o_data['value']):.3f} to {max(self.d18o_data['value']):.3f}")
            print(f"  - Ice volume range: {min(self.d18o_data['ice_volume']):.1f} to {max(self.d18o_data['ice_volume']):.1f} %")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def compute_kernel(self, x1, x2, params):
        """
        Compute composite kernel between two sets of points
        Includes RBF + Periodic components to capture Milankovitch cycles
        
        Parameters:
        -----------
        x1 : array-like
            First set of points
        x2 : array-like
            Second set of points
        params : dict
            Kernel parameters
            
        Returns:
        --------
        array-like
            Kernel matrix
        """
        # Compute squared distances
        x1 = np.asarray(x1).reshape(-1, 1)
        x2 = np.asarray(x2).reshape(-1, 1)
        
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        dist = np.sqrt(sqdist)
        
        # RBF/Squared Exponential kernel component
        K_rbf = params['amplitude']**2 * np.exp(-0.5 * sqdist / params['lengthscale']**2)
        
        # Add periodic components for Milankovitch cycles
        # 100kyr eccentricity cycle
        period_100k = 100.0
        K_100k = 0.7 * np.exp(-2 * np.sin(np.pi * dist / period_100k)**2 / 0.5**2)
        
        # 41kyr obliquity cycle
        period_41k = 41.0
        K_41k = 0.4 * np.exp(-2 * np.sin(np.pi * dist / period_41k)**2 / 0.5**2)
        
        # 23kyr precession cycle
        period_23k = 23.0
        K_23k = 0.2 * np.exp(-2 * np.sin(np.pi * dist / period_23k)**2 / 0.5**2)
        
        # Combine all kernel components
        K = K_rbf + K_100k + K_41k + K_23k
        
        return K
        
    def gp_predict(self, train_x, train_y, test_x, params):
        """
        Make GP predictions
        
        Parameters:
        -----------
        train_x : array-like
            Training inputs
        train_y : array-like
            Training targets
        test_x : array-like
            Test inputs
        params : dict
            Kernel parameters
            
        Returns:
        --------
        tuple
            (mean, variance) predictions
        """
        # Ensure arrays
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        
        # Compute kernels
        K = self.compute_kernel(train_x, train_x, params)
        K_s = self.compute_kernel(train_x, test_x, params)
        K_ss = self.compute_kernel(test_x, test_x, params)
        
        # Add noise
        K = K + params['noise']**2 * np.eye(len(train_x))
        
        # Add small jitter for stability
        jitter = 1e-6
        K = K + jitter * np.eye(len(train_x))
        
        # Compute predictions using direct matrix operations (matplotlib style)
        try:
            # Direct matrix inversion (faster and simpler)
            K_inv = np.linalg.inv(K)
            mean = K_s.T @ K_inv @ train_y
            var = K_ss - K_s.T @ K_inv @ K_s
        except:
            # Fall back to more stable Cholesky if needed
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, train_y))
            mean = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            var = K_ss - v.T @ v
        
        # Get diagonal variance
        if var.ndim > 1:
            var = np.diag(var)
            
        # Ensure positive variance
        var = np.maximum(var, 1e-6)
        
        return mean, var
        
    def optimize_hyperparameters(self):
        """
        Optimize kernel hyperparameters using maximum likelihood
        Uses bounds that allow for more variation
        """
        print("Optimizing hyperparameters...")
        
        # Define log marginal likelihood function
        def log_marginal_likelihood(params, x, y, kernel_base):
            # Extract parameters: [log(lengthscale), log(amplitude), log(noise)]
            params_dict = kernel_base.copy()
            params_dict['lengthscale'] = np.exp(params[0])
            params_dict['amplitude'] = np.exp(params[1])
            params_dict['noise'] = np.exp(params[2])
            
            # Compute kernel matrix
            K = self.compute_kernel(x, x, params_dict)
            
            # Add noise
            K = K + params_dict['noise']**2 * np.eye(len(x))
            
            # Add jitter for stability
            jitter = 1e-6
            K = K + jitter * np.eye(len(x))
            
            # Compute log marginal likelihood
            try:
                # Direct approach
                K_inv = np.linalg.inv(K)
                logdet = np.log(np.linalg.det(K))
                lml = -0.5 * (y @ K_inv @ y + logdet + len(y) * np.log(2 * np.pi))
                return -lml  # return negative for minimization
            except:
                # Cholesky if direct fails
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    lml = -0.5 * (y @ alpha + 2 * np.sum(np.log(np.diag(L))) + len(y) * np.log(2 * np.pi))
                    return -lml
                except:
                    # If all fails, return large value
                    return 1e6
        
        # Optimize SST hyperparameters
        if self.uk37_data is not None:
            print("Optimizing SST hyperparameters...")
            
            # Initial parameters [log(lengthscale), log(amplitude), log(noise)]
            init_params = np.log([self.sst_kernel['lengthscale'], 
                                 self.sst_kernel['amplitude'], 
                                 self.sst_kernel['noise']])
            
            # Define bounds - SMALLER LOWER BOUNDS to allow more detail
            bounds = [
                (np.log(5.0), np.log(50.0)),     # lengthscale (lower for more detail)
                (np.log(0.5), np.log(5.0)),      # amplitude (higher for more variation)
                (np.log(0.01), np.log(1.0))      # noise (lower for less smoothing)
            ]
            
            # Optimize with L-BFGS-B
            result = optimize.minimize(
                lambda params: log_marginal_likelihood(params, self.uk37_data['age'], self.uk37_data['sst'], self.sst_kernel),
                init_params,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Update parameters
            self.sst_kernel['lengthscale'] = np.exp(result.x[0])
            self.sst_kernel['amplitude'] = np.exp(result.x[1])
            self.sst_kernel['noise'] = np.exp(result.x[2])
            
            print(f"  Optimized SST parameters: lengthscale={self.sst_kernel['lengthscale']:.2f}, "
                  f"amplitude={self.sst_kernel['amplitude']:.2f}, noise={self.sst_kernel['noise']:.2f}")
        
        # Optimize ice volume hyperparameters
        if self.d18o_data is not None:
            print("Optimizing ice volume hyperparameters...")
            
            # Initial parameters [log(lengthscale), log(amplitude), log(noise)]
            init_params = np.log([self.ice_kernel['lengthscale'], 
                                 self.ice_kernel['amplitude'], 
                                 self.ice_kernel['noise']])
            
            # Define bounds - SMALLER LOWER BOUNDS to allow more detail
            bounds = [
                (np.log(5.0), np.log(70.0)),     # lengthscale (lower for more detail)
                (np.log(0.5), np.log(8.0)),      # amplitude (higher for more variation)
                (np.log(0.1), np.log(2.0))       # noise (lower for less smoothing)
            ]
            
            # Optimize
            result = optimize.minimize(
                lambda params: log_marginal_likelihood(params, self.d18o_data['age'], self.d18o_data['ice_volume'], self.ice_kernel),
                init_params,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Update parameters
            self.ice_kernel['lengthscale'] = np.exp(result.x[0])
            self.ice_kernel['amplitude'] = np.exp(result.x[1])
            self.ice_kernel['noise'] = np.exp(result.x[2])
            
            print(f"  Optimized ice volume parameters: lengthscale={self.ice_kernel['lengthscale']:.2f}, "
                  f"amplitude={self.ice_kernel['amplitude']:.2f}, noise={self.ice_kernel['noise']:.2f}")
    
    def sample_posterior_direct(self, x, y, params, n_samples=100):
        """
        Sample directly from the posterior distribution (matplotlib style)
        
        Parameters:
        -----------
        x : array-like
            Input points
        y : array-like
            Target values
        params : dict
            Kernel parameters
        n_samples : int
            Number of samples
            
        Returns:
        --------
        dict
            Posterior samples and statistics
        """
        # Compute kernel and mean function
        K = self.compute_kernel(x, x, params)
        K_noisy = K + params['noise']**2 * np.eye(len(x))
        
        # Add small jitter for stability
        jitter = 1e-6
        K_noisy = K_noisy + jitter * np.eye(len(x))
        
        # Compute posterior mean and covariance
        try:
            # Compute posterior mean
            K_inv = np.linalg.inv(K_noisy)
            mean = K @ K_inv @ y
            
            # Compute posterior covariance
            cov = K - K @ K_inv @ K
            cov = cov + jitter * np.eye(len(x))  # Add jitter for stability
            
            # Generate samples from multivariate normal
            samples = np.random.multivariate_normal(mean, cov, size=n_samples).T
        except:
            # If matrix inversion fails, use simpler approach
            print("Warning: Matrix inversion failed, using simpler sampling approach")
            # Just sample around the mean with noise
            mean, var = self.gp_predict(x, y, x, params)
            std = np.sqrt(var)
            samples = np.zeros((len(x), n_samples))
            for i in range(n_samples):
                samples[:, i] = mean + np.random.normal(0, std)
        
        # Compute statistics
        posterior = {
            'samples': samples,
            'mean': np.mean(samples, axis=1),
            'std': np.std(samples, axis=1),
            'lower_95': np.percentile(samples, 2.5, axis=1),
            'upper_95': np.percentile(samples, 97.5, axis=1)
        }
        
        return posterior
        
    def fit(self, optimize=True, n_samples=100):
        """
        Fit the model to the data using direct posterior sampling (matplotlib style)
        
        Parameters:
        -----------
        optimize : bool
            Whether to optimize hyperparameters first
        n_samples : int
            Number of posterior samples
        """
        if self.uk37_data is None or self.d18o_data is None:
            print("No data loaded. Please load data first.")
            return False
            
        print("\nFitting model with direct sampling (matplotlib style)...")
        
        # Optimize hyperparameters
        if optimize:
            self.optimize_hyperparameters()
            
        # Sample from SST posterior
        print("Sampling from SST posterior...")
        self.sst_posterior = self.sample_posterior_direct(
            self.uk37_data['age'],
            self.uk37_data['sst'],
            self.sst_kernel,
            n_samples
        )
        
        # Sample from ice volume posterior
        print("Sampling from ice volume posterior...")
        self.ice_posterior = self.sample_posterior_direct(
            self.d18o_data['age'],
            self.d18o_data['ice_volume'],
            self.ice_kernel,
            n_samples
        )
        
        # Detect transitions
        self._detect_transitions()
        
        return True
    
    def predict(self, test_x):
        """
        Make predictions at test points
        
        Parameters:
        -----------
        test_x : array-like
            Test inputs
            
        Returns:
        --------
        tuple
            (sst_mean, ice_mean, sst_std, ice_std)
        """
        if self.uk37_data is None or self.d18o_data is None:
            print("No data loaded. Please load data first.")
            return None, None, None, None
            
        if self.sst_posterior is None or self.ice_posterior is None:
            print("Model not fitted. Please fit the model first.")
            return None, None, None, None
            
        # SST predictions
        sst_mean, sst_var = self.gp_predict(
            self.uk37_data['age'], 
            self.sst_posterior['mean'], 
            test_x,
            self.sst_kernel
        )
        
        # Ice volume predictions
        ice_mean, ice_var = self.gp_predict(
            self.d18o_data['age'], 
            self.ice_posterior['mean'], 
            test_x,
            self.ice_kernel
        )
        
        return sst_mean, ice_mean, np.sqrt(sst_var), np.sqrt(ice_var)
    
    def _detect_transitions(self):
        """Detect transitions using rate of change analysis"""
        if self.sst_posterior is None or self.ice_posterior is None:
            return
            
        print("Detecting climate transitions...")
        
        # Detect SST transitions
        try:
            # Get rate of change
            sst_x = self.uk37_data['age']
            sst_y = self.sst_posterior['mean']
            
            # Sort by age
            sort_idx = np.argsort(sst_x)
            sst_x = sst_x[sort_idx]
            sst_y = sst_y[sort_idx]
            
            # Minimal smoothing to preserve detail
            sst_y_smooth = gaussian_filter1d(sst_y, sigma=1.0)  # Reduced sigma for less smoothing
            
            # Compute rate of change
            sst_rate = np.abs(np.gradient(sst_y_smooth, sst_x))
            
            # Find peaks
            threshold = np.percentile(sst_rate, 95)
            peaks, _ = signal.find_peaks(sst_rate, height=threshold, distance=20)
            
            if len(peaks) > 0:
                # Get transition ages
                sst_transitions = sorted(sst_x[peaks])
                
                # Limit to top 5
                if len(sst_transitions) > 5:
                    # Get the top 5 by rate magnitude
                    peak_heights = sst_rate[peaks]
                    top_idxs = np.argsort(peak_heights)[-5:]
                    sst_transitions = sorted([sst_x[peaks[i]] for i in top_idxs])
                    
                self.transitions['SST'] = sst_transitions
                print(f"Detected {len(sst_transitions)} SST transitions at: {', '.join([f'{t:.1f} ka' for t in sst_transitions])}")
            else:
                self.transitions['SST'] = []
                print("No SST transitions detected")
                
        except Exception as e:
            print(f"Error detecting SST transitions: {str(e)}")
            self.transitions['SST'] = []
            
        # Detect ice volume transitions
        try:
            # Get rate of change
            ice_x = self.d18o_data['age']
            ice_y = self.ice_posterior['mean']
            
            # Sort by age
            sort_idx = np.argsort(ice_x)
            ice_x = ice_x[sort_idx]
            ice_y = ice_y[sort_idx]
            
            # Minimal smoothing to preserve detail
            ice_y_smooth = gaussian_filter1d(ice_y, sigma=1.0)  # Reduced sigma for less smoothing
            
            # Compute rate of change
            ice_rate = np.abs(np.gradient(ice_y_smooth, ice_x))
            
            # Find peaks
            threshold = np.percentile(ice_rate, 95)
            peaks, _ = signal.find_peaks(ice_rate, height=threshold, distance=20)
            
            if len(peaks) > 0:
                # Get transition ages
                ice_transitions = sorted(ice_x[peaks])
                
                # Limit to top 5
                if len(ice_transitions) > 5:
                    # Get the top 5 by rate magnitude
                    peak_heights = ice_rate[peaks]
                    top_idxs = np.argsort(peak_heights)[-5:]
                    ice_transitions = sorted([ice_x[peaks[i]] for i in top_idxs])
                    
                self.transitions['Ice Volume'] = ice_transitions
                print(f"Detected {len(ice_transitions)} ice volume transitions at: {', '.join([f'{t:.1f} ka' for t in ice_transitions])}")
            else:
                self.transitions['Ice Volume'] = []
                print("No ice volume transitions detected")
                
        except Exception as e:
            print(f"Error detecting ice volume transitions: {str(e)}")
            self.transitions['Ice Volume'] = []
            
    def plot_reconstruction(self, test_x, output_dir=None):
        """
        Plot the reconstructed SST and ice volume with matplotlib-style visualization
        
        Parameters:
        -----------
        test_x : array-like
            Test inputs
        output_dir : str, optional
            Output directory
        """
        if self.uk37_data is None or self.d18o_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        if self.sst_posterior is None or self.ice_posterior is None:
            print("Model not fitted. Please fit the model first.")
            return None
            
        # Make predictions
        sst_mean, ice_mean, sst_std, ice_std = self.predict(test_x)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot SST reconstruction - MATPLOTLIB STYLE
        # Individual samples for more detailed uncertainty visualization
        if self.sst_posterior['samples'] is not None:
            # Plot a subset of posterior samples
            n_to_plot = min(20, self.sst_posterior['samples'].shape[1])
            samples_to_plot = np.random.choice(self.sst_posterior['samples'].shape[1], n_to_plot, replace=False)
            
            # Interpolate each sample to test points
            for i in samples_to_plot:
                sample_mean, _ = self.gp_predict(
                    self.uk37_data['age'],
                    self.sst_posterior['samples'][:, i],
                    test_x,
                    self.sst_kernel
                )
                ax1.plot(test_x, sample_mean, 'b-', linewidth=0.3, alpha=0.3)
        
        # Mean and confidence interval
        ax1.plot(test_x, sst_mean, 'b-', linewidth=2, label='SST Reconstruction')
        ax1.fill_between(
            test_x, 
            sst_mean - 1.96 * sst_std, 
            sst_mean + 1.96 * sst_std, 
            color='b', alpha=0.2, label='95% CI'
        )
        
        # Plot UK37 data
        ax1.scatter(
            self.uk37_data['age'],
            self.uk37_data['sst'],
            marker='o',
            color='green',
            s=30,
            alpha=0.7,
            label='UK37 derived SST'
        )
        
        # Mark SST transitions
        for trans in self.transitions['SST']:
            ax1.axvline(x=trans, color='r', linestyle='--', alpha=0.7)
            y_range = ax1.get_ylim()
            ax1.text(
                trans, 
                y_range[0] + 0.95*(y_range[1]-y_range[0]), 
                f'{trans:.1f}', 
                color='r', 
                rotation=90, 
                ha='right'
            )
            
        ax1.set_ylabel('Sea Surface Temperature (°C)', fontsize=12)
        ax1.set_title('SST Reconstruction from UK37 Measurements', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot ice volume reconstruction - MATPLOTLIB STYLE
        # Individual samples for more detailed uncertainty visualization
        if self.ice_posterior['samples'] is not None:
            # Plot a subset of posterior samples
            n_to_plot = min(20, self.ice_posterior['samples'].shape[1])
            samples_to_plot = np.random.choice(self.ice_posterior['samples'].shape[1], n_to_plot, replace=False)
            
            # Interpolate each sample to test points
            for i in samples_to_plot:
                sample_mean, _ = self.gp_predict(
                    self.d18o_data['age'],
                    self.ice_posterior['samples'][:, i],
                    test_x,
                    self.ice_kernel
                )
                ax2.plot(test_x, sample_mean, 'g-', linewidth=0.3, alpha=0.3)
        
        # Mean and confidence interval
        ax2.plot(test_x, ice_mean, 'g-', linewidth=2, label='Ice Volume Reconstruction')
        ax2.fill_between(
            test_x, 
            ice_mean - 1.96 * ice_std, 
            ice_mean + 1.96 * ice_std, 
            color='g', alpha=0.2, label='95% CI'
        )
        
        # Plot d18O data
        ax2.scatter(
            self.d18o_data['age'],
            self.d18o_data['ice_volume'],
            marker='s',
            color='orange',
            s=30,
            alpha=0.7,
            label='δ18O derived Ice Volume'
        )
        
        # Mark ice volume transitions
        for trans in self.transitions['Ice Volume']:
            ax2.axvline(x=trans, color='r', linestyle='--', alpha=0.7)
            y_range = ax2.get_ylim()
            ax2.text(
                trans, 
                y_range[0] + 0.95*(y_range[1]-y_range[0]), 
                f'{trans:.1f}', 
                color='r', 
                rotation=90, 
                ha='right'
            )
            
        ax2.set_xlabel('Age (kyr)', fontsize=12)
        ax2.set_ylabel('Global Ice Volume (%)', fontsize=12)
        ax2.set_title('Ice Volume Reconstruction from δ18O Measurements', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Reverse x-axis to show older ages on the right
        ax1.set_xlim(max(test_x), min(test_x))
        
        plt.tight_layout()
        
        # Save figure
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'reconstruction.png'), dpi=300, bbox_inches='tight')
            
        return fig