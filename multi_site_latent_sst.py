"""
multi_site_latent_sst.py - Multi-site latent SST reconstruction with spatial correlation modeling

This module implements a sophisticated multi-site latent variable extraction approach for 
Sea Surface Temperature (SST) reconstruction, incorporating the Linear Model of Coregionalization 
(LMC) to model spatial correlations across multiple geographic sites.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, PeriodicKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import math
from scipy import special
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
from scipy.spatial.distance import pdist, squareform
import ipywidgets as widgets
from IPython.display import display

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set up directory for results
output_dir = "multi_site_results"
os.makedirs(output_dir, exist_ok=True)

# Constants for synthetic data
TIME_MIN = 0      # Start time in kyr BP
TIME_MAX = 500    # End time in kyr BP
N_POINTS = 200    # Number of data points per site
TEST_SIZE = 0.2   # Proportion of data for testing
N_SITES = 5       # Number of geographic sites

# Geographic site locations (latitude, longitude)
# Spread across different ocean basins to model spatial variability
SITE_LOCATIONS = [
    (20.0, -70.0),    # North Atlantic
    (-10.0, -30.0),   # South Atlantic
    (30.0, 150.0),    # North Pacific
    (-15.0, 100.0),   # Indian Ocean
    (0.0, -110.0)     # Equatorial Pacific
]

# Proxy calibration parameters with VERY HIGH NOISE for challenging reconstruction
# Each site has slightly different calibration parameters to reflect regional differences
# Format: [slope, intercept, error_std]

# δ18O = α1 * SST + β1 + ε1, where ε1 ~ N(0, σ1²)
D18O_CALIBRATIONS = [
    [-0.22, 3.0, 1.2],   # North Atlantic - high noise
    [-0.24, 3.1, 1.0],   # South Atlantic - high noise
    [-0.21, 2.9, 1.5],   # North Pacific - very high noise
    [-0.23, 3.0, 1.3],   # Indian Ocean - high noise
    [-0.22, 2.8, 1.1]    # Equatorial Pacific - high noise
]

# UK'37 = α2 * SST + β2 + ε2, where ε2 ~ N(0, σ2²)
UK37_CALIBRATIONS = [
    [0.033, 0.044, 0.3],   # North Atlantic - moderate noise
    [0.034, 0.040, 0.4],   # South Atlantic - high noise
    [0.032, 0.045, 0.35],  # North Pacific - moderate-high noise
    [0.033, 0.042, 0.5],   # Indian Ocean - very high noise
    [0.034, 0.043, 0.45]   # Equatorial Pacific - high noise
]

# Spatial correlation parameters
# Controls how strongly the latent SST fields are correlated across sites
SPATIAL_CORRELATION_LENGTH = 3000  # km
TEMPORAL_CORRELATION_LENGTH = 10   # kyr


class MultiSiteSyntheticData:
    """
    Generate synthetic paleoclimate data across multiple geographic sites with 
    spatial correlations and site-specific proxy calibrations.
    
    This implements a spatially-aware synthetic data generator that creates
    correlated SST time series across multiple sites with realistic climate variability.
    """
    
    def __init__(self, time_min=TIME_MIN, time_max=TIME_MAX, n_points=N_POINTS, 
                 site_locations=SITE_LOCATIONS, random_seed=42):
        """Initialize with time range, number of sites, and their locations."""
        np.random.seed(random_seed)
        self.time_min = time_min
        self.time_max = time_max
        self.n_points = n_points
        self.site_locations = site_locations
        self.n_sites = len(site_locations)
        
        # Calculate distance matrix between sites (in km)
        self.distances = self._calculate_distances()
        
        # Create spatial correlation matrix based on distances
        self.spatial_correlation = self._create_spatial_correlation_matrix()
    
    def _calculate_distances(self):
        """Calculate great-circle distances between sites in kilometers."""
        distances = np.zeros((self.n_sites, self.n_sites))
        
        for i in range(self.n_sites):
            lat1, lon1 = self.site_locations[i]
            for j in range(i, self.n_sites):
                lat2, lon2 = self.site_locations[j]
                distances[i, j] = self._haversine_distance(lat1, lon1, lat2, lon2)
                distances[j, i] = distances[i, j]  # Symmetric
        
        return distances
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points in kilometers."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def _create_spatial_correlation_matrix(self):
        """Create spatial correlation matrix using a squared exponential kernel."""
        # Use squared exponential (RBF) kernel for spatial correlation
        # C_ij = exp(-d_ij^2 / (2 * l^2)), where d_ij is the distance and l is the length scale
        correlation_matrix = np.exp(-self.distances**2 / (2 * SPATIAL_CORRELATION_LENGTH**2))
        
        return correlation_matrix
    
    def generate_realistic_age_models(self):
        """
        Generate realistic age models for each site with irregularly spaced measurements
        and site-specific sampling resolution.
        """
        # Each site has a slightly different sampling scheme and resolution
        site_ages = []
        
        for site in range(self.n_sites):
            # Base spacing between measurements
            base_spacing = np.random.uniform(1.5, 3.0)  # Different resolution for each site
            
            # Start with approximately evenly spaced points
            base_ages = np.linspace(self.time_min, self.time_max, self.n_points)
            
            # Add perturbations to create irregular spacing
            perturbations = np.random.gamma(1, base_spacing, size=self.n_points)
            perturbations = perturbations / perturbations.sum() * (self.time_max - self.time_min) * 0.3
            
            # Ensure monotonically increasing ages with minimum spacing
            ages = np.zeros(self.n_points)
            ages[0] = self.time_min
            for i in range(1, self.n_points):
                min_spacing = (self.time_max - self.time_min) / (self.n_points * 2)
                ages[i] = max(ages[i-1] + min_spacing, base_ages[i] + perturbations[i])
            
            # Include sections with higher resolution (e.g., specific climate events)
            cluster_centers = np.random.choice(np.arange(10, self.n_points-10), size=3, replace=False)
            for center in cluster_centers:
                window = slice(max(0, center-5), min(self.n_points, center+5))
                ages[window] = np.linspace(ages[max(0, center-5)], ages[min(self.n_points-1, center+5)], 
                                          len(ages[window]))
            
            site_ages.append(ages)
        
        return site_ages
    
    def generate_spatially_correlated_sst(self, site_ages):
        """
        Generate synthetic SST data with realistic climate features that are spatially
        correlated across sites according to their geographic distances.
        
        Parameters:
            site_ages: List of age arrays for each site
            
        Returns:
            List of true SST arrays for each site
        """
        # Determine the maximum number of time points across all sites
        max_points = max(len(ages) for ages in site_ages)
        
        # Create common temporal features (similar across all sites)
        # These represent global climate patterns
        
        # Base orbital components (mimicking Milankovitch cycles)
        # Combined reference signal over a dense time grid
        reference_ages = np.linspace(self.time_min, self.time_max, 1000)
        
        # 100 kyr eccentricity cycle
        eccentricity = 2.0 * np.sin(2 * np.pi * reference_ages / 100)
        
        # 41 kyr obliquity cycle
        obliquity = 1.0 * np.sin(2 * np.pi * reference_ages / 41 + 0.5)
        
        # 23 kyr precession cycle
        precession = 0.7 * np.sin(2 * np.pi * reference_ages / 23 + 0.3)
        
        # Millennial-scale oscillations (Dansgaard-Oeschger events)
        millennial = 0.8 * np.sin(2 * np.pi * reference_ages / 1.5) * np.exp(-((reference_ages % 10) / 2)**2)
        
        # Add abrupt climate transitions (Heinrich events, terminations)
        abrupt_events = np.zeros_like(reference_ages)
        
        # Define transition points for abrupt events
        transition_points = [50, 130, 240, 340, 430]
        for point in transition_points:
            # Create a sigmoidal transition
            transition = 1.2 * special.expit((reference_ages - point) * 3)
            abrupt_events += transition
        
        # Combine global signals
        global_signal = eccentricity + obliquity + precession + millennial + abrupt_events
        
        # Generate site-specific SST realizations with spatial correlation
        site_specific_sst = []
        
        # Create spatially-correlated innovations (deviations from global signal)
        # We'll use multivariate normal distribution with the spatial correlation matrix
        innovations = []
        for _ in range(len(reference_ages)):
            # Generate correlated random values for this time point across all sites
            # The covariance matrix ensures proper spatial correlation
            innovation = np.random.multivariate_normal(
                mean=np.zeros(self.n_sites),
                cov=2.0 * self.spatial_correlation  # Scale factor controls local variability strength
            )
            innovations.append(innovation)
        
        # Convert to array for easier manipulation
        innovations = np.array(innovations)  # Shape: [time_points, n_sites]
        
        # Add site-specific features
        for site in range(self.n_sites):
            # Get ages for this site
            ages = site_ages[site]
            
            # Interpolate global signal to site's age model
            site_global = np.interp(ages, reference_ages, global_signal)
            
            # Interpolate site-specific innovations to site's age model
            site_innovations = np.interp(ages, reference_ages, innovations[:, site])
            
            # Add site-specific mean temperature and latitudinal effect
            # Sites at lower latitudes (closer to equator) are warmer
            latitude = abs(self.site_locations[site][0])  # Absolute latitude
            latitudinal_effect = -0.3 * latitude  # Temperature decreases with latitude
            
            # Create the baseline temperature
            baseline = 20.0 + latitudinal_effect  # Base temperature
            
            # Long-term trend may vary slightly by site
            trend_slope = -0.005 + 0.002 * np.random.randn()  # Slight variation in cooling trend
            trend = trend_slope * ages
            
            # Combine all components
            sst = baseline + trend + site_global + site_innovations
            
            # Add fine-scale temporal noise
            fine_noise = 0.3 * np.random.randn(len(ages))
            
            # Generate autocorrelated noise (red noise)
            red_noise = np.zeros_like(ages)
            red_noise[0] = np.random.randn()
            for i in range(1, len(ages)):
                red_noise[i] = 0.7 * red_noise[i-1] + 0.3 * np.random.randn()
            red_noise = 0.5 * red_noise / np.std(red_noise)
            
            # Final SST time series
            sst += fine_noise + red_noise
            
            site_specific_sst.append(sst)
        
        return site_specific_sst
    
    def generate_proxies(self, site_ages, site_sst):
        """
        Generate both proxy types simultaneously for each site.
        Returns all proxy data with calibrations specific to each site.
        
        The proxies include extremely high noise levels, especially for sites
        with challenging preservation conditions.
        """
        all_proxies = {
            'd18o': [],
            'uk37': []
        }
        
        for site in range(self.n_sites):
            ages = site_ages[site]
            sst = site_sst[site]
            
            # Get site-specific calibration parameters
            d18o_alpha, d18o_beta, d18o_sigma = D18O_CALIBRATIONS[site]
            uk37_alpha, uk37_beta, uk37_sigma = UK37_CALIBRATIONS[site]
            
            # Calculate mean values based on calibration equations
            d18o_mean = d18o_alpha * sst + d18o_beta
            uk37_mean = uk37_alpha * sst + uk37_beta
            
            # Generate correlated noise within each site
            n_samples = len(sst)
            
            # Define the correlation between proxy noise at this site
            noise_correlation = 0.3  # Correlation coefficient
            
            # Define the covariance matrix for this site's proxies
            cov_matrix = np.array([
                [d18o_sigma**2, noise_correlation * d18o_sigma * uk37_sigma],
                [noise_correlation * d18o_sigma * uk37_sigma, uk37_sigma**2]
            ])
            
            # Cholesky decomposition
            L = np.linalg.cholesky(cov_matrix)
            
            # Generate uncorrelated standard normal samples
            uncorrelated = np.random.normal(size=(2, n_samples))
            
            # Transform to correlated samples
            correlated = L @ uncorrelated
            d18o_noise, uk37_noise = correlated[0, :], correlated[1, :]
            
            # Add heteroscedastic effects (variation based on SST value)
            # More extreme temperatures have higher uncertainty
            d18o_hetero = 0.2 * np.abs(sst - np.mean(sst)) / np.std(sst)
            uk37_hetero = 0.15 * (1 + np.exp(-(sst - 15)**2 / 50))
            
            d18o_noise = d18o_noise * (1 + d18o_hetero)
            uk37_noise = uk37_noise * uk37_hetero
            
            # Create final proxy values
            d18o_values = d18o_mean + d18o_noise
            uk37_values = uk37_mean + uk37_noise
            
            # Ensure UK'37 values are within realistic range (0 to 1)
            uk37_values = np.clip(uk37_values, 0, 1)
            
            # Add to dictionary
            all_proxies['d18o'].append(d18o_values)
            all_proxies['uk37'].append(uk37_values)
        
        return all_proxies
    
    def generate_dataset(self):
        """Generate a complete multi-site synthetic dataset with irregular sampling."""
        # Generate age models for each site
        site_ages = self.generate_realistic_age_models()
        
        # Generate spatially-correlated true SST
        site_sst = self.generate_spatially_correlated_sst(site_ages)
        
        # Generate proxies for each site with high noise
        all_proxies = self.generate_proxies(site_ages, site_sst)
        
        # Create dataset
        dataset = {
            'sites': self.site_locations,
            'ages': site_ages,
            'true_sst': site_sst,
            'proxies': all_proxies,
            'distances': self.distances,
            'spatial_correlation': self.spatial_correlation
        }
        
        return dataset
    
    def plot_dataset(self, dataset, show_calibrated=True):
        """Plot the synthetic multi-site dataset with spatial correlation visualization."""
        n_sites = len(dataset['sites'])
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Plot site locations on a map
        ax_map = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax_map)
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='tan', lake_color='lightblue')
        
        # Plot sites
        site_x, site_y = [], []
        for site in range(n_sites):
            lat, lon = dataset['sites'][site]
            x, y = m(lon, lat)
            site_x.append(x)
            site_y.append(y)
            m.plot(x, y, 'ro', markersize=8, label=f'Site {site+1}')
            ax_map.text(x, y, f' {site+1}', fontsize=12)
        
        # Draw lines between sites to show connections
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                # Line thickness based on correlation strength
                corr = dataset['spatial_correlation'][i, j]
                if corr > 0.2:  # Only show significant correlations
                    ax_map.plot([site_x[i], site_x[j]], [site_y[i], site_y[j]], 'b-', 
                               alpha=corr, linewidth=corr*3)
        
        ax_map.set_title('Site Locations and Spatial Correlations')
        
        # 2. Plot spatial correlation matrix
        ax_corr = plt.subplot2grid((3, 3), (0, 2))
        im = ax_corr.imshow(dataset['spatial_correlation'], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax_corr, label='Correlation')
        ax_corr.set_xticks(range(n_sites))
        ax_corr.set_yticks(range(n_sites))
        ax_corr.set_xticklabels([f'Site {i+1}' for i in range(n_sites)])
        ax_corr.set_yticklabels([f'Site {i+1}' for i in range(n_sites)])
        ax_corr.set_title('Spatial Correlation Matrix')
        
        # 3. Plot time series for each site
        for site in range(min(n_sites, 5)):  # Show max 5 sites for clarity
            ax = plt.subplot2grid((3, 3), (site+1, 0), colspan=3)
            
            # Get data
            ages = dataset['ages'][site]
            true_sst = dataset['true_sst'][site]
            d18o_values = dataset['proxies']['d18o'][site]
            uk37_values = dataset['proxies']['uk37'][site]
            
            # Calculate calibrated SST from proxies if requested
            if show_calibrated:
                d18o_alpha, d18o_beta, _ = D18O_CALIBRATIONS[site]
                uk37_alpha, uk37_beta, _ = UK37_CALIBRATIONS[site]
                
                d18o_sst = (d18o_values - d18o_beta) / d18o_alpha
                uk37_sst = (uk37_values - uk37_beta) / uk37_alpha
            
            # Plot true SST
            ax.plot(ages, true_sst, 'k-', linewidth=1.5, label='True SST')
            
            # Plot proxies or their calibrated values
            if show_calibrated:
                ax.plot(ages, d18o_sst, 'bo', markersize=3, alpha=0.5, label='δ¹⁸O Calibrated')
                ax.plot(ages, uk37_sst, 'ro', markersize=3, alpha=0.5, label='UK\'37 Calibrated')
            else:
                # Create a twin axis for proxies
                ax2 = ax.twinx()
                ax2.plot(ages, d18o_values, 'bo', markersize=3, alpha=0.5, label='δ¹⁸O')
                ax2.plot(ages, uk37_values, 'ro', markersize=3, alpha=0.5, label='UK\'37')
                ax2.set_ylabel('Proxy Values')
                ax2.legend(loc='upper right')
            
            # Calculate statistics
            d18o_corr = np.corrcoef(true_sst, d18o_values)[0, 1]
            uk37_corr = np.corrcoef(true_sst, uk37_values)[0, 1]
            
            # Add site location and statistics
            lat, lon = dataset['sites'][site]
            stats_text = (
                f"Site {site+1}: ({lat:.1f}°N, {lon:.1f}°E)\n"
                f"δ¹⁸O-SST Correlation: {d18o_corr:.3f}\n"
                f"UK'37-SST Correlation: {uk37_corr:.3f}\n"
            )
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Age (kyr BP)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Site {site+1} Time Series')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
        
        plt.tight_layout()
        return fig


class BatchIndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    """
    Multi-task (multi-site) GP model with a batch of independent GPs.
    
    This is a baseline approach where each site has its own GP model,
    with no spatial correlation modeling.
    """
    
    def __init__(self, train_x, train_y, num_tasks):
        """
        Initialize the model with independent GPs for each site.
        
        Parameters:
            train_x: Training input ages
            train_y: Training proxy values
            num_tasks: Number of sites
        """
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            train_x.size(0), batch_shape=torch.Size([num_tasks])
        )
        
        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output is a vector-valued function
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance modules for each task
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))
        
        # RBF + Periodic kernel for each site
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_tasks])) + 
            ScaleKernel(PeriodicKernel(batch_shape=torch.Size([num_tasks]))),
            batch_shape=torch.Size([num_tasks])
        )
    
    def forward(self, x):
        """Forward pass for the independent GP model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SpatialLMCModel(gpytorch.models.ApproximateGP):
    """
    Linear Model of Coregionalization (LMC) for multi-site SST reconstruction.
    
    This model explicitly captures spatial correlations between sites
    through a shared set of latent functions.
    """
    
    def __init__(self, train_x, train_y, num_sites, num_latents=2):
        """
        Initialize the LMC model with shared latent processes.
        
        Parameters:
            train_x: Training input ages across all sites
            train_y: Training proxy values across all sites
            num_sites: Number of geographic sites
            num_latents: Number of latent processes to use (usually 2-3)
        """
        # Reorganize x and y into batches
        batch_shape = torch.Size([num_latents])
        
        # Each latent function gets its own variational distribution
        inducing_points = train_x[:100].clone()  # Use subset of points as inducing
        
        # Variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=batch_shape
        )
        
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance modules for each latent process
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        
        # RBF + Periodic kernel combination for climate modeling
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape) + 
            ScaleKernel(PeriodicKernel(batch_shape=batch_shape)),
            batch_shape=batch_shape
        )
        
        # LMC mixing coefficients (learned)
        # This implements the mixing/weights of the latent processes for each site
        self.register_parameter(
            "lmc_coefficients",
            torch.nn.Parameter(torch.randn(num_sites, num_latents))
        )
    
    def forward(self, x):
        """Forward pass for the LMC model."""
        # Get outputs from latent processes
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Return a MVN distribution
        return MultivariateNormal(mean_x, covar_x)
    
    def get_marginal_predictions(self, x, site_indices):
        """
        Get site-specific predictions by combining the latent processes
        with the appropriate LMC coefficients.
        
        Parameters:
            x: Input ages to predict at
            site_indices: Indices of sites to predict for
            
        Returns:
            List of site-specific distributions
        """
        # Get outputs from latent processes
        latent_function = self(x)
        latent_mean = latent_function.mean
        latent_covar = latent_function.covariance_matrix
        
        # Initialize lists for site-specific means and covariances
        site_means = []
        site_covars = []
        
        # For each site, combine the latent processes with LMC coefficients
        for site_idx in site_indices:
            # Get LMC coefficients for this site
            site_coefficients = self.lmc_coefficients[site_idx]
            
            # Compute site-specific mean
            site_mean = latent_mean @ site_coefficients
            
            # Compute site-specific covariance
            site_covar = site_coefficients @ latent_covar @ site_coefficients.T
            
            # Add to lists
            site_means.append(site_mean)
            site_covars.append(site_covar)
        
        # Return site-specific distributions
        return site_means, site_covars


class MultiSiteMultiProxyLikelihood(gpytorch.likelihoods.Likelihood):
    """
    Custom likelihood for multi-site, multi-proxy reconstruction.
    
    This likelihood models the relationship between the latent SST at each site
    and the observed proxy values according to site-specific calibration equations.
    """
    
    def __init__(self, num_sites, d18o_calibrations, uk37_calibrations):
        """
        Initialize the multi-site, multi-proxy likelihood.
        
        Parameters:
            num_sites: Number of geographic sites
            d18o_calibrations: List of δ18O calibration parameters for each site
            uk37_calibrations: List of UK'37 calibration parameters for each site
        """
        super(MultiSiteMultiProxyLikelihood, self).__init__()
        
        self.num_sites = num_sites
        
        # Register calibration parameters as buffers
        self.register_buffer('d18o_alpha', torch.tensor(
            [params[0] for params in d18o_calibrations], dtype=torch.float32))
        self.register_buffer('d18o_beta', torch.tensor(
            [params[1] for params in d18o_calibrations], dtype=torch.float32))
        self.register_buffer('d18o_sigma', torch.tensor(
            [params[2] for params in d18o_calibrations], dtype=torch.float32))
        
        self.register_buffer('uk37_alpha', torch.tensor(
            [params[0] for params in uk37_calibrations], dtype=torch.float32))
        self.register_buffer('uk37_beta', torch.tensor(
            [params[1] for params in uk37_calibrations], dtype=torch.float32))
        self.register_buffer('uk37_sigma', torch.tensor(
            [params[2] for params in uk37_calibrations], dtype=torch.float32))
    
    def forward(self, latent_sst, site_indices):
        """
        Forward pass mapping latent SST to proxy observation distributions.
        
        Parameters:
            latent_sst: The latent SST distributions from the GP model
            site_indices: Indices of sites corresponding to the latent_sst
            
        Returns:
            A joint distribution over proxy observations
        """
        # Get mean and variance of latent SST
        mean_sst = latent_sst.mean
        var_sst = latent_sst.variance
        
        # Initialize lists for proxy means and variances
        d18o_means, d18o_vars = [], []
        uk37_means, uk37_vars = [], []
        
        # For each site, apply the calibration equations
        for i, site_idx in enumerate(site_indices):
            # Apply d18O calibration equation: d18O = α*SST + β + ε
            d18o_mean = self.d18o_alpha[site_idx] * mean_sst[i] + self.d18o_beta[site_idx]
            d18o_var = (self.d18o_alpha[site_idx]**2) * var_sst[i] + self.d18o_sigma[site_idx]**2
            
            # Apply UK'37 calibration equation: UK'37 = α*SST + β + ε
            uk37_mean = self.uk37_alpha[site_idx] * mean_sst[i] + self.uk37_beta[site_idx]
            uk37_var = (self.uk37_alpha[site_idx]**2) * var_sst[i] + self.uk37_sigma[site_idx]**2
            
            # Add to lists
            d18o_means.append(d18o_mean)
            d18o_vars.append(d18o_var)
            uk37_means.append(uk37_mean)
            uk37_vars.append(uk37_var)
        
        # Stack means and variances
        d18o_means = torch.stack(d18o_means)
        d18o_vars = torch.stack(d18o_vars)
        uk37_means = torch.stack(uk37_means)
        uk37_vars = torch.stack(uk37_vars)
        
        # Return distributions for both proxies
        d18o_dist = torch.distributions.Normal(d18o_means, d18o_vars.sqrt())
        uk37_dist = torch.distributions.Normal(uk37_means, uk37_vars.sqrt())
        
        return d18o_dist, uk37_dist
    
    def log_marginal(self, observations, latent_sst, site_indices):
        """
        Compute the log marginal likelihood of proxy observations given latent SST.
        
        Parameters:
            observations: Tuple of (d18o_values, uk37_values)
            latent_sst: The latent SST distributions from the GP model
            site_indices: Indices of sites corresponding to the latent_sst
            
        Returns:
            Log likelihood of observations
        """
        d18o_values, uk37_values = observations
        
        # Get proxy distributions from forward pass
        d18o_dist, uk37_dist = self.forward(latent_sst, site_indices)
        
        # Compute log probabilities
        d18o_log_prob = d18o_dist.log_prob(d18o_values)
        uk37_log_prob = uk37_dist.log_prob(uk37_values)
        
        # Sum log probabilities across all observations
        return d18o_log_prob.sum() + uk37_log_prob.sum()


class VariationalMultiSiteGP:
    """
    Wrapper class for the full multi-site Gaussian Process model
    with variational inference.
    """
    
    def __init__(self, dataset, use_spatial_correlation=True, num_latents=2):
        """
        Initialize the multi-site GP model.
        
        Parameters:
            dataset: Multi-site dataset with ages, proxies, and site information
            use_spatial_correlation: Whether to use the LMC model or independent GPs
            num_latents: Number of latent processes for the LMC model
        """
        self.dataset = dataset
        self.num_sites = len(dataset['sites'])
        self.use_spatial_correlation = use_spatial_correlation
        self.num_latents = num_latents
        
        # Convert dataset to PyTorch tensors
        self.prepare_data()
        
        # Initialize model components
        self.initialize_model()
    
    def prepare_data(self):
        """Prepare data for the GP model."""
        # Extract data from dataset
        site_ages = self.dataset['ages']
        d18o_values = self.dataset['proxies']['d18o']
        uk37_values = self.dataset['proxies']['uk37']
        
        # Container for prepared data
        self.train_data = {
            'ages': [],
            'd18o': [],
            'uk37': [],
            'site_indices': []
        }
        
        # Combine data from all sites
        for site in range(self.num_sites):
            ages = torch.tensor(site_ages[site], dtype=torch.float32).reshape(-1, 1)
            d18o = torch.tensor(d18o_values[site], dtype=torch.float32)
            uk37 = torch.tensor(uk37_values[site], dtype=torch.float32)
            
            self.train_data['ages'].append(ages)
            self.train_data['d18o'].append(d18o)
            self.train_data['uk37'].append(uk37)
            self.train_data['site_indices'].extend([site] * len(ages))
        
        # Concatenate data
        self.train_data['ages'] = torch.cat(self.train_data['ages'])
        self.train_data['d18o'] = torch.cat(self.train_data['d18o'])
        self.train_data['uk37'] = torch.cat(self.train_data['uk37'])
        self.train_data['site_indices'] = torch.tensor(self.train_data['site_indices'])
    
    def initialize_model(self):
        """Initialize the GP model and likelihood."""
        # Initialize model
        if self.use_spatial_correlation:
            # LMC model with spatial correlations
            self.model = SpatialLMCModel(
                self.train_data['ages'],
                torch.stack([self.train_data['d18o'], self.train_data['uk37']], dim=1),
                self.num_sites,
                self.num_latents
            )
        else:
            # Independent GPs for each site
            self.model = BatchIndependentMultitaskGPModel(
                self.train_data['ages'],
                torch.stack([self.train_data['d18o'], self.train_data['uk37']], dim=1),
                self.num_sites
            )
        
        # Initialize likelihood
        self.likelihood = MultiSiteMultiProxyLikelihood(
            self.num_sites,
            D18O_CALIBRATIONS,
            UK37_CALIBRATIONS
        )
    
    def train_model(self, num_epochs=200, learning_rate=0.01, verbose=True):
        """
        Train the model using variational inference.
        
        Parameters:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
            verbose: Whether to print training progress
        """
        # Set to training mode
        self.model.train()
        self.likelihood.train()
        
        # Define optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=learning_rate)
        
        # Define loss function (ELBO)
        mll = VariationalELBO(
            self.likelihood,
            self.model,
            num_data=len(self.train_data['ages'])
        )
        
        # Training loop
        losses = []
        
        for i in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(self.train_data['ages'])
            
            # Compute loss
            observations = (self.train_data['d18o'], self.train_data['uk37'])
            loss = -mll(output, observations, self.train_data['site_indices'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Print progress
            if verbose and (i + 1) % 20 == 0:
                print(f"Epoch {i+1}/{num_epochs} - Loss: {loss.item():.4f}")
        
        # Set to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        return losses
    
    def predict(self, ages, site_indices):
        """
        Make predictions at the given ages for specific sites.
        
        Parameters:
            ages: Ages to predict at (tensor)
            site_indices: Indices of sites to predict for
            
        Returns:
            mean_sst, lower_ci, upper_ci
        """
        # Set to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get latent SST distributions
            if self.use_spatial_correlation:
                # For LMC model, get site-specific predictions
                latent_means, latent_covars = self.model.get_marginal_predictions(ages, site_indices)
                
                # Extract means and standard deviations
                mean_sst = torch.stack(latent_means)
                std_sst = torch.stack([torch.diag(cov).sqrt() for cov in latent_covars])
            else:
                # For independent GP model, get predictions directly
                latent_dist = self.model(ages)
                
                # Extract site-specific predictions
                mean_sst = latent_dist.mean.reshape(len(site_indices), -1)
                std_sst = latent_dist.variance.sqrt().reshape(len(site_indices), -1)
        
        # Compute confidence intervals
        lower_ci = mean_sst - 2 * std_sst
        upper_ci = mean_sst + 2 * std_sst
        
        return mean_sst, lower_ci, upper_ci
    
    def evaluate(self, true_sst, predicted_sst):
        """
        Evaluate the quality of the SST reconstruction.
        
        Parameters:
            true_sst: True latent SST values
            predicted_sst: Predicted SST values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Flatten arrays
        true_sst_flat = np.concatenate(true_sst)
        predicted_sst_flat = predicted_sst.flatten()
        
        # Match array lengths (use minimum length)
        min_len = min(len(true_sst_flat), len(predicted_sst_flat))
        true_sst_flat = true_sst_flat[:min_len]
        predicted_sst_flat = predicted_sst_flat[:min_len]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_sst_flat, predicted_sst_flat))
        mae = mean_absolute_error(true_sst_flat, predicted_sst_flat)
        r2 = r2_score(true_sst_flat, predicted_sst_flat)
        bias = np.mean(predicted_sst_flat - true_sst_flat)
        std_err = np.std(predicted_sst_flat - true_sst_flat)
        
        # Return as dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'bias': bias,
            'std_err': std_err
        }
        
        return metrics


def plot_multi_site_reconstruction(dataset, predictions, metrics, model_name):
    """
    Plot the multi-site SST reconstruction along with evaluation metrics.
    
    Parameters:
        dataset: Multi-site dataset
        predictions: Dictionary with prediction results
        metrics: Evaluation metrics
        model_name: Name of the model (for title)
        
    Returns:
        matplotlib figure
    """
    n_sites = len(dataset['sites'])
    
    # Create figure
    fig = plt.figure(figsize=(15, 4 * n_sites))
    
    # Create a map in the first row
    ax_map = plt.subplot2grid((n_sites, 3), (0, 0), colspan=3, rowspan=1)
    m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax_map)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='tan', lake_color='lightblue')
    
    # Plot sites
    site_x, site_y = [], []
    for site in range(n_sites):
        lat, lon = dataset['sites'][site]
        x, y = m(lon, lat)
        site_x.append(x)
        site_y.append(y)
        
        # Color based on reconstruction quality
        r2 = metrics['site_r2'][site]
        if r2 > 0.7:
            color = 'g'  # Good
        elif r2 > 0.4:
            color = 'y'  # Moderate
        else:
            color = 'r'  # Poor
        
        m.plot(x, y, 'o', markersize=8, color=color, label=f'Site {site+1}')
        ax_map.text(x, y, f' {site+1}', fontsize=12)
    
    # Draw connections between sites
    for i in range(n_sites):
        for j in range(i+1, n_sites):
            # Line thickness based on correlation strength
            corr = dataset['spatial_correlation'][i, j]
            if corr > 0.2:  # Only show significant correlations
                ax_map.plot([site_x[i], site_x[j]], [site_y[i], site_y[j]], 'b-', 
                           alpha=corr, linewidth=corr*3)
    
    ax_map.set_title(f'Multi-Site Reconstruction Results - {model_name}')
    
    # Add overall metrics text
    metrics_text = (
        f"Overall Metrics:\n"
        f"RMSE: {metrics['rmse']:.3f}°C\n"
        f"R²: {metrics['r2']:.3f}\n"
        f"MAE: {metrics['mae']:.3f}°C"
    )
    ax_map.text(0.02, 0.98, metrics_text, transform=ax_map.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot time series for each site
    for site in range(n_sites):
        ax = plt.subplot2grid((n_sites, 3), (site+1, 0), colspan=3)
        
        # Get data
        ages = dataset['ages'][site]
        true_sst = dataset['true_sst'][site]
        pred_ages = predictions['ages'][site].numpy()
        pred_mean = predictions['mean'][site].numpy()
        pred_lower = predictions['lower'][site].numpy()
        pred_upper = predictions['upper'][site].numpy()
        
        # Plot true SST
        ax.plot(ages, true_sst, 'k-', linewidth=1.5, label='True SST')
        
        # Plot reconstructed SST
        ax.plot(pred_ages, pred_mean, 'g-', linewidth=2, label='Reconstructed SST')
        ax.fill_between(pred_ages, pred_lower, pred_upper, color='g', alpha=0.2,
                       label='95% Confidence Interval')
        
        # Add site-specific metrics
        site_metrics = metrics['site_metrics'][site]
        site_text = (
            f"Site {site+1} Metrics:\n"
            f"RMSE: {site_metrics['rmse']:.3f}°C\n"
            f"R²: {site_metrics['r2']:.3f}\n"
            f"MAE: {site_metrics['mae']:.3f}°C"
        )
        ax.text(0.02, 0.98, site_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add site location
        lat, lon = dataset['sites'][site]
        loc_text = f"Site {site+1}: ({lat:.1f}°N, {lon:.1f}°E)"
        ax.text(0.98, 0.98, loc_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Age (kyr BP)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Site {site+1} Reconstruction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


def create_interactive_visualization(dataset, lmc_predictions, independent_predictions):
    """
    Create interactive visualization comparing LMC vs Independent models.
    
    Parameters:
        dataset: Multi-site dataset
        lmc_predictions: Predictions from LMC model
        independent_predictions: Predictions from Independent model
        
    Returns:
        Interactive figure
    """
    n_sites = len(dataset['sites'])
    
    # Create spatial correlation heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(dataset['spatial_correlation'], cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(n_sites), [f'Site {i+1}' for i in range(n_sites)])
    plt.yticks(range(n_sites), [f'Site {i+1}' for i in range(n_sites)])
    plt.title('Spatial Correlation Matrix')
    
    # Create map with site locations
    plt.figure(figsize=(10, 6))
    m = Basemap(projection='robin', lon_0=0, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='tan', lake_color='lightblue')
    
    # Plot sites
    for site in range(n_sites):
        lat, lon = dataset['sites'][site]
        x, y = m(lon, lat)
        m.plot(x, y, 'ro', markersize=8)
        plt.text(x, y, f' {site+1}', fontsize=12)
    
    plt.title('Site Locations')
    
    # Create a slider to select the site
    site_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_sites-1,
        step=1,
        description='Site:',
        continuous_update=False
    )
    
    # Create a dropdown to select the model
    model_dropdown = widgets.Dropdown(
        options=[('LMC Model', 'lmc'), ('Independent GPs', 'independent')],
        value='lmc',
        description='Model:'
    )
    
    # Create a function to update the plot based on slider and dropdown values
    def update_plot(site, model):
        # Get data
        ages = dataset['ages'][site]
        true_sst = dataset['true_sst'][site]
        
        if model == 'lmc':
            pred_ages = lmc_predictions['ages'][site].numpy()
            pred_mean = lmc_predictions['mean'][site].numpy()
            pred_lower = lmc_predictions['lower'][site].numpy()
            pred_upper = lmc_predictions['upper'][site].numpy()
            model_name = 'LMC Model'
        else:
            pred_ages = independent_predictions['ages'][site].numpy()
            pred_mean = independent_predictions['mean'][site].numpy()
            pred_lower = independent_predictions['lower'][site].numpy()
            pred_upper = independent_predictions['upper'][site].numpy()
            model_name = 'Independent GPs'
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot true SST
        plt.plot(ages, true_sst, 'k-', linewidth=1.5, label='True SST')
        
        # Plot reconstructed SST
        plt.plot(pred_ages, pred_mean, 'g-', linewidth=2, label=f'{model_name} Reconstruction')
        plt.fill_between(pred_ages, pred_lower, pred_upper, color='g', alpha=0.2,
                        label='95% Confidence Interval')
        
        # Add site information
        lat, lon = dataset['sites'][site]
        plt.title(f'Site {site+1} ({lat:.1f}°N, {lon:.1f}°E) - {model_name}')
        plt.xlabel('Age (kyr BP)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    # Create interactive output
    interactive_output = widgets.interactive_output(
        update_plot,
        {'site': site_slider, 'model': model_dropdown}
    )
    
    # Display widgets and output
    display(widgets.HBox([site_slider, model_dropdown]))
    display(interactive_output)


def create_animation(dataset, predictions):
    """
    Create an animation showing the temporal evolution of SST across all sites.
    
    Parameters:
        dataset: Multi-site dataset
        predictions: Dictionary with prediction results
        
    Returns:
        Animation
    """
    n_sites = len(dataset['sites'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create map
    m = Basemap(projection='robin', lon_0=0, resolution='c', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='tan', lake_color='lightblue')
    
    # Extract site coordinates for map
    site_x, site_y = [], []
    for site in range(n_sites):
        lat, lon = dataset['sites'][site]
        x, y = m(lon, lat)
        site_x.append(x)
        site_y.append(y)
    
    # Initialize scatter plot
    scatter = ax.scatter(site_x, site_y, c=[15]*n_sites, cmap='plasma', 
                         s=100, vmin=10, vmax=30, edgecolor='k')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('SST (°C)')
    
    # Add time label
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Get interpolated predictions for animation
    # Ensure all sites have the same time points
    common_ages = np.linspace(TIME_MIN, TIME_MAX, 200)
    interp_preds = []
    
    for site in range(n_sites):
        pred_ages = predictions['ages'][site].numpy()
        pred_mean = predictions['mean'][site].numpy()
        
        # Interpolate to common time points
        interp_pred = np.interp(common_ages, pred_ages, pred_mean)
        interp_preds.append(interp_pred)
    
    interp_preds = np.array(interp_preds)
    
    # Animation update function
    def update(frame):
        # Get SST values for this time point
        time_point = common_ages[frame]
        sst_values = interp_preds[:, frame]
        
        # Update scatter colors
        scatter.set_array(sst_values)
        
        # Update time text
        time_text.set_text(f'Time: {time_point:.0f} kyr BP')
        
        return scatter, time_text
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(common_ages),
        interval=100, blit=True
    )
    
    # Set title
    ax.set_title('Temporal Evolution of SST Reconstruction')
    
    # Save animation
    ani.save(os.path.join(output_dir, 'sst_evolution.mp4'), writer='ffmpeg', dpi=200)
    
    return ani


def generate_multi_site_heatmap(dataset, predictions, metrics):
    """
    Generate a heatmap visualization of the spatiotemporal SST patterns.
    
    Parameters:
        dataset: Multi-site dataset
        predictions: Dictionary with prediction results
        metrics: Evaluation metrics
        
    Returns:
        matplotlib figure
    """
    n_sites = len(dataset['sites'])
    
    # Create interpolated predictions for visualization
    # Ensure all sites have the same time points
    common_ages = np.linspace(TIME_MIN, TIME_MAX, 200)
    
    # Extract true and predicted SST values
    true_sst_matrix = np.zeros((n_sites, len(common_ages)))
    pred_sst_matrix = np.zeros((n_sites, len(common_ages)))
    
    for site in range(n_sites):
        # True SST
        site_ages = dataset['ages'][site]
        site_sst = dataset['true_sst'][site]
        true_sst_matrix[site] = np.interp(common_ages, site_ages, site_sst)
        
        # Predicted SST
        pred_ages = predictions['ages'][site].numpy()
        pred_mean = predictions['mean'][site].numpy()
        pred_sst_matrix[site] = np.interp(common_ages, pred_ages, pred_mean)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                       gridspec_kw={'height_ratios': [1, 1, 0.5]})
    
    # Plot true SST heatmap
    im1 = ax1.pcolormesh(common_ages, np.arange(n_sites), true_sst_matrix, 
                        cmap='plasma', vmin=10, vmax=30)
    ax1.set_title('True Latent SST')
    ax1.set_ylabel('Site')
    ax1.set_yticks(np.arange(n_sites) + 0.5)
    ax1.set_yticklabels([f'Site {i+1}' for i in range(n_sites)])
    ax1.set_xlabel('Age (kyr BP)')
    plt.colorbar(im1, ax=ax1, label='SST (°C)')
    
    # Plot predicted SST heatmap
    im2 = ax2.pcolormesh(common_ages, np.arange(n_sites), pred_sst_matrix, 
                        cmap='plasma', vmin=10, vmax=30)
    ax2.set_title('Reconstructed SST')
    ax2.set_ylabel('Site')
    ax2.set_yticks(np.arange(n_sites) + 0.5)
    ax2.set_yticklabels([f'Site {i+1}' for i in range(n_sites)])
    ax2.set_xlabel('Age (kyr BP)')
    plt.colorbar(im2, ax=ax2, label='SST (°C)')
    
    # Plot reconstruction error
    error_matrix = true_sst_matrix - pred_sst_matrix
    im3 = ax3.pcolormesh(common_ages, np.arange(n_sites), error_matrix, 
                        cmap='RdBu_r', vmin=-5, vmax=5)
    ax3.set_title('Reconstruction Error (True - Predicted)')
    ax3.set_ylabel('Site')
    ax3.set_yticks(np.arange(n_sites) + 0.5)
    ax3.set_yticklabels([f'Site {i+1}' for i in range(n_sites)])
    ax3.set_xlabel('Age (kyr BP)')
    plt.colorbar(im3, ax=ax3, label='Error (°C)')
    
    # Add site information
    site_info = []
    for site in range(n_sites):
        lat, lon = dataset['sites'][site]
        site_info.append(f'Site {site+1}: ({lat:.1f}°N, {lon:.1f}°E), R²: {metrics["site_r2"][site]:.3f}')
    
    fig.text(0.5, 0.01, '\n'.join(site_info), ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig


def run_multi_site_experiment():
    """
    Run the multi-site latent SST reconstruction experiment.
    """
    print("Starting Multi-Site Latent SST Reconstruction Experiment...")
    print("This experiment implements spatially-aware GP reconstruction across multiple sites")
    start_time = time.time()
    
    # Step 1: Generate synthetic data with high noise and spatial correlations
    print("\nGenerating synthetic multi-site data with high noise and spatial correlations...")
    data_generator = MultiSiteSyntheticData(n_points=N_POINTS)
    dataset = data_generator.generate_dataset()
    
    # Plot the synthetic dataset
    print("\nPlotting synthetic multi-site dataset...")
    fig_data = data_generator.plot_dataset(dataset)
    fig_data.savefig(os.path.join(output_dir, "multi_site_synthetic_data.png"), 
                    dpi=300, bbox_inches='tight')
    plt.close(fig_data)
    
    # Step 2: Initialize and train the LMC model with spatial correlations
    print("\nTraining LMC model with spatial correlations...")
    lmc_model = VariationalMultiSiteGP(dataset, use_spatial_correlation=True, num_latents=3)
    lmc_losses = lmc_model.train_model(num_epochs=200, learning_rate=0.01)
    
    # Step 3: Initialize and train the independent GP model (no spatial correlations)
    print("\nTraining independent GP model (no spatial correlations)...")
    independent_model = VariationalMultiSiteGP(dataset, use_spatial_correlation=False)
    ind_losses = independent_model.train_model(num_epochs=200, learning_rate=0.01)
    
    # Step 4: Make predictions for both models
    print("\nMaking predictions...")
    
    # Prepare prediction ages for each site
    pred_ages = []
    for site in range(lmc_model.num_sites):
        ages = torch.linspace(TIME_MIN, TIME_MAX, 100).reshape(-1, 1)
        pred_ages.append(ages)
    
    # LMC model predictions
    lmc_predictions = {
        'ages': pred_ages,
        'mean': [],
        'lower': [],
        'upper': []
    }
    
    for site in range(lmc_model.num_sites):
        mean, lower, upper = lmc_model.predict(pred_ages[site], [site])
        lmc_predictions['mean'].append(mean)
        lmc_predictions['lower'].append(lower)
        lmc_predictions['upper'].append(upper)
    
    # Independent model predictions
    ind_predictions = {
        'ages': pred_ages,
        'mean': [],
        'lower': [],
        'upper': []
    }
    
    for site in range(independent_model.num_sites):
        mean, lower, upper = independent_model.predict(pred_ages[site], [site])
        ind_predictions['mean'].append(mean)
        ind_predictions['lower'].append(lower)
        ind_predictions['upper'].append(upper)
    
    # Step 5: Evaluate both models
    print("\nEvaluating model performance...")
    
    # LMC model evaluation
    lmc_site_metrics = []
    lmc_site_r2 = []
    
    for site in range(lmc_model.num_sites):
        # Get true and predicted SST
        true_sst = dataset['true_sst'][site]
        pred_mean = lmc_predictions['mean'][site].numpy()
        
        # Calculate site-specific metrics
        site_metrics = lmc_model.evaluate([true_sst], pred_mean)
        lmc_site_metrics.append(site_metrics)
        lmc_site_r2.append(site_metrics['r2'])
    
    # Calculate overall metrics
    lmc_metrics = lmc_model.evaluate(dataset['true_sst'], 
                                    np.concatenate([m.numpy() for m in lmc_predictions['mean']]))
    lmc_metrics['site_metrics'] = lmc_site_metrics
    lmc_metrics['site_r2'] = lmc_site_r2
    
    # Independent model evaluation
    ind_site_metrics = []
    ind_site_r2 = []
    
    for site in range(independent_model.num_sites):
        # Get true and predicted SST
        true_sst = dataset['true_sst'][site]
        pred_mean = ind_predictions['mean'][site].numpy()
        
        # Calculate site-specific metrics
        site_metrics = independent_model.evaluate([true_sst], pred_mean)
        ind_site_metrics.append(site_metrics)
        ind_site_r2.append(site_metrics['r2'])
    
    # Calculate overall metrics
    ind_metrics = independent_model.evaluate(dataset['true_sst'], 
                                           np.concatenate([m.numpy() for m in ind_predictions['mean']]))
    ind_metrics['site_metrics'] = ind_site_metrics
    ind_metrics['site_r2'] = ind_site_r2
    
    # Step 6: Generate visualizations
    print("\nGenerating visualizations...")
    
    # LMC model reconstruction plot
    fig_lmc = plot_multi_site_reconstruction(dataset, lmc_predictions, lmc_metrics, 
                                           "LMC Model with Spatial Correlations")
    fig_lmc.savefig(os.path.join(output_dir, "lmc_reconstruction.png"), 
                   dpi=300, bbox_inches='tight')
    plt.close(fig_lmc)
    
    # Independent model reconstruction plot
    fig_ind = plot_multi_site_reconstruction(dataset, ind_predictions, ind_metrics,
                                           "Independent GPs (No Spatial Correlations)")
    fig_ind.savefig(os.path.join(output_dir, "independent_reconstruction.png"), 
                   dpi=300, bbox_inches='tight')
    plt.close(fig_ind)
    
    # Heatmap visualization
    fig_heat = generate_multi_site_heatmap(dataset, lmc_predictions, lmc_metrics)
    fig_heat.savefig(os.path.join(output_dir, "spatiotemporal_heatmap.png"), 
                    dpi=300, bbox_inches='tight')
    plt.close(fig_heat)
    
    # Animation
    print("\nCreating animation of SST evolution...")
    animation = create_animation(dataset, lmc_predictions)
    
    # Step 7: Print results summary
    print("\n===== SUMMARY OF RESULTS =====\n")
    
    print("Input Data Characteristics:")
    print(f"  Number of Sites: {lmc_model.num_sites}")
    print(f"  Time Range: {TIME_MIN}-{TIME_MAX} kyr BP")
    print(f"  Data Points per Site: ~{N_POINTS}")
    print(f"  δ18O Noise Level (σ): 1.0-1.5")
    print(f"  UK'37 Noise Level (σ): 0.3-0.5")
    
    print("\nLMC Model with Spatial Correlations:")
    print(f"  Overall RMSE: {lmc_metrics['rmse']:.4f}°C")
    print(f"  Overall R²: {lmc_metrics['r2']:.4f}")
    print(f"  Overall MAE: {lmc_metrics['mae']:.4f}°C")
    
    print("\n  Site-Specific Performance:")
    for site in range(lmc_model.num_sites):
        print(f"    Site {site+1}: RMSE={lmc_metrics['site_metrics'][site]['rmse']:.4f}°C, "
              f"R²={lmc_metrics['site_metrics'][site]['r2']:.4f}")
    
    print("\nIndependent GP Model (No Spatial Correlations):")
    print(f"  Overall RMSE: {ind_metrics['rmse']:.4f}°C")
    print(f"  Overall R²: {ind_metrics['r2']:.4f}")
    print(f"  Overall MAE: {ind_metrics['mae']:.4f}°C")
    
    print("\n  Site-Specific Performance:")
    for site in range(independent_model.num_sites):
        print(f"    Site {site+1}: RMSE={ind_metrics['site_metrics'][site]['rmse']:.4f}°C, "
              f"R²={ind_metrics['site_metrics'][site]['r2']:.4f}")
    
    # Calculate performance improvement
    rmse_improvement = ((ind_metrics['rmse'] - lmc_metrics['rmse']) / 
                        ind_metrics['rmse']) * 100
    r2_improvement = ((lmc_metrics['r2'] - ind_metrics['r2']) / 
                     max(0.01, ind_metrics['r2'])) * 100
    
    print("\nPerformance Improvement from Spatial Correlation Modeling:")
    print(f"  RMSE Reduction: {rmse_improvement:.2f}%")
    print(f"  R² Improvement: {r2_improvement:.2f}%")
    
    # Save metrics to CSV
    results_df = pd.DataFrame({
        'Model': ['LMC Model', 'Independent GPs'],
        'RMSE': [lmc_metrics['rmse'], ind_metrics['rmse']],
        'MAE': [lmc_metrics['mae'], ind_metrics['mae']],
        'R2': [lmc_metrics['r2'], ind_metrics['r2']],
        'Bias': [lmc_metrics['bias'], ind_metrics['bias']],
        'StdErr': [lmc_metrics['std_err'], ind_metrics['std_err']]
    })
    
    results_df.to_csv(os.path.join(output_dir, "multi_site_metrics.csv"), index=False)
    
    # Site-specific metrics
    site_results = []
    for site in range(lmc_model.num_sites):
        site_results.append({
            'Site': site+1,
            'LMC_RMSE': lmc_metrics['site_metrics'][site]['rmse'],
            'LMC_R2': lmc_metrics['site_metrics'][site]['r2'],
            'Independent_RMSE': ind_metrics['site_metrics'][site]['rmse'],
            'Independent_R2': ind_metrics['site_metrics'][site]['r2'],
            'RMSE_Improvement': ((ind_metrics['site_metrics'][site]['rmse'] - 
                                 lmc_metrics['site_metrics'][site]['rmse']) / 
                                ind_metrics['site_metrics'][site]['rmse']) * 100
        })
    
    site_df = pd.DataFrame(site_results)
    site_df.to_csv(os.path.join(output_dir, "site_specific_metrics.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}")
    
    # Print runtime
    end_time = time.time()
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds.")
    
    return dataset, lmc_predictions, ind_predictions, lmc_metrics, ind_metrics


if __name__ == "__main__":
    run_multi_site_experiment()