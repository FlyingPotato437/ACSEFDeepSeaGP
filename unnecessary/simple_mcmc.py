"""
Simple MCMC Implementation for Bayesian GP State-Space Models

This module provides a mathematically robust yet simple MCMC implementation 
for GPyTorch models. It uses random walk Metropolis sampling with parameter-specific
scaling to ensure good mixing without requiring gradients.
"""

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from tqdm.auto import tqdm
import os

class SimpleMCMC:
    """
    A simplified MCMC implementation for GPyTorch models that focuses on
    robustness rather than efficiency.
    
    This implementation uses random walk Metropolis sampling without gradients,
    making it compatible with any GPyTorch model.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        n_samples: int = 500,
        burn_in: int = 200,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the MCMC sampler.
        
        Args:
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            n_samples: Number of posterior samples to generate
            burn_in: Number of burn-in samples to discard
            random_state: Random seed for reproducibility 
            verbose: Whether to print progress information
        """
        self.model = model
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.verbose = verbose
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Extract model parameters
        self.params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.detach().clone()
        
        # Define parameter-specific scales for proposal distribution
        self.param_scales = {}
        for name in self.params:
            if 'lengthscale' in name:
                self.param_scales[name] = 0.1  # Larger changes for lengthscale
            elif 'noise' in name:
                self.param_scales[name] = 0.05  # Smaller changes for noise
            elif 'outputscale' in name:
                self.param_scales[name] = 0.1  # Medium changes for outputscale
            elif 'period' in name:
                self.param_scales[name] = 0.2  # Larger changes for period
            else:
                self.param_scales[name] = 0.1  # Default
                
        # Storage for samples and diagnostics
        self.samples = {}
        self.log_probs = []
        self.acceptance_rate = 0.0
    
    def _log_prior(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log prior probability of parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log prior probability
        """
        log_prior = 0.0
        
        # Calculate log prior based on parameter types
        for name, param in parameters.items():
            param_value = param.item() if param.numel() == 1 else param[0].item()
            
            # Lengthscale parameters (positive, so use log-normal prior)
            if 'lengthscale' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Log-normal prior centered at 1.0 with sigma=1.0
                log_prior += -0.5 * (np.log(param_value))**2 - np.log(param_value)
            
            # Noise parameters (positive, smaller values preferred)
            elif 'noise' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Half-normal prior centered at 0 with sigma=0.1
                log_prior += -0.5 * (param_value / 0.1)**2
            
            # Outputscale parameters (positive)
            elif 'outputscale' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Log-normal prior
                log_prior += -0.5 * (np.log(param_value))**2 - np.log(param_value)
            
            # Period parameters for periodic kernels (positive, centered around common cycles)
            elif 'period' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Normal prior centered at 41 (for Milankovitch cycles)
                log_prior += -0.5 * ((param_value - 41.0) / 10.0)**2
        
        return log_prior
    
    def _log_likelihood(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log likelihood of parameters given the data.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log likelihood value
        """
        # Store original parameters to restore later
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.detach().clone()
        
        # Set model parameters to proposed values
        for name, value in parameters.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)
        
        # Compute log likelihood
        try:
            # Get training data
            if hasattr(self.model, 'train_inputs') and self.model.train_inputs:
                x_train = self.model.train_inputs[0]
            elif hasattr(self.model, 'train_x'):
                x_train = self.model.train_x
            else:
                raise ValueError("Cannot find training inputs in model")
                
            if hasattr(self.model, 'train_targets'):
                y_train = self.model.train_targets
            elif hasattr(self.model, 'train_y'):
                y_train = self.model.train_y
            else:
                raise ValueError("Cannot find training targets in model")
            
            # Compute log likelihood using marginal log likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                self.model.eval()
                output = self.model(x_train)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                log_lik = mll(output, y_train).item()
        except Exception as e:
            # If there's an error, return very negative log likelihood
            log_lik = -1e10
            if self.verbose:
                print(f"Likelihood error: {str(e)}")
        
        # Restore original parameters
        for name, value in original_params.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)
        
        return log_lik
    
    def _log_posterior(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log posterior probability (log_likelihood + log_prior).
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability
        """
        log_prior = self._log_prior(parameters)
        
        # If prior is -inf, we can skip likelihood calculation
        if log_prior == -np.inf:
            return -np.inf
        
        log_lik = self._log_likelihood(parameters)
        return log_lik + log_prior
    
    def run_sampling(self, progress_bar: bool = True) -> Dict[str, np.ndarray]:
        """
        Run MCMC sampling using random walk Metropolis algorithm.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary of parameter samples
        """
        if self.verbose:
            print("Starting MCMC sampling...")
        
        # Initialize storage for samples
        for name in self.params:
            self.samples[name] = []
        
        # Start with current model parameters
        current_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                current_params[name] = param.detach().clone()
        
        # Calculate initial log posterior
        current_log_post = self._log_posterior(current_params)
        
        # Setup for progress tracking
        total_iterations = self.burn_in + self.n_samples
        accepted = 0
        
        # Setup progress bar if requested
        if progress_bar:
            pbar = tqdm(total=total_iterations, desc="MCMC Sampling")
        
        # Main sampling loop
        for i in range(total_iterations):
            # Propose new parameters by perturbing one parameter at a time
            for name in current_params:
                # Create copy of current parameters to modify
                proposed_params = {k: v.clone() for k, v in current_params.items()}
                
                # Generate proposal by adding random noise
                scale = self.param_scales[name]
                
                # Choose proposal scale based on parameter dimensionality
                if proposed_params[name].numel() == 1:  # Scalar parameter
                    # Add random noise to scalar parameter
                    noise = torch.randn(1).item() * scale * proposed_params[name].abs().item()
                    proposed_params[name] = proposed_params[name] + noise
                else:  # Vector parameter
                    # Add random noise to each element
                    noise = torch.randn_like(proposed_params[name]) * scale * proposed_params[name].abs()
                    proposed_params[name] = proposed_params[name] + noise
                
                # Ensure parameters that should be positive stay positive
                if any(s in name for s in ['lengthscale', 'noise', 'outputscale', 'period']):
                    proposed_params[name] = torch.abs(proposed_params[name])
                
                # Calculate log posterior for proposal
                proposed_log_post = self._log_posterior(proposed_params)
                
                # Metropolis acceptance criterion
                log_ratio = proposed_log_post - current_log_post
                
                # Accept or reject proposal
                if np.log(np.random.random()) < log_ratio:
                    # Accept proposal
                    current_params[name] = proposed_params[name].clone()
                    current_log_post = proposed_log_post
                    accepted += 1
            
            # Store sample after burn-in
            if i >= self.burn_in:
                for name, param in current_params.items():
                    self.samples[name].append(param.detach().cpu().clone().numpy())
                self.log_probs.append(current_log_post)
            
            # Update progress bar
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix({
                    "log_prob": f"{current_log_post:.2f}",
                    "accept": f"{accepted/(i+1):.2f}"
                })
        
        # Close progress bar
        if progress_bar:
            pbar.close()
        
        # Calculate acceptance rate
        self.acceptance_rate = accepted / (total_iterations * len(current_params))
        
        # Convert sample lists to numpy arrays
        for name in self.samples:
            if self.samples[name]:
                self.samples[name] = np.array(self.samples[name])
        
        if self.verbose:
            print(f"MCMC sampling completed - {self.n_samples} samples with {self.acceptance_rate:.2f} acceptance rate")
        
        return self.samples
    
    def sample_posterior_predictive(self, x_test, n_samples=20):
        """
        Generate predictions using posterior samples.
        
        Args:
            x_test: Test input points
            n_samples: Number of samples to use
            
        Returns:
            Dictionary with prediction results
        """
        if not hasattr(self, 'samples') or not self.samples:
            raise ValueError("No samples available. Run sampling first.")
        
        # Convert x_test to tensor if needed
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test, dtype=torch.float32)
        
        # Get the device of the model
        device = next(self.model.parameters()).device
        x_test = x_test.to(device)
        
        # Store original parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.detach().clone()
        
        # Sample indices
        sample_indices = np.random.choice(
            len(self.samples[list(self.samples.keys())[0]]), 
            size=min(n_samples, len(self.samples[list(self.samples.keys())[0]])),
            replace=False
        )
        
        # Generate predictions for each parameter set
        predictions = []
        
        for idx in sample_indices:
            # Set model parameters to sampled values
            for name in self.samples:
                param = dict(self.model.named_parameters()).get(name)
                if param is not None:
                    sample = self.samples[name][idx]
                    
                    # Handle different dimensionality
                    if isinstance(sample, np.ndarray):
                        if param.numel() == 1 and sample.size > 1:
                            # Take first element for scalar parameters
                            sample_tensor = torch.tensor(sample.flat[0], dtype=torch.float32, device=device)
                        else:
                            # Match shape with parameter
                            try:
                                sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
                                if sample_tensor.shape != param.shape:
                                    sample_tensor = sample_tensor.reshape(param.shape)
                            except:
                                # Use original parameter as fallback
                                sample_tensor = param.detach().clone()
                    else:
                        sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
                    
                    # Set parameter value
                    with torch.no_grad():
                        param.copy_(sample_tensor)
            
            # Generate prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.model(x_test)
                prediction = self.likelihood(output).sample()
                predictions.append(prediction.cpu().numpy())
        
        # Restore original parameters
        for name, value in original_params.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'samples': predictions
        }
    
    def plot_traces(self, figure_path=None):
        """
        Plot parameter traces and posterior distributions.
        
        Args:
            figure_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, 'samples') or not self.samples:
            raise ValueError("No samples available. Run sampling first.")
        
        # Count parameters
        n_params = len(self.samples)
        
        # Create figure
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 4 * n_params))
        
        # Handle single parameter case
        if n_params == 1:
            axes = axes.reshape(1, 2)
        
        # Plot each parameter
        for i, (name, samples) in enumerate(self.samples.items()):
            # Get flattened samples for the first element if multidimensional
            if samples.ndim > 1:
                samples_flat = samples[:, 0].flatten()
            else:
                samples_flat = samples.flatten()
            
            # Trace plot
            axes[i, 0].plot(samples_flat)
            axes[i, 0].set_title(f"Trace for {name}")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram
            axes[i, 1].hist(samples_flat, bins=30, density=True, alpha=0.7)
            
            # Add mean and CI
            mean = np.mean(samples_flat)
            q025, q975 = np.percentile(samples_flat, [2.5, 97.5])
            
            axes[i, 1].axvline(mean, color='r', linestyle='-', alpha=0.7)
            axes[i, 1].axvline(q025, color='r', linestyle='--', alpha=0.5)
            axes[i, 1].axvline(q975, color='r', linestyle='--', alpha=0.5)
            
            # Add text with statistics
            axes[i, 1].text(0.05, 0.95, f"Mean: {mean:.4f}\n95% CI: [{q025:.4f}, {q975:.4f}]",
                          transform=axes[i, 1].transAxes, va='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i, 1].set_title(f"Posterior for {name}")
            axes[i, 1].set_xlabel("Value")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].grid(True, alpha=0.3)
        
        # Add tight layout
        plt.tight_layout()
        
        # Save if path provided
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            
            # Also create log probability plot
            if self.log_probs:
                fig_log, ax = plt.subplots(figsize=(10, 5))
                ax.plot(self.log_probs)
                ax.set_title("Log Posterior Trace")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Log Posterior")
                ax.grid(True, alpha=0.3)
                
                log_path = figure_path.replace('.png', '_logprob.png')
                fig_log.savefig(log_path, dpi=300, bbox_inches='tight')
        
        return fig


def test_simple_mcmc_on_sparse_data(n_points=20, irregularity=0.7):
    """
    Test the SimpleMCMC implementation on sparse synthetic data.
    
    Args:
        n_points: Number of data points
        irregularity: Irregularity of data sampling (0-1)
        
    Returns:
        Dictionary with results
    """
    from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse
    import os
    
    # Create output directory
    output_dir = "data/results/simple_mcmc_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing SimpleMCMC on sparse data with {n_points} points and {irregularity} irregularity...")
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data_sparse(
        n_points=n_points,
        age_min=0,
        age_max=200,
        irregularity=irregularity,
        proxy_types=['d18O'],
        random_state=42
    )
    
    # Create and fit model
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O'],
        kernel_type='rbf',  # Simpler kernel for testing
        n_mcmc_samples=100,
        random_state=42
    )
    
    model.fit(synthetic_data['proxy_data'], training_iterations=200)
    
    # Extract data for visualization
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    
    # Create test points for predictions
    test_ages = np.linspace(0, 200, 200)
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    
    # Apply SimpleMCMC to the fitted model
    mcmc = SimpleMCMC(
        model=model.gp_model,
        likelihood=model.likelihood,
        n_samples=100,
        burn_in=100,
        random_state=42
    )
    
    # Run sampling
    samples = mcmc.run_sampling(progress_bar=True)
    
    # Plot parameter traces
    mcmc.plot_traces(figure_path=f"{output_dir}/parameter_traces.png")
    
    # Generate posterior predictions
    predictions = mcmc.sample_posterior_predictive(torch.tensor(test_ages, dtype=torch.float32), n_samples=20)
    
    # Update model with samples for future use
    model.posterior_samples = samples
    
    # Create comparison plot with original model prediction
    mean, lower, upper = model.predict(test_ages)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot truth and data
    for ax in axes:
        ax.plot(test_ages, test_true_sst, 'k-', linewidth=2, label='True SST')
        
        # Plot proxy data
        d18o_data = synthetic_data['proxy_data']['d18O']
        d18o_ages = d18o_data['age']
        d18o_values = d18o_data['value']
        d18o_sst = (d18o_values - 3.0) * -4.54545
        ax.scatter(d18o_ages, d18o_sst, marker='o', color='green', s=40, alpha=0.7, label='d18O data')
    
    # Plot original model prediction
    axes[0].plot(test_ages, mean, 'b-', linewidth=2, label='Original Model Mean')
    axes[0].fill_between(test_ages, lower, upper, color='b', alpha=0.2, label='Original Model 95% CI')
    axes[0].set_title('Original Model Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MCMC prediction
    axes[1].plot(test_ages, predictions['mean'], 'r-', linewidth=2, label='MCMC Mean')
    axes[1].fill_between(test_ages, predictions['lower_ci'], predictions['upper_ci'], color='r', alpha=0.2, label='MCMC 95% CI')
    
    # Plot some sample predictions
    for i in range(min(5, predictions['samples'].shape[0])):
        axes[1].plot(test_ages, predictions['samples'][i], 'r-', linewidth=0.5, alpha=0.2)
    
    axes[1].set_title('SimpleMCMC Posterior Prediction')
    axes[1].set_xlabel('Age (kyr)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Reverse x-axis for geological convention
    for ax in axes:
        ax.set_xlim(max(test_ages), min(test_ages))
        ax.set_ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    
    print(f"Test completed. Results saved to {output_dir}")
    
    return {
        'model': model,
        'mcmc': mcmc,
        'synthetic_data': synthetic_data,
        'predictions': predictions
    }


if __name__ == "__main__":
    test_simple_mcmc_on_sparse_data(20, 0.7)