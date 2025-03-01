"""
MCMC Sampling Module for Bayesian Gaussian Process State-Space Models

This module provides robust MCMC implementations for sampling from the posterior
distributions of GP model parameters and generating uncertainty estimates for
paleoclimate reconstructions.
"""

import numpy as np
import torch
import gpytorch
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm


class MCMCSampler:
    """
    Base class for MCMC sampling from GP model posteriors.
    
    This implementation focuses on generating samples for uncertainty
    quantification in paleoclimate reconstructions.
    """
    
    def __init__(
        self,
        model: Any,
        likelihood: gpytorch.likelihoods.Likelihood,
        n_samples: int = 1000,
        burn_in: int = 200,
        thinning: int = 2,
        step_size: float = 0.1,
        target_acceptance: float = 0.6,
        adaptation_steps: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the MCMC sampler.
        
        Args:
            model: GP model to sample from
            likelihood: Model likelihood
            n_samples: Number of samples to draw
            burn_in: Number of burn-in steps
            thinning: Thinning factor to reduce autocorrelation
            step_size: Initial step size for proposals
            target_acceptance: Target acceptance rate for adaptation
            adaptation_steps: Number of steps for step size adaptation
            random_state: Random seed
        """
        self.model = model
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.step_size = step_size
        self.target_acceptance = target_acceptance
        self.adaptation_steps = adaptation_steps
        
        # Set random seeds
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Storage for samples and diagnostics
        self.samples = {}
        self.acceptance_rate = 0.0
        self.log_probs = []
        
        # Get parameter names and initial values
        self.param_names = []
        self.param_shapes = {}
        self.initial_values = {}
        
        # Extract initial parameter values from model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
                self.initial_values[name] = param.detach().clone()
    
    def log_posterior(self, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate log posterior probability of parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability
        """
        # Set model parameters
        for name, param in parameters.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                param_obj.data = param.data
        
        # Calculate log likelihood
        try:
            output = self.model(self.model.train_inputs[0])
            log_likelihood = self.model.likelihood.log_marginal(
                output, self.model.train_targets
            )
        except Exception as e:
            # Return very low probability for numerical errors
            return torch.tensor(-1e10, device=self._device())
        
        # Calculate log prior
        log_prior = 0.0
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                for prior_name, prior, closure, _ in self.model.named_priors():
                    if prior_name == name:
                        try:
                            log_prior = log_prior + prior.log_prob(closure()).sum()
                        except Exception:
                            # Skip if prior computation fails
                            pass
        
        return log_likelihood + log_prior
    
    def _get_param_by_name(self, name: str) -> Optional[torch.nn.Parameter]:
        """Helper to get parameter object by name."""
        for param_name, param in self.model.named_parameters():
            if param_name == name:
                return param
        return None
    
    def _device(self) -> torch.device:
        """Helper to get the device of the model."""
        return next(self.model.parameters()).device
    
    def run_hmc(self, progress_bar: bool = True):
        """
        Run Hamiltonian Monte Carlo sampling.
        
        Args:
            progress_bar: Whether to show a progress bar
        """
        device = self._device()
        
        # Initialize storage for samples
        total_samples = self.burn_in + self.n_samples * self.thinning
        
        # Initialize parameter vectors for HMC
        current_params = {}
        for name in self.param_names:
            current_params[name] = self._get_param_by_name(name).detach().clone()
        
        # Calculate initial log probability
        current_log_prob = self.log_posterior(current_params)
        
        # Initialize step size adaptation
        step_size = torch.tensor(self.step_size, device=device)
        log_step_size = torch.log(step_size)
        
        # Acceptance tracking
        n_accepted = 0
        
        # Sample collection
        samples = {name: [] for name in self.param_names}
        log_probs = []
        
        # Progress tracking
        iterator = range(total_samples)
        if progress_bar:
            iterator = tqdm(iterator, desc="HMC Sampling")
        
        # Main sampling loop
        for i in iterator:
            # Store sample (after burn-in and accounting for thinning)
            if i >= self.burn_in and (i - self.burn_in) % self.thinning == 0:
                for name in self.param_names:
                    samples[name].append(current_params[name].detach().cpu().clone())
                log_probs.append(current_log_prob.item())
            
            # Momentum variables
            momentum = {}
            for name, shape in self.param_shapes.items():
                momentum[name] = torch.randn(shape, device=device)
            
            # Initial half step for momentum
            proposed_momentum = self._leapfrog_step_begin(
                current_params, momentum, step_size
            )
            
            # Full leapfrog step for position
            proposed_params = {}
            for name in self.param_names:
                proposed_params[name] = current_params[name] + step_size * proposed_momentum[name]
            
            # Calculate gradients at the new position
            proposed_log_prob = self.log_posterior(proposed_params)
            
            # Final half step for momentum
            proposed_momentum = self._leapfrog_step_end(
                proposed_params, proposed_momentum, step_size
            )
            
            # Calculate Hamiltonian (energy) for both states
            current_energy = -current_log_prob + self._kinetic_energy(momentum)
            proposed_energy = -proposed_log_prob + self._kinetic_energy(proposed_momentum)
            
            # Metropolis-Hastings acceptance
            energy_change = proposed_energy - current_energy
            if torch.rand(1, device=device) < torch.exp(-energy_change):
                current_params = proposed_params
                current_log_prob = proposed_log_prob
                n_accepted += 1
            
            # Step size adaptation during burn-in
            if i < self.adaptation_steps:
                # Adapt step size to achieve target acceptance rate
                accept_stat = 1.0 if energy_change < 0 else torch.exp(-energy_change).item()
                log_step_size = self._adapt_step_size(
                    log_step_size, self.target_acceptance, accept_stat, i
                )
                step_size = torch.exp(log_step_size)
        
        # Calculate final acceptance rate
        self.acceptance_rate = n_accepted / total_samples
        
        # Convert collected samples to numpy arrays
        for name in self.param_names:
            stacked_samples = torch.stack(samples[name])
            self.samples[name] = stacked_samples.numpy()
        
        self.log_probs = log_probs
        
        print(f"Sampling complete. Acceptance rate: {self.acceptance_rate:.2f}")
        
        # Return to evaluation mode and set parameters to posterior means
        self._set_posterior_means()
    
    def _leapfrog_step_begin(
        self, 
        params: Dict[str, torch.Tensor], 
        momentum: Dict[str, torch.Tensor], 
        step_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """First half leapfrog step for momentum."""
        # Set params
        for name, param in params.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                param_obj.data = param.data
        
        # Zero gradients
        if hasattr(self.model, 'zero_grad'):
            self.model.zero_grad()
        else:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # Compute log posterior
        log_prob = self.log_posterior(params)
        
        # Calculate gradients
        log_prob.backward()
        
        # Update momentum
        proposed_momentum = {}
        for name in self.param_names:
            param_obj = self._get_param_by_name(name)
            if param_obj is not None and param_obj.grad is not None:
                proposed_momentum[name] = momentum[name] + 0.5 * step_size * param_obj.grad
            else:
                proposed_momentum[name] = momentum[name]
                
        return proposed_momentum
    
    def _leapfrog_step_end(
        self, 
        params: Dict[str, torch.Tensor], 
        momentum: Dict[str, torch.Tensor], 
        step_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Second half leapfrog step for momentum."""
        # Set params
        for name, param in params.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                param_obj.data = param.data
        
        # Zero gradients
        if hasattr(self.model, 'zero_grad'):
            self.model.zero_grad()
        else:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # Compute log posterior
        log_prob = self.log_posterior(params)
        
        # Calculate gradients
        log_prob.backward()
        
        # Update momentum
        proposed_momentum = {}
        for name in self.param_names:
            param_obj = self._get_param_by_name(name)
            if param_obj is not None and param_obj.grad is not None:
                proposed_momentum[name] = momentum[name] + 0.5 * step_size * param_obj.grad
            else:
                proposed_momentum[name] = momentum[name]
                
        return proposed_momentum
    
    def _kinetic_energy(self, momentum: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate kinetic energy of momentum variables."""
        energy = torch.tensor(0.0, device=self._device())
        for name, p in momentum.items():
            energy = energy + 0.5 * torch.sum(p**2)
        return energy
    
    def _adapt_step_size(
        self, 
        log_step_size: torch.Tensor, 
        target_accept: float, 
        accept_stat: float, 
        iter_idx: int
    ) -> torch.Tensor:
        """Adapt the step size to achieve target acceptance rate."""
        # Robbins-Monro sequence for step size adaptation
        gamma = 0.05  # Adaptation rate
        t0 = 10.0     # Stabilization constant
        exponent = -0.75  # Decay rate
        
        # Update using Robbins-Monro recursion
        factor = (target_accept - accept_stat) * gamma * (iter_idx + t0)**exponent
        new_log_step_size = log_step_size + factor
        
        return new_log_step_size
    
    def _set_posterior_means(self):
        """Set model parameters to posterior means after sampling."""
        # Calculate posterior means
        posterior_means = {}
        for name in self.param_names:
            samples_np = self.samples[name]
            mean_value = torch.tensor(np.mean(samples_np, axis=0), device=self._device())
            posterior_means[name] = mean_value
        
        # Set parameters to posterior means
        for name, value in posterior_means.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                param_obj.data = value
    
    def get_samples(
        self,
        test_x: np.ndarray,
        return_numpy: bool = True,
        num_pred_samples: int = 100
    ) -> Dict:
        """
        Generate prediction samples at test points.
        
        Args:
            test_x: Test points
            return_numpy: Whether to return numpy arrays
            num_pred_samples: Number of prediction samples to generate
            
        Returns:
            Dictionary with prediction samples and statistics
        """
        # Convert test points to tensor
        device = self._device()
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32).reshape(-1, 1).to(device)
        
        # Set model and likelihood to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Storage for prediction samples
        pred_means = []
        pred_samples = []
        
        # Randomly select parameter samples for predictions
        if num_pred_samples < len(self.samples[self.param_names[0]]):
            sample_indices = np.random.choice(
                len(self.samples[self.param_names[0]]), 
                num_pred_samples, 
                replace=False
            )
        else:
            sample_indices = np.arange(len(self.samples[self.param_names[0]]))
        
        # Generate predictions for each sample
        for idx in sample_indices:
            # Set model parameters to this sample
            for name in self.param_names:
                param = self._get_param_by_name(name)
                if param is not None:
                    sample_value = torch.tensor(
                        self.samples[name][idx], 
                        dtype=torch.float32,
                        device=device
                    )
                    param.data = sample_value
            
            # Generate prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = self.likelihood(self.model(test_x_tensor))
                
                # Save mean prediction
                mean = posterior.mean.cpu().numpy() if return_numpy else posterior.mean
                pred_means.append(mean)
                
                # Draw sample from posterior
                sample = posterior.sample().cpu().numpy() if return_numpy else posterior.sample()
                pred_samples.append(sample)
        
        # Convert lists to arrays if needed
        if return_numpy:
            pred_means = np.array(pred_means)
            pred_samples = np.array(pred_samples)
            
            # Calculate statistics
            mean_prediction = np.mean(pred_means, axis=0)
            std_prediction = np.std(pred_means, axis=0)
            lower_ci = np.percentile(pred_samples, 2.5, axis=0)
            upper_ci = np.percentile(pred_samples, 97.5, axis=0)
            
            # Return results as dictionary
            return {
                'mean': mean_prediction,
                'std': std_prediction,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'samples': pred_samples
            }
        else:
            # Return raw tensor results
            return {
                'mean_samples': pred_means,
                'samples': pred_samples
            }
    
    def plot_samples(self, parameter_name: str, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the samples for a specific parameter.
        
        Args:
            parameter_name: Name of the parameter to plot
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if parameter_name not in self.samples:
            raise ValueError(f"Parameter {parameter_name} not found in samples")
        
        # Get samples for the parameter
        param_samples = self.samples[parameter_name]
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot histogram
        axes[0].hist(param_samples, bins=30, density=True)
        axes[0].set_title(f'Posterior Distribution: {parameter_name}')
        axes[0].axvline(np.mean(param_samples), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(param_samples):.4f}')
        axes[0].axvline(np.median(param_samples), color='g', linestyle=':', 
                       label=f'Median: {np.median(param_samples):.4f}')
        axes[0].legend()
        
        # Plot trace
        axes[1].plot(param_samples)
        axes[1].set_title(f'Trace Plot: {parameter_name}')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Parameter Value')
        
        # Plot autocorrelation
        n_lags = min(50, len(param_samples) // 5)
        autocorr = self._autocorrelation(param_samples, n_lags)
        axes[2].bar(range(len(autocorr)), autocorr)
        axes[2].set_title('Autocorrelation')
        axes[2].set_xlabel('Lag')
        axes[2].set_ylabel('Autocorrelation')
        
        plt.tight_layout()
        return fig
    
    def _autocorrelation(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function for samples."""
        x = x.flatten()
        x_mean = np.mean(x)
        x_centered = x - x_mean
        
        # Normalization by variance
        norm = np.sum(x_centered ** 2)
        
        # Calculate autocorrelation for lags 0 to max_lag
        autocorr = []
        for lag in range(max_lag + 1):
            # Correlation at lag k
            corr = np.sum(x_centered[lag:] * x_centered[:-lag] if lag > 0 else x_centered ** 2)
            autocorr.append(corr / norm if norm > 0 else 0)
            
        return np.array(autocorr)
    
    def get_posterior_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all parameter posterior distributions.
        
        Returns:
            DataFrame with posterior statistics
        """
        stats = []
        
        for name in self.param_names:
            param_samples = self.samples[name]
            shape = self.param_shapes[name]
            
            # Handle multidimensional parameters
            if len(shape) > 0:
                flat_samples = param_samples.reshape(len(param_samples), -1)
                for i in range(flat_samples.shape[1]):
                    samples_i = flat_samples[:, i]
                    
                    # Calculate statistics
                    stats.append({
                        'parameter': f"{name}_{i}",
                        'mean': np.mean(samples_i),
                        'std': np.std(samples_i),
                        'median': np.median(samples_i),
                        '2.5%': np.percentile(samples_i, 2.5),
                        '97.5%': np.percentile(samples_i, 97.5),
                        'effective_sample_size': self._effective_sample_size(samples_i)
                    })
            else:
                # Scalar parameter
                samples = param_samples.flatten()
                
                # Calculate statistics
                stats.append({
                    'parameter': name,
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'median': np.median(samples),
                    '2.5%': np.percentile(samples, 2.5),
                    '97.5%': np.percentile(samples, 97.5),
                    'effective_sample_size': self._effective_sample_size(samples)
                })
        
        return pd.DataFrame(stats)
    
    def _effective_sample_size(self, samples: np.ndarray) -> float:
        """Calculate effective sample size accounting for autocorrelation."""
        n = len(samples)
        if n <= 5:
            return n
        
        # Calculate autocorrelation
        acf = self._autocorrelation(samples, min(100, n // 5))
        
        # Find where autocorrelation drops below 0.05
        cutoff = np.where(np.abs(acf) < 0.05)[0]
        if len(cutoff) > 0:
            # Use the first point where autocorrelation is negligible
            max_lag = cutoff[0]
        else:
            # Use half the available lags if no clear cutoff
            max_lag = len(acf) // 2
        
        # Calculate effective sample size using initial monotone sequence estimator
        rho = acf[1:max_lag+1]  # Exclude lag 0
        ess = n / (1 + 2 * np.sum(rho))
        
        # Ensure ESS is positive and at most n
        return min(max(1, ess), n)


class HeteroscedasticMCMCSampler(MCMCSampler):
    """
    MCMC sampler for heteroscedastic GP models with observation-specific noise.
    
    This extends the base MCMC sampler to handle varying noise levels.
    """
    
    def __init__(
        self,
        model: Any,
        likelihood: gpytorch.likelihoods.Likelihood,
        noise_levels: torch.Tensor,
        **kwargs
    ):
        """
        Initialize the heteroscedastic MCMC sampler.
        
        Args:
            model: GP model to sample from
            likelihood: Model likelihood
            noise_levels: Observation-specific noise levels
            **kwargs: Additional arguments for the base sampler
        """
        super().__init__(model, likelihood, **kwargs)
        self.noise_levels = noise_levels
    
    def log_posterior(self, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate log posterior probability for heteroscedastic model.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability
        """
        # Set model parameters
        for name, param in parameters.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                param_obj.data = param.data
        
        # Calculate heteroscedastic log likelihood
        try:
            output = self.model(self.model.train_inputs[0])
            mean = output.mean
            
            # Calculate likelihood with observation-specific noise
            train_y = self.model.train_targets
            residuals = train_y - mean
            
            # Negative log likelihood
            log_likelihood = -0.5 * torch.sum(residuals**2 / self.noise_levels**2) - \
                             torch.sum(torch.log(self.noise_levels)) - \
                             0.5 * len(train_y) * torch.log(torch.tensor(2 * np.pi))
        except Exception as e:
            # Return very low probability for numerical errors
            return torch.tensor(-1e10, device=self._device())
        
        # Calculate log prior
        log_prior = 0.0
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                for prior_name, prior, closure, _ in self.model.named_priors():
                    if prior_name == name:
                        try:
                            log_prior = log_prior + prior.log_prob(closure()).sum()
                        except Exception:
                            # Skip if prior computation fails
                            pass
        
        return log_likelihood + log_prior