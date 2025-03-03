"""
MCMC Sampling for Bayesian GP State-Space Models

This module implements Hamiltonian Monte Carlo sampling for Bayesian GP models,
with support for standard and heteroscedastic noise models.
"""

import numpy as np
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm.auto import tqdm
import time


class MCMCSampler:
    """
    MCMC Sampler for Bayesian Gaussian Process State-Space Models
    
    Implements Hamiltonian Monte Carlo (HMC) for sampling from the posterior
    distribution of model parameters in GPyTorch models.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        n_samples: int = 500,
        burn_in: int = 100,
        thinning: int = 2,
        step_size: float = 0.01,
        target_acceptance: float = 0.75,
        adaptation_steps: int = 50,
        random_state: int = 42
    ):
        """
        Initialize MCMC sampler.
        
        Args:
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            n_samples: Number of posterior samples to collect
            burn_in: Number of burn-in steps
            thinning: Thinning factor for samples
            step_size: Initial HMC step size
            target_acceptance: Target acceptance rate
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
        
        # Calculate log prior - safe version that handles various GPyTorch versions
        log_prior = 0.0
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                # Try to handle various versions of GPyTorch
                try:
                    # Try standard newer method
                    for prior_name, prior, closure, _ in self.model.named_priors():
                        if prior_name == name:
                            try:
                                log_prior += prior.log_prob(closure()).sum()
                            except Exception:
                                pass
                except ValueError:
                    # Fall back to older method
                    try:
                        for prior_name, prior, closure in self.model.named_priors():
                            if prior_name == name:
                                try:
                                    log_prior += prior.log_prob(closure()).sum()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    # Final fallback: just skip priors if they cause problems
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
        for name in self.param_names:
            self.samples[name] = []
        
        # Initial parameter values
        current_params = {}
        for name, value in self.initial_values.items():
            current_params[name] = value.clone()
        
        # Initialize step size
        step_size = self.step_size
        
        # Compute initial log probability
        current_log_prob = self.log_posterior(current_params)
        
        # Track acceptance
        n_accepted = 0
        
        # Set up tqdm progress bar if requested
        total_iterations = self.burn_in + self.n_samples * self.thinning
        if progress_bar:
            pbar = tqdm(total=total_iterations, desc="MCMC Sampling")
        
        # Burn-in and sampling
        for i in range(total_iterations):
            # Store samples after burn-in, with thinning
            if i >= self.burn_in and (i - self.burn_in) % self.thinning == 0:
                for name, param in current_params.items():
                    self.samples[name].append(param.detach().clone())
            
            # Initialize momentum for each parameter
            momentum = {}
            for name, param in current_params.items():
                momentum[name] = torch.randn_like(param, device=device)
            
            # Compute kinetic energy
            kinetic_energy = sum(torch.sum(m ** 2) for m in momentum.values()) / 2
            
            # Clone current parameters and log prob
            proposed_params = {name: param.clone() for name, param in current_params.items()}
            proposed_log_prob = current_log_prob.clone()
            proposed_momentum = {name: m.clone() for name, m in momentum.items()}
            
            # Sample random number of leapfrog steps
            L = int(np.random.uniform(1, 10))
            
            # Leapfrog integration
            for j in range(L):
                # Half step for momentum
                for name in proposed_params:
                    proposed_params[name].requires_grad_(True)
                
                # Compute gradients for potential energy (negative log probability)
                potential_energy = -self.log_posterior(proposed_params)
                potential_energy.backward()
                
                # Update momentum
                for name, param in proposed_params.items():
                    proposed_momentum[name] = proposed_momentum[name] - 0.5 * step_size * param.grad
                    param.grad = None
                
                # Full step for position
                for name in proposed_params:
                    proposed_params[name] = proposed_params[name] + step_size * proposed_momentum[name]
                    proposed_params[name].detach_()
                
                # Compute gradients for updated position
                for name in proposed_params:
                    proposed_params[name].requires_grad_(True)
                
                # Remove old gradients if they exist
                for name, param in proposed_params.items():
                    if param.grad is not None:
                        param.grad.zero_()
                
                # Compute gradients for potential energy
                potential_energy = -self.log_posterior(proposed_params)
                potential_energy.backward()
                
                # Update momentum
                for name, param in proposed_params.items():
                    proposed_momentum[name] = proposed_momentum[name] - 0.5 * step_size * param.grad
                    param.grad = None
                    proposed_params[name].detach_()
            
            # Compute proposed log probability
            proposed_log_prob = self.log_posterior(proposed_params)
            
            # Compute proposed kinetic energy
            proposed_kinetic_energy = sum(torch.sum(m ** 2) for m in proposed_momentum.values()) / 2
            
            # Metropolis-Hastings acceptance
            log_accept_ratio = proposed_log_prob - current_log_prob + kinetic_energy - proposed_kinetic_energy
            
            # Accept or reject
            if torch.log(torch.rand(1, device=device)) < log_accept_ratio:
                current_params = {name: param.detach().clone() for name, param in proposed_params.items()}
                current_log_prob = proposed_log_prob.detach().clone()
                n_accepted += 1
            
            # Adapt step size during burn-in
            if i < self.adaptation_steps:
                # Increase step size if acceptance is too high, decrease if too low
                if float(n_accepted) / (i + 1) > self.target_acceptance:
                    step_size *= 1.05
                else:
                    step_size *= 0.95
            
            # Log iterations
            self.log_probs.append(current_log_prob.item())
            
            # Update progress bar
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix({"log_prob": f"{current_log_prob.item():.2f}",
                                 "step_size": f"{step_size:.5f}",
                                 "accept": f"{n_accepted/(i+1):.2f}"})
        
        # Close progress bar
        if progress_bar:
            pbar.close()
        
        # Calculate final acceptance rate
        self.acceptance_rate = n_accepted / total_iterations
        
        # Convert samples to tensors
        for name in self.samples:
            self.samples[name] = torch.stack(self.samples[name], dim=0)
    
    def plot_diagnostics(self, figure_path: Optional[str] = None):
        """
        Plot MCMC diagnostics including trace plots and posterior histograms.
        
        Args:
            figure_path: Optional path to save the figure
        """
        # Check if samples exist
        if not self.samples:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        # Create figure
        n_params = len(self.param_names)
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))
        
        # Make sure axes is 2D even for a single parameter
        if n_params == 1:
            axes = axes.reshape(1, 2)
        
        # Plot trace and histogram for each parameter
        for i, name in enumerate(self.param_names):
            samples = self.samples[name].cpu().numpy()
            
            # For vector parameters, take the first element
            if len(self.param_shapes[name]) > 0:
                samples = samples[..., 0]
            
            # Trace plot
            axes[i, 0].plot(samples)
            axes[i, 0].set_title(f"Trace for {name}")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram
            axes[i, 1].hist(samples, bins=30, density=True)
            axes[i, 1].set_title(f"Posterior for {name}")
            axes[i, 1].set_xlabel("Value")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].grid(True, alpha=0.3)
        
        # Plot log probability
        if len(self.log_probs) > 0:
            fig_logprob, ax_logprob = plt.subplots(figsize=(10, 4))
            ax_logprob.plot(self.log_probs)
            ax_logprob.set_title("Log Probability Trace")
            ax_logprob.set_xlabel("Iteration")
            ax_logprob.set_ylabel("Log Probability")
            ax_logprob.grid(True, alpha=0.3)
            
            if figure_path:
                log_prob_path = figure_path.replace('.png', '_logprob.png')
                fig_logprob.savefig(log_prob_path, dpi=300, bbox_inches='tight')
        
        # Add tight layout
        fig.tight_layout()
        
        # Save if path provided
        if figure_path:
            fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_posterior_samples(self, n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get posterior samples for prediction.
        
        Args:
            n_samples: Number of samples to return (default: all)
            
        Returns:
            Dictionary of parameter samples
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        if n_samples is None:
            return self.samples
        
        # Sample random indices
        n_available = len(self.samples[self.param_names[0]])
        indices = torch.randperm(n_available)[:n_samples]
        
        # Return samples at selected indices
        return {name: samples[indices] for name, samples in self.samples.items()}
    
    def get_posterior_mean(self) -> Dict[str, torch.Tensor]:
        """
        Get posterior mean for each parameter.
        
        Returns:
            Dictionary of mean parameter values
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        return {name: torch.mean(samples, dim=0) for name, samples in self.samples.items()}
    
    def get_posterior_std(self) -> Dict[str, torch.Tensor]:
        """
        Get posterior standard deviation for each parameter.
        
        Returns:
            Dictionary of standard deviation values
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        return {name: torch.std(samples, dim=0) for name, samples in self.samples.items()}
    
    def apply_posterior_mean(self):
        """
        Set model parameters to their posterior mean values.
        """
        posterior_mean = self.get_posterior_mean()
        for name, value in posterior_mean.items():
            param = self._get_param_by_name(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)


class HeteroscedasticMCMCSampler(MCMCSampler):
    """
    MCMC Sampler for models with heteroscedastic noise.
    
    This extends the base MCMCSampler to handle models where
    different observations have different noise levels.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        noise_levels: torch.Tensor,
        **kwargs
    ):
        """
        Initialize heteroscedastic MCMC sampler.
        
        Args:
            model: GP model
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
        
        # Calculate log prior - safe version that handles various GPyTorch versions
        log_prior = 0.0
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                # Try to handle various versions of GPyTorch
                try:
                    # Try standard newer method
                    for prior_name, prior, closure, _ in self.model.named_priors():
                        if prior_name == name:
                            try:
                                log_prior += prior.log_prob(closure()).sum()
                            except Exception:
                                pass
                except ValueError:
                    # Fall back to older method
                    try:
                        for prior_name, prior, closure in self.model.named_priors():
                            if prior_name == name:
                                try:
                                    log_prior += prior.log_prob(closure()).sum()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    # Final fallback: just skip priors if they cause problems
                    pass
        
        return log_likelihood + log_prior