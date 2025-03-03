"""
Enhanced MCMC Sampler for Bayesian GP State-Space Models

This module implements a more robust Hamiltonian Monte Carlo (HMC) sampler
with compatibility fixes for modern GPyTorch versions, improved diagnostics,
and specialized functionality for paleoclimate models.
"""

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# For MCMC diagnostics
from scipy.stats import gaussian_kde


class EnhancedMCMCSampler:
    """
    Enhanced MCMC Sampler for Bayesian Gaussian Process Models
    
    Features:
    - Robust HMC sampling with auto-tuning
    - GPyTorch compatibility across versions
    - Comprehensive diagnostics
    - Efficient sampling from posterior predictive distribution
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        n_samples: int = 1000,
        burn_in: int = 200,
        thinning: int = 2,
        step_size: float = 0.01,
        target_acceptance: float = 0.75,
        adaptation_steps: int = 100,
        max_leapfrog_steps: int = 20,
        jitter: float = 1e-5,
        random_state: int = 42,
        parameters_to_sample: Optional[List[str]] = None
    ):
        """
        Initialize the enhanced MCMC sampler.
        
        Args:
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            n_samples: Number of posterior samples to collect
            burn_in: Number of burn-in steps
            thinning: Thinning factor for samples
            step_size: Initial HMC step size
            target_acceptance: Target acceptance rate
            adaptation_steps: Number of steps for step size adaptation
            max_leapfrog_steps: Maximum number of leapfrog steps
            jitter: Jitter for numerical stability
            random_state: Random seed
            parameters_to_sample: Optional list of parameter names to sample (defaults to all)
        """
        self.model = model
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.step_size = step_size
        self.target_acceptance = target_acceptance
        self.adaptation_steps = adaptation_steps
        self.max_leapfrog_steps = max_leapfrog_steps
        self.jitter = jitter
        
        # Set random seeds
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Storage for samples and diagnostics
        self.samples = {}
        self.acceptance_rate = 0.0
        self.log_probs = []
        self.step_sizes = []
        
        # Extract parameter information
        self.param_names = []
        self.param_shapes = {}
        self.initial_values = {}
        
        # Extract initial parameter values from model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if parameters_to_sample is None or name in parameters_to_sample:
                    self.param_names.append(name)
                    self.param_shapes[name] = param.shape
                    self.initial_values[name] = param.detach().clone()
        
        # Store parameter constraints for bounded parameters
        self.param_constraints = {}
        for name in self.param_names:
            if hasattr(self.model, name + "_constraint"):
                self.param_constraints[name] = getattr(self.model, name + "_constraint")
    
    def _get_device(self) -> torch.device:
        """Helper to get the device of the model."""
        return next(self.model.parameters()).device
    
    def _get_param_by_name(self, name: str) -> Optional[torch.nn.Parameter]:
        """Helper to get parameter object by name."""
        for param_name, param in self.model.named_parameters():
            if param_name == name:
                return param
        return None
    
    def log_posterior(self, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate log posterior probability of parameters with improved
        version compatibility.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability
        """
        # First update model parameters - ensure parameters require grad
        for name, param in parameters.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                # Ensure parameter requires grad
                param.requires_grad_(True)
                # Update the parameter value
                param_obj.data.copy_(param.data)
        
        # Calculate log likelihood
        try:
            # Handle different input formats across GPyTorch versions
            if hasattr(self.model, 'train_inputs') and self.model.train_inputs is not None:
                train_inputs = self.model.train_inputs[0]
            elif hasattr(self.model, 'train_x'):
                train_inputs = self.model.train_x
            else:
                raise AttributeError("Model has no train_inputs or train_x attribute")
            
            # Get train targets
            if hasattr(self.model, 'train_targets'):
                train_targets = self.model.train_targets
            elif hasattr(self.model, 'train_y'):
                train_targets = self.model.train_y
            else:
                raise AttributeError("Model has no train_targets or train_y attribute")
            
            # Put model in eval mode to avoid batch norm/dropout issues
            self.model.eval()
            
            # Additional safety settings for numerical stability
            with gpytorch.settings.cholesky_jitter(self.jitter):
                # Compute model output
                output = self.model(train_inputs)
                
                # Check if the model's likelihood has log_marginal method
                if hasattr(self.likelihood, 'log_marginal'):
                    log_likelihood = self.likelihood.log_marginal(
                        output, train_targets
                    )
                else:
                    # Fallback to manual calculation
                    # Compute p(y|f,θ) where f is the GP output
                    residuals = train_targets - output.mean
                    variance = output.variance + self.jitter
                    
                    log_likelihood = -0.5 * torch.sum(residuals**2 / variance) - \
                                    0.5 * torch.sum(torch.log(2 * np.pi * variance))
        except Exception as e:
            # If something goes wrong, return very low probability
            print(f"Likelihood computation error: {str(e)}")
            return torch.tensor(-1e10, device=self._get_device(), requires_grad=True)
        
        # Calculate log prior - compatible across GPyTorch versions
        log_prior = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        try:
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and name in parameters:
                    # Version-compatible way to get priors
                    try:
                        # First try to access named_priors directly if it exists
                        if hasattr(self.model, 'named_priors'):
                            # Handle different return formats across versions
                            try:
                                # Try with 4-tuple (newer GPyTorch)
                                for prior_name, prior, closure, _ in self.model.named_priors():
                                    if prior_name == name:
                                        try:
                                            prior_logprob = prior.log_prob(closure())
                                            if prior_logprob.requires_grad:
                                                log_prior = log_prior + prior_logprob.sum()
                                        except Exception as e:
                                            print(f"Prior error for {prior_name}: {str(e)}")
                                            pass
                            except ValueError:
                                # Try with 3-tuple (older GPyTorch)
                                try:
                                    for prior_name, prior, closure in self.model.named_priors():
                                        if prior_name == name:
                                            try:
                                                prior_logprob = prior.log_prob(closure())
                                                if prior_logprob.requires_grad:
                                                    log_prior = log_prior + prior_logprob.sum()
                                            except Exception as e:
                                                print(f"Prior error for {prior_name}: {str(e)}")
                                                pass
                                except Exception as e:
                                    print(f"Named priors error: {str(e)}")
                                    pass
                        
                        # If we couldn't get priors through named_priors, try to access directly
                        elif hasattr(self.model, f'{name}_prior'):
                            prior = getattr(self.model, f'{name}_prior')
                            prior_logprob = prior.log_prob(parameter)
                            if prior_logprob.requires_grad:
                                log_prior = log_prior + prior_logprob.sum()
                    except Exception as e:
                        # If something goes wrong with priors, just continue
                        print(f"Prior handling error for {name}: {str(e)}")
                        pass
        except Exception as e:
            print(f"General prior computation error: {str(e)}")
            pass
        
        # Ensure the result requires grad
        result = log_likelihood + log_prior
        if not result.requires_grad:
            print("Warning: log posterior does not require gradient")
            # Create a differentiable copy by adding a tiny gradient-requiring term
            result = result + 0.0 * sum(p.sum() for p in parameters.values() if p.requires_grad)
            
        return result
    
    def run_hmc(self, progress_bar: bool = True, print_summary: bool = True):
        """
        Run Hamiltonian Monte Carlo sampling with enhanced robustness.
        
        Args:
            progress_bar: Whether to show progress bar
            print_summary: Whether to print summary statistics after sampling
        """
        # Set device and initialize the sampler state
        device = self._get_device()
        
        # Initialize storage for samples
        for name in self.param_names:
            self.samples[name] = []
        
        # Initialize parameters
        current_params = {}
        for name, value in self.initial_values.items():
            current_params[name] = value.clone()
        
        # Get initial log probability
        current_log_prob = self.log_posterior(current_params)
        self.log_probs.append(current_log_prob.item())
        
        # Initialize step size
        step_size = self.step_size
        self.step_sizes.append(step_size)
        
        # Track acceptance
        n_accepted = 0
        
        # Total iterations including burn-in and thinning
        total_iterations = self.burn_in + self.n_samples * self.thinning
        
        # Set up progress bar if requested
        if progress_bar:
            pbar = tqdm(total=total_iterations, desc="MCMC Sampling")
        
        # HMC sampling loop
        for i in range(total_iterations):
            # Store samples after burn-in, respecting thinning
            if i >= self.burn_in and (i - self.burn_in) % self.thinning == 0:
                for name, param in current_params.items():
                    self.samples[name].append(param.detach().clone())
            
            # Initialize momentum
            momentum = {}
            for name, param in current_params.items():
                momentum[name] = torch.randn_like(param, device=device)
            
            # Compute kinetic energy
            kinetic_energy = 0.5 * sum(torch.sum(m**2) for m in momentum.values())
            
            # Clone current parameters and log prob
            proposed_params = {name: param.clone() for name, param in current_params.items()}
            proposed_momentum = {name: m.clone() for name, m in momentum.items()}
            
            # Randomly choose number of leapfrog steps
            L = np.random.randint(1, self.max_leapfrog_steps + 1)
            
            # Leapfrog integration
            # Half step for momentum
            grad_dict = self._compute_gradients(proposed_params)
            for name in proposed_momentum:
                proposed_momentum[name] = proposed_momentum[name] + 0.5 * step_size * grad_dict[name]
            
            # Full steps for position and momentum
            for j in range(L):
                # Full step for position
                for name in proposed_params:
                    proposed_params[name] = proposed_params[name] + step_size * proposed_momentum[name]
                    
                    # Apply constraints if they exist
                    if name in self.param_constraints:
                        constraint = self.param_constraints[name]
                        proposed_params[name].data = constraint.transform(proposed_params[name].data)
                
                # Full step for momentum (except at the end)
                if j < L - 1:
                    grad_dict = self._compute_gradients(proposed_params)
                    for name in proposed_momentum:
                        proposed_momentum[name] = proposed_momentum[name] + step_size * grad_dict[name]
            
            # Half step for momentum at the end
            grad_dict = self._compute_gradients(proposed_params)
            for name in proposed_momentum:
                proposed_momentum[name] = proposed_momentum[name] + 0.5 * step_size * grad_dict[name]
            
            # Negate momentum for reversibility
            for name in proposed_momentum:
                proposed_momentum[name] = -proposed_momentum[name]
            
            # Compute proposed log probability and kinetic energy
            proposed_log_prob = self.log_posterior(proposed_params)
            proposed_kinetic_energy = 0.5 * sum(torch.sum(m**2) for m in proposed_momentum.values())
            
            # Metropolis-Hastings acceptance criterion
            log_accept_ratio = proposed_log_prob - current_log_prob + kinetic_energy - proposed_kinetic_energy
            
            # Accept or reject
            if torch.log(torch.rand(1, device=device)) < log_accept_ratio:
                current_params = {name: param.detach().clone() for name, param in proposed_params.items()}
                current_log_prob = proposed_log_prob.detach().clone()
                n_accepted += 1
            
            # Store diagnostics
            self.log_probs.append(current_log_prob.item())
            
            # Adapt step size during burn-in
            if i < self.adaptation_steps:
                # Increase step size if acceptance is too high, decrease if too low
                if n_accepted / (i + 1) > self.target_acceptance:
                    step_size *= 1.02
                else:
                    step_size *= 0.98
                
                self.step_sizes.append(step_size)
            
            # Update progress bar
            if progress_bar:
                accept_rate = n_accepted / (i + 1)
                pbar.update(1)
                pbar.set_postfix({
                    "log_prob": f"{current_log_prob.item():.2f}",
                    "step": f"{step_size:.5f}",
                    "accept": f"{accept_rate:.2f}"
                })
        
        # Close progress bar if used
        if progress_bar:
            pbar.close()
        
        # Calculate final acceptance rate
        self.acceptance_rate = n_accepted / total_iterations
        
        # Convert sample lists to tensors
        for name in self.samples:
            if self.samples[name]:  # Check if we have samples
                self.samples[name] = torch.stack(self.samples[name])
        
        # Print summary if requested
        if print_summary:
            self._print_summary()
    
    def _compute_gradients(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute gradients of negative log posterior.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Dictionary of gradients for each parameter
        """
        # Create parameter copies that require gradients
        param_copies = {}
        for name, param in parameters.items():
            param_copy = param.detach().clone()
            param_copy.requires_grad_(True)
            param_copies[name] = param_copy
        
        try:
            # Compute negative log posterior
            negative_log_posterior = -self.log_posterior(param_copies)
            
            # Check if the result requires grad
            if not negative_log_posterior.requires_grad:
                print("Warning: negative_log_posterior doesn't require grad")
                # Add a small differentiable term
                negative_log_posterior = negative_log_posterior + 0.0 * sum(p.sum() for p in param_copies.values())
            
            # Compute gradients
            negative_log_posterior.backward(retain_graph=False)
            
        except Exception as e:
            print(f"Gradient computation error: {str(e)}")
            # Return zero gradients if computation fails
            return {name: torch.zeros_like(param) for name, param in parameters.items()}
        
        # Collect gradients
        grad_dict = {}
        for name, param in param_copies.items():
            if param.grad is not None:
                # Negate for maximization and detach/clone for safety
                grad_dict[name] = -param.grad.detach().clone()
            else:
                print(f"No gradient for parameter: {name}")
                grad_dict[name] = torch.zeros_like(param)
        
        return grad_dict
    
    def _print_summary(self):
        """Print summary statistics of the MCMC run."""
        print("\n=== MCMC Sampling Summary ===")
        print(f"Number of samples: {self.n_samples}")
        print(f"Acceptance rate: {self.acceptance_rate:.2f}")
        print(f"Final step size: {self.step_sizes[-1]:.5f}")
        
        # Print parameter summaries
        print("\nParameter estimates:")
        for name in self.param_names:
            samples = self.samples[name].cpu().numpy()
            
            # Handle parameters with different shapes
            if len(self.param_shapes[name]) == 0:  # Scalar
                mean = np.mean(samples)
                std = np.std(samples)
                lower, upper = np.percentile(samples, [2.5, 97.5])
                
                print(f"  {name}: {mean:.4f} ± {std:.4f} [95% CI: {lower:.4f}, {upper:.4f}]")
            else:
                # For vector/matrix parameters, just print shape and mean
                print(f"  {name}: shape={self.param_shapes[name]}, mean={np.mean(samples):.4f}")
    
    def get_posterior_samples(self, n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get posterior samples for prediction.
        
        Args:
            n_samples: Optional number of samples to return (default: all)
            
        Returns:
            Dictionary of parameter samples
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        if n_samples is None:
            return self.samples
        
        # Sample a subset
        n_available = len(list(self.samples.values())[0])
        indices = torch.randperm(n_available)[:n_samples]
        
        return {name: samples[indices] for name, samples in self.samples.items()}
    
    def get_posterior_mean(self) -> Dict[str, torch.Tensor]:
        """
        Get posterior mean for each parameter.
        
        Returns:
            Dictionary of mean parameter values
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        return {name: torch.mean(samples, dim=0) for name, samples in self.samples.items()}
    
    def get_posterior_median(self) -> Dict[str, torch.Tensor]:
        """
        Get posterior median for each parameter.
        
        Returns:
            Dictionary of median parameter values
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        return {name: torch.quantile(samples, 0.5, dim=0) for name, samples in self.samples.items()}
    
    def get_posterior_interval(self, alpha: float = 0.05) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get credible intervals for each parameter.
        
        Args:
            alpha: Significance level (e.g., 0.05 for 95% interval)
            
        Returns:
            Dictionary of (lower, upper) interval tuples
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        intervals = {}
        for name, samples in self.samples.items():
            lower = torch.quantile(samples, alpha/2, dim=0)
            upper = torch.quantile(samples, 1-alpha/2, dim=0)
            intervals[name] = (lower, upper)
        
        return intervals
    
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
    
    def plot_trace(self, figure_path: Optional[str] = None):
        """
        Plot MCMC trace plots for scalar parameters.
        
        Args:
            figure_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        # Count scalar parameters
        scalar_params = [name for name, shape in self.param_shapes.items() 
                         if len(shape) == 0]
        
        if not scalar_params:
            print("No scalar parameters to plot.")
            return None
        
        # Create figure
        n_params = len(scalar_params)
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))
        
        # Handle single parameter case
        if n_params == 1:
            axes = axes.reshape(1, 2)
        
        # Plot each parameter
        for i, name in enumerate(scalar_params):
            samples = self.samples[name].cpu().numpy()
            
            # Trace plot
            axes[i, 0].plot(samples)
            axes[i, 0].set_title(f"Trace for {name}")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True, alpha=0.3)
            
            # Posterior histogram with KDE
            kde = gaussian_kde(samples)
            x = np.linspace(np.min(samples), np.max(samples), 200)
            axes[i, 1].hist(samples, bins=30, density=True, alpha=0.6)
            axes[i, 1].plot(x, kde(x), 'r-', linewidth=2)
            axes[i, 1].set_title(f"Posterior for {name}")
            axes[i, 1].set_xlabel("Value")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add mean and 95% CI
            mean = np.mean(samples)
            q025, q975 = np.percentile(samples, [2.5, 97.5])
            axes[i, 1].axvline(mean, color='k', linestyle='-', alpha=0.5)
            axes[i, 1].axvline(q025, color='k', linestyle='--', alpha=0.5)
            axes[i, 1].axvline(q975, color='k', linestyle='--', alpha=0.5)
            
            # Add text annotation
            axes[i, 1].text(0.05, 0.95, f"Mean = {mean:.4f}\n95% CI: [{q025:.4f}, {q975:.4f}]",
                        transform=axes[i, 1].transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add figure for sampling diagnostics
        fig_diag, axes_diag = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot log probability
        axes_diag[0].plot(self.log_probs)
        axes_diag[0].set_title("Log Probability")
        axes_diag[0].set_xlabel("Iteration")
        axes_diag[0].set_ylabel("Log Probability")
        axes_diag[0].axvline(self.burn_in, color='r', linestyle='--', alpha=0.5)
        axes_diag[0].text(self.burn_in, min(self.log_probs),
                     ' Burn-in end', color='r', ha='left', va='bottom')
        axes_diag[0].grid(True, alpha=0.3)
        
        # Plot step size
        if self.step_sizes:
            axes_diag[1].plot(self.step_sizes)
            axes_diag[1].set_title("Step Size Adaptation")
            axes_diag[1].set_xlabel("Iteration")
            axes_diag[1].set_ylabel("Step Size")
            axes_diag[1].set_yscale('log')
            axes_diag[1].axvline(self.adaptation_steps, color='r', linestyle='--', alpha=0.5)
            axes_diag[1].text(self.adaptation_steps, min(self.step_sizes),
                         ' Adaptation end', color='r', ha='left', va='bottom')
            axes_diag[1].grid(True, alpha=0.3)
        
        # Add tight layout
        fig.tight_layout()
        fig_diag.tight_layout()
        
        # Save if path provided
        if figure_path:
            fig.savefig(figure_path, dpi=300, bbox_inches='tight')
            diag_path = figure_path.replace('.png', '_diagnostics.png')
            fig_diag.savefig(diag_path, dpi=300, bbox_inches='tight')
        
        return fig, fig_diag
    
    def sample_posterior_predictive(
        self,
        x_pred: torch.Tensor,
        n_samples: int = 100,
        batch_size: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Draw samples from the posterior predictive distribution.
        
        Args:
            x_pred: Prediction locations
            n_samples: Number of posterior samples to generate
            batch_size: Batch size for prediction (to manage memory)
            
        Returns:
            Dictionary with keys 'mean', 'samples', 'lower_ci', 'upper_ci'
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        # Use batch prediction if batch_size is specified
        batch_size = batch_size or n_samples
        
        # Get model device
        device = self._get_device()
        
        # Ensure x_pred is on the correct device
        if isinstance(x_pred, torch.Tensor):
            x_pred = x_pred.to(device)
        else:
            x_pred = torch.tensor(x_pred, dtype=torch.float32, device=device)
        
        # Reshape if needed
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(-1)
        
        # Storage for predictions
        all_samples = []
        
        # Generate samples from posterior distribution
        n_total_samples = len(next(iter(self.samples.values())))
        sample_indices = torch.randperm(n_total_samples)[:n_samples]
        
        # Batch processing
        for i in range(0, n_samples, batch_size):
            batch_indices = sample_indices[i:min(i + batch_size, n_samples)]
            batch_size_actual = len(batch_indices)
            
            # Get parameter samples for this batch
            batch_params = {}
            for name in self.param_names:
                batch_params[name] = self.samples[name][batch_indices]
            
            # Storage for this batch
            batch_samples = torch.zeros(batch_size_actual, x_pred.size(0), device=device)
            
            # Generate predictions for each parameter sample
            for j in range(batch_size_actual):
                # Set model parameters to sample values
                for name in self.param_names:
                    param = self._get_param_by_name(name)
                    param.data = batch_params[name][j].data
                
                # Make prediction
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    output = self.model(x_pred)
                    post_pred = self.likelihood(output)
                    sample = post_pred.sample()
                
                # Store sample
                batch_samples[j] = sample
            
            # Add to all samples
            all_samples.append(batch_samples)
        
        # Combine batches
        all_samples = torch.cat(all_samples, dim=0)
        
        # Compute statistics
        mean = torch.mean(all_samples, dim=0)
        lower_ci = torch.quantile(all_samples, 0.025, dim=0)
        upper_ci = torch.quantile(all_samples, 0.975, dim=0)
        
        # Return as numpy arrays
        return {
            'mean': mean.cpu().numpy(),
            'samples': all_samples.cpu().numpy(),
            'lower_ci': lower_ci.cpu().numpy(),
            'upper_ci': upper_ci.cpu().numpy()
        }
    
    def save_samples(self, file_path: str):
        """
        Save MCMC samples to a file.
        
        Args:
            file_path: Path to save the samples
        """
        if not self.samples:
            raise ValueError("No samples available. Run MCMC first.")
        
        # Convert samples to numpy
        samples_np = {}
        for name, samples in self.samples.items():
            samples_np[name] = samples.cpu().numpy()
        
        # Add metadata
        metadata = {
            'acceptance_rate': self.acceptance_rate,
            'n_samples': self.n_samples,
            'burn_in': self.burn_in,
            'thinning': self.thinning,
            'random_state': self.random_state,
            'log_probs': np.array(self.log_probs),
            'step_sizes': np.array(self.step_sizes) if self.step_sizes else None
        }
        
        # Save
        np.savez(file_path, **samples_np, **{'metadata': metadata})
    
    def load_samples(self, file_path: str):
        """
        Load MCMC samples from a file.
        
        Args:
            file_path: Path to the samples file
        """
        # Load file
        data = np.load(file_path, allow_pickle=True)
        
        # Extract samples
        self.samples = {}
        device = self._get_device()
        
        for name in self.param_names:
            if name in data:
                self.samples[name] = torch.tensor(data[name], device=device)
        
        # Extract metadata
        if 'metadata' in data:
            metadata = data['metadata'].item()
            self.acceptance_rate = metadata.get('acceptance_rate', 0.0)
            self.n_samples = metadata.get('n_samples', len(next(iter(self.samples.values()))))
            self.burn_in = metadata.get('burn_in', 0)
            self.thinning = metadata.get('thinning', 1)
            self.log_probs = metadata.get('log_probs', []).tolist()
            
            if 'step_sizes' in metadata and metadata['step_sizes'] is not None:
                self.step_sizes = metadata['step_sizes'].tolist()


# Specialization for heteroscedastic noise
class HeteroscedasticEnhancedMCMCSampler(EnhancedMCMCSampler):
    """
    Enhanced MCMC Sampler for models with heteroscedastic noise.
    
    This sampler handles models where different observations have different
    noise levels, common in paleoclimate proxy data.
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
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            noise_levels: Tensor of noise standard deviations for each observation
            **kwargs: Additional arguments for base sampler
        """
        super().__init__(model, likelihood, **kwargs)
        self.noise_levels = noise_levels.to(self._get_device())
    
    def log_posterior(self, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate log posterior with heteroscedastic likelihood.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability
        """
        # First update model parameters
        for name, param in parameters.items():
            param_obj = self._get_param_by_name(name)
            if param_obj is not None:
                param_obj.data = param.data
        
        # Calculate heteroscedastic log likelihood
        try:
            # Get train data
            if hasattr(self.model, 'train_inputs') and self.model.train_inputs is not None:
                train_inputs = self.model.train_inputs[0]
            elif hasattr(self.model, 'train_x'):
                train_inputs = self.model.train_x
            else:
                raise AttributeError("Model has no train_inputs or train_x attribute")
            
            # Get train targets
            if hasattr(self.model, 'train_targets'):
                train_targets = self.model.train_targets
            elif hasattr(self.model, 'train_y'):
                train_targets = self.model.train_y
            else:
                raise AttributeError("Model has no train_targets or train_y attribute")
            
            # Compute model output
            output = self.model(train_inputs)
            mean = output.mean
            
            # Calculate log likelihood with observation-specific noise
            residuals = train_targets - mean
            
            # Compute negative log likelihood
            log_likelihood = -0.5 * torch.sum(residuals**2 / (self.noise_levels**2)) - \
                             torch.sum(torch.log(self.noise_levels)) - \
                             0.5 * len(train_targets) * torch.log(torch.tensor(2 * np.pi))
        except Exception as e:
            # If something goes wrong, return very low probability
            return torch.tensor(-1e10, device=self._get_device())
        
        # Calculate log prior (same as parent class)
        log_prior = super()._calculate_log_prior(parameters)
        
        return log_likelihood + log_prior
    
    def _calculate_log_prior(self, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate log prior probability (helper method).
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log prior probability
        """
        log_prior = 0.0
        
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and name in parameters:
                # Version-compatible way to get priors
                try:
                    # First try to access named_priors directly if it exists
                    if hasattr(self.model, 'named_priors'):
                        # Handle different return formats across versions
                        try:
                            # Try with 4-tuple (newer GPyTorch)
                            for prior_name, prior, closure, _ in self.model.named_priors():
                                if prior_name == name:
                                    try:
                                        log_prior = log_prior + prior.log_prob(closure()).sum()
                                    except:
                                        pass
                        except ValueError:
                            # Try with 3-tuple (older GPyTorch)
                            try:
                                for prior_name, prior, closure in self.model.named_priors():
                                    if prior_name == name:
                                        try:
                                            log_prior = log_prior + prior.log_prob(closure()).sum()
                                        except:
                                            pass
                            except:
                                pass
                    
                    # If we couldn't get priors through named_priors, try to access directly
                    elif hasattr(self.model, f'{name}_prior'):
                        prior = getattr(self.model, f'{name}_prior')
                        log_prior = log_prior + prior.log_prob(parameter).sum()
                except Exception:
                    # If something goes wrong with priors, just continue
                    pass
        
        return log_prior