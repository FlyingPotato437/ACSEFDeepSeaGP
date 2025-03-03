"""
Improved MCMC Implementation for Bayesian GP-State Space Models

This module provides a simple but robust MCMC implementation that
doesn't rely on auto-differentiation or gradients, making it more
compatible with GPyTorch and more reliable for complex models.
"""

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from tqdm.auto import tqdm

class RobustMCMCSampler:
    """
    A robust MCMC sampler using Random Walk Metropolis algorithm.
    
    This implementation does not rely on gradients, making it more compatible
    with GPyTorch models that may have issues with autodifferentiation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        n_samples: int = 1000,
        burn_in: int = 200,
        proposal_scale: float = 0.1,
        parameter_scales: Optional[Dict[str, float]] = None,
        random_state: int = 42
    ):
        """
        Initialize the MCMC sampler.
        
        Args:
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            n_samples: Number of samples to generate
            burn_in: Number of burn-in samples to discard
            proposal_scale: Global scale of proposal distribution
            parameter_scales: Dictionary of parameter-specific scales
            random_state: Random seed
        """
        self.model = model
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.proposal_scale = proposal_scale
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Initialize parameter information
        self._initialize_parameters()
        
        # Parameter-specific scales
        if parameter_scales is None:
            self.parameter_scales = {name: 1.0 for name in self.param_names}
        else:
            self.parameter_scales = parameter_scales
            # Ensure all parameters have a scale
            for name in self.param_names:
                if name not in self.parameter_scales:
                    self.parameter_scales[name] = 1.0
        
        # Storage for samples and diagnostics
        self.samples = {name: [] for name in self.param_names}
        self.log_probs = []
        self.acceptance_rate = 0.0
    
    def _initialize_parameters(self):
        """Initialize parameter information from the model."""
        self.param_names = []
        self.param_shapes = {}
        self.initial_values = {}
        
        # Extract parameters that require gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
                self.initial_values[name] = param.detach().clone()
        
        # Store constraints for bounded parameters
        self.param_constraints = {}
        for name in self.param_names:
            constraint_name = name.replace('.', '_') + '_constraint'
            parts = name.split('.')
            
            # Check different ways the constraint might be stored
            obj = self.model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            
            if obj is not None:
                # Check direct attribute
                direct_constraint = getattr(obj, parts[-1] + '_constraint', None)
                if direct_constraint is not None:
                    self.param_constraints[name] = direct_constraint
            
            # Check if constraint is stored at model level
            model_constraint = getattr(self.model, constraint_name, None)
            if model_constraint is not None:
                self.param_constraints[name] = model_constraint
    
    def _get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    def _log_posterior(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log posterior probability of parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability as a float
        """
        # Save original parameters to restore later
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.detach().clone()
        
        # Update model parameters with proposal
        for name, value in parameters.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)
        
        # Calculate log likelihood
        try:
            # Put model in eval mode for likelihood calculation
            self.model.eval()
            
            # Special handling for different GPyTorch versions
            x_input = None
            y_target = None
            
            # Try to get training data
            if hasattr(self.model, 'train_inputs') and self.model.train_inputs:
                x_input = self.model.train_inputs[0]
            elif hasattr(self.model, 'train_x'):
                x_input = self.model.train_x
            
            if hasattr(self.model, 'train_targets'):
                y_target = self.model.train_targets
            elif hasattr(self.model, 'train_y'):
                y_target = self.model.train_y
            
            if x_input is None or y_target is None:
                raise ValueError("Cannot find training data in model")
            
            # Compute model output
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.model(x_input)
                
                # Get negative log likelihood from mll
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                log_likelihood = mll(output, y_target).item()
        except Exception as e:
            # If there's an error, return very negative log probability
            log_likelihood = -1e10
        
        # Calculate log prior (a simplified version)
        log_prior = 0.0
        
        # For simplicity, just use simple priors
        for name, value in parameters.items():
            # For lengthscales, outputscales, use log-normal priors
            if 'lengthscale' in name or 'outputscale' in name:
                # Log-normal prior centered at current value
                mean = np.log(original_params[name].item())
                std = 1.0
                log_value = torch.log(value).item() if value.item() > 0 else -1e10
                log_prior += -0.5 * ((log_value - mean) / std) ** 2
            # For noise parameters, prefer smaller values
            elif 'noise' in name:
                # Half-normal prior centered at 0
                log_prior += -0.5 * (value.item() / 0.1) ** 2
        
        # Restore original parameters
        for name, param_value in original_params.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(param_value)
        
        return log_likelihood + log_prior
    
    def run_sampling(self, progress_bar: bool = True):
        """
        Run MCMC sampling using Random Walk Metropolis algorithm.
        
        Args:
            progress_bar: Whether to show a progress bar
        """
        # Make sure model is in eval mode
        self.model.eval()
        self.likelihood.eval()
        
        # Extract current parameter values
        current_params = {}
        for name in self.param_names:
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                current_params[name] = param.detach().clone()
        
        # Calculate initial log probability
        current_log_prob = self._log_posterior(current_params)
        
        # Set up progress bar
        total_iterations = self.burn_in + self.n_samples
        if progress_bar:
            pbar = tqdm(total=total_iterations, desc="MCMC")
        
        # Storage for acceptance monitoring
        accepted = 0
        
        # Main sampling loop
        for i in range(total_iterations):
            # Generate proposal parameters
            proposal_params = {}
            for name, param in current_params.items():
                # Scale the proposal based on parameter type
                scale = self.proposal_scale * self.parameter_scales.get(name, 1.0)
                
                # For multi-dimensional parameters, add noise to each element
                if param.numel() > 1:
                    noise = torch.randn_like(param) * scale
                    proposal = param + noise
                else:
                    # For scalar parameters
                    noise = torch.randn(1, device=self._get_device()).item() * scale
                    proposal = param + noise
                
                # Apply parameter constraints if they exist
                if name in self.param_constraints:
                    constraint = self.param_constraints[name]
                    try:
                        # Try to apply constraint
                        proposal = constraint.transform(proposal)
                    except Exception:
                        # If constraint fails, ensure proposal is positive
                        proposal = torch.abs(proposal)
                
                proposal_params[name] = proposal
            
            # Calculate proposal log probability
            proposal_log_prob = self._log_posterior(proposal_params)
            
            # Metropolis acceptance criterion
            log_alpha = proposal_log_prob - current_log_prob
            
            # Accept or reject
            if np.log(np.random.random()) < log_alpha:
                # Accept the proposal
                current_params = {
                    name: param.detach().clone() for name, param in proposal_params.items()
                }
                current_log_prob = proposal_log_prob
                accepted += 1
            
            # Store sample after burn-in
            if i >= self.burn_in:
                for name, param in current_params.items():
                    self.samples[name].append(param.detach().clone())
                self.log_probs.append(current_log_prob)
            
            # Update progress bar
            if progress_bar:
                accept_rate = accepted / (i + 1)
                pbar.set_postfix({
                    "log_prob": f"{current_log_prob:.2f}",
                    "accept": f"{accept_rate:.2f}"
                })
                pbar.update(1)
        
        # Close progress bar
        if progress_bar:
            pbar.close()
        
        # Calculate acceptance rate
        self.acceptance_rate = accepted / total_iterations
        
        # Convert samples to tensors for easier manipulation
        for name in self.param_names:
            if self.samples[name]:
                self.samples[name] = torch.stack(self.samples[name], dim=0)
        
        print(f"MCMC completed with {self.n_samples} samples (acceptance rate: {self.acceptance_rate:.2f})")
    
    def get_posterior_mean(self) -> Dict[str, torch.Tensor]:
        """
        Get posterior mean for each parameter.
        
        Returns:
            Dictionary of mean parameter values
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run sampling first.")
        
        return {name: torch.mean(samples, dim=0) for name, samples in self.samples.items()}
    
    def get_posterior_interval(self, alpha: float = 0.05) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get credible intervals for each parameter.
        
        Args:
            alpha: Significance level (e.g., 0.05 for 95% interval)
            
        Returns:
            Dictionary of (lower, upper) interval tuples
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run sampling first.")
        
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
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)
    
    def plot_traces(self, figure_path: Optional[str] = None):
        """
        Plot MCMC trace plots for scalar parameters.
        
        Args:
            figure_path: Optional path to save figure
            
        Returns:
            matplotlib figure
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run sampling first.")
        
        # Get scalar parameters
        scalar_params = [name for name, shape in self.param_shapes.items() 
                         if np.prod(shape) == 1]
        
        if not scalar_params:
            print("No scalar parameters to plot.")
            return None
        
        # Create figure
        fig, axes = plt.subplots(len(scalar_params), 2, figsize=(12, 3 * len(scalar_params)))
        
        # Handle single parameter case
        if len(scalar_params) == 1:
            axes = axes.reshape(1, 2)
        
        # Plot each parameter
        for i, name in enumerate(scalar_params):
            samples = self.samples[name].cpu().numpy()
            
            # Reshape if needed
            if samples.ndim > 1:
                samples = samples.reshape(-1)
            
            # Trace plot
            axes[i, 0].plot(samples)
            axes[i, 0].set_title(f"Trace for {name}")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True, alpha=0.3)
            
            # Posterior histogram
            axes[i, 1].hist(samples, bins=30, density=True, alpha=0.6)
            axes[i, 1].set_title(f"Posterior for {name}")
            axes[i, 1].set_xlabel("Value")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add mean and credible intervals
            mean = np.mean(samples)
            q025, q975 = np.percentile(samples, [2.5, 97.5])
            
            axes[i, 1].axvline(mean, color='r', linestyle='-', alpha=0.7)
            axes[i, 1].axvline(q025, color='r', linestyle='--', alpha=0.5)
            axes[i, 1].axvline(q975, color='r', linestyle='--', alpha=0.5)
            
            # Add text annotation
            axes[i, 1].text(0.05, 0.95, f"Mean = {mean:.4f}\n95% CI: [{q025:.4f}, {q975:.4f}]",
                         transform=axes[i, 1].transAxes, va='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def sample_posterior_predictive(
        self, 
        x_test: torch.Tensor, 
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Sample from the posterior predictive distribution.
        
        Args:
            x_test: Test points
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with posterior predictive samples and statistics
        """
        if not self.samples:
            raise RuntimeError("No samples available. Run sampling first.")
        
        # Ensure x_test is a tensor on the correct device
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test, dtype=torch.float32, device=self._get_device())
        
        # Original parameters (to restore later)
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.detach().clone()
        
        # Storage for predictions
        pred_samples = []
        
        # Sample indices to use (if we have more samples than requested)
        if n_samples <= len(self.log_probs):
            # Pick random indices
            indices = np.random.choice(len(self.log_probs), n_samples, replace=False)
        else:
            # Use all samples with replacement
            indices = np.random.choice(len(self.log_probs), n_samples, replace=True)
        
        # Generate predictions for each sampled parameter set
        self.model.eval()
        self.likelihood.eval()
        
        for idx in indices:
            # Set model parameters to sampled values
            for name in self.param_names:
                param = dict(self.model.named_parameters()).get(name)
                if param is not None:
                    with torch.no_grad():
                        param.copy_(self.samples[name][idx])
            
            # Make prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.model(x_test)
                sample = self.likelihood(output).sample()
            
            # Store prediction
            pred_samples.append(sample.cpu().numpy())
        
        # Restore original parameters
        for name, param_value in original_params.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(param_value)
        
        # Convert to numpy array
        pred_samples = np.array(pred_samples)
        
        # Calculate statistics
        mean = np.mean(pred_samples, axis=0)
        lower_ci = np.percentile(pred_samples, 2.5, axis=0)
        upper_ci = np.percentile(pred_samples, 97.5, axis=0)
        
        return {
            'mean': mean,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'samples': pred_samples
        }


class AdaptiveRobustMCMCSampler(RobustMCMCSampler):
    """
    An adaptive version of the Robust MCMC sampler that adjusts
    proposal scales during burn-in to achieve target acceptance rate.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        n_samples: int = 1000,
        burn_in: int = 200,
        proposal_scale: float = 0.1,
        parameter_scales: Optional[Dict[str, float]] = None,
        target_acceptance: float = 0.234,
        adaptation_window: int = 50,
        random_state: int = 42
    ):
        """
        Initialize the adaptive MCMC sampler.
        
        Args:
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            n_samples: Number of samples to generate
            burn_in: Number of burn-in samples to discard
            proposal_scale: Initial global scale of proposal distribution
            parameter_scales: Dictionary of parameter-specific scales
            target_acceptance: Target acceptance rate (optimal is ~0.234 for d>4)
            adaptation_window: Window size for adaptation
            random_state: Random seed
        """
        super().__init__(
            model=model,
            likelihood=likelihood,
            n_samples=n_samples,
            burn_in=burn_in,
            proposal_scale=proposal_scale,
            parameter_scales=parameter_scales,
            random_state=random_state
        )
        
        self.target_acceptance = target_acceptance
        self.adaptation_window = adaptation_window
    
    def run_sampling(self, progress_bar: bool = True):
        """
        Run adaptive MCMC sampling using Random Walk Metropolis algorithm.
        
        Args:
            progress_bar: Whether to show a progress bar
        """
        # Make sure model is in eval mode
        self.model.eval()
        self.likelihood.eval()
        
        # Extract current parameter values
        current_params = {}
        for name in self.param_names:
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                current_params[name] = param.detach().clone()
        
        # Calculate initial log probability
        current_log_prob = self._log_posterior(current_params)
        
        # Set up progress bar
        total_iterations = self.burn_in + self.n_samples
        if progress_bar:
            pbar = tqdm(total=total_iterations, desc="Adaptive MCMC")
        
        # Storage for acceptance monitoring
        accepted = 0
        recent_acceptances = []
        
        # Storage for adapting parameter scales
        param_adapt_count = {name: 0 for name in self.param_names}
        
        # Main sampling loop
        for i in range(total_iterations):
            # Generate proposal parameters
            proposal_params = {}
            proposed_param_name = None  # Track which parameter we're updating
            
            # Basic Random Walk Metropolis - update one parameter at a time
            if i % len(self.param_names) == 0:
                # Randomize parameter order for each cycle
                param_order = list(self.param_names)
                np.random.shuffle(param_order)
            
            # Get parameter to update in this iteration
            proposed_param_name = param_order[i % len(self.param_names)]
            
            # Copy all current parameters
            for name, param in current_params.items():
                if name == proposed_param_name:
                    # Generate proposal for this parameter
                    scale = self.proposal_scale * self.parameter_scales.get(name, 1.0)
                    
                    # For multi-dimensional parameters, add noise to each element
                    if param.numel() > 1:
                        noise = torch.randn_like(param) * scale
                        proposal = param + noise
                    else:
                        # For scalar parameters
                        noise = torch.randn(1, device=self._get_device()).item() * scale
                        proposal = param + noise
                    
                    # Apply parameter constraints if they exist
                    if name in self.param_constraints:
                        constraint = self.param_constraints[name]
                        try:
                            # Try to apply constraint
                            proposal = constraint.transform(proposal)
                        except Exception:
                            # If constraint fails, ensure proposal is positive
                            proposal = torch.abs(proposal)
                    
                    proposal_params[name] = proposal
                else:
                    # Keep other parameters the same
                    proposal_params[name] = param.clone()
            
            # Calculate proposal log probability
            proposal_log_prob = self._log_posterior(proposal_params)
            
            # Metropolis acceptance criterion
            log_alpha = proposal_log_prob - current_log_prob
            
            # Accept or reject
            accept = False
            if np.log(np.random.random()) < log_alpha:
                # Accept the proposal
                current_params = {
                    name: param.detach().clone() for name, param in proposal_params.items()
                }
                current_log_prob = proposal_log_prob
                accepted += 1
                accept = True
            
            # Track acceptance for adaptation
            recent_acceptances.append(accept)
            param_adapt_count[proposed_param_name] += 1
            
            # Adapt proposal scales during burn-in
            if i < self.burn_in and i > self.adaptation_window:
                # Only adapt occasionally
                if i % self.adaptation_window == 0:
                    recent_rate = sum(recent_acceptances[-self.adaptation_window:]) / self.adaptation_window
                    
                    # Update global scale based on recent acceptance rate
                    if recent_rate < self.target_acceptance:
                        # Too few acceptances, decrease scale
                        self.proposal_scale *= 0.9
                    else:
                        # Too many acceptances, increase scale
                        self.proposal_scale *= 1.1
                    
                    # Also update individual parameter scales if we have enough data
                    for name in self.param_names:
                        if param_adapt_count[name] >= 10:
                            # Reset counter
                            param_adapt_count[name] = 0
                            
                            # Get acceptance for this parameter
                            param_accepts = []
                            for j in range(len(recent_acceptances) - self.adaptation_window, len(recent_acceptances)):
                                if j >= 0 and param_order[j % len(self.param_names)] == name:
                                    param_accepts.append(recent_acceptances[j])
                            
                            if len(param_accepts) > 0:
                                param_rate = sum(param_accepts) / len(param_accepts)
                                
                                # Adjust parameter-specific scale
                                if param_rate < self.target_acceptance:
                                    self.parameter_scales[name] *= 0.9
                                else:
                                    self.parameter_scales[name] *= 1.1
            
            # Store sample after burn-in
            if i >= self.burn_in:
                for name, param in current_params.items():
                    self.samples[name].append(param.detach().clone())
                self.log_probs.append(current_log_prob)
            
            # Update progress bar
            if progress_bar:
                accept_rate = accepted / (i + 1)
                pbar.set_postfix({
                    "log_prob": f"{current_log_prob:.2f}",
                    "accept": f"{accept_rate:.2f}",
                    "scale": f"{self.proposal_scale:.4f}"
                })
                pbar.update(1)
        
        # Close progress bar
        if progress_bar:
            pbar.close()
        
        # Calculate acceptance rate
        self.acceptance_rate = accepted / total_iterations
        
        # Convert samples to tensors for easier manipulation
        for name in self.param_names:
            if self.samples[name]:
                self.samples[name] = torch.stack(self.samples[name], dim=0)
        
        print(f"MCMC completed with {self.n_samples} samples (acceptance rate: {self.acceptance_rate:.2f})")
        print(f"Final parameter scales: {self.parameter_scales}")


def test_robust_mcmc(model_file: str = "data/results/bayesian_gp/best_model.pt"):
    """
    Test the robust MCMC implementation on a saved model.
    
    Args:
        model_file: Path to saved model file
    """
    import os
    
    # Output directory
    output_dir = "data/results/mcmc_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load the model
    try:
        model_state = torch.load(model_file)
        print(f"Loaded model from {model_file}")
    except Exception as e:
        print(f"Could not load model: {str(e)}")
        print("Generating synthetic data and training a new model instead...")
        
        # Import model and data generation
        from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data_sparse(
            n_points=60,
            age_min=0,
            age_max=300,
            irregularity=0.7,
            proxy_types=['d18O', 'UK37'],
            random_state=42
        )
        
        # Extract data
        proxy_data = synthetic_data['proxy_data']
        true_sst = synthetic_data['true_sst']
        regular_ages = synthetic_data['regular_ages']
        
        # Create and fit model
        model = BayesianGPStateSpaceModel(
            proxy_types=['d18O', 'UK37'],
            kernel_type='combined',
            n_mcmc_samples=500,
            random_state=42
        )
        
        model.fit(proxy_data, training_iterations=300)
        
        # Save test data
        np.savez(
            f"{output_dir}/test_data.npz",
            regular_ages=regular_ages,
            true_sst=true_sst,
            **{f"{proxy}_age": data['age'] for proxy, data in proxy_data.items()},
            **{f"{proxy}_value": data['value'] for proxy, data in proxy_data.items()}
        )
        
        # Extract GP model and likelihood
        gp_model = model.gp_model
        likelihood = model.likelihood
        
        print("Model created and fitted successfully")
    
    # Set up MCMC sampler
    sampler = AdaptiveRobustMCMCSampler(
        model=gp_model,
        likelihood=likelihood,
        n_samples=500,
        burn_in=200,
        proposal_scale=0.05,
        target_acceptance=0.4,
        adaptation_window=50,
        random_state=42
    )
    
    # Run sampling
    print("Running MCMC sampling...")
    sampler.run_sampling(progress_bar=True)
    
    # Plot traces
    print("Plotting parameter traces...")
    fig_traces = sampler.plot_traces(figure_path=f"{output_dir}/parameter_traces.png")
    
    # Sample from posterior predictive
    print("Generating posterior predictive samples...")
    test_x = torch.linspace(0, 300, 300, device=sampler._get_device())
    post_pred = sampler.sample_posterior_predictive(test_x, n_samples=50)
    
    # Load test data if available
    try:
        test_data = np.load(f"{output_dir}/test_data.npz")
        regular_ages = test_data['regular_ages']
        true_sst = test_data['true_sst']
        
        # Create dictionary of proxy data
        proxy_data = {}
        for proxy in ['d18O', 'UK37']:
            if f"{proxy}_age" in test_data and f"{proxy}_value" in test_data:
                proxy_data[proxy] = {
                    'age': test_data[f"{proxy}_age"],
                    'value': test_data[f"{proxy}_value"]
                }
    except Exception:
        print("No test data found, plotting without ground truth")
        regular_ages = None
        true_sst = None
        proxy_data = None
    
    # Plot reconstruction
    print("Plotting reconstruction...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert test_x to numpy for plotting
    test_x_np = test_x.cpu().numpy()
    
    # Plot true SST if available
    if regular_ages is not None and true_sst is not None:
        test_true_sst = np.interp(test_x_np, regular_ages, true_sst)
        ax.plot(test_x_np, test_true_sst, 'k-', linewidth=2, label='True SST')
    
    # Plot mean and credible intervals
    ax.plot(test_x_np, post_pred['mean'], 'b-', linewidth=2, label='Mean Prediction')
    ax.fill_between(test_x_np, post_pred['lower_ci'], post_pred['upper_ci'], color='b', alpha=0.2, label='95% CI')
    
    # Plot some samples for visualization
    for i in range(min(10, post_pred['samples'].shape[0])):
        ax.plot(test_x_np, post_pred['samples'][i], 'b-', linewidth=0.5, alpha=0.2)
    
    # Plot proxy data if available
    if proxy_data is not None:
        markers = ['o', 's']
        colors = ['green', 'orange']
        
        for i, (proxy_type, data) in enumerate(proxy_data.items()):
            if proxy_type == 'd18O':
                scaling = -4.54545  # inverse_slope
                intercept = 3.0
            else:  # UK37
                scaling = 30.303   # inverse_slope
                intercept = 0.044
            
            # Convert to SST
            sst_values = (data['value'] - intercept) * scaling
            
            ax.scatter(data['age'], sst_values, color=colors[i], marker=markers[i], s=40, alpha=0.7,
                      label=f'{proxy_type} derived SST')
    
    # Formatting
    ax.set_xlabel('Age (kyr)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Bayesian GP Reconstruction with MCMC Uncertainty')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Reverse x-axis for geological convention
    ax.set_xlim(max(test_x_np), min(test_x_np))
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/mcmc_reconstruction.png", dpi=300, bbox_inches='tight')
    
    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    test_robust_mcmc()