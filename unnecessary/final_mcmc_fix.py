"""
Final MCMC Implementation for Bayesian Gaussian Process Models

This module provides the most robust implementation of MCMC for Bayesian GP models,
specifically designed to address numerical issues with GPyTorch models in paleoclimate
reconstruction. Key features:

1. Component-wise Metropolis algorithm for better mixing
2. Adaptive proposal distributions with custom scales for each parameter
3. Proper prior distributions matched to parameter constraints
4. Robust error handling and fallbacks for cholesky decomposition issues
5. Comprehensive diagnostics for monitoring convergence
"""

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm.auto import tqdm
import os

# Make sure output directory exists
output_dir = "data/results/final_mcmc"
os.makedirs(output_dir, exist_ok=True)

class RobustMCMC:
    """
    A mathematically robust implementation of MCMC for Bayesian GP models.
    
    This class implements a component-wise Metropolis algorithm with adaptive proposal
    distributions, specifically designed to handle numerical issues in GP models
    and provide reliable uncertainty quantification.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        likelihood: Any,
        n_samples: int = 1000,
        burn_in: int = 300,
        proposal_scale: float = 0.05,
        parameter_scales: Optional[Dict[str, float]] = None,
        target_acceptance: float = 0.25,
        adaptation_window: int = 50,
        random_state: int = 42
    ):
        """
        Initialize the MCMC sampler.
        
        Args:
            model: GPyTorch model
            likelihood: GPyTorch likelihood
            n_samples: Number of samples to generate
            burn_in: Number of burn-in samples to discard
            proposal_scale: Initial global scale of proposal distribution
            parameter_scales: Dictionary of parameter-specific scales
            target_acceptance: Target acceptance rate
            adaptation_window: Window size for adaptation
            random_state: Random seed
        """
        self.model = model
        self.likelihood = likelihood
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.proposal_scale = proposal_scale
        self.target_acceptance = target_acceptance
        self.adaptation_window = adaptation_window
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Extract parameter information
        self._initialize_parameters()
        
        # Parameter-specific scales
        if parameter_scales is None:
            self.parameter_scales = {
                name: 0.1 for name in self.param_names
            }
            
            # Adjust scales based on parameter names
            for name in self.param_names:
                if "lengthscale" in name:
                    self.parameter_scales[name] = 0.1
                elif "noise" in name:
                    self.parameter_scales[name] = 0.02  # Smaller steps for noise
                elif "outputscale" in name:
                    self.parameter_scales[name] = 0.1
                elif "period" in name:
                    self.parameter_scales[name] = 0.5  # Larger steps for period
                elif "mean" in name:
                    self.parameter_scales[name] = 0.05
        else:
            self.parameter_scales = parameter_scales
            # Ensure all parameters have a scale
            for name in self.param_names:
                if name not in self.parameter_scales:
                    self.parameter_scales[name] = 0.1
        
        # Storage for samples and diagnostics
        self.samples = {name: [] for name in self.param_names}
        self.log_probs = []
        self.acceptance_rate = 0.0
        self.acceptance_history = []
        self.param_scales_history = {}
        for name in self.param_names:
            self.param_scales_history[name] = []
    
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
            # Set constraints for common parameter types
            if any(s in name for s in ['lengthscale', 'noise', 'outputscale', 'period']):
                self.param_constraints[name] = 'positive'
    
    def _get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    def _log_prior(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log prior probability for parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log prior probability
        """
        log_prior = 0.0
        
        # Use appropriate priors based on parameter types
        for name, value in parameters.items():
            param_value = value.item() if value.numel() == 1 else value[0].item()
            
            # For lengthscales, outputscales, use log-normal priors (more robust for positive parameters)
            if 'lengthscale' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Log-normal prior
                log_value = np.log(param_value)
                log_prior += -0.5 * (log_value ** 2) - log_value - 0.5 * np.log(2 * np.pi)
            
            # For outputscales, use log-normal priors
            elif 'outputscale' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Log-normal prior
                log_value = np.log(param_value)
                log_prior += -0.5 * (log_value ** 2) - log_value - 0.5 * np.log(2 * np.pi)
            
            # For noise parameters, use half-normal prior
            elif 'noise' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Half-normal prior
                log_prior += -0.5 * (param_value / 0.1) ** 2
            
            # For periodic parameters, use normal priors
            elif 'period' in name:
                if param_value <= 0:
                    return -np.inf  # Invalid value
                # Normal prior centered at 41 (for Milankovitch cycles)
                log_prior += -0.5 * ((param_value - 41.0) / 5.0) ** 2
        
        return log_prior
    
    def _log_likelihood(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log likelihood of parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log likelihood value
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
            
            # Get training data
            x_input = None
            y_target = None
            
            # Try to get training data from different model attributes
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
            
            # Compute model output with extra numerical stability settings
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                output = self.model(x_input)
                
                # Get negative log likelihood from mll
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                log_likelihood = mll(output, y_target).item()
        except Exception as e:
            # If there's an error, return very negative log probability
            log_likelihood = -1e10
        
        # Restore original parameters
        for name, param_value in original_params.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(param_value)
        
        return log_likelihood
    
    def _log_posterior(self, parameters: Dict[str, torch.Tensor]) -> float:
        """
        Calculate log posterior probability.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Log posterior probability
        """
        # Calculate log prior first as a quick check
        log_prior = self._log_prior(parameters)
        
        # If prior is -infinity, we can skip likelihood calculation
        if log_prior == -np.inf:
            return -np.inf
        
        # Calculate log likelihood
        log_likelihood = self._log_likelihood(parameters)
        
        # Return log posterior
        return log_likelihood + log_prior
    
    def run_sampling(self, progress_bar: bool = True) -> Dict[str, np.ndarray]:
        """
        Run MCMC sampling using component-wise Metropolis algorithm.
        
        Args:
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dictionary of parameter samples
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
        accepted = {name: 0 for name in self.param_names}
        recent_acceptances = {name: [] for name in self.param_names}
        
        # Storage for updates
        param_order = list(self.param_names)
        
        # Store adaptation data
        for name in self.param_names:
            self.param_scales_history[name].append(self.parameter_scales[name])
        
        # Main sampling loop
        for i in range(total_iterations):
            # Randomize parameter order for every iteration
            if i % len(self.param_names) == 0:
                np.random.shuffle(param_order)
            
            # Update one parameter at a time (component-wise Metropolis)
            for param_name in param_order:
                # Copy all current parameters
                proposed_params = {name: param.clone() for name, param in current_params.items()}
                
                # Generate proposal for this parameter
                scale = self.proposal_scale * self.parameter_scales[param_name]
                current_value = current_params[param_name]
                
                # Add scaled noise based on parameter dimension
                if current_value.numel() == 1:  # Scalar parameter
                    # For positive parameters, use multiplicative noise (log-normal)
                    if self.param_constraints.get(param_name) == 'positive':
                        multiplier = np.exp(np.random.normal(0, scale))
                        proposed_value = current_value * multiplier
                    else:
                        # Additive noise for unconstrained parameters
                        perturbation = torch.randn(1, device=self._get_device()).item() * scale
                        proposed_value = current_value + perturbation
                else:  # Vector/matrix parameter
                    # Use element-wise perturbation
                    if self.param_constraints.get(param_name) == 'positive':
                        multiplier = torch.exp(torch.randn_like(current_value) * scale)
                        proposed_value = current_value * multiplier
                    else:
                        perturbation = torch.randn_like(current_value) * scale
                        proposed_value = current_value + perturbation
                
                # Ensure positivity for certain parameters
                if self.param_constraints.get(param_name) == 'positive':
                    proposed_value = torch.abs(proposed_value)
                
                proposed_params[param_name] = proposed_value
                
                # Calculate proposal log probability
                proposed_log_prob = self._log_posterior(proposed_params)
                
                # Metropolis acceptance criterion
                log_alpha = proposed_log_prob - current_log_prob
                
                # Accept or reject
                accepted_proposal = False
                if np.log(np.random.random()) < log_alpha:
                    # Accept the proposal
                    current_params[param_name] = proposed_value.detach().clone()
                    current_log_prob = proposed_log_prob
                    accepted[param_name] += 1
                    accepted_proposal = True
                
                # Track acceptance for adaptation
                recent_acceptances[param_name].append(accepted_proposal)
                
                # Adapt parameter scales during burn-in
                if i < self.burn_in and len(recent_acceptances[param_name]) >= self.adaptation_window:
                    # Compute recent acceptance rate for this parameter
                    recent_rate = sum(recent_acceptances[param_name][-self.adaptation_window:]) / self.adaptation_window
                    
                    # Update scale for this parameter
                    if recent_rate < self.target_acceptance:
                        # Too few acceptances, decrease scale
                        self.parameter_scales[param_name] *= 0.9
                    else:
                        # Too many acceptances, increase scale
                        self.parameter_scales[param_name] *= 1.1
                    
                    # Record adaptation
                    self.param_scales_history[param_name].append(self.parameter_scales[param_name])
            
            # Store acceptance rate for this iteration
            total_accepted = sum(accepted.values())
            total_proposed = (i + 1) * len(self.param_names)
            current_acceptance = total_accepted / total_proposed
            self.acceptance_history.append(current_acceptance)
            
            # Store sample after burn-in
            if i >= self.burn_in:
                for name, param in current_params.items():
                    self.samples[name].append(param.detach().clone().cpu().numpy())
                self.log_probs.append(current_log_prob)
            
            # Update progress bar
            if progress_bar:
                pbar.set_postfix({
                    "log_prob": f"{current_log_prob:.2f}",
                    "accept": f"{current_acceptance:.2f}"
                })
                pbar.update(1)
        
        # Close progress bar
        if progress_bar:
            pbar.close()
        
        # Calculate acceptance rate
        total_accepted = sum(accepted.values())
        total_proposed = total_iterations * len(self.param_names)
        self.acceptance_rate = total_accepted / total_proposed
        
        # Convert sample lists to numpy arrays
        for name in self.param_names:
            if self.samples[name]:
                self.samples[name] = np.array(self.samples[name])
        
        print(f"MCMC completed: {self.n_samples} samples with {self.acceptance_rate:.2f} acceptance rate")
        return self.samples
    
    def get_parameter_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for sampled parameters.
        
        Returns:
            Dictionary with parameter statistics
        """
        if not self.samples or not all(len(s) > 0 for s in self.samples.values()):
            raise ValueError("No samples available. Run sampling first.")
        
        stats = {}
        for name, samples in self.samples.items():
            # For vector parameters, just use the first element
            if samples.ndim > 2:
                samples_flat = samples[:, 0, 0]
            elif samples.ndim == 2 and samples.shape[1] > 1:
                samples_flat = samples[:, 0]
            else:
                samples_flat = samples.flatten()
            
            mean = np.mean(samples_flat)
            std = np.std(samples_flat)
            q025, q50, q975 = np.percentile(samples_flat, [2.5, 50, 97.5])
            
            stats[name] = {
                'mean': mean,
                'median': q50,
                'std': std,
                'lower_95': q025,
                'upper_95': q975
            }
        
        return stats
    
    def plot_traces(self, figure_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot MCMC traces for all parameters.
        
        Args:
            figure_path: Optional path to save figure
            
        Returns:
            Tuple of matplotlib figures (parameter traces, diagnostics)
        """
        if not self.samples or not all(len(s) > 0 for s in self.samples.values()):
            raise ValueError("No samples available. Run sampling first.")
        
        param_names = list(self.samples.keys())
        n_params = len(param_names)
        
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 4 * n_params))
        
        # Handle single parameter case
        if n_params == 1:
            axes = axes.reshape(1, 2)
        
        # Plot each parameter
        for i, name in enumerate(param_names):
            samples = self.samples[name]
            
            # If parameter is multi-dimensional, just use the first element
            if samples.ndim > 2:
                samples_flat = samples[:, 0, 0]
            elif samples.ndim == 2 and samples.shape[1] > 1:
                samples_flat = samples[:, 0]
            else:
                samples_flat = samples.flatten()
            
            # Trace plot
            axes[i, 0].plot(samples_flat)
            axes[i, 0].set_title(f"Trace for {name}")
            axes[i, 0].set_xlabel("Iteration")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram
            axes[i, 1].hist(samples_flat, bins=30, alpha=0.7, density=True)
            
            # Add mean and CI
            mean = np.mean(samples_flat)
            q025, q975 = np.percentile(samples_flat, [2.5, 97.5])
            
            axes[i, 1].axvline(mean, color='r', linestyle='-', alpha=0.7)
            axes[i, 1].axvline(q025, color='r', linestyle='--', alpha=0.5)
            axes[i, 1].axvline(q975, color='r', linestyle='--', alpha=0.5)
            
            # Add text annotation
            axes[i, 1].text(0.05, 0.95, f"Mean: {mean:.4f}\n95% CI: [{q025:.4f}, {q975:.4f}]",
                          transform=axes[i, 1].transAxes, va='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i, 1].set_title(f"Posterior for {name}")
            axes[i, 1].set_xlabel("Value")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create diagnostic plots
        fig_diag, axes_diag = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot log probability
        if self.log_probs:
            axes_diag[0].plot(self.log_probs)
            axes_diag[0].set_title("Log Probability Trace")
            axes_diag[0].set_xlabel("Iteration")
            axes_diag[0].set_ylabel("Log Probability")
            axes_diag[0].grid(True, alpha=0.3)
        
        # Plot acceptance rate
        if self.acceptance_history:
            axes_diag[1].plot(self.acceptance_history)
            axes_diag[1].set_title("Acceptance Rate")
            axes_diag[1].set_xlabel("Iteration")
            axes_diag[1].set_ylabel("Acceptance Rate")
            axes_diag[1].axhline(self.target_acceptance, color='r', linestyle='--')
            axes_diag[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if figure_path:
            fig.savefig(figure_path, dpi=300, bbox_inches='tight')
            diag_path = figure_path.replace('.png', '_diagnostics.png')
            fig_diag.savefig(diag_path, dpi=300, bbox_inches='tight')
            
            # Create parameter scale adaptation plots
            fig_scales, axes_scales = plt.subplots(len(self.param_names), 1, figsize=(12, 3 * len(self.param_names)))
            if len(self.param_names) == 1:
                axes_scales = [axes_scales]
                
            for i, name in enumerate(self.param_names):
                axes_scales[i].plot(self.param_scales_history[name])
                axes_scales[i].set_title(f"Scale Adaptation for {name}")
                axes_scales[i].set_xlabel("Adaptation Step")
                axes_scales[i].set_ylabel("Proposal Scale")
                axes_scales[i].set_yscale('log')
                axes_scales[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            scales_path = figure_path.replace('.png', '_scales.png')
            fig_scales.savefig(scales_path, dpi=300, bbox_inches='tight')
        
        return fig, fig_diag
    
    def sample_posterior_predictive(self, x_test: torch.Tensor, n_samples: int = 20) -> Dict[str, np.ndarray]:
        """
        Generate predictions using posterior samples.
        
        Args:
            x_test: Test input points
            n_samples: Number of samples to use
            
        Returns:
            Dictionary with prediction results
        """
        if not self.samples or not all(len(s) > 0 for s in self.samples.values()):
            raise ValueError("No samples available. Run sampling first.")
        
        # Ensure x_test is a tensor on the right device
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test, dtype=torch.float32)
            
        # Ensure x_test is on the same device as the model
        device = self._get_device()
        x_test = x_test.to(device)
        
        # Save original parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.detach().clone()
        
        # Select random samples
        sample_indices = np.random.choice(len(self.samples[self.param_names[0]]), 
                                         size=min(n_samples, len(self.samples[self.param_names[0]])),
                                         replace=False)
        
        # Generate predictions for each sampled parameter set
        all_predictions = []
        
        for idx in sample_indices:
            # Set model parameters to this sample
            for name in self.param_names:
                param = dict(self.model.named_parameters()).get(name)
                if param is not None:
                    sample_value = self.samples[name][idx]
                    
                    # Handle multidimensional parameters
                    if isinstance(sample_value, np.ndarray):
                        if param.numel() == 1 and sample_value.size > 1:
                            # Take first element if parameter should be scalar
                            sample_tensor = torch.tensor(sample_value.flat[0], 
                                                      dtype=torch.float32, device=device)
                        else:
                            # Try to match the shape
                            try:
                                sample_tensor = torch.tensor(sample_value, 
                                                          dtype=torch.float32, device=device)
                                
                                # Reshape if needed
                                if sample_tensor.shape != param.shape:
                                    sample_tensor = sample_tensor.reshape(param.shape)
                            except:
                                # Fallback: use original parameter shape
                                print(f"Warning: Shape mismatch for {name}. Using original parameter.")
                                sample_tensor = param.detach().clone()
                    else:
                        sample_tensor = torch.tensor(sample_value, dtype=torch.float32, device=device)
                    
                    # Copy the parameter
                    with torch.no_grad():
                        try:
                            param.copy_(sample_tensor)
                        except:
                            print(f"Warning: Failed to set parameter {name}. Using original parameter.")
            
            # Generate prediction
            try:
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    output = self.model(x_test)
                    pred = self.likelihood(output).sample()
                    all_predictions.append(pred.cpu().numpy())
            except Exception as e:
                print(f"Warning: Prediction failed for sample {idx}: {str(e)}")
        
        # Restore original parameters
        for name, value in original_params.items():
            param = dict(self.model.named_parameters()).get(name)
            if param is not None:
                with torch.no_grad():
                    param.copy_(value)
        
        # Calculate statistics
        if all_predictions:
            predictions = np.array(all_predictions)
            mean = np.mean(predictions, axis=0)
            lower_ci = np.percentile(predictions, 2.5, axis=0)
            upper_ci = np.percentile(predictions, 97.5, axis=0)
        else:
            print("Warning: No valid predictions generated.")
            # Create dummy outputs matching x_test shape
            x_np = x_test.cpu().numpy()
            mean = np.zeros_like(x_np)
            lower_ci = np.zeros_like(x_np)
            upper_ci = np.zeros_like(x_np)
            predictions = np.zeros((min(1, n_samples), len(x_np)))
        
        return {
            'mean': mean,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'samples': predictions
        }


def test_robust_mcmc():
    """
    Test the RobustMCMC implementation on paleoclimate data.
    
    Returns:
        Dictionary with test results
    """
    from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse
    
    print("Testing RobustMCMC implementation...")
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data_sparse(
        n_points=30,
        age_min=0,
        age_max=200,
        irregularity=0.8,
        proxy_types=['d18O', 'UK37'],
        random_state=42
    )
    
    # Extract data
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    proxy_data = synthetic_data['proxy_data']
    
    # Train Bayesian GP model
    print("Training Bayesian GP model...")
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37'],
        kernel_type='combined',
        n_mcmc_samples=500,
        random_state=42
    )
    
    model.fit(proxy_data, training_iterations=200)
    
    # Original model predictions
    test_ages = np.linspace(0, 200, 300)
    test_true_sst = np.interp(test_ages, regular_ages, true_sst)
    original_mean, original_lower, original_upper = model.predict(test_ages)
    
    # Apply RobustMCMC to the fitted model
    print("Running RobustMCMC sampling...")
    mcmc = RobustMCMC(
        model=model.gp_model,
        likelihood=model.likelihood,
        n_samples=400,
        burn_in=200,
        random_state=42
    )
    
    # Run sampling
    samples = mcmc.run_sampling(progress_bar=True)
    
    # Plot parameter traces
    mcmc.plot_traces(figure_path=f"{output_dir}/parameter_traces.png")
    
    # Generate posterior predictions
    print("Generating posterior predictions...")
    predictions = mcmc.sample_posterior_predictive(
        torch.tensor(test_ages, dtype=torch.float32), 
        n_samples=50
    )
    
    # Create comparison plot
    print("Creating comparison plots...")
    
    # Plot data and models
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    
    # Plot data and true SST for all subplots
    for ax in axes:
        ax.plot(test_ages, test_true_sst, 'k-', linewidth=2, label='True SST')
        
        # Plot proxy data
        markers = ['o', 's']
        colors = ['blue', 'green']
        
        for i, (proxy_type, data) in enumerate(proxy_data.items()):
            if proxy_type == 'd18O':
                sst_values = (data['value'] - 3.0) * -4.54545
            else:  # UK37
                sst_values = (data['value'] - 0.044) * 30.303
            
            ax.scatter(data['age'], sst_values, color=colors[i], marker=markers[i], 
                      s=40, alpha=0.7, label=f'{proxy_type} data')
    
    # Plot original model prediction (Bayesian GP with perturbation)
    ax = axes[0]
    ax.plot(test_ages, original_mean, 'b-', linewidth=2, label='Original Model')
    ax.fill_between(test_ages, original_lower, original_upper, color='b', alpha=0.2, label='95% CI')
    ax.set_title('Bayesian GP with Parameter Perturbation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('SST (°C)')
    
    # Plot MCMC prediction
    ax = axes[1]
    ax.plot(test_ages, predictions['mean'], 'r-', linewidth=2, label='RobustMCMC')
    ax.fill_between(test_ages, predictions['lower_ci'], predictions['upper_ci'], 
                   color='r', alpha=0.2, label='95% CI')
    ax.set_title('Bayesian GP with RobustMCMC Sampling')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('SST (°C)')
    
    # Plot MCMC samples
    ax = axes[2]
    ax.plot(test_ages, test_true_sst, 'k-', linewidth=2, label='True SST')
    
    # Plot a subset of posterior samples
    for i in range(min(10, predictions['samples'].shape[0])):
        ax.plot(test_ages, predictions['samples'][i], 'r-', linewidth=0.5, alpha=0.15)
    
    # Add mean and CI
    ax.plot(test_ages, predictions['mean'], 'r-', linewidth=2, label='Mean')
    ax.fill_between(test_ages, predictions['lower_ci'], predictions['upper_ci'], 
                   color='r', alpha=0.2, label='95% CI')
    
    ax.set_title('MCMC Posterior Samples')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Age (kyr)')
    ax.set_ylabel('SST (°C)')
    
    # Reverse x-axis for geological convention
    for ax in axes:
        ax.set_xlim(max(test_ages), min(test_ages))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mcmc_comparison.png", dpi=300, bbox_inches='tight')
    
    # Calculate metrics
    original_metrics = {
        'rmse': np.sqrt(np.mean((original_mean - test_true_sst)**2)),
        'mae': np.mean(np.abs(original_mean - test_true_sst)),
        'coverage': np.mean((original_lower <= test_true_sst) & (test_true_sst <= original_upper)),
        'ci_width': np.mean(original_upper - original_lower)
    }
    
    mcmc_metrics = {
        'rmse': np.sqrt(np.mean((predictions['mean'] - test_true_sst)**2)),
        'mae': np.mean(np.abs(predictions['mean'] - test_true_sst)),
        'coverage': np.mean((predictions['lower_ci'] <= test_true_sst) & 
                           (test_true_sst <= predictions['upper_ci'])),
        'ci_width': np.mean(predictions['upper_ci'] - predictions['lower_ci'])
    }
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Original Model: RMSE={original_metrics['rmse']:.4f}, Coverage={original_metrics['coverage']:.2f}, CI Width={original_metrics['ci_width']:.2f}")
    print(f"RobustMCMC: RMSE={mcmc_metrics['rmse']:.4f}, Coverage={mcmc_metrics['coverage']:.2f}, CI Width={mcmc_metrics['ci_width']:.2f}")
    
    # Create metrics comparison plot
    labels = ['Original', 'RobustMCMC']
    metrics_data = [
        [original_metrics['rmse'], mcmc_metrics['rmse']],
        [original_metrics['coverage'], mcmc_metrics['coverage']],
        [original_metrics['ci_width'], mcmc_metrics['ci_width']]
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    # Plot bars
    rects1 = ax.bar(x - width, metrics_data[0], width, label='RMSE (°C)')
    rects2 = ax.bar(x, metrics_data[1], width, label='Coverage')
    rects3 = ax.bar(x + width, metrics_data[2], width, label='CI Width (°C)')
    
    # Add labels
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
    
    print(f"Results saved to {output_dir}")
    
    return {
        'model': model,
        'mcmc': mcmc,
        'synthetic_data': synthetic_data,
        'original_metrics': original_metrics,
        'mcmc_metrics': mcmc_metrics,
        'predictions': predictions
    }


if __name__ == "__main__":
    test_robust_mcmc()