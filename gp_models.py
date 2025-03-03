"""
Standard Gaussian Process Models for Paleoclimate Reconstruction

This module implements traditional GP models with RBF and Matérn kernels
for comparison with the Bayesian GP State-Space model.
"""

import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, Optional

# Configure GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseGP:
    """
    Base class for Gaussian Process models.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Initialize the GP model.
        
        Args:
            x: Input ages
            y: Output proxy-derived SST values
        """
        # Convert data to tensors
        self.x_train = torch.tensor(x, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(y, dtype=torch.float32).to(device)
        
        # Initialize likelihood and model
        self.likelihood = None
        self.model = None
        self.is_fitted = False
    
    def fit(self, proxy_data_dict: Dict, training_iterations: int = 300, run_mcmc: bool = False):
        """
        Fit the Bayesian GP State-Space model to the proxy data.
        
        Args:
            proxy_data_dict: Dictionary with proxy data. Each key is a proxy type,
                        and each value is a dict with 'age' and 'value' arrays.
            training_iterations: Number of iterations for optimizer
            run_mcmc: Whether to run MCMC sampling after optimization
            
        Returns:
            self: The fitted model
        """
        # Preprocess data
        train_x, train_y = self._preprocess_data(proxy_data_dict)
        
        # Store training data
        self.train_x = train_x
        self.train_y = train_y
        
        # Initialize model
        self.model, self.likelihood = self._init_model(train_x, train_y)
        
        print("Model initialized. Training...")
        # Set model to training mode
        self.model.train()
        self.likelihood.train()
        
        # Use the Adam optimizer with reduced learning rate
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}
        ], lr=0.03)  # Reduced learning rate to prevent overfitting
        
        # "Loss" for GP is the negative log marginal likelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Initialize loss history
        losses = []
        
        # Training loop
        with gpytorch.settings.cholesky_jitter(1e-3):
            # Use try-except for numerical stability
            try:
                for i in range(training_iterations):
                    optimizer.zero_grad()
                    output = self.model(self.model.train_inputs[0])
                    
                    # Catch and handle numerical issues in loss calculation
                    try:
                        loss = -self.mll(output, self.model.train_targets)
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    except RuntimeError as e:
                        if "singular" in str(e).lower() or "cholesky" in str(e).lower() or "positive definite" in str(e).lower():
                            print(f"Numerical issue at iteration {i+1}: {e}")
                            print("Adjusting optimization parameters and continuing...")
                            # Add more jitter to the model
                            with torch.no_grad():
                                if hasattr(self.likelihood, 'noise'):
                                    self.likelihood.noise = self.likelihood.noise * 1.05
                        else:
                            raise e
                    
                    if (i+1) % 50 == 0:
                        print(f'Iteration {i+1}/{training_iterations} - Loss: {losses[-1] if losses else "N/A"}')
                        # Check for early stopping
                        if i > 100 and self._early_stopping_check(losses):
                            print(f"Early stopping at iteration {i+1}")
                            break
            
            except Exception as e:
                print(f"Training halted early due to: {str(e)}")
                print(f"Completed {len(losses)} iterations before error")
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Run MCMC sampling if requested
        if run_mcmc:
            print("Running MCMC sampling for uncertainty quantification...")
            self._run_mcmc_sampling()
        
        self.is_fitted = True
        return self
    
    def predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions at test points.
        
        Args:
            x_test: Test input ages
            
        Returns:
            mean, lower_ci, upper_ci
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x_test_tensor))
            
            # Get mean and confidence intervals
            mean = posterior.mean.cpu().numpy()
            lower, upper = posterior.confidence_region()
            lower_ci = lower.cpu().numpy()
            upper_ci = upper.cpu().numpy()
        
        return mean, lower_ci, upper_ci
    
    def get_params(self) -> dict:
        """
        Get the fitted model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        params = {}
        
        with torch.no_grad():
            # Get noise
            params['noise'] = self.likelihood.noise.item()
            
            # Get mean
            params['mean'] = self.model.mean_module.constant.item()
        
        return params


class RBFKernelGP(BaseGP):
    """
    GP model with RBF kernel.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Initialize RBF kernel GP.
        
        Args:
            x: Input ages
            y: Output proxy-derived SST values
        """
        super().__init__(x, y)
        
        # Initialize likelihood
        self.likelihood = GaussianLikelihood().to(device)
        
        # Define RBF kernel GP model
        class RBFGPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(RBFKernel())
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)
        
        # Create model
        self.model = RBFGPModel(self.x_train, self.y_train, self.likelihood).to(device)
    
    def get_params(self) -> dict:
        """
        Get RBF kernel parameters.
        
        Returns:
            Dictionary of model parameters
        """
        params = super().get_params()
        
        with torch.no_grad():
            # RBF kernel parameters
            params['lengthscale'] = self.model.covar_module.base_kernel.lengthscale.item()
            params['outputscale'] = self.model.covar_module.outputscale.item()
        
        return params


class MaternGP(BaseGP):
    """
    GP model with Matérn kernel.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, nu: float = 2.5):
        """
        Initialize Matérn kernel GP.
        
        Args:
            x: Input ages
            y: Output proxy-derived SST values
            nu: Smoothness parameter (default: 2.5)
        """
        super().__init__(x, y)
        
        self.nu = nu
        
        # Initialize likelihood
        self.likelihood = GaussianLikelihood().to(device)
        
        # Define Matérn kernel GP model
        class MaternGPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood, nu):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(MaternKernel(nu=nu))
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)
        
        # Create model
        self.model = MaternGPModel(self.x_train, self.y_train, self.likelihood, self.nu).to(device)
    
    def get_params(self) -> dict:
        """
        Get Matérn kernel parameters.
        
        Returns:
            Dictionary of model parameters
        """
        params = super().get_params()
        
        with torch.no_grad():
            # Matérn kernel parameters
            params['lengthscale'] = self.model.covar_module.base_kernel.lengthscale.item()
            params['outputscale'] = self.model.covar_module.outputscale.item()
            params['nu'] = self.nu
        
        return params