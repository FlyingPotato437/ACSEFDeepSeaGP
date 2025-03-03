"""
Debug script for MCMC implementation
"""

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import os

from bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_data_sparse

# Create output directory
output_dir = "data/results/debug_mcmc"
os.makedirs(output_dir, exist_ok=True)

def run_debug_test():
    """Run a simplified test for debugging MCMC"""
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data_sparse(
        n_points=20,
        age_min=0,
        age_max=100,
        irregularity=0.5,
        proxy_types=['d18O'],  # Use single proxy for simplicity
        random_state=42
    )
    
    # Create and fit the model, using fallback naive MCMC
    print("Training model with naive MCMC...")
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O'],
        kernel_type='rbf',  # Use simpler kernel
        n_mcmc_samples=100,  # Use fewer samples
        random_state=42
    )
    
    model.fit(synthetic_data['proxy_data'], training_iterations=200)
    
    # Extract data for plotting
    regular_ages = synthetic_data['regular_ages']
    true_sst = synthetic_data['true_sst']
    
    # Make predictions
    test_ages = np.linspace(0, 100, 200)
    mean, lower, upper, samples = model.predict(test_ages, return_samples=True, n_samples=20)
    
    # Create a basic plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_ages, mean, 'b-', label='Mean Prediction')
    plt.fill_between(test_ages, lower, upper, color='b', alpha=0.2, label='95% CI')
    plt.plot(regular_ages, true_sst, 'k--', label='True SST')
    
    # Plot data points
    d18o_data = synthetic_data['proxy_data']['d18O']
    d18o_ages = d18o_data['age']
    d18o_values = d18o_data['value']
    d18o_sst = (d18o_values - 3.0) * -4.54545
    plt.scatter(d18o_ages, d18o_sst, marker='o', color='green', label='d18O data')
    
    # Plot posterior samples
    for i in range(min(5, samples.shape[0])):
        plt.plot(test_ages, samples[i], 'b-', linewidth=0.5, alpha=0.3)
    
    plt.xlabel('Age (kyr)')
    plt.ylabel('SST (°C)')
    plt.title('Bayesian GP Reconstruction with Naive MCMC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(max(test_ages), min(test_ages))  # Reverse x-axis
    
    plt.savefig(f"{output_dir}/naive_mcmc_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Implement manual simple MCMC with direct parameter perturbation
    print("Implementing manual parameter perturbation MCMC...")
    # Store original parameters
    original_params = {}
    for name, param in model.gp_model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.detach().clone()
    
    # Manual MCMC samples
    n_samples = 100
    perturbed_means = []
    perturbed_lowers = []
    perturbed_uppers = []
    
    for i in range(n_samples):
        # Perturb parameters randomly
        for name, param in model.gp_model.named_parameters():
            if param.requires_grad:
                # Calculate scale based on parameter type
                if 'lengthscale' in name:
                    scale = 0.1
                elif 'noise' in name:
                    scale = 0.01
                elif 'outputscale' in name:
                    scale = 0.1
                else:
                    scale = 0.1
                
                # Add random perturbation
                with torch.no_grad():
                    perturbed_value = param.data + torch.randn_like(param.data) * scale * param.data
                    # Ensure positivity for certain parameters
                    if any(s in name for s in ['lengthscale', 'noise', 'outputscale']):
                        perturbed_value = torch.abs(perturbed_value)
                    param.data.copy_(perturbed_value)
        
        # Make prediction with perturbed parameters
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(test_ages, dtype=torch.float32)
            output = model.gp_model(test_x)
            prediction = model.likelihood(output).sample()
            
            # Store prediction
            perturbed_means.append(prediction.cpu().numpy())
    
    # Restore original parameters
    for name, param_value in original_params.items():
        param = dict(model.gp_model.named_parameters()).get(name)
        if param is not None:
            with torch.no_grad():
                param.copy_(param_value)
    
    # Calculate statistics from perturbed samples
    perturbed_means = np.array(perturbed_means)
    perturbed_mean = np.mean(perturbed_means, axis=0)
    perturbed_lower = np.percentile(perturbed_means, 2.5, axis=0)
    perturbed_upper = np.percentile(perturbed_means, 97.5, axis=0)
    
    # Create plot comparing naive and manual MCMC
    plt.figure(figsize=(10, 6))
    
    # Original MCMC
    plt.plot(test_ages, mean, 'b-', linewidth=2, label='Naive MCMC Mean')
    plt.fill_between(test_ages, lower, upper, color='b', alpha=0.2, label='Naive MCMC 95% CI')
    
    # Manual perturbation MCMC
    plt.plot(test_ages, perturbed_mean, 'r-', linewidth=2, label='Manual MCMC Mean')
    plt.fill_between(test_ages, perturbed_lower, perturbed_upper, color='r', alpha=0.2, label='Manual MCMC 95% CI')
    
    # Truth and data
    plt.plot(regular_ages, true_sst, 'k--', label='True SST')
    plt.scatter(d18o_ages, d18o_sst, marker='o', color='green', label='d18O data')
    
    plt.xlabel('Age (kyr)')
    plt.ylabel('SST (°C)')
    plt.title('Comparison of MCMC Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(max(test_ages), min(test_ages))  # Reverse x-axis
    
    plt.savefig(f"{output_dir}/mcmc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Debug test completed. Results saved to {output_dir}")

if __name__ == "__main__":
    run_debug_test()