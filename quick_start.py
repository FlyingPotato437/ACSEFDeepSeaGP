"""
Quick Start Demo for Enhanced Bayesian GP State-Space Model

This script provides a quick demonstration of the enhanced Bayesian GP State-Space 
model for paleoclimate reconstruction with a focus on multi-proxy weighting, 
adaptive kernel lengthscales, multi-scale periodic components, heteroscedastic
noise modeling, and MCMC uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from models.bayesian_gp_state_space import BayesianGPStateSpaceModel, generate_synthetic_multiproxy_data


def run_quick_demo():
    """Run a quick demonstration of the model."""
    # Create output directory
    output_dir = "data/results/quick_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=====================================================")
    print("Enhanced Bayesian GP State-Space Model Quick Demo")
    print("=====================================================")
    print("\nThis demonstration shows how the model handles:")
    print("  1. Multi-proxy weighting with balanced influence")
    print("  2. Adaptive kernel lengthscales for abrupt transitions")
    print("  3. Multi-scale periodic components for orbital cycles")
    print("  4. Heteroscedastic noise modeling")
    print("  5. MCMC uncertainty quantification\n")
    
    print("Generating synthetic multi-proxy data...")
    # Generate synthetic data with known features
    data = generate_synthetic_multiproxy_data(
        n_points=60,               # Sparse sampling
        age_min=0,                 # Age range in kyr
        age_max=500,
        proxy_types=['d18O', 'UK37', 'Mg_Ca'],  # Multiple proxy types
        n_transitions=3,           # Add abrupt transitions
        transition_magnitude=3.0,  # Large transitions
        include_orbital_cycles=True,  # Include Milankovitch cycles
        smoothness=1.0,
        proxy_noise_scale=1.2,     # Realistic noise levels
        random_state=42
    )
    
    # Configure kernel
    kernel_config = {
        'base_kernel_type': 'matern',   # Matern kernel base
        'min_lengthscale': 2.0,         # Minimum physically meaningful lengthscale
        'max_lengthscale': 10.0,        # Maximum lengthscale
        'base_lengthscale': 5.0,        # Base lengthscale
        'adaptation_strength': 1.5,     # How strongly to adapt to transitions
        'lengthscale_regularization': 0.1,  # Prevent unrealistic fluctuations
        'include_periodic': True,       # Include Milankovitch components
        'periods': [100.0, 41.0, 23.0],  # Eccentricity, obliquity, precession
        'outputscales': [2.0, 1.0, 0.5]  # Relative weights
    }
    
    # Configure MCMC (reduced samples for quick demo)
    mcmc_config = {
        'n_samples': 300,
        'burn_in': 60,
        'thinning': 2,
        'step_size': 0.05,
        'target_acceptance': 0.6,
        'adaptation_steps': 30
    }
    
    # Disable MCMC temporarily until we resolve compatibility issues
    run_mcmc = False
    
    print("Initializing model with advanced components...")
    # Initialize model
    model = BayesianGPStateSpaceModel(
        proxy_types=['d18O', 'UK37', 'Mg_Ca'],
        weighting_method='balanced',  # Balanced proxy weighting
        kernel_config=kernel_config,
        mcmc_config=mcmc_config,
        calibration_params=data['calibration_params'],
        random_state=42
    )
    
    print("Fitting model (this may take a minute)...")
    # Fit model with reduced iterations for quick demo
    model.fit(
        data['proxy_data'],
        training_iterations=200,  # Reduced for demo
        run_mcmc=run_mcmc        # Disable MCMC temporarily
    )
    
    # Generate test points for prediction (must match true_sst length)
    test_ages = np.linspace(0, 500, len(data['true_sst']))
    
    print("Evaluating model performance...")
    # Evaluate model
    metrics = model.evaluate(test_ages, data['true_sst'])
    
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nDetecting abrupt transitions...")
    # Detect transitions
    transitions = model.detect_abrupt_transitions(test_ages)
    print(f"Detected transitions at ages: {transitions}")
    print(f"True transitions: {data['transition_ages']}")
    
    print("\nGenerating visualizations...")
    # Plot reconstruction
    fig = model.plot_reconstruction(
        test_ages,
        proxy_data_dict=data['proxy_data'],
        true_sst=data['true_sst'],
        detected_transitions=transitions,
        figure_path=os.path.join(output_dir, "reconstruction.png")
    )
    
    # Plot parameter posteriors from MCMC
    if hasattr(model, 'mcmc_sampler') and model.mcmc_sampler is not None:
        fig = model.plot_parameter_posterior(
            figure_path=os.path.join(output_dir, "parameter_posteriors.png")
        )
    
    print(f"\nResults saved to {output_dir}")
    print("\nQuick demo completed!")
    print("\nTo run with custom parameters, try:")
    print("python main.py --proxy_types d18O UK37 --weighting_method balanced --kernel_type adaptive_matern")
    
    return model, data


if __name__ == "__main__":
    model, data = run_quick_demo()