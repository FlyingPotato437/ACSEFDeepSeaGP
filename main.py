"""
Main script for running the Enhanced Bayesian Gaussian Process State-Space Model
for paleoclimate reconstruction.

This script provides an entry point for running the model with different configurations
and data sources, generating visualizations, and comparing performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional, Union

# Import model components
from models.bayesian_gp_state_space import (
    BayesianGPStateSpaceModel,
    generate_synthetic_multiproxy_data
)
from utils.proxy_calibration import DEFAULT_CALIBRATION_PARAMS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Enhanced Bayesian GP State-Space Model for paleoclimate reconstruction."
    )
    
    # Data options
    parser.add_argument(
        "--data_type", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="Type of data to use (synthetic or real)"
    )
    parser.add_argument(
        "--proxy_types", type=str, nargs="+", default=["d18O", "UK37", "Mg_Ca"],
        help="List of proxy types to use"
    )
    parser.add_argument(
        "--data_file", type=str, default=None,
        help="Path to real data file (if data_type is 'real')"
    )
    
    # Model configuration
    parser.add_argument(
        "--weighting_method", type=str, default="balanced",
        choices=["balanced", "error", "snr", "equal"],
        help="Method for weighting different proxies"
    )
    parser.add_argument(
        "--kernel_type", type=str, default="adaptive_matern",
        choices=["adaptive_matern", "adaptive_rbf"],
        help="Type of kernel to use"
    )
    parser.add_argument(
        "--include_periodic", action="store_true", default=True,
        help="Include periodic components for Milankovitch cycles"
    )
    parser.add_argument(
        "--base_lengthscale", type=float, default=5.0,
        help="Base lengthscale for the kernel"
    )
    parser.add_argument(
        "--adaptation_strength", type=float, default=1.5,
        help="Strength of lengthscale adaptation"
    )
    
    # MCMC options
    parser.add_argument(
        "--run_mcmc", action="store_true", default=True,
        help="Run MCMC sampling for uncertainty quantification"
    )
    parser.add_argument(
        "--mcmc_samples", type=int, default=1000,
        help="Number of MCMC samples"
    )
    
    # Training options
    parser.add_argument(
        "--training_iterations", type=int, default=500,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir", type=str, default="data/results/enhanced_bayesian_gp",
        help="Directory to save results"
    )
    
    return parser.parse_args()


def load_real_data(filename: str, proxy_types: List[str]) -> Dict:
    """
    Load real proxy data from file.
    
    Args:
        filename: Path to data file
        proxy_types: List of proxy types to extract
        
    Returns:
        Dictionary with proxy data
    """
    import pandas as pd
    
    # Load data file
    try:
        data = pd.read_csv(filename)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)
    
    # Convert to desired format
    proxy_data = {}
    
    for proxy_type in proxy_types:
        # Check if columns exist
        age_col = f"{proxy_type}_age"
        value_col = f"{proxy_type}_value"
        
        if age_col in data.columns and value_col in data.columns:
            # Extract data
            proxy_ages = data[age_col].values
            proxy_values = data[value_col].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(proxy_ages) & ~np.isnan(proxy_values)
            
            proxy_data[proxy_type] = {
                'age': proxy_ages[valid_mask],
                'value': proxy_values[valid_mask]
            }
        else:
            print(f"Warning: Columns for {proxy_type} not found in data file")
    
    return {'proxy_data': proxy_data}


def main():
    """Main entry point for running the model."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate data
    if args.data_type == "synthetic":
        print("Generating synthetic multi-proxy data...")
        data = generate_synthetic_multiproxy_data(
            n_points=80,
            age_min=0,
            age_max=500,
            proxy_types=args.proxy_types,
            n_transitions=3,
            transition_magnitude=3.0,
            include_orbital_cycles=True,
            smoothness=1.0,
            proxy_noise_scale=1.0,
            random_state=args.random_state
        )
    else:
        print(f"Loading real data from {args.data_file}...")
        data = load_real_data(args.data_file, args.proxy_types)
    
    # Configure kernel
    kernel_config = {
        'base_kernel_type': args.kernel_type.split('_')[1],  # 'matern' or 'rbf'
        'min_lengthscale': 2.0,
        'max_lengthscale': 10.0,
        'base_lengthscale': args.base_lengthscale,
        'adaptation_strength': args.adaptation_strength,
        'lengthscale_regularization': 0.1,
        'include_periodic': args.include_periodic,
        'periods': [100.0, 41.0, 23.0],  # Milankovitch cycles
        'outputscales': [2.0, 1.0, 0.5]  # Initial weights for periodic components
    }
    
    # Configure MCMC
    mcmc_config = {
        'n_samples': args.mcmc_samples,
        'burn_in': args.mcmc_samples // 5,  # 20% burn-in
        'thinning': 2,
        'step_size': 0.05,
        'target_acceptance': 0.6,
        'adaptation_steps': args.mcmc_samples // 10
    }
    
    # Initialize model
    print("Initializing model...")
    model = BayesianGPStateSpaceModel(
        proxy_types=args.proxy_types,
        weighting_method=args.weighting_method,
        kernel_config=kernel_config,
        mcmc_config=mcmc_config,
        calibration_params=data.get('calibration_params', DEFAULT_CALIBRATION_PARAMS),
        random_state=args.random_state
    )
    
    # Fit model
    print("Fitting model...")
    model.fit(
        data['proxy_data'],
        training_iterations=args.training_iterations,
        run_mcmc=args.run_mcmc
    )
    
    # Generate test points covering the full age range
    if args.data_type == "synthetic":
        test_ages = np.linspace(0, 500, 500)
        true_sst = data['true_sst']
    else:
        # For real data, determine age range from proxy data
        min_age = float('inf')
        max_age = float('-inf')
        
        for proxy_type, proxy_data in data['proxy_data'].items():
            min_age = min(min_age, np.min(proxy_data['age']))
            max_age = max(max_age, np.max(proxy_data['age']))
        
        test_ages = np.linspace(min_age, max_age, 500)
        true_sst = None
    
    # Evaluate if we have true values
    if true_sst is not None:
        print("Evaluating model...")
        metrics = model.evaluate(test_ages, true_sst)
        
        print("Performance metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save metrics to file
        import json
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    # Detect transitions
    print("Detecting transitions...")
    transitions = model.detect_abrupt_transitions(test_ages)
    print(f"Detected transitions at ages: {transitions}")
    
    if args.data_type == "synthetic":
        print(f"True transitions: {data['transition_ages']}")
    
    # Plot reconstruction
    print("Generating visualizations...")
    fig = model.plot_reconstruction(
        test_ages,
        proxy_data_dict=data['proxy_data'],
        true_sst=true_sst if args.data_type == "synthetic" else None,
        detected_transitions=transitions,
        figure_path=os.path.join(args.output_dir, "reconstruction.png")
    )
    
    # Plot parameter posteriors if MCMC was run
    if args.run_mcmc and hasattr(model, 'mcmc_sampler') and model.mcmc_sampler is not None:
        fig = model.plot_parameter_posterior(
            figure_path=os.path.join(args.output_dir, "parameter_posteriors.png")
        )
    
    print(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()