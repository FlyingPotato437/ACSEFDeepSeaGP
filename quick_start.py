#!/usr/bin/env python3
"""
Matplotlib-Style quick start demo for paleoclimate reconstruction
With detailed structure and less smoothing
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import the matplotlib-inspired model
from models.multi_output_bayesian_gp_state_space import MultiOutputBayesianGP

# Suppress warnings
warnings.filterwarnings("ignore")


def run_quick_demo(csv_file="ODP722.csv", output_dir="outputs"):
    """
    Run the demonstration using matplotlib-style approach
    
    Parameters:
    -----------
    csv_file : str, optional
        Path to CSV file
    output_dir : str, optional
        Output directory
    """
    try:
        print("\nStarting enhanced demo with matplotlib-style approach...")
        print("=" * 60)
        print("MULTI-OUTPUT BAYESIAN GP MODEL")
        print("With Composite Kernels and Detailed Structure")
        print("=" * 60)
        
        print("This demonstration shows:")
        print("  1. Composite kernels with periodic components for Milankovitch cycles")
        print("  2. Direct posterior sampling for detailed structure")
        print("  3. Multi-output model with more realistic uncertainty")
        print("  4. Transition detection with high sensitivity")
        
        print(f"\nUsing device: cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        model = MultiOutputBayesianGP(random_state=42)
        
        # Load data with maximum age of 800 kyr
        print(f"\nLoading data from {csv_file}...")
        if not model.load_data(csv_file, max_age=800):
            print("Error loading data. Exiting.")
            return False
        
        # Fit model
        print("\nFitting model with hyperparameter optimization and direct sampling...")
        start_time = time.time()
        
        # Using matplotlib-style direct sampling
        model.fit(optimize=True, n_samples=100)
        
        # Print execution time
        exec_time = time.time() - start_time
        print(f"\nModel fitting completed in {exec_time:.2f} seconds")
        
        # Generate predictions
        print("\nGenerating high-resolution predictions on test grid...")
        test_ages = np.linspace(0, 800, 1000)
        
        # Plot reconstruction
        print("\nCreating visualization plots...")
        fig = model.plot_reconstruction(test_ages, output_dir)
        
        print("\nDemo completed successfully!")
        print(f"Results saved in {output_dir}/")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR during model fitting: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nTry running the script again or check that the data file is correct.")
        print("\nDemo failed to complete. Please check the error messages above.")
        return False


if __name__ == "__main__":
    # Get CSV file from command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "ODP722.csv"
        
    # Get output directory from command line argument
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "outputs"
        
    # Run demo
    run_quick_demo(csv_file, output_dir)