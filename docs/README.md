# Bayesian GP State-Space Model for Paleoclimate Reconstruction

This project implements an enhanced Bayesian Gaussian Process State-Space Model specifically designed for paleoclimate reconstruction. The model excels at capturing both gradual climate changes and abrupt transitions.

## Key Features

- **Adaptive Kernel Lengthscales**: Automatically adjusts model flexibility based on climate rate of change, enabling accurate reconstruction of both stable periods and abrupt transitions
- **Multi-proxy Integration**: Balanced weighting mechanism for combining δ18O, UK37, Mg/Ca, and other proxies
- **Multi-scale Orbital Components**: Models Milankovitch cycles (100kyr, 41kyr, 23kyr) for improved long-term climate reconstructions
- **Heteroscedastic Noise Model**: Handles observation-specific uncertainty based on proxy type, sample age, and quality
- **Robust MCMC Sampling**: Uncertainty quantification through Hamiltonian Monte Carlo sampling

## Model Structure

The model has a modular design with specialized components:

- `kernels/adaptive_kernel.py`: Rate estimation and adaptive lengthscale kernels
- `utils/proxy_calibration.py`: Balanced proxy weighting and heteroscedastic noise
- `mcmc/sampler.py`: Robust MCMC implementation
- `models/bayesian_gp_state_space.py`: Main model implementation

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- GPyTorch
- NumPy, SciPy, Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/paleoclimate-gp-model.git
cd paleoclimate-gp-model

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo

Run the quick demonstration script:

```bash
python quick_start.py
```

This will:
1. Generate synthetic multi-proxy data with abrupt transitions
2. Initialize the model with advanced components
3. Fit the model to the data
4. Detect abrupt climate transitions
5. Generate visualization of the reconstruction

## Using with Your Own Data

The model can be used with custom data through the `BayesianGPStateSpaceModel` class. Here's a basic example:

```python
from models.bayesian_gp_state_space import BayesianGPStateSpaceModel

# Configure your model
model = BayesianGPStateSpaceModel(
    proxy_types=['d18O', 'UK37', 'Mg_Ca'],  # Specify proxy types
    weighting_method='balanced',            # Options: 'balanced', 'error', 'snr', 'equal'
    kernel_config={
        'base_kernel_type': 'matern',       # Base kernel type
        'base_lengthscale': 5.0,            # Base lengthscale in kyr
        'adaptation_strength': 1.5,         # How strongly to adapt to transitions
        'include_periodic': True,           # Include orbital components
        'periods': [100.0, 41.0, 23.0]      # Orbital periods in kyr
    }
)

# Fit to your data
model.fit(
    proxy_data={
        'd18O': {'age': age_d18o, 'value': value_d18o},
        'UK37': {'age': age_uk37, 'value': value_uk37}
    },
    training_iterations=500
)

# Make predictions
test_ages = np.linspace(0, 500, 1000)
mean, lower_ci, upper_ci = model.predict(test_ages)

# Detect transitions
transitions = model.detect_abrupt_transitions(test_ages)

# Plot results
model.plot_reconstruction(test_ages, proxy_data_dict=your_proxy_data)
```

## Advanced Usage

### Custom Proxy Calibration

You can provide custom calibration parameters for your proxies:

```python
custom_calibration = {
    'd18O': {
        'slope': -0.25,       # °C per ‰
        'intercept': 3.5,     # ‰
        'error_std': 0.12,    # ‰
        'inverse_slope': -4.0 # ‰ per °C (1/slope)
    },
    # Add other proxies as needed
}

model = BayesianGPStateSpaceModel(
    proxy_types=['d18O'],
    calibration_params=custom_calibration
)
```

### MCMC for Uncertainty Quantification

To run MCMC sampling for uncertainty quantification:

```python
# Configure MCMC
mcmc_config = {
    'n_samples': 500,
    'burn_in': 100,
    'step_size': 0.01,
    'target_acceptance': 0.75
}

model = BayesianGPStateSpaceModel(
    proxy_types=['d18O', 'UK37'],
    mcmc_config=mcmc_config
)

# Enable MCMC during fitting
model.fit(proxy_data, run_mcmc=True)

# Plot posterior parameter distributions
model.plot_parameter_posterior(figure_path='parameter_posterior.png')
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This model builds upon Gaussian Process methodology from GPyTorch and paleoclimate reconstruction methods, with a focus on improving the handling of abrupt climate transitions.