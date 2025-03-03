# Enhanced Bayesian GP State-Space Model for Paleoclimate Reconstruction

This project implements a sophisticated Bayesian Gaussian Process State-Space Model specifically designed for paleoclimate reconstruction, with enhanced capabilities for handling abrupt climate transitions, multiple proxies, and data irregularities.

## Key Features

- **Adaptive Kernel Lengthscales**: Automatically adjusts model flexibility based on climate rate of change
- **Multi-proxy Integration**: Balanced weighting mechanism for combining δ18O, UK37, Mg/Ca proxies
- **Multi-scale Orbital Components**: Models Milankovitch cycles (100kyr, 41kyr, 23kyr)
- **Heteroscedastic Noise Model**: Handles observation-specific uncertainty
- **Advanced MCMC Sampling**: Robust uncertainty quantification with GPyTorch compatibility
- **Temperature-Dependent Calibration**: Improved proxy-temperature relationships
- **Flexible Data Import**: Support for irregular/sparse data from various formats

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/paleoclimate-gp-model.git
cd paleoclimate-gp-model

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch gpytorch numpy scipy matplotlib pandas pywt tqdm
```

## Quick Start

Run the quick demonstration:

```bash
python quick_start.py
```

This will generate synthetic multi-proxy data, fit the model, detect transitions, and create visualizations.

## Using with Custom Data

Import data from various formats:

```python
from utils.data_loader import load_paleoclimate_data

# Load data from CSV/Excel
proxy_data = load_paleoclimate_data(
    "your_data.csv",
    proxy_columns={
        'd18O': 'delta18O_column',
        'UK37': 'alkenone_column',
        'Mg_Ca': 'MgCa_column'
    },
    age_column='age_kyr'
)

# Or use auto-detection
proxy_data = load_paleoclimate_data("your_data.csv")
```

Run the model:

```python
from models.bayesian_gp_state_space import BayesianGPStateSpaceModel

# Configure and initialize model
model = BayesianGPStateSpaceModel(
    proxy_types=['d18O', 'UK37', 'Mg_Ca'],
    weighting_method='balanced',
    kernel_config={
        'base_kernel_type': 'matern',
        'base_lengthscale': 5.0,
        'adaptation_strength': 1.5
    }
)

# Fit model to data
model.fit(proxy_data, training_iterations=500)

# Predict and detect transitions
test_ages = np.linspace(0, 500, 1000)
mean, lower_ci, upper_ci = model.predict(test_ages)
transitions = model.detect_abrupt_transitions(test_ages)

# Visualize results
model.plot_reconstruction(test_ages, proxy_data_dict=proxy_data, detected_transitions=transitions)
```

## Advanced Usage

### Temperature-Dependent Calibration

```python
from utils.temperature_dependent_calibration import temperature_to_d18o, d18o_to_temperature

# Define calibration parameters
params = {
    'a': -0.22,      # Linear coefficient
    'b': 3.0,        # Intercept
    'c': 0.02,       # Nonlinearity coefficient
    'threshold': 10.0  # Temperature threshold
}

# Convert temperature to proxy values
d18o_values = temperature_to_d18o(temperatures, params)

# Convert proxy values to temperature
reconstructed_temps = d18o_to_temperature(d18o_values, params)
```

### Advanced Transition Detection

```python
from models.improved_transition_detection import ensemble_transition_detector

# Detect transitions using ensemble approach
transitions = ensemble_transition_detector(
    ages, temperatures,
    detectors=['derivative', 'wavelet', 'bayesian'],
    min_separation=20.0,
    min_magnitude=1.5
)

# Visualize detected transitions
from models.improved_transition_detection import plot_detected_transitions
plot_detected_transitions(ages, temperatures, transitions)
```

### Enhanced MCMC Sampling

```python
from mcmc.enhanced_sampler import EnhancedMCMCSampler

# Configure MCMC
mcmc_sampler = EnhancedMCMCSampler(
    model, likelihood,
    n_samples=1000,
    burn_in=200,
    step_size=0.01
)

# Run sampler
mcmc_sampler.run_hmc(progress_bar=True)

# Visualize posterior distributions
mcmc_sampler.plot_trace(figure_path="posterior_trace.png")

# Make predictions with uncertainty
pred_dict = mcmc_sampler.sample_posterior_predictive(test_ages, n_samples=100)
```

### Non-Stationary Covariance Functions

```python
from kernels.non_stationary_kernel import TemperatureDependentKernel

# Configure kernel
kernel = TemperatureDependentKernel(
    base_kernel_type='matern',
    cold_lengthscale=4.0,
    warm_lengthscale=8.0,
    transition_midpoint=10.0
)

# Update with temperature values
kernel.update_temperature_values(temperatures)
```

## Directory Structure

- `kernels/`: Kernel implementations (adaptive, non-stationary)
- `mcmc/`: MCMC sampling implementations
- `models/`: Main model and transition detection implementations
- `utils/`: Data loading, proxy calibration, and utility functions
- `docs/`: Detailed documentation
- `quick_start.py`: Demonstration script
- `main.py`: Command-line interface

## Documentation

For detailed documentation, see the `docs/` directory:

- [Mathematical Details](docs/mathematical_details.md)
- [Data Import Guide](docs/data_import_guide.md)
- [Transition Detection](docs/transition_detection.md)
- [MCMC Sampling](docs/mcmc_sampling.md)
- [API Reference](docs/api_reference.md)

## Citation

If you use this model in your research, please cite:

```
Author, A. (2025). Enhanced Bayesian Gaussian Process State-Space Model 
for Paleoclimate Reconstruction. GitHub repository: 
https://github.com/yourusername/paleoclimate-gp-model
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.