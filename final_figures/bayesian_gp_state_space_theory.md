# Theoretical Foundations of Bayesian Gaussian Process State-Space Models for Paleoclimate Reconstruction

This document presents the mathematical framework underlying the Bayesian Gaussian Process State-Space Model developed for high-fidelity paleoclimate reconstruction, especially for detecting abrupt transitions in climate signals with sparse and irregular proxy data.

## 1. Bayesian State-Space Formulation

### 1.1 State-Space Model Structure

We formulate the paleoclimate reconstruction problem as a state-space model where:

- $\mathbf{x}_t$ represents the hidden state (true SST) at time $t$
- $\mathbf{y}_t$ represents the observed proxy measurements at time $t$

The state-space model consists of two key equations:

1. **State Transition Equation**:
   $\mathbf{x}_t = f(\mathbf{x}_{t-1}) + \mathbf{w}_t$

2. **Measurement Equation**:
   $\mathbf{y}_t = h(\mathbf{x}_t) + \mathbf{v}_t$

Where:
- $f(\cdot)$ is the state transition function
- $h(\cdot)$ is the measurement function (calibration equations)
- $\mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$ is the process noise
- $\mathbf{v}_t \sim \mathcal{N}(0, \mathbf{R})$ is the measurement noise

### 1.2 Gaussian Process Prior for State Dynamics

Instead of specifying an explicit form for $f(\cdot)$, we use a Gaussian Process prior:

$\mathbf{x} \sim \mathcal{GP}(m(\mathbf{t}), k(\mathbf{t}, \mathbf{t}'))$

Where:
- $m(\mathbf{t})$ is the mean function (typically set to a constant or linear function)
- $k(\mathbf{t}, \mathbf{t}')$ is the kernel function that encodes our prior beliefs about the temporal dynamics

For paleoclimate reconstruction, we use a combined kernel to capture both long-term trends and orbital cycles:

$k(\mathbf{t}, \mathbf{t}') = k_{\text{RBF}}(\mathbf{t}, \mathbf{t}') + k_{\text{Periodic}}(\mathbf{t}, \mathbf{t}')$

### 1.3 Proxy-Specific Calibration Equations

For each proxy type, we have a specific calibration equation that relates the proxy measurement to SST:

- δ¹⁸O: $y_{\text{d18O}} = \beta_{\text{d18O}} + \alpha_{\text{d18O}} \cdot x_{\text{SST}} + \varepsilon_{\text{d18O}}$
- UK'37: $y_{\text{UK37}} = \beta_{\text{UK37}} + \alpha_{\text{UK37}} \cdot x_{\text{SST}} + \varepsilon_{\text{UK37}}$
- Mg/Ca: $y_{\text{MgCa}} = \beta_{\text{MgCa}} + \alpha_{\text{MgCa}} \cdot x_{\text{SST}} + \varepsilon_{\text{MgCa}}$

Where:
- $\alpha$ and $\beta$ are the slope and intercept parameters
- $\varepsilon \sim \mathcal{N}(0, \sigma^2_{\text{proxy}})$ represents the proxy-specific measurement noise

## 2. Posterior Inference

### 2.1 Joint Posterior Distribution

The posterior distribution of the latent SST given all observations is:

$p(\mathbf{x}_{1:T} | \mathbf{y}_{1:T}) \propto p(\mathbf{x}_{1:T}) \prod_{t=1}^{T} p(\mathbf{y}_t | \mathbf{x}_t)$

Where:
- $p(\mathbf{x}_{1:T})$ is the GP prior over the entire time series
- $p(\mathbf{y}_t | \mathbf{x}_t)$ is the likelihood from the measurement equation

### 2.2 Marginal Likelihood for Parameter Optimization

For parameter optimization, we maximize the log marginal likelihood:

$\log p(\mathbf{y}_{1:T} | \boldsymbol{\theta}) = \log \int p(\mathbf{y}_{1:T} | \mathbf{x}_{1:T}) p(\mathbf{x}_{1:T} | \boldsymbol{\theta}) d\mathbf{x}_{1:T}$

Where $\boldsymbol{\theta}$ represents all model parameters including:
- GP kernel hyperparameters (lengthscales, outputscales, periods)
- Noise parameters for each proxy

For the linear-Gaussian case with a single proxy, this has a closed form. For multiple proxies with different sampling schedules, we use approximation methods.

### 2.3 Optimal Proxy Weighting

Each proxy contributes information with different uncertainty levels. The optimal weight for each proxy is inversely proportional to its error variance in temperature units:

$w_i = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$

Where:
- $\sigma_i^2 = \sigma_{\text{proxy},i}^2 \cdot |\alpha_i|^{-2}$ is the proxy error converted to temperature units
- $\alpha_i$ is the slope of the calibration equation for proxy $i$

## 3. Handling Sparse and Irregular Data

### 3.1 GP Formulation for Irregular Sampling

One of the key advantages of the GP framework is its natural handling of irregularly sampled data. For times $\mathbf{t} = [t_1, t_2, \ldots, t_n]$, the joint distribution is simply:

$\mathbf{x} \sim \mathcal{N}(m(\mathbf{t}), K(\mathbf{t}, \mathbf{t}))$

Where $K(\mathbf{t}, \mathbf{t})$ is the kernel matrix with entries $K_{ij} = k(t_i, t_j)$.

### 3.2 Inference at Arbitrary Time Points

For prediction at new time points $\mathbf{t_*}$, we compute:

$p(\mathbf{x_*}|\mathbf{y}, \mathbf{t}, \mathbf{t_*}) = \mathcal{N}(\boldsymbol{\mu_*}, \boldsymbol{\Sigma_*})$

Where:
- $\boldsymbol{\mu_*} = m(\mathbf{t_*}) + K(\mathbf{t_*}, \mathbf{t})K(\mathbf{t}, \mathbf{t})^{-1}(\mathbf{y} - m(\mathbf{t}))$
- $\boldsymbol{\Sigma_*} = K(\mathbf{t_*}, \mathbf{t_*}) - K(\mathbf{t_*}, \mathbf{t})K(\mathbf{t}, \mathbf{t})^{-1}K(\mathbf{t}, \mathbf{t_*})$

This allows reconstruction at any time resolution regardless of the original sampling.

### 3.3 Multi-Proxy Integration with Different Sampling Schedules

For multiple proxies with different sampling schedules:

1. Convert each proxy to SST estimates using calibration equations
2. Weight the proxy-derived SST estimates based on their uncertainties
3. Fit the GP model to all weighted estimates simultaneously

## 4. Detecting Abrupt Transitions

### 4.1 Rate of Change Analysis

We detect abrupt transitions by analyzing the rate of change of the reconstructed SST. The posterior distribution of the derivative is also a GP:

$\frac{d\mathbf{x}}{dt} \sim \mathcal{GP}\left(\frac{dm(\mathbf{t})}{dt}, \frac{\partial^2 k(\mathbf{t}, \mathbf{t}')}{\partial t \partial t'}\right)$

### 4.2 Bayesian Change Point Detection

For each time point, we compute the normalized rate of change:

$r_t = \frac{|\dot{x}_t|}{\sigma_{\dot{x}_t}}$

Where:
- $\dot{x}_t$ is the estimated rate of change
- $\sigma_{\dot{x}_t}$ is the standard deviation of the rate of change

Time points with $r_t > \tau$ (where $\tau$ is a threshold, typically set to the 95th percentile) are identified as potential change points.

### 4.3 Uncertainty in Change Point Detection

The uncertainty in detected change points is quantified by sampling multiple realizations from the posterior distribution and analyzing the distribution of detected change points.

## 5. Full Bayesian Treatment

### 5.1 Prior Distributions on Parameters

We place physically informed prior distributions on all model parameters:

- Lengthscales: $\ell \sim \text{LogNormal}(\mu_\ell, \sigma_\ell^2)$
- Period length: $p \sim \text{Normal}(\mu_p, \sigma_p^2)$ (centered on orbital cycles)
- Outputscales: $\sigma_f^2 \sim \text{LogNormal}(\mu_f, \sigma_f^2)$
- Noise parameters: $\sigma_n^2 \sim \text{LogNormal}(\mu_n, \sigma_n^2)$

### 5.2 MCMC for Full Posterior

For full Bayesian inference, we use Markov Chain Monte Carlo (MCMC) to sample from the joint posterior:

$p(\mathbf{x}_{1:T}, \boldsymbol{\theta} | \mathbf{y}_{1:T})$

This allows us to propagate parameter uncertainty into our predictions and provides more robust uncertainty quantification.

## 6. Advantages Over Traditional Methods

### 6.1 Comparison with Classical GP Regression

Classical GP regression typically:

1. First converts proxies to SST using calibration equations
2. Then applies GP regression to the converted data

Our Bayesian GP State-Space approach:

1. Treats SST as a latent state with GP dynamics
2. Integrates calibration equations directly into the measurement model
3. Provides a fully Bayesian treatment with uncertainty propagation

### 6.2 Advantages for Paleoclimate Reconstruction

Our approach offers several advantages for paleoclimate reconstruction:

1. **Improved handling of sparse, irregular data**: By leveraging the GP framework's natural ability to handle non-uniform sampling
2. **Optimal information integration**: By properly weighting multiple proxy types based on their uncertainties
3. **Detection of abrupt transitions**: Through Bayesian change point analysis
4. **Full uncertainty quantification**: By propagating uncertainty from all sources - proxy measurements, calibration parameters, and temporal modeling

### 6.3 Mathematical Rigor in Uncertainty Propagation

Our model rigorously propagates uncertainty through the entire reconstruction pipeline, accounting for:

1. Proxy measurement noise
2. Calibration parameter uncertainty 
3. Temporal modeling uncertainty
4. Parameter estimation uncertainty

This comprehensive uncertainty quantification is essential for reliable interpretation of paleoclimate records, especially when identifying significant climate events and transitions.

## 7. Implementation Details

The model is implemented using a combination of PyTorch and GPyTorch, leveraging automatic differentiation for efficient parameter optimization. For full Bayesian inference, we implement a simplified MCMC scheme for posterior sampling.

The complete implementation includes:
- Custom likelihood functions for each proxy type
- Specialized covariance functions for capturing orbital cycles
- Efficient inference methods for sparse and irregular data
- Advanced visualization tools for uncertainty representation
- Rigorous evaluation metrics for model comparison

---

*This document provides the theoretical foundation for the Bayesian GP State-Space model implemented in `bayesian_gp_state_space.py` for paleoclimate reconstruction.*