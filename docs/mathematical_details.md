# Mathematical Details of the Enhanced Bayesian GP State-Space Model

This document provides in-depth mathematical explanations of the key components of our enhanced Bayesian Gaussian Process State-Space Model for paleoclimate reconstruction.

## 1. Non-Stationary Covariance Functions

### 1.1 Adaptive Lengthscale Kernel

The adaptive lengthscale kernel adjusts its flexibility based on the rate of climate change, allowing it to model both stable periods and abrupt transitions effectively.

The kernel is defined as:

$$k_{\text{adaptive}}(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2 \ell(x) \ell(x')}\right)$$

where $\ell(x)$ is the location-dependent lengthscale function given by:

$$\ell(x) = \frac{\ell_{\text{base}}}{1 + \alpha \cdot r(x)}$$

Here:
- $\ell_{\text{base}}$ is the base lengthscale (typically 5-10 kyr)
- $\alpha$ is the adaptation strength controlling how strongly the lengthscale responds to rate changes
- $r(x)$ is the normalized rate of change (0-1) at location $x$

For computation, we use a centralized lengthscale value for each pair of points:

$$\ell_{x,x'} = \frac{\ell(x) + \ell(x')}{2}$$

### 1.2 Temperature-Dependent Kernel

The temperature-dependent kernel allows kernel parameters to vary based on the absolute temperature, accounting for different climate regimes (glacial vs. interglacial).

$$k_{\text{temp}}(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2 \ell(T(x), T(x'))}\right)$$

where $\ell(T(x), T(x'))$ is a function of the temperatures at $x$ and $x'$:

$$\ell(T) = \ell_{\text{cold}} + \frac{\ell_{\text{warm}} - \ell_{\text{cold}}}{1 + \exp(-(T - T_{\text{threshold}})/\omega)}$$

where:
- $\ell_{\text{cold}}$ is the lengthscale for cold (glacial) periods
- $\ell_{\text{warm}}$ is the lengthscale for warm (interglacial) periods
- $T_{\text{threshold}}$ is the temperature threshold separating regimes
- $\omega$ controls the width of the transition region

### 1.3 Spectral Mixture Kernel with Trend

For capturing complex cyclic patterns and long-term trends:

$$k_{\text{SM}}(x, x') = \sum_{i=1}^Q w_i \exp\left(-\frac{2\pi^2(x-x')^2}{\ell_i^2}\right) \cos(2\pi\mu_i(x-x')) + \sum_{j=0}^{d} \beta_j (x \cdot x')^j$$

where:
- $Q$ is the number of mixture components
- $w_i$ are the mixture weights
- $\ell_i$ are the lengthscales
- $\mu_i$ are the frequencies
- $\beta_j$ are the polynomial trend coefficients
- $d$ is the degree of the polynomial trend

## 2. Multi-Scale Periodic Components

The multi-scale periodic kernel specifically models orbital forcing at Milankovitch cycles:

$$k_{\text{periodic}}(x, x') = \sum_{i=1}^3 w_i \exp\left(-\frac{2\sin^2(\pi|x-x'|/p_i)}{l_i^2}\right)$$

where:
- $p_1 = 100$ kyr (eccentricity)
- $p_2 = 41$ kyr (obliquity) 
- $p_3 = 23$ kyr (precession)
- $w_i$ are the relative weights of each component
- $l_i$ control the regularity of each cycle

## 3. Temperature-Dependent Proxy Calibrations

### 3.1 δ¹⁸O Calibration

The temperature-dependent δ¹⁸O calibration is given by:

$$\delta^{18}\text{O} = b + a \cdot T - c \cdot \sigma(T - T_{\text{threshold}}) \cdot T$$

where $\sigma()$ is the sigmoid function providing a smooth transition between regimes:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 3.2 UK'37 Calibration

The UK'37 calibration with temperature-dependent behavior:

$$\text{UK}'_{37} = b + (a + c \cdot \sigma(T - T_{\text{threshold}})) \cdot T$$

with additional saturation effect at high temperatures:

$$\text{UK}'_{37} = \text{UK}'_{37} \cdot (1 - e^{-(T/40)^2})$$

### 3.3 Mg/Ca Calibration

The Mg/Ca calibration with exponential form and temperature-dependent exponent:

$$\text{Mg/Ca} = a \cdot \exp((b + c \cdot \sigma(T - T_{\text{threshold}})) \cdot T)$$

## 4. Advanced Transition Detection Algorithms

### 4.1 Multi-Scale Derivative Detector

This detector computes derivatives at multiple scales and combines them with weighting:

$$r_{\text{combined}}(t) = \sum_{i=1}^S w_i \cdot \left| \frac{d}{dt} f_{\sigma_i}(t) \right|$$

where:
- $f_{\sigma_i}(t)$ is the time series smoothed with scale $\sigma_i$
- $w_i$ is the weight for scale $i$ (proportional to $1/\sigma_i$)
- $S$ is the number of scales used

### 4.2 Wavelet-Based Detector

The wavelet-based detector uses the wavelet transform to identify transitions:

$$W_\psi f(s, t) = \frac{1}{\sqrt{s}} \int_{-\infty}^{\infty} f(\tau) \psi^*\left(\frac{\tau-t}{s}\right) d\tau$$

where:
- $\psi$ is the wavelet function (e.g., Haar wavelet for sharp transitions)
- $s$ is the scale parameter
- $t$ is the time parameter

The transitions are detected where wavelet coefficients exceed a threshold:

$$t_{\text{transition}} = \{t : |W_\psi f(s, t)| > \lambda \cdot \text{median}(|W_\psi f(s, t)|/0.6745)\}$$

### 4.3 Bayesian Change Point Detector

The Bayesian change point detector computes posterior probabilities of change points:

$$P(\text{change at } t) = 1 - p_{\text{value}}(z)$$

where:

$$z = \frac{|\mu_{\text{before}} - \mu_{\text{after}}|}{\sigma_{\text{pooled}} \cdot \sqrt{2/n_{\text{window}}}}$$

## 5. Enhanced MCMC Implementation

### 5.1 Hamiltonian Monte Carlo (HMC)

The HMC algorithm uses Hamiltonian dynamics to efficiently explore the parameter space:

$$H(\theta, p) = -\log p(\theta | D) + \frac{1}{2}p^T M^{-1} p$$

The leapfrog integrator update rules are:

$$p_{t+\epsilon/2} = p_t - \frac{\epsilon}{2} \nabla_\theta U(\theta_t)$$
$$\theta_{t+\epsilon} = \theta_t + \epsilon M^{-1} p_{t+\epsilon/2}$$
$$p_{t+\epsilon} = p_{t+\epsilon/2} - \frac{\epsilon}{2} \nabla_\theta U(\theta_{t+\epsilon})$$

where:
- $U(\theta) = -\log p(\theta | D)$ is the potential energy (negative log posterior)
- $p$ are the momentum variables
- $M$ is the mass matrix (typically identity)
- $\epsilon$ is the step size

### 5.2 Adaptive Step Size

The step size is adapted during burn-in to achieve the target acceptance rate:

$$\epsilon_{i+1} = \begin{cases}
\epsilon_i \cdot 1.02 & \text{if } a_i > a_{\text{target}}\\
\epsilon_i \cdot 0.98 & \text{otherwise}
\end{cases}$$

where:
- $a_i$ is the current acceptance rate
- $a_{\text{target}}$ is the target acceptance rate (typically 0.6-0.8)

## 6. Heteroscedastic Noise Model

The heteroscedastic noise model allows for observation-specific noise levels:

$$y_i \sim \mathcal{N}(f(x_i), \sigma_i^2)$$

with noise levels modeled as:

$$\sigma_i^2 = \sigma_{\text{base}}^2(p_i) \cdot (1 + \gamma \cdot \text{age}_i) \cdot (1 + \beta \cdot r(x_i))$$

where:
- $\sigma_{\text{base}}^2(p_i)$ is the base noise level for proxy type $p_i$
- $\gamma$ controls age-dependent uncertainty increase
- $\beta$ controls additional uncertainty near transitions
- $r(x_i)$ is the normalized rate of change at $x_i$

## 7. Balanced Multi-Proxy Weighting

The balanced weighting scheme ensures no single proxy dominates, with weights given by:

$$w_i = \frac{\sqrt{\frac{\sigma_i^{-2}}{\max_j \sigma_j^{-2}}}}{\sum_k \sqrt{\frac{\sigma_k^{-2}}{\max_j \sigma_j^{-2}}}}$$

where:
- $\sigma_i$ is the temperature-equivalent error for proxy $i$
- $\sigma_i^{-2}$ is the precision (inverse variance)

This square-root transformation prevents high-precision proxies from completely dominating the weighting.

## 8. Joint Posterior Distribution

The full Bayesian model has the joint posterior:

$$p(\theta, f | D) \propto p(D | f, \theta) \cdot p(f | \theta) \cdot p(\theta)$$

where:
- $\theta$ are the hyperparameters (lengthscales, noise parameters, etc.)
- $f$ is the GP function values
- $D$ is the observed data

The MCMC procedure samples from the marginal posterior:

$$p(\theta | D) = \int p(\theta, f | D) df$$

which for a GP with Gaussian likelihood can be computed analytically as:

$$p(\theta | D) \propto p(D | \theta) \cdot p(\theta)$$

with:

$$p(D | \theta) = \mathcal{N}(y | 0, K_\theta + \Sigma)$$

where:
- $K_\theta$ is the kernel matrix (covariance function)
- $\Sigma$ is the diagonal matrix of observation noise variances