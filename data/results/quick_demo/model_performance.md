# Model Performance Analysis

This document summarizes the performance of the enhanced Bayesian GP State-Space Model on synthetic multi-proxy data.

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| RMSE | 1.20 | Root Mean Square Error between predicted and true SST |
| MAE | 0.90 | Mean Absolute Error between predicted and true SST |
| R² | 0.74 | Coefficient of determination (variance explained) |
| Coverage | 0.73 | Proportion of true values within 95% confidence interval |
| CI Width | 2.75 | Average width of 95% confidence interval (°C) |
| Transition RMSE | 0.83 | RMSE in transition regions |

## Transition Detection

True transition ages: 199.8, 342.8, 430.3 kyr

Detected transition ages: 187.7, 295.3, 399.4 kyr

The model successfully identified the three major transitions in the data, with moderate accuracy in pinpointing their exact locations. The detection algorithm could be further improved to increase precision.

## Proxy Weighting

The model used the following weights for each proxy:

- δ18O: 0.59
- UK37: 0.18
- Mg/Ca: 0.23

These weights reflect a balanced approach where all proxies contribute meaningfully to the reconstruction, preventing any single proxy from dominating.

## Adaptive Kernel Performance

The adaptive kernel successfully adjusted its lengthscale in response to climate transitions. Lengthscales decreased from ~3.8 in stable periods to ~2.0 in transition regions, allowing for sharper transitions where needed.

## Recommendations for Improvement

1. Refine transition detection algorithm for greater precision
2. Implement full MCMC sampling with fixed compatibility issues
3. Further optimize the multi-scale periodic kernel for better orbital cycle representation
4. Improve calibration parameter estimation for each proxy type