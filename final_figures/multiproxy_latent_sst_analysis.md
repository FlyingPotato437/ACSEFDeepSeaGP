# Multi-Proxy Latent SST Variable Extraction

## Overview

This document summarizes the results of our multi-proxy latent variable extraction experiment for Sea Surface Temperature (SST) reconstruction. We implemented a mathematically rigorous approach to extract the underlying latent SST variable from multiple proxy measurements (δ¹⁸O and UK'37) by properly modeling the calibration equations and error propagation.

## Mathematical Foundation

Our approach explicitly models the latent variable extraction problem using calibration equations:

1. **Calibration Equations**:
   - δ¹⁸O = -4.38 × SST + 16.9 + ε₁, where ε₁ ~ N(0, 0.8²)
   - UK'37 = 0.033 × SST + 0.044 + ε₂, where ε₂ ~ N(0, 0.25²)

2. **Error Propagation**:
   - SST from δ¹⁸O: Var(SST) = Var(δ¹⁸O)/α₁² = 0.8²/4.38² = 0.033°C²
   - SST from UK'37: Var(SST) = Var(UK'37)/α₂² = 0.25²/0.033² = 57.4°C²

3. **Optimal Weighting**:
   - We calculate weights inversely proportional to error variance
   - w₁ = (1/Var₁)/Σ(1/Varᵢ)
   - w₂ = (1/Var₂)/Σ(1/Varᵢ)

4. **Combined Estimate**:
   - SST = w₁×SST₁ + w₂×SST₂, where SSTᵢ is derived from proxy i

## Methods

We implemented two approaches to the latent variable extraction problem:

1. **Direct Weighted Combination**:
   - Convert each proxy to temperature using calibration equations
   - Apply statistically optimal weights based on error propagation
   - Combine to form minimum variance unbiased estimator

2. **GP-Smoothed Multi-Proxy Approach**:
   - Build on the weighted combination
   - Apply GP model with combined kernel to add temporal structure
   - Use optimized hyperparameters for kernel components

## Results

| Method | RMSE (°C) | MAE (°C) | R² | Bias (°C) | Std Error (°C) |
|--------|-----------|----------|-----|-----------|----------------|
| Multi-Proxy GP | 0.8615 | 0.6386 | 0.9554 | 0.0029 | 0.8615 |
| Weighted Combination | 0.2149 | 0.1710 | 0.9972 | 0.0062 | 0.2148 |
| δ¹⁸O Simple | 0.2155 | 0.1715 | 0.9972 | 0.0062 | 0.2154 |
| UK'37 Simple | 2.4846 | 1.9473 | 0.6289 | -0.0127 | 2.4846 |

### Signal-to-Noise Ratio
- δ¹⁸O: 22.33
- UK'37: 0.54

### Optimal Weights
- δ¹⁸O: 99.94%
- UK'37: 0.06%

## Key Findings

1. **Optimal Weighting Dominance**: The direct weighted combination approach gives almost all weight (99.94%) to the δ¹⁸O proxy due to its much higher SNR, effectively ignoring the UK'37 proxy. This explains why the weighted combination performs nearly identically to the δ¹⁸O simple calibration.

2. **GP Smoothing Trade-off**: The GP-smoothed approach actually performs worse (0.86°C RMSE) than the direct weighted combination (0.21°C RMSE). This suggests that the GP's temporal smoothing introduces bias in this case, trading off point accuracy for temporal coherence.

3. **UK'37 Limited Contribution**: With its very low sensitivity (0.033 units/°C) and moderate noise (0.25), UK'37 has an extremely poor SNR (0.54), making it nearly useless for SST reconstruction when a higher SNR proxy like δ¹⁸O is available.

4. **Mathematical Rigor Matters**: By properly modeling the error propagation through the calibration equations, we achieved near-optimal results with the direct weighted combination (99.7% of maximum possible R²).

## Conclusions

This experiment demonstrates the importance of proper mathematical modeling in latent variable extraction from multiple proxies. The key insights are:

1. **Error Propagation**: Different proxies have dramatically different error characteristics after propagating through their calibration equations. This must be accounted for in any multi-proxy reconstruction.

2. **Signal-to-Noise Ratio**: SNR is the critical factor determining a proxy's usefulness. In our case, δ¹⁸O has ~41x higher SNR than UK'37, making it the dominant information source.

3. **Trade-offs in GP Modeling**: While GP models add valuable temporal structure, they may introduce a bias-variance trade-off that can reduce point accuracy. The choice between direct weighted combination and GP-smoothed approaches depends on the specific reconstruction goals.

4. **Optimal Resource Allocation**: When designing multi-proxy studies, resources should be allocated toward proxies with higher sensitivities rather than equally among all proxy types.

This analysis provides a mathematical foundation for extracting the latent SST variable from paleoclimate proxies in a way that properly accounts for their different sensitivities and error characteristics.