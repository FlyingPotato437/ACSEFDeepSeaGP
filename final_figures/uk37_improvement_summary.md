# UK'37 Proxy Performance Improvement: Effect of Noise Reduction

## Summary

This document summarizes the dramatic improvement in UK'37 proxy performance achieved by reducing noise levels by 80%. The experiment used the previously tuned combined kernel (RBF + Periodic) with the optimized hyperparameters from the δ¹⁸O tuning experiment.

## Experimental Design

* **Approach**: Reduced UK'37 noise standard deviation by 80% (from 0.4 to 0.08)
* **Proxy Sensitivity**: Maintained at 0.033 units/°C (unchanged)
* **GP Model**: Combined kernel (RBF + Periodic) with tuned hyperparameters:
  - RBF lengthscale: 6.9208
  - Periodic lengthscale: 5.3162
  - Periodic period length: 0.6752

## Performance Metrics Comparison (GP vs TRUE Latent SST)

| Metric | Original UK'37 | Improved UK'37 | Improvement |
|--------|---------------|----------------|-------------|
| Signal-to-Noise Ratio | 0.3365 | 1.6824 | 5.0x higher |
| RMSE (°C) | 4.3904 | 1.4548 | 66.9% reduction |
| MAE (°C) | 3.5516 | 1.1551 | 67.5% reduction |
| R² | -0.2212 | 0.8659 | Negative → Positive |
| Systematic Error (°C) | -0.3057 | 0.1595 | Improved |
| Random Error (°C) | 4.3797 | 1.4460 | 67.0% reduction |

## Key Findings

1. **Signal-to-Noise Ratio is Critical**: Reducing UK'37 noise by 80% improved the SNR from 0.34 to 1.68 (5.0x higher), making the latent SST signal much more distinguishable from noise.

2. **GP Model Performance Improved Dramatically**:
   - RMSE decreased from 4.39°C to 1.45°C (66.9% reduction)
   - R² improved from -0.22 (worse than mean prediction) to 0.87 (excellent fit)
   - Random error reduced by 67.0%, making predictions much more precise

3. **Low-Sensitivity Proxies Can Work with Sufficient SNR**: The UK'37 proxy still has a relatively low sensitivity (0.033 units/°C), but with reduced noise, the GP model can effectively extract the underlying temperature signal.

4. **Tuned Combined Kernel Effectiveness**: The tuned combined kernel (RBF + Periodic) effectively captures both long-term trends and cyclic patterns in the data when the signal exceeds the noise.

## Conclusion

This experiment demonstrates that the poor performance of UK'37 in previous tests was primarily due to its low signal-to-noise ratio rather than a fundamental limitation of the GP model or the combined kernel. When noise is reduced to a level where the signal can be detected, the GP model with the tuned combined kernel performs excellently in reconstructing the latent SST from UK'37 proxy data.

The experiment achieved an R² of 0.87, indicating that 87% of the variance in the true latent SST is captured by the GP model when trained on the reduced-noise UK'37 proxy data. This is comparable to the performance achieved with the δ¹⁸O proxy, which has inherently higher sensitivity.

## Implications

1. **Laboratory Precision Matters**: For proxies with low sensitivity like UK'37, improving laboratory measurement precision to reduce noise can dramatically improve reconstruction performance.

2. **Combined Kernel Approach is Robust**: The combined kernel (RBF + Periodic) approach works effectively for both high-SNR (δ¹⁸O) and improved-SNR (UK'37) proxies, suggesting it's a robust approach for paleoclimate reconstruction.

3. **Multi-Proxy Integration**: With improved UK'37 performance, future work could explore integrating multiple proxies (δ¹⁸O and UK'37) to further improve latent SST reconstruction.