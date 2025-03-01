"""
create_uk37_dashboard.py - Create a summary dashboard for UK'37 noise reduction experiment
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

# Set up directory for results
input_dir = "final_figures"
output_dir = input_dir
os.makedirs(output_dir, exist_ok=True)

# Load the previously generated images
data_img = mpimg.imread(os.path.join(input_dir, "uk37_reduced_noise_data.png"))
pred_img = mpimg.imread(os.path.join(input_dir, "uk37_reduced_noise_predictions.png"))
perf_img = mpimg.imread(os.path.join(input_dir, "uk37_performance_comparison.png"))
residuals_img = mpimg.imread(os.path.join(input_dir, "uk37_reduced_noise_residuals.png"))

# Define metrics from the experiment results
original_metrics = {
    'snr': 0.3365,
    'rmse': 4.3904,
    'mae': 3.5516,
    'r2': -0.2212,
    'bias': -0.3057,
    'std_err': 4.3797
}

improved_metrics = {
    'snr': 1.6824,
    'rmse': 1.4548,
    'mae': 1.1551,
    'r2': 0.8659,
    'bias': 0.1595,
    'std_err': 1.4460
}

# Calculate improvements
improvements = {
    'snr': improved_metrics['snr'] / original_metrics['snr'],
    'rmse': (original_metrics['rmse'] - improved_metrics['rmse']) / original_metrics['rmse'] * 100,
    'mae': (original_metrics['mae'] - improved_metrics['mae']) / original_metrics['mae'] * 100,
    'std_err': (original_metrics['std_err'] - improved_metrics['std_err']) / original_metrics['std_err'] * 100
}

# Create dashboard figure
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.8, 1, 1.8, 1])

# Add title
fig.suptitle('UK\'37 Proxy Performance Improvement Dashboard\nEffect of 80% Noise Reduction on Latent SST Reconstruction', 
              fontsize=24, fontweight='bold', y=0.98)

# Add synthetic data visualization (top left)
ax_data = fig.add_subplot(gs[0, 0])
ax_data.imshow(data_img)
ax_data.axis('off')
ax_data.set_title('Synthetic UK\'37 Proxy Data with Reduced Noise', fontsize=18, pad=10)

# Add GP predictions visualization (top right)
ax_pred = fig.add_subplot(gs[0, 1])
ax_pred.imshow(pred_img)
ax_pred.axis('off')
ax_pred.set_title('UK\'37 GP Model Predictions vs TRUE Latent SST', fontsize=18, pad=10)

# Add metrics table (middle row, spanning both columns)
ax_metrics = fig.add_subplot(gs[1, :])
ax_metrics.axis('off')

# Create table data
metrics_labels = ['Signal-to-Noise Ratio (SNR)', 'RMSE (°C)', 'MAE (°C)', 'R²', 
                'Systematic Error (°C)', 'Random Error (°C)']
metrics_original = [f"{original_metrics['snr']:.4f}", 
                  f"{original_metrics['rmse']:.4f}", 
                  f"{original_metrics['mae']:.4f}", 
                  f"{original_metrics['r2']:.4f}", 
                  f"{original_metrics['bias']:.4f}", 
                  f"{original_metrics['std_err']:.4f}"]
metrics_improved = [f"{improved_metrics['snr']:.4f}", 
                  f"{improved_metrics['rmse']:.4f}", 
                  f"{improved_metrics['mae']:.4f}", 
                  f"{improved_metrics['r2']:.4f}", 
                  f"{improved_metrics['bias']:.4f}", 
                  f"{improved_metrics['std_err']:.4f}"]

# Format improvement values
snr_text = f"{improvements['snr']:.1f}x higher"
rmse_text = f"{improvements['rmse']:.1f}% reduction"
mae_text = f"{improvements['mae']:.1f}% reduction"
r2_text = "Negative → Positive"
bias_text = "Improved"
std_text = f"{improvements['std_err']:.1f}% reduction"

metrics_improvement = [snr_text, rmse_text, mae_text, r2_text, bias_text, std_text]

# Create a simpler table with no complex coloring
table = ax_metrics.table(
    cellText=[metrics_labels, metrics_original, metrics_improved, metrics_improvement],
    colLabels=['Metric', 'Original UK\'37', 'Improved UK\'37', 'Improvement'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2)

# Add performance comparison visualization (bottom left)
ax_perf = fig.add_subplot(gs[2, 0])
ax_perf.imshow(perf_img)
ax_perf.axis('off')
ax_perf.set_title('Performance Metrics Comparison', fontsize=18, pad=10)

# Add residuals visualization (bottom right)
ax_resid = fig.add_subplot(gs[2, 1])
ax_resid.imshow(residuals_img)
ax_resid.axis('off')
ax_resid.set_title('Residual Analysis for Reduced-Noise UK\'37', fontsize=18, pad=10)

# Add key findings (bottom row)
ax_findings = fig.add_subplot(gs[3, :])
ax_findings.axis('off')

findings_text = """
Key Findings:

1. Reducing UK'37 noise by 80% improved Signal-to-Noise Ratio from 0.34 to 1.68 (5.0x higher)
2. GP model performance improved dramatically:
   - RMSE decreased from 4.39°C to 1.45°C (66.9% reduction)
   - R² improved from -0.22 (worse than mean) to 0.87 (excellent fit)
3. Random error reduced by 67.0%, making predictions much more precise
4. The low-sensitivity UK'37 proxy (0.033 units/°C) can reconstruct latent SST effectively 
   with sufficient noise reduction
5. The tuned combined kernel (RBF + Periodic) effectively captures temperature patterns 
   when signal exceeds noise
"""

ax_findings.text(0.5, 0.5, findings_text, fontsize=16, 
                ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

# Fine-tune layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the dashboard
fig.savefig(os.path.join(output_dir, "uk37_improvement_dashboard.png"), 
            dpi=300, bbox_inches='tight')

print(f"Dashboard created and saved to {output_dir}/uk37_improvement_dashboard.png")