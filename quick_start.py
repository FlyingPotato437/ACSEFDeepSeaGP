import numpy as np
import matplotlib.pyplot as plt
import os

# Import from our flat structure
from ar_models import AR1Model, AR2Model
from gp_models import PhysicsInformedGP
from synthetic_data import SyntheticPaleoData
from model_comparison import ModelComparison
from calibration import proxy_to_sst
from visualization import plot_time_series, plot_power_spectrum

# Create output directory
if not os.path.exists('data/results'):
    os.makedirs('data/results')

print("Paleoclimate Reconstruction Demo")
print("===============================")

# Step 1: Generate synthetic data with Milankovitch cycles
print("\nGenerating synthetic paleoclimate data...")
synth = SyntheticPaleoData(start_time=0, end_time=600, noise_level=0.5, random_seed=42)

# Use standard Milankovitch periods and amplitudes
synth.set_cycle_parameters('eccentricity', period=100.0, amplitude=2.0)
synth.set_cycle_parameters('obliquity', period=41.0, amplitude=1.0)
synth.set_cycle_parameters('precession', period=23.0, amplitude=0.5)

# Generate dataset with regular sampling
train_dataset = synth.generate_dataset(n_points=200, regular=True, proxies=['UK37'])
print(f"  Generated training dataset with {len(train_dataset['time_points'])} points")

# Generate test dataset with irregular sampling
test_dataset = synth.generate_dataset(n_points=50, regular=False, proxies=['UK37'], age_error=1.0)
print(f"  Generated test dataset with {len(test_dataset['time_points'])} points")

# Plot the synthetic data
fig = synth.plot_dataset(train_dataset)
plt.savefig('data/results/synthetic_data.png')
plt.close()

# Step 2: Train different models
print("\nTraining paleoclimate reconstruction models...")

# Extract data
train_times = train_dataset['time_points']
train_proxies = train_dataset['proxy_data']['UK37']['values']
train_sst = train_dataset['true_sst']

# Convert proxies to SST using standard calibration
train_proxy_sst = proxy_to_sst(train_proxies, proxy_type='UK37')

# AR1 Model
print("  Training AR1 model...")
ar1_model = AR1Model(process_noise=0.1, observation_noise=0.1, optimize_params=True)
ar1_model.fit(train_times, train_proxy_sst)
print(f"  AR1 parameters: {ar1_model.get_params()}")

# AR2 Model
print("  Training AR2 model...")
ar2_model = AR2Model(process_noise=0.1, observation_noise=0.1, optimize_params=True)
ar2_model.fit(train_times, train_proxy_sst)
print(f"  AR2 parameters: {ar2_model.get_params()}")

# GP with RBF kernel
print("  Training GP model with RBF kernel...")
gp_rbf = PhysicsInformedGP(kernel='rbf', normalize=True, optimize_hyperparams=True)
gp_rbf.fit(train_times, train_proxy_sst)

# GP with Milankovitch kernel
print("  Training GP model with Milankovitch kernel...")
gp_mil = PhysicsInformedGP(kernel='milankovitch', normalize=True, optimize_hyperparams=True)
gp_mil.fit(train_times, train_proxy_sst)

# Step 3: Evaluate models
print("\nEvaluating models on test data...")

# Prepare test data
test_times = test_dataset['time_points']
test_sst = test_dataset['true_sst']

# Create model comparison
models = [ar1_model, ar2_model, gp_rbf, gp_mil]
model_names = ['AR1', 'AR2', 'GP-RBF', 'GP-Milankovitch']
comparison = ModelComparison(models, model_names)

# Evaluate models
metrics = comparison.evaluate(test_times, test_sst)
print("\nEvaluation metrics:")
print(metrics.to_string(index=False))

# Step 4: Create visualizations
print("\nCreating visualizations...")

# Plot reconstructions
fig = comparison.plot_reconstructions(figsize=(12, 6))
plt.savefig('data/results/model_reconstructions.png')
plt.close()

# Plot power spectra
fig = comparison.plot_power_spectra(figsize=(12, 10))
plt.savefig('data/results/model_power_spectra.png')
plt.close()

# Plot comprehensive comparison
fig = comparison.plot_comprehensive_comparison(figsize=(16, 12))
plt.savefig('data/results/comprehensive_comparison.png')
plt.close()

# Step 5: Generate summary report
print("\nGenerating summary report...")
summary = comparison.get_summary()
print("\n" + summary)

# Save summary to file
with open('data/results/model_comparison_summary.txt', 'w') as f:
    f.write(summary)

print("\nResults saved to 'data/results/'")
print("Demo completed successfully!")