import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import coherence

# No changes needed for this file as it only imports standard libraries


def calculate_rmse(y_true, y_pred):
    """
    Calculate the root mean square error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    rmse : float
        Root mean square error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_bic(y_true, y_pred, n_params):
    """
    Calculate the Bayesian Information Criterion (BIC).
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_params : int
        Number of model parameters
        
    Returns:
    --------
    bic : float
        Bayesian Information Criterion
    """
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate maximum log-likelihood (assuming Gaussian errors)
    log_likelihood = -n/2 * np.log(2*np.pi*mse) - n/2
    
    # Calculate BIC
    bic = -2 * log_likelihood + n_params * np.log(n)
    
    return bic


def calculate_aic(y_true, y_pred, n_params):
    """
    Calculate the Akaike Information Criterion (AIC).
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_params : int
        Number of model parameters
        
    Returns:
    --------
    aic : float
        Akaike Information Criterion
    """
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate maximum log-likelihood (assuming Gaussian errors)
    log_likelihood = -n/2 * np.log(2*np.pi*mse) - n/2
    
    # Calculate AIC
    aic = -2 * log_likelihood + 2 * n_params
    
    return aic


def calculate_spectral_metrics(y_true, y_pred, times, frequencies=None):
    """
    Calculate spectral analysis metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    times : array-like
        Time points
    frequencies : array-like, optional
        Frequencies of interest (cycles/kyr)
        
    Returns:
    --------
    metrics : dict
        Dictionary of spectral metrics
    """
    # Check if time points are regularly spaced 
    dt = np.diff(times)
    is_regular = np.allclose(dt, dt[0], rtol=1e-3)
    
    if not is_regular:
        return {
            'warning': 'Irregular sampling detected. Spectral metrics not calculated.',
            'coherence': None,
            'power_ratio': None
        }
    
    # Sample rate (samples per kyr)
    fs = 1.0 / dt[0]
    
    # Set default frequencies if not provided
    if frequencies is None:
        # Milankovitch cycles: eccentricity, obliquity, precession
        frequencies = [1/100.0, 1/41.0, 1/23.0]  # cycles/kyr
    
    # Calculate coherence
    f, Cxy = coherence(y_true, y_pred, fs=fs, nperseg=min(256, len(y_true)//2))
    
    # Find coherence at specified frequencies
    coherence_values = {}
    for freq in frequencies:
        # Find closest frequency in the coherence spectrum
        idx = np.argmin(np.abs(f - freq))
        
        # Get coherence value
        coherence_values[f'coherence_{1/freq:.1f}kyr'] = Cxy[idx]
    
    # Calculate power spectra
    from scipy import signal
    f, Pxx = signal.periodogram(y_true, fs=fs)
    f, Pyy = signal.periodogram(y_pred, fs=fs)
    
    # Calculate power ratios at specified frequencies
    power_ratios = {}
    for freq in frequencies:
        # Find closest frequency in the spectrum
        idx = np.argmin(np.abs(f - freq))
        
        # Calculate power ratio (predicted / true)
        power_ratios[f'power_ratio_{1/freq:.1f}kyr'] = Pyy[idx] / (Pxx[idx] + 1e-10)
    
    # Combine metrics
    metrics = {
        'warning': None,
        **coherence_values,
        **power_ratios
    }
    
    return metrics


def run_model_evaluation(model, X_train, y_train, X_test, y_test):
    """
    Run comprehensive model evaluation.
    
    Parameters:
    -----------
    model : object
        Fitted model object with predict method
    X_train : array-like
        Training time points
    y_train : array-like
        Training temperature values
    X_test : array-like
        Test time points
    y_test : array-like
        Test temperature values
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate basic metrics
    train_rmse = calculate_rmse(y_train, y_train_pred)
    test_rmse = calculate_rmse(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Get number of parameters
    if hasattr(model, 'params'):
        n_params = len(model.params)
    else:
        # Default for GP model
        n_params = 5
    
    # Calculate information criteria
    test_bic = calculate_bic(y_test, y_test_pred, n_params)
    test_aic = calculate_aic(y_test, y_test_pred, n_params)
    
    # Calculate spectral metrics
    spectral_metrics = calculate_spectral_metrics(y_test, y_test_pred, X_test)
    
    # Combine all metrics
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_params': n_params,
        'bic': test_bic,
        'aic': test_aic,
        **spectral_metrics
    }
    
    return metrics


def calculate_calibration_error(y_true, y_pred, alpha=0.05):
    """
    Calculate calibration error metrics for uncertainty quantification.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values (mean predictions)
    alpha : float, default=0.05
        Significance level for prediction intervals (default 5%)
        
    Returns:
    --------
    metrics : dict
        Dictionary of calibration error metrics
    """
    if not hasattr(y_pred, 'reshape'):
        y_pred = np.array(y_pred)
    
    if len(y_pred.shape) < 2:
        # If only mean predictions are provided, return bias
        bias = np.mean(y_pred - y_true)
        return {'bias': bias}
    
    # Unpack predictions (assuming format: [mean, std])
    y_mean = y_pred[:, 0]
    y_std = y_pred[:, 1]
    
    # Calculate normalized error
    z = (y_true - y_mean) / y_std
    
    # Calculate metrics
    bias = np.mean(y_mean - y_true)
    rmse = np.sqrt(np.mean((y_mean - y_true)**2))
    
    # Proportion of true values within prediction intervals
    z_crit = stats.norm.ppf(1 - alpha/2)
    coverage = np.mean(np.abs(z) <= z_crit)
    
    # Expected coverage
    expected_coverage = 1 - alpha
    
    # Interval width
    interval_width = 2 * z_crit * np.mean(y_std)
    
    # Calculate CRPS (Continuous Ranked Probability Score)
    # Assuming Gaussian predictive distributions
    crps = np.mean(y_std * (1/np.sqrt(np.pi) - 2*stats.norm.pdf(z) - z*(2*stats.norm.cdf(z) - 1)))
    
    # Compute predictive log-likelihood
    log_likelihood = np.mean(-0.5*np.log(2*np.pi*y_std**2) - 0.5*(y_true - y_mean)**2/y_std**2)
    
    # Check if calibration is significantly different from expected
    n = len(y_true)
    p_value = stats.binom_test(
        int(n * coverage), n, expected_coverage
    )
    
    # Return all metrics
    metrics = {
        'bias': bias,
        'rmse': rmse,
        'coverage': coverage,
        'expected_coverage': expected_coverage,
        'interval_width': interval_width,
        'crps': crps,
        'log_likelihood': log_likelihood,
        'p_value': p_value,
        'is_calibrated': p_value > 0.05
    }
    
    return metrics


def plot_calibration_curve(y_true, y_pred_mean, y_pred_std, figsize=(10, 8)):
    """
    Plot calibration curve for uncertainty quantification.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred_mean : array-like
        Predicted mean values
    y_pred_std : array-like
        Predicted standard deviations
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Calculate normalized error
    z = (y_true - y_pred_mean) / y_pred_std
    
    # Subplot 1: Q-Q plot of normalized errors
    stats.probplot(z, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot of Normalized Errors")
    ax1.grid(True)
    
    # Subplot 2: Calibration curve (empirical vs theoretical probabilities)
    # Generate theoretical quantiles
    theoretical_quantiles = np.linspace(0.01, 0.99, 99)
    theoretical_z = stats.norm.ppf(theoretical_quantiles)
    
    # Generate empirical quantiles
    z_sorted = np.sort(z)
    empirical_quantiles = np.linspace(0, 1, len(z_sorted), endpoint=False) + 1/(2*len(z_sorted))
    
    # Plot
    ax2.plot(theoretical_quantiles, empirical_quantiles, 'bo', markersize=4)
    ax2.plot([0, 1], [0, 1], 'r-', lw=2)
    ax2.set_xlabel('Theoretical Probability')
    ax2.set_ylabel('Empirical Probability')
    ax2.set_title('Calibration Curve')
    ax2.grid(True)
    
    # Calculate metrics
    metrics = calculate_calibration_error(y_true, np.column_stack([y_pred_mean, y_pred_std]))
    
    # Add metrics as text
    text = (f"Bias: {metrics['bias']:.3f}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"Coverage (95%): {metrics['coverage']:.3f}\n"
            f"Expected Coverage: {metrics['expected_coverage']:.3f}\n"
            f"Interval Width: {metrics['interval_width']:.3f}\n"
            f"CRPS: {metrics['crps']:.3f}\n"
            f"Log Likelihood: {metrics['log_likelihood']:.3f}\n"
            f"Calibrated: {metrics['is_calibrated']}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig


def compute_confidence_intervals(model, X, alpha=0.05):
    """
    Compute confidence intervals for model predictions.
    
    Parameters:
    -----------
    model : object
        Fitted model object with predict method that returns uncertainty
    X : array-like
        Time points
    alpha : float, default=0.05
        Significance level (default 5%)
        
    Returns:
    --------
    y_mean : array-like
        Mean predictions
    y_lower : array-like
        Lower confidence bounds
    y_upper : array-like
        Upper confidence bounds
    """
    # Get predictions with uncertainty
    if hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
        # Model has predict method with return_std parameter
        y_mean, y_std = model.predict(X, return_std=True)
        
        # Calculate critical value for the desired confidence level
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Calculate confidence intervals
        y_lower = y_mean - z_crit * y_std
        y_upper = y_mean + z_crit * y_std
    
    elif hasattr(model, 'predict_interval'):
        # Model has predict_interval method
        y_mean = model.predict(X)
        y_lower, y_upper = model.predict_interval(X, alpha=alpha)
    
    elif hasattr(model, 'sample_posterior'):
        # Model can sample from posterior
        y_mean = model.predict(X)
        
        # Draw posterior samples
        samples = model.sample_posterior(X, n_samples=1000)
        
        # Calculate quantiles
        y_lower = np.percentile(samples, 100 * alpha/2, axis=0)
        y_upper = np.percentile(samples, 100 * (1 - alpha/2), axis=0)
    
    else:
        # Fallback: return mean predictions without uncertainty
        y_mean = model.predict(X)
        y_lower = y_mean
        y_upper = y_mean
    
    return y_mean, y_lower, y_upper


def cross_validate_model(model_class, X, y, n_folds=5, random_state=None, **model_params):
    """
    Perform cross-validation for a model.
    
    Parameters:
    -----------
    model_class : class
        Model class to instantiate
    X : array-like
        Input features (time points)
    y : array-like
        Target values (temperature)
    n_folds : int, default=5
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility
    **model_params :
        Parameters to pass to the model constructor
        
    Returns:
    --------
    cv_results : pd.DataFrame
        DataFrame with cross-validation results
    """
    from sklearn.model_selection import KFold
    
    # Initialize cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Initialize results
    results = []
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = run_model_evaluation(model, X_train, y_train, X_test, y_test)
        
        # Add fold information
        metrics['fold'] = fold
        
        # Append to results
        results.append(metrics)
    
    # Convert to DataFrame
    cv_results = pd.DataFrame(results)
    
    # Add summary statistics
    summary = cv_results.drop('fold', axis=1).mean().to_dict()
    summary['fold'] = 'mean'
    
    std_summary = cv_results.drop('fold', axis=1).std().to_dict()
    std_summary = {f'{k}_std': v for k, v in std_summary.items()}
    summary.update(std_summary)
    
    cv_results = cv_results.append(summary, ignore_index=True)
    
    return cv_results


def plot_residual_analysis(y_true, y_pred, figsize=(12, 10)):
    """
    Plot comprehensive residual analysis.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    figsize : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Residuals vs. Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    
    # Add lowess trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lowess_y = lowess(residuals, y_pred, frac=0.6, it=3, return_sorted=False)
        axes[0, 0].plot(y_pred, lowess_y, 'r-', linewidth=2)
    except:
        pass
    
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs. Predicted')
    axes[0, 0].grid(True)
    
    # 2. Histogram of Residuals
    axes[0, 1].hist(residuals, bins=20, alpha=0.6, color='blue', edgecolor='black')
    
    # Add normal distribution curve
    x = np.linspace(min(residuals), max(residuals), 100)
    mu, sigma = np.mean(residuals), np.std(residuals)
    y = stats.norm.pdf(x, mu, sigma) * len(residuals) * (max(residuals) - min(residuals)) / 20
    axes[0, 1].plot(x, y, 'r-', linewidth=2)
    
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].grid(True)
    
    # 3. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    axes[1, 0].grid(True)
    
    # 4. Residual Autocorrelation
    from statsmodels.graphics.tsaplots import plot_acf
    try:
        plot_acf(residuals, lags=20, ax=axes[1, 1])
    except:
        # Fallback if statsmodels not available
        axes[1, 1].xcorr(residuals, residuals, maxlags=20)
        axes[1, 1].set_title('Autocorrelation of Residuals')
    
    axes[1, 1].grid(True)
    
    # Calculate residual statistics
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    skew_resid = stats.skew(residuals)
    kurt_resid = stats.kurtosis(residuals)
    
    # Add residual statistics as text
    text = (f"Mean: {mean_resid:.4f}\n"
            f"Std Dev: {std_resid:.4f}\n"
            f"Skewness: {skew_resid:.4f}\n"
            f"Kurtosis: {kurt_resid:.4f}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0, 1].text(0.95, 0.95, text, transform=axes[0, 1].transAxes, fontsize=10,
                   horizontalalignment='right', verticalalignment='top', bbox=props)
    
    # Test for normality
    shapiro_test = stats.shapiro(residuals)
    shapiro_text = f"Shapiro-Wilk Test:\nW={shapiro_test[0]:.4f}, p={shapiro_test[1]:.4f}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 0].text(0.95, 0.05, shapiro_text, transform=axes[1, 0].transAxes, fontsize=10,
                   horizontalalignment='right', verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.suptitle('Residual Analysis', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    return fig


def compare_training_curves(models, X_train, y_train, n_subsets=10, figsize=(10, 6)):
    """
    Compare model performance as a function of training data size.
    
    Parameters:
    -----------
    models : list
        List of (model_name, model_class, model_params) tuples
    X_train : array-like
        Training time points
    y_train : array-like
        Training temperature values
    n_subsets : int, default=10
        Number of subsets to evaluate
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Convert to numpy arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    # Create subsets of increasing size
    subset_sizes = np.linspace(0.1, 1.0, n_subsets)
    n_train = len(X_train)
    
    # Initialize results
    results = {model_name: {'rmse': [], 'r2': []} for model_name, _, _ in models}
    
    # Train and evaluate models on each subset
    for i, size in enumerate(subset_sizes):
        # Calculate subset size
        n_subset = int(n_train * size)
        
        # Use first n_subset points
        X_subset = X_train[:n_subset]
        y_subset = y_train[:n_subset]
        
        # Use remaining points as validation
        X_val = X_train[n_subset:]
        y_val = y_train[n_subset:]
        
        # Skip if validation set is empty
        if len(X_val) == 0:
            X_val = X_train
            y_val = y_train
        
        # Train and evaluate each model
        for model_name, model_class, model_params in models:
            # Train model
            model = model_class(**model_params)
            model.fit(X_subset, y_subset)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            rmse = calculate_rmse(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Store results
            results[model_name]['rmse'].append(rmse)
            results[model_name]['r2'].append(r2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot RMSE
    for model_name, _, _ in models:
        ax1.plot(subset_sizes * n_train, results[model_name]['rmse'], 'o-', label=model_name)
    
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs. Training Set Size')
    ax1.grid(True)
    ax1.legend()
    
    # Plot R2
    for model_name, _, _ in models:
        ax2.plot(subset_sizes * n_train, results[model_name]['r2'], 'o-', label=model_name)
    
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('R²')
    ax2.set_title('R² vs. Training Set Size')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    return fig