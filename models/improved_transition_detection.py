"""
Improved Transition Detection Algorithms for Paleoclimate Time Series

This module implements advanced methods for detecting abrupt climate transitions
in paleoclimate time series, with mathematical foundations in change point detection,
wavelet analysis, and derivative estimation.
"""

import numpy as np
import pywt
from scipy import signal
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt


def multi_scale_derivative_detector(
    time_points: np.ndarray,
    values: np.ndarray,
    scales: List[float] = [2.0, 5.0, 10.0, 20.0],
    threshold_percentile: float = 99.0,
    min_separation: float = 20.0,
    min_magnitude: float = 1.5,
    filter_type: str = 'gaussian',
    return_derivatives: bool = False
) -> Union[List[float], Tuple[List[float], Dict]]:
    """
    Multi-scale derivative-based transition detector.
    
    This method computes derivatives at multiple scales (smoothing levels) and
    combines them to identify significant transitions while reducing noise sensitivity.
    
    Args:
        time_points: Array of time points
        values: Array of values
        scales: List of scales (smoothing levels) to use for derivative calculation
        threshold_percentile: Percentile threshold for transition detection
        min_separation: Minimum separation between transitions
        min_magnitude: Minimum magnitude change to qualify as transition
        filter_type: Type of smoothing filter ('gaussian', 'savgol')
        return_derivatives: Whether to return derivative data
        
    Returns:
        List of transition points and optionally derivative data
    """
    # Ensure inputs are numpy arrays
    time_points = np.asarray(time_points)
    values = np.asarray(values)
    
    # Initialize storage for multi-scale derivatives
    ms_derivatives = []
    scale_weights = []
    
    # Process each scale
    for scale in scales:
        # Determine the appropriate window size based on scale and time resolution
        dt = np.median(np.diff(time_points))
        window_size = max(3, int(scale / dt))
        
        # Ensure window size is odd for symmetry
        if window_size % 2 == 0:
            window_size += 1
        
        # Apply smoothing based on filter type
        if filter_type == 'gaussian':
            # Compute standard deviation in samples
            sigma = scale / dt / 3  # 3-sigma rule
            smoothed = gaussian_filter1d(values, sigma)
        elif filter_type == 'savgol':
            # Savitzky-Golay filter (polynomial smoothing)
            poly_order = min(3, window_size - 1)
            smoothed = signal.savgol_filter(values, window_size, poly_order)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        # Compute derivatives
        # We use central differences for more accurate derivative estimation
        dx = np.diff(time_points)
        dy = np.diff(smoothed)
        derivatives = np.zeros_like(values)
        derivatives[1:-1] = 0.5 * (dy[1:] / dx[1:] + dy[:-1] / dx[:-1])
        derivatives[0] = dy[0] / dx[0]
        derivatives[-1] = dy[-1] / dx[-1]
        
        # Weight by scale (smaller scales get higher weight as they're more precise)
        scale_weight = 1.0 / scale
        
        # Store
        ms_derivatives.append(derivatives)
        scale_weights.append(scale_weight)
    
    # Normalize weights
    scale_weights = np.array(scale_weights) / np.sum(scale_weights)
    
    # Compute weighted combination of derivatives
    combined_derivative = np.zeros_like(values)
    for i, deriv in enumerate(ms_derivatives):
        combined_derivative += scale_weights[i] * deriv
    
    # Use absolute value for transition detection
    abs_derivative = np.abs(combined_derivative)
    
    # Determine threshold
    threshold = np.percentile(abs_derivative, threshold_percentile)
    
    # Find peaks above threshold
    peak_indices = signal.find_peaks(abs_derivative, height=threshold)[0]
    
    # If no peaks found, return empty list
    if len(peak_indices) == 0:
        if return_derivatives:
            return [], {
                'derivatives': ms_derivatives,
                'combined_derivative': combined_derivative,
                'threshold': threshold
            }
        return []
    
    # Filter by magnitude change
    filtered_indices = []
    for idx in peak_indices:
        # Define window for checking magnitude change
        window_size = int(min_separation / np.median(np.diff(time_points)))
        left_idx = max(0, idx - window_size)
        right_idx = min(len(values) - 1, idx + window_size)
        
        # Compute magnitude change
        magnitude_change = np.abs(values[right_idx] - values[left_idx])
        
        if magnitude_change >= min_magnitude:
            filtered_indices.append(idx)
    
    # If no transitions pass magnitude filter, return empty list
    if len(filtered_indices) == 0:
        if return_derivatives:
            return [], {
                'derivatives': ms_derivatives,
                'combined_derivative': combined_derivative,
                'threshold': threshold
            }
        return []
    
    # Group by proximity to handle multiple detections of the same transition
    filtered_indices = np.sort(filtered_indices)
    groups = [[filtered_indices[0]]]
    
    for i in range(1, len(filtered_indices)):
        curr_idx = filtered_indices[i]
        prev_idx = filtered_indices[i-1]
        
        # Convert index distance to time distance
        time_diff = time_points[curr_idx] - time_points[prev_idx]
        
        if time_diff <= min_separation:
            # Add to current group
            groups[-1].append(curr_idx)
        else:
            # Start new group
            groups.append([curr_idx])
    
    # For each group, select the index with maximum derivative
    transition_indices = []
    for group in groups:
        max_idx = group[np.argmax(abs_derivative[group])]
        transition_indices.append(max_idx)
    
    # Convert indices to time points
    transition_points = [time_points[idx] for idx in transition_indices]
    
    if return_derivatives:
        return transition_points, {
            'derivatives': ms_derivatives,
            'combined_derivative': combined_derivative,
            'threshold': threshold,
            'indices': transition_indices
        }
    
    return transition_points


def wavelet_based_transition_detector(
    time_points: np.ndarray,
    values: np.ndarray,
    wavelet: str = 'haar',
    max_scale: Optional[int] = None,
    threshold_factor: float = 3.0,
    min_separation: float = 20.0,
    min_magnitude: float = 1.5,
    return_coeffs: bool = False
) -> Union[List[float], Tuple[List[float], Dict]]:
    """
    Wavelet-based transition detector.
    
    This method uses wavelet decomposition to identify abrupt transitions,
    which appear as high wavelet coefficients at specific scales and locations.
    
    Args:
        time_points: Array of time points
        values: Array of values
        wavelet: Wavelet to use (e.g., 'haar', 'db4')
        max_scale: Maximum wavelet scale to consider
        threshold_factor: Factor times standard deviation for thresholding
        min_separation: Minimum separation between transitions
        min_magnitude: Minimum magnitude change to qualify as transition
        return_coeffs: Whether to return wavelet coefficients
        
    Returns:
        List of transition points and optionally wavelet data
    """
    # Ensure inputs are numpy arrays
    time_points = np.asarray(time_points)
    values = np.asarray(values)
    
    # Determine max scale if not provided
    if max_scale is None:
        # Heuristic: use scales that can detect transitions at least 5 data points wide
        max_scale = int(np.log2(len(values) / 5))
        max_scale = max(1, min(max_scale, int(np.log2(len(values))) - 2))
    
    # Compute wavelet transform
    coeffs = pywt.wavedec(values, wavelet, level=max_scale)
    
    # Extract detail coefficients (without the approximation)
    detail_coeffs = coeffs[1:]
    
    # Combine detail coefficients with proper weighting
    combined_coeffs = np.zeros_like(values)
    
    for scale, scale_coeffs in enumerate(detail_coeffs):
        # Interpolate to original length
        scale_idx = max_scale - scale
        scale_weight = 2.0**scale_idx  # Higher weight for smaller scales
        
        # Pad to match original length
        padded_coeffs = np.zeros_like(values)
        
        # Handle boundary effects - center the coefficients
        start_idx = (len(values) - len(scale_coeffs)) // 2
        padded_coeffs[start_idx:start_idx + len(scale_coeffs)] = scale_coeffs
        
        # Add weighted contribution
        combined_coeffs += scale_weight * padded_coeffs
    
    # Take absolute value for detection
    abs_coeffs = np.abs(combined_coeffs)
    
    # Compute threshold based on noise level
    noise_std = np.median(abs_coeffs) / 0.6745  # Robust estimator of standard deviation
    threshold = threshold_factor * noise_std
    
    # Find indices exceeding threshold
    candidate_indices = np.where(abs_coeffs > threshold)[0]
    
    # If no candidates, return empty list
    if len(candidate_indices) == 0:
        if return_coeffs:
            return [], {'coeffs': coeffs, 'combined': combined_coeffs, 'threshold': threshold}
        return []
    
    # Filter by magnitude change
    filtered_indices = []
    for idx in candidate_indices:
        # Define window for checking magnitude change
        window_size = int(min_separation / np.median(np.diff(time_points)))
        left_idx = max(0, idx - window_size)
        right_idx = min(len(values) - 1, idx + window_size)
        
        # Compute magnitude change
        magnitude_change = np.abs(values[right_idx] - values[left_idx])
        
        if magnitude_change >= min_magnitude:
            filtered_indices.append(idx)
    
    # If no transitions pass magnitude filter, return empty list
    if len(filtered_indices) == 0:
        if return_coeffs:
            return [], {'coeffs': coeffs, 'combined': combined_coeffs, 'threshold': threshold}
        return []
    
    # Group by proximity to handle multiple detections of the same transition
    filtered_indices = np.sort(filtered_indices)
    groups = [[filtered_indices[0]]]
    
    for i in range(1, len(filtered_indices)):
        curr_idx = filtered_indices[i]
        prev_idx = filtered_indices[i-1]
        
        # Convert index distance to time distance
        time_diff = time_points[curr_idx] - time_points[prev_idx]
        
        if time_diff <= min_separation:
            # Add to current group
            groups[-1].append(curr_idx)
        else:
            # Start new group
            groups.append([curr_idx])
    
    # For each group, select the index with maximum coefficient
    transition_indices = []
    for group in groups:
        max_idx = group[np.argmax(abs_coeffs[group])]
        transition_indices.append(max_idx)
    
    # Convert indices to time points
    transition_points = [time_points[idx] for idx in transition_indices]
    
    if return_coeffs:
        return transition_points, {
            'coeffs': coeffs,
            'combined': combined_coeffs,
            'threshold': threshold,
            'indices': transition_indices
        }
    
    return transition_points


def bayesian_change_point_detector(
    time_points: np.ndarray,
    values: np.ndarray,
    window_size: int = 10,
    threshold_prob: float = 0.95,
    min_separation: float = 20.0,
    min_magnitude: float = 1.5,
    return_probs: bool = False
) -> Union[List[float], Tuple[List[float], np.ndarray]]:
    """
    Bayesian change point detector.
    
    This method uses Bayesian inference to compute posterior probabilities
    of change points at each location in the time series.
    
    Args:
        time_points: Array of time points
        values: Array of values
        window_size: Window size for local segments
        threshold_prob: Probability threshold for transition detection
        min_separation: Minimum separation between transitions
        min_magnitude: Minimum magnitude change to qualify as transition
        return_probs: Whether to return probability values
        
    Returns:
        List of transition points and optionally probability array
    """
    # Ensure inputs are numpy arrays
    time_points = np.asarray(time_points)
    values = np.asarray(values)
    
    n = len(values)
    change_point_probs = np.zeros(n)
    
    # Loop through possible change point locations
    for i in range(window_size, n - window_size):
        # Get segments before and after potential change point
        segment1 = values[i - window_size:i]
        segment2 = values[i:i + window_size]
        
        # Compute statistics for each segment
        mean1, std1 = np.mean(segment1), np.std(segment1)
        mean2, std2 = np.mean(segment2), np.std(segment2)
        
        # Compute probability of change based on difference in means
        # Assuming normal distribution for simplicity
        delta = abs(mean2 - mean1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        
        if pooled_std > 0:
            # Calculate z-score of difference
            z = delta / (pooled_std * np.sqrt(2/window_size))
            
            # Convert to probability
            change_prob = 2 * (1 - norm.cdf(z))  # Two-tailed p-value
            change_prob = 1 - change_prob        # Probability of change
            
            change_point_probs[i] = change_prob
    
    # Find peaks above threshold
    peak_indices = []
    for i in range(1, n-1):
        if (change_point_probs[i] > change_point_probs[i-1] and 
            change_point_probs[i] > change_point_probs[i+1] and
            change_point_probs[i] >= threshold_prob):
            peak_indices.append(i)
    
    # If no peaks found, return empty list
    if len(peak_indices) == 0:
        if return_probs:
            return [], change_point_probs
        return []
    
    # Filter by magnitude change
    filtered_indices = []
    for idx in peak_indices:
        # Define window for checking magnitude change
        window_size_time = int(min_separation / np.median(np.diff(time_points)))
        left_idx = max(0, idx - window_size_time)
        right_idx = min(len(values) - 1, idx + window_size_time)
        
        # Compute magnitude change
        magnitude_change = np.abs(values[right_idx] - values[left_idx])
        
        if magnitude_change >= min_magnitude:
            filtered_indices.append(idx)
    
    # If no transitions pass magnitude filter, return empty list
    if len(filtered_indices) == 0:
        if return_probs:
            return [], change_point_probs
        return []
    
    # Group by proximity to handle multiple detections of the same transition
    filtered_indices = np.sort(filtered_indices)
    groups = [[filtered_indices[0]]]
    
    for i in range(1, len(filtered_indices)):
        curr_idx = filtered_indices[i]
        prev_idx = filtered_indices[i-1]
        
        # Convert index distance to time distance
        time_diff = time_points[curr_idx] - time_points[prev_idx]
        
        if time_diff <= min_separation:
            # Add to current group
            groups[-1].append(curr_idx)
        else:
            # Start new group
            groups.append([curr_idx])
    
    # For each group, select the index with maximum probability
    transition_indices = []
    for group in groups:
        max_idx = group[np.argmax(change_point_probs[group])]
        transition_indices.append(max_idx)
    
    # Convert indices to time points
    transition_points = [time_points[idx] for idx in transition_indices]
    
    if return_probs:
        return transition_points, change_point_probs
    
    return transition_points


def ensemble_transition_detector(
    time_points: np.ndarray,
    values: np.ndarray,
    detectors: Optional[List[str]] = None,
    min_separation: float = 20.0,
    min_magnitude: float = 1.5,
    min_detectors: int = 2,
    return_details: bool = False,
    **detector_kwargs
) -> Union[List[float], Tuple[List[float], Dict]]:
    """
    Ensemble method combining multiple transition detection algorithms.
    
    This method runs multiple detection algorithms and combines their results
    through consensus, improving robustness and reducing false positives.
    
    Args:
        time_points: Array of time points
        values: Array of values
        detectors: List of detectors to use ('derivative', 'wavelet', 'bayesian')
        min_separation: Minimum separation between transitions
        min_magnitude: Minimum magnitude change to qualify as transition
        min_detectors: Minimum number of detectors that must agree for a transition
        return_details: Whether to return detailed results
        **detector_kwargs: Additional arguments for individual detectors
        
    Returns:
        List of transition points and optionally detailed results
    """
    # Ensure inputs are numpy arrays
    time_points = np.asarray(time_points)
    values = np.asarray(values)
    
    # Default to all detectors if not specified
    if detectors is None:
        detectors = ['derivative', 'wavelet', 'bayesian']
    
    # Run each detector
    all_transitions = {}
    detector_results = {}
    
    for detector in detectors:
        if detector == 'derivative':
            # Extract arguments for this detector
            scales = detector_kwargs.get('derivative_scales', [2.0, 5.0, 10.0, 20.0])
            threshold_percentile = detector_kwargs.get('derivative_threshold', 99.0)
            filter_type = detector_kwargs.get('derivative_filter', 'gaussian')
            
            transitions, details = multi_scale_derivative_detector(
                time_points, values,
                scales=scales,
                threshold_percentile=threshold_percentile,
                min_separation=min_separation,
                min_magnitude=min_magnitude,
                filter_type=filter_type,
                return_derivatives=True
            )
            
        elif detector == 'wavelet':
            # Extract arguments for this detector
            wavelet = detector_kwargs.get('wavelet_type', 'haar')
            max_scale = detector_kwargs.get('wavelet_max_scale', None)
            threshold_factor = detector_kwargs.get('wavelet_threshold', 3.0)
            
            transitions, details = wavelet_based_transition_detector(
                time_points, values,
                wavelet=wavelet,
                max_scale=max_scale,
                threshold_factor=threshold_factor,
                min_separation=min_separation,
                min_magnitude=min_magnitude,
                return_coeffs=True
            )
            
        elif detector == 'bayesian':
            # Extract arguments for this detector
            window_size = detector_kwargs.get('bayesian_window', 10)
            threshold_prob = detector_kwargs.get('bayesian_threshold', 0.95)
            
            transitions, details = bayesian_change_point_detector(
                time_points, values,
                window_size=window_size,
                threshold_prob=threshold_prob,
                min_separation=min_separation,
                min_magnitude=min_magnitude,
                return_probs=True
            )
            
        else:
            raise ValueError(f"Unknown detector type: {detector}")
        
        all_transitions[detector] = transitions
        detector_results[detector] = details
    
    # Combine results through consensus
    # First, create a grid of all possible transition points
    all_points = []
    for detector, transitions in all_transitions.items():
        all_points.extend(transitions)
    
    # If no transitions detected by any method, return empty list
    if len(all_points) == 0:
        if return_details:
            return [], {
                'detector_results': detector_results,
                'detector_transitions': all_transitions
            }
        return []
    
    # Group points that are close to each other
    all_points = np.sort(all_points)
    groups = [[all_points[0]]]
    
    for i in range(1, len(all_points)):
        curr_point = all_points[i]
        prev_point = all_points[i-1]
        
        if curr_point - prev_point <= min_separation:
            # Add to current group
            groups[-1].append(curr_point)
        else:
            # Start new group
            groups.append([curr_point])
    
    # For each group, count how many detectors found a transition in that group
    consensus_transitions = []
    
    for group in groups:
        group_center = np.mean(group)
        detector_votes = 0
        
        for detector, transitions in all_transitions.items():
            # Check if any transition from this detector is in the group
            for trans_point in transitions:
                if min(group) <= trans_point <= max(group):
                    detector_votes += 1
                    break
        
        # Add to consensus if enough detectors agree
        if detector_votes >= min_detectors:
            consensus_transitions.append(group_center)
    
    if return_details:
        return consensus_transitions, {
            'detector_results': detector_results,
            'detector_transitions': all_transitions,
            'groups': groups
        }
    
    return consensus_transitions


def plot_detected_transitions(
    time_points: np.ndarray,
    values: np.ndarray,
    transitions: List[float],
    detector_name: str = 'Ensemble',
    time_label: str = 'Age (kyr)',
    value_label: str = 'Temperature (°C)',
    figure_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series with detected transitions.
    
    Args:
        time_points: Array of time points
        values: Array of values
        transitions: List of detected transition points
        detector_name: Name of the detector used
        time_label: Label for time axis
        value_label: Label for value axis
        figure_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    ax.plot(time_points, values, 'b-', linewidth=1.5)
    
    # Mark transitions
    for t in transitions:
        ax.axvline(t, color='r', linestyle='--', alpha=0.7)
        
        # Add label at the top
        ymin, ymax = ax.get_ylim()
        ax.text(t, ymax, f'{t:.1f}', rotation=90, va='top', ha='right', color='r')
    
    # Add labels and title
    ax.set_xlabel(time_label)
    ax.set_ylabel(value_label)
    ax.set_title(f'Detected Transitions using {detector_name} Method (n={len(transitions)})')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if figure_path:
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    
    return fig