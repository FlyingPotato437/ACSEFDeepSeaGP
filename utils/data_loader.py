"""
Data Loading and Processing Utilities for Paleoclimate Data

This module provides functions for loading and processing paleoclimate data from
various file formats (CSV, Excel, etc.), handling irregular sampling, data gaps,
and different proxy formats.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import warnings
from scipy.interpolate import interp1d


def load_paleoclimate_data(
    file_path: str,
    proxy_columns: Optional[Dict[str, str]] = None,
    age_column: str = 'age',
    depth_column: Optional[str] = None,
    age_model_file: Optional[str] = None,
    age_unit: str = 'kyr',
    value_transform: Optional[Dict[str, callable]] = None,
    skip_rows: Optional[int] = None,
    sheet_name: Optional[Union[str, int]] = 0,
    delimiter: str = ',',
    na_values: List[str] = ['NA', 'NaN', '-999', '-999.9', 'n/a', 'null'],
    verbose: bool = False
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load and process paleoclimate data from various file formats.
    
    Args:
        file_path: Path to the data file (CSV, Excel, etc.)
        proxy_columns: Dictionary mapping proxy types to column names
        age_column: Name of the column containing age data
        depth_column: Name of the column containing depth data (if available)
        age_model_file: Path to an age model file for depth-to-age conversion (if needed)
        age_unit: Unit of age data ('kyr', 'yr', 'ka', etc.)
        value_transform: Dictionary of transform functions for each proxy
        skip_rows: Number of rows to skip at the beginning of the file
        sheet_name: Sheet name for Excel files
        delimiter: Delimiter for CSV files
        na_values: List of strings to interpret as NaN
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary of proxy data in the format expected by the BayesianGPStateSpaceModel
    """
    # Determine file type from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if verbose:
        print(f"Loading data from {file_path}")
    
    # Load data based on file type
    if file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows, na_values=na_values)
    elif file_ext == '.csv':
        df = pd.read_csv(file_path, delimiter=delimiter, skiprows=skip_rows, na_values=na_values)
    elif file_ext == '.txt':
        df = pd.read_csv(file_path, delimiter=delimiter, skiprows=skip_rows, na_values=na_values)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Check if age data exists, otherwise convert from depth using age model
    if age_column in df.columns:
        age_data = df[age_column].values
    elif depth_column is not None and depth_column in df.columns:
        if age_model_file is None:
            raise ValueError("Age model file required for depth-to-age conversion")
        
        # Convert depth to age using the age model
        age_data = _convert_depth_to_age(df[depth_column].values, age_model_file)
    else:
        raise ValueError(f"No age or depth data found in columns: {age_column}, {depth_column}")
    
    # Convert age to kyr if needed
    age_data = _standardize_age_unit(age_data, age_unit)
    
    # Process proxy data
    proxy_data = {}
    
    if proxy_columns is None:
        # Attempt to automatically detect proxy columns
        proxy_columns = _auto_detect_proxy_columns(df)
        if verbose:
            print(f"Auto-detected proxy columns: {proxy_columns}")
    
    for proxy_type, column in proxy_columns.items():
        if column in df.columns:
            # Extract values
            proxy_values = df[column].values
            
            # Apply transform if specified
            if value_transform is not None and proxy_type in value_transform:
                proxy_values = value_transform[proxy_type](proxy_values)
            
            # Store data with only valid entries (no NaN)
            valid_mask = ~np.isnan(proxy_values) & ~np.isnan(age_data)
            
            if np.sum(valid_mask) > 0:
                proxy_data[proxy_type] = {
                    'age': age_data[valid_mask],
                    'value': proxy_values[valid_mask]
                }
                
                if verbose:
                    print(f"Loaded {proxy_type} data: {np.sum(valid_mask)} valid points")
            else:
                warnings.warn(f"No valid data found for proxy {proxy_type}")
    
    return proxy_data


def _standardize_age_unit(age_data: np.ndarray, age_unit: str) -> np.ndarray:
    """
    Convert age data to standard units (kyr).
    
    Args:
        age_data: Array of age values
        age_unit: Current age unit ('kyr', 'yr', 'ka', etc.)
        
    Returns:
        Standardized age data in kyr
    """
    age_unit = age_unit.lower()
    
    if age_unit in ['kyr', 'ka', 'kyr bp', 'ka bp']:
        # Already in kyr
        return age_data
    elif age_unit in ['yr', 'y', 'yr bp', 'y bp']:
        # Convert from years to kiloyears
        return age_data / 1000.0
    elif age_unit in ['myr', 'ma', 'myr bp', 'ma bp']:
        # Convert from million years to kiloyears
        return age_data * 1000.0
    else:
        warnings.warn(f"Unknown age unit: {age_unit}. Assuming kyr.")
        return age_data


def _convert_depth_to_age(depth_data: np.ndarray, age_model_file: str) -> np.ndarray:
    """
    Convert depth data to age using an age model.
    
    Args:
        depth_data: Array of depth values
        age_model_file: Path to the age model file
        
    Returns:
        Converted age data
    """
    # Load age model
    file_ext = os.path.splitext(age_model_file)[1].lower()
    
    if file_ext in ['.xlsx', '.xls']:
        age_model = pd.read_excel(age_model_file)
    else:  # Assume CSV or similar
        age_model = pd.read_csv(age_model_file)
    
    # Identify depth and age columns in the age model
    depth_col = None
    age_col = None
    
    for col in age_model.columns:
        if re.search(r'depth|mcd|mbsf|m$|cm$|mm$', col.lower()):
            depth_col = col
        elif re.search(r'age|kyr|ka|date|yr|year', col.lower()):
            age_col = col
    
    if depth_col is None or age_col is None:
        # Try common column names
        if 'depth' in age_model.columns:
            depth_col = 'depth'
        if 'age' in age_model.columns:
            age_col = 'age'
        
        if depth_col is None or age_col is None:
            raise ValueError("Could not identify depth and age columns in age model")
    
    # Create interpolation function
    age_model_func = interp1d(
        age_model[depth_col].values,
        age_model[age_col].values,
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    # Convert depths to ages
    return age_model_func(depth_data)


def _auto_detect_proxy_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Attempt to automatically detect proxy columns in the dataframe.
    
    Args:
        df: Pandas dataframe containing proxy data
        
    Returns:
        Dictionary mapping proxy types to column names
    """
    proxy_columns = {}
    
    # Look for common proxy column patterns
    for col in df.columns:
        col_lower = col.lower()
        
        # δ18O proxies
        if re.search(r'(d18o|delta\s*18\s*o|\δ18o|δ18o|d18|d\s*18\s*o)', col_lower):
            proxy_columns['d18O'] = col
        
        # UK'37 proxies
        elif re.search(r'(uk37|uk\'37|uk\'\'37|uk|alkenone)', col_lower):
            proxy_columns['UK37'] = col
        
        # Mg/Ca proxies
        elif re.search(r'(mg\/ca|mg_ca|mg-ca|mgca)', col_lower):
            proxy_columns['Mg_Ca'] = col
        
        # TEX86 proxies
        elif re.search(r'(tex86|tex|gdgt)', col_lower):
            proxy_columns['TEX86'] = col
    
    return proxy_columns


def resample_proxy_data(
    proxy_data: Dict[str, Dict[str, np.ndarray]],
    age_points: np.ndarray,
    method: str = 'linear',
    min_points_required: int = 2,
    max_gap: Optional[float] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Resample proxy data to common age points with gap handling.
    
    Args:
        proxy_data: Dictionary of proxy data
        age_points: Common age points for resampling
        method: Interpolation method ('linear', 'nearest', 'cubic', etc.)
        min_points_required: Minimum number of points required for interpolation
        max_gap: Maximum allowed gap between points in kyr (optional)
        
    Returns:
        Resampled proxy data
    """
    resampled_data = {}
    
    for proxy_type, data in proxy_data.items():
        ages = data['age']
        values = data['value']
        
        # Check if we have enough points for interpolation
        if len(ages) < min_points_required:
            warnings.warn(f"Insufficient data points for {proxy_type}. Skipping.")
            continue
        
        # Create interpolation function
        f = interp1d(
            ages, values,
            kind=method,
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Interpolate to common age points
        resampled_values = f(age_points)
        
        # Handle gaps if max_gap is specified
        if max_gap is not None:
            # Find gaps in the original data
            age_diffs = np.diff(np.sort(ages))
            
            # Create mask for points within max_gap of an original data point
            valid_mask = np.zeros_like(age_points, dtype=bool)
            
            for age in ages:
                within_gap = np.abs(age_points - age) <= max_gap
                valid_mask = valid_mask | within_gap
            
            # Set values outside the max_gap to NaN
            resampled_values[~valid_mask] = np.nan
        
        # Store resampled data
        resampled_data[proxy_type] = {
            'age': age_points,
            'value': resampled_values
        }
    
    return resampled_data


def combine_proxy_datasets(
    datasets: List[Dict[str, Dict[str, np.ndarray]]],
    age_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Combine multiple proxy datasets, potentially from different sites or studies.
    
    Args:
        datasets: List of proxy datasets to combine
        age_range: Optional age range to filter data (min_age, max_age)
        
    Returns:
        Combined proxy dataset
    """
    combined_data = {}
    
    # Iterate through datasets
    for dataset in datasets:
        for proxy_type, data in dataset.items():
            ages = data['age']
            values = data['value']
            
            # Apply age filter if specified
            if age_range is not None:
                min_age, max_age = age_range
                valid_mask = (ages >= min_age) & (ages <= max_age)
                ages = ages[valid_mask]
                values = values[valid_mask]
            
            # Add to combined dataset
            if proxy_type not in combined_data:
                combined_data[proxy_type] = {
                    'age': ages,
                    'value': values
                }
            else:
                # Append to existing data
                combined_data[proxy_type]['age'] = np.concatenate([
                    combined_data[proxy_type]['age'], ages
                ])
                combined_data[proxy_type]['value'] = np.concatenate([
                    combined_data[proxy_type]['value'], values
                ])
    
    # Sort all combined datasets by age
    for proxy_type in combined_data:
        sort_idx = np.argsort(combined_data[proxy_type]['age'])
        combined_data[proxy_type]['age'] = combined_data[proxy_type]['age'][sort_idx]
        combined_data[proxy_type]['value'] = combined_data[proxy_type]['value'][sort_idx]
    
    return combined_data


def export_results(
    results: Dict,
    file_path: str,
    include_proxy_data: bool = True,
    include_transitions: bool = True,
    format: str = 'csv'
):
    """
    Export reconstruction results to file.
    
    Args:
        results: Dictionary of results
        file_path: Path to save the results
        include_proxy_data: Whether to include original proxy data
        include_transitions: Whether to include detected transitions
        format: Output format ('csv' or 'excel')
    """
    # Extract data
    ages = results['ages']
    mean = results['mean']
    lower_ci = results.get('lower_ci')
    upper_ci = results.get('upper_ci')
    
    # Create dataframe
    data = {
        'age': ages,
        'sst_mean': mean
    }
    
    if lower_ci is not None and upper_ci is not None:
        data['sst_lower_95ci'] = lower_ci
        data['sst_upper_95ci'] = upper_ci
    
    if include_transitions and 'transitions' in results:
        # Create indicator column for transitions
        transitions = results['transitions']
        is_transition = np.zeros_like(ages)
        
        for t in transitions:
            # Find closest age point
            idx = np.argmin(np.abs(ages - t))
            is_transition[idx] = 1
        
        data['is_transition'] = is_transition
    
    # Include proxy data if requested
    if include_proxy_data and 'proxy_data' in results:
        proxy_data = results['proxy_data']
        
        for proxy_type, proxy_values in proxy_data.items():
            # Interpolate proxy data to the common age points
            proxy_ages = proxy_values['age']
            proxy_values = proxy_values['value']
            
            # Create interpolation function
            f = interp1d(
                proxy_ages, proxy_values,
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Interpolate to common age points
            interp_values = f(ages)
            
            # Add to dataframe
            data[f'{proxy_type}_value'] = interp_values
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Export based on format
    if format.lower() == 'csv':
        df.to_csv(file_path, index=False)
    elif format.lower() in ['excel', 'xlsx']:
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {format}")