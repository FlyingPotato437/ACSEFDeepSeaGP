# Data Import Guide

This guide provides detailed instructions for importing your own paleoclimate data into the Bayesian GP State-Space Model.

## Supported Data Formats

The model supports various data formats through the `load_paleoclimate_data` function:

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **Text files** (.txt) with various delimiters

## Basic Data Import

The simplest way to import data is to use the automatic detection features:

```python
from utils.data_loader import load_paleoclimate_data

# Auto-detect columns and import data
proxy_data = load_paleoclimate_data("your_data.csv")
```

This will:
1. Auto-detect the age/depth column
2. Auto-detect proxy columns based on common naming patterns
3. Validate and import the data

## Specifying Column Names

For more control, you can explicitly specify column names:

```python
proxy_data = load_paleoclimate_data(
    "your_data.csv",
    proxy_columns={
        'd18O': 'delta_18O',       # δ18O column name
        'UK37': 'UK37_index',      # UK'37 column name
        'Mg_Ca': 'Mg_Ca_ratio'     # Mg/Ca column name
    },
    age_column='Age_kyr_BP',       # Age column name
    delimiter=','                  # CSV delimiter
)
```

## Depth-to-Age Conversion

If your data uses depth instead of age, you can provide an age model for conversion:

```python
proxy_data = load_paleoclimate_data(
    "core_data.csv",
    depth_column='Depth_cm',
    age_model_file='age_model.csv'
)
```

The age model file should contain depth-age pairs. The function will:
1. Load the age model
2. Detect depth and age columns
3. Apply interpolation to convert depths to ages

## Handling Different Age Units

You can specify the age unit to ensure proper standardization:

```python
proxy_data = load_paleoclimate_data(
    "your_data.csv",
    age_column='Age',
    age_unit='yr'    # Options: 'yr', 'kyr', 'myr', 'ka', etc.
)
```

This will automatically convert all ages to kiloyears (kyr) internally.

## Data Transformation

You can apply transformations to proxy values during import:

```python
def uk37_to_sst(uk37_values):
    """Convert UK'37 to SST using standard calibration."""
    return (uk37_values - 0.044) / 0.033

proxy_data = load_paleoclimate_data(
    "uk37_data.csv",
    proxy_columns={'UK37': 'UK37_index'},
    value_transform={'UK37': uk37_to_sst}
)
```

## Handling Multiple Files

To combine data from multiple files or cores:

```python
from utils.data_loader import load_paleoclimate_data, combine_proxy_datasets

# Load individual datasets
core1_data = load_paleoclimate_data("core1.csv")
core2_data = load_paleoclimate_data("core2.csv")

# Combine datasets
combined_data = combine_proxy_datasets(
    [core1_data, core2_data],
    age_range=(0, 500)  # Optional age range filter
)
```

## Resampling and Gap Handling

For irregular data, you can resample to a common age grid:

```python
from utils.data_loader import resample_proxy_data
import numpy as np

# Define common age points
age_points = np.arange(0, 500, 1)  # 1 kyr resolution

# Resample with gap handling
resampled_data = resample_proxy_data(
    proxy_data,
    age_points=age_points,
    method='linear',  # Interpolation method
    max_gap=20        # Maximum gap size in kyr
)
```

This will:
1. Interpolate data to the common age grid
2. Mark points as NaN if they are more than `max_gap` away from real data

## Handling Missing Values

The model can handle NaN values in the input data. Missing values are automatically detected and excluded from the fitting process.

## Excel-Specific Options

For Excel files, you can specify the sheet name:

```python
proxy_data = load_paleoclimate_data(
    "multisheet_data.xlsx",
    sheet_name="Site 1 Data",
    proxy_columns={...}
)
```

## Verbose Output

For debugging, you can enable verbose output:

```python
proxy_data = load_paleoclimate_data(
    "your_data.csv",
    verbose=True
)
```

This will print information about detected columns, data points, etc.

## Expected Data Structure

The function returns a dictionary with the following structure:

```python
{
    'proxy_type1': {
        'age': np.array([...]),  # Age values
        'value': np.array([...])  # Proxy values
    },
    'proxy_type2': {
        'age': np.array([...]),
        'value': np.array([...])
    },
    ...
}
```

This is the format expected by the BayesianGPStateSpaceModel.fit() method.

## Exporting Results

After running the model, you can export the results:

```python
from utils.data_loader import export_results

# Export reconstruction results
export_results(
    {
        'ages': test_ages,
        'mean': mean_prediction,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'transitions': detected_transitions,
        'proxy_data': proxy_data
    },
    file_path="reconstruction_results.csv",
    include_proxy_data=True,
    include_transitions=True
)
```

This creates a CSV or Excel file with all results for further analysis or plotting in other software.

## Common Issues and Solutions

### Issue: Column Names Not Detected
**Solution**: Explicitly specify column names with the `proxy_columns` parameter.

### Issue: Data Points Being Excluded
**Solution**: Check for NaN values in your data or use the `verbose=True` option to see which points are being excluded.

### Issue: Age Units Incorrect
**Solution**: Specify the `age_unit` parameter to ensure proper conversion to kyr.

### Issue: Multiple Proxy Types in One Column
**Solution**: Pre-process your data to split different proxy types into separate columns before import.

### Issue: Excel File with Multiple Headers
**Solution**: Use the `skip_rows` parameter to skip additional header rows.

## Example: Complete Data Import Workflow

```python
from utils.data_loader import load_paleoclimate_data, resample_proxy_data, export_results
import numpy as np
from models.bayesian_gp_state_space import BayesianGPStateSpaceModel

# 1. Load data from multiple cores
core1_data = load_paleoclimate_data(
    "core1.xlsx",
    proxy_columns={'d18O': 'delta18O', 'UK37': 'Alkenone'},
    age_column='Age_kyr',
    sheet_name="Proxy Data"
)

core2_data = load_paleoclimate_data(
    "core2.csv",
    proxy_columns={'Mg_Ca': 'MgCa_ratio'},
    age_column='age',
    delimiter=','
)

# 2. Combine datasets
from utils.data_loader import combine_proxy_datasets
combined_data = combine_proxy_datasets([core1_data, core2_data])

# 3. Create a common age grid at 1 kyr resolution
age_grid = np.arange(0, 500, 1)

# 4. Resample to common grid with gap handling
resampled_data = resample_proxy_data(
    combined_data,
    age_points=age_grid,
    method='linear',
    max_gap=20
)

# 5. Run the model
model = BayesianGPStateSpaceModel(
    proxy_types=['d18O', 'UK37', 'Mg_Ca'],
    weighting_method='balanced'
)

model.fit(resampled_data, training_iterations=500)

# 6. Make predictions and detect transitions
test_ages = np.linspace(0, 500, 1000)
mean, lower_ci, upper_ci = model.predict(test_ages)
transitions = model.detect_abrupt_transitions(test_ages)

# 7. Visualize results
model.plot_reconstruction(
    test_ages, 
    proxy_data_dict=combined_data,
    detected_transitions=transitions,
    figure_path="reconstruction.png"
)

# 8. Export results
export_results(
    {
        'ages': test_ages,
        'mean': mean,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'transitions': transitions,
        'proxy_data': combined_data
    },
    file_path="results.xlsx",
    format='excel'
)
```