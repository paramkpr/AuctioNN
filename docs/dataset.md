# AuctionDataset (BAD - NOT UPDATED)

**`src.data_processing.datasets.AuctionDataset(dataframe, target_column='conversion_flag')`**

A PyTorch `Dataset` designed to handle cleaned auction data. This class takes a preprocessed pandas DataFrame, separates the features from the target variable, and provides an interface for iterating over individual samples (feature dictionary and target tensor).

## Parameters

*   **`dataframe`** (`pd.DataFrame`): The input DataFrame containing the cleaned and preprocessed auction data. Must be a pandas DataFrame.
*   **`target_column`** (`str`, *optional*, default=`'conversion_flag'`): The name of the column in the `dataframe` that represents the target variable. This column must exist in the provided DataFrame.

## Attributes

*   **`target`** (`torch.Tensor`): A tensor containing the target variable values extracted from the `target_column`. The data type is set to `torch.float32`, suitable for many binary classification loss functions.
*   **`features`** (`pd.DataFrame`): A DataFrame containing only the feature columns derived from the input `dataframe`. Columns like `target_column`, `unique_id`, `impression_dttm_utc`, `conv_dttm_utc`, and `dte` are automatically dropped. You might need to adjust the drop logic within the class if your preprocessing yields different non-feature columns.
*   **`feature_names`** (`list[str]`): A list containing the names of the columns considered as features.

## Methods

### `__len__(self)`

Returns the total number of samples (rows) in the dataset.

*   **Returns**: `int` - The number of samples.

### `__getitem__(self, idx)`

Retrieves the features and target for a specific sample index.

*   **Parameters**:
    *   **`idx`** (`int` or `torch.Tensor`): The index of the sample to retrieve.
*   **Returns**: `tuple` - A tuple containing:
    1.  `dict`: A dictionary where keys are feature names (`str`) and values are the corresponding feature values for the sample at the given index.
    2.  `torch.Tensor`: A tensor containing the target value for the sample.

## Raises

*   **`TypeError`**: If the input `dataframe` is not a pandas DataFrame.
*   **`ValueError`**: If the specified `target_column` does not exist in the input `dataframe`.

## Example Usage

```python
import pandas as pd
from torch.utils.data import DataLoader
from src.data_processing.datasets import AuctionDataset # Assuming the class is in this path

# Assume 'cleaned_data.csv' contains your preprocessed data
try:
    df = pd.read_csv('cleaned_data.csv')
except FileNotFoundError:
    print("Error: cleaned_data.csv not found. Please provide a valid path.")
    # Create a dummy DataFrame for demonstration if file not found
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'device_type': ['mobile', 'desktop', 'mobile', 'tablet', 'desktop'],
        'conversion_flag': [0, 1, 0, 1, 0],
        'unique_id': [101, 102, 103, 104, 105],
        'impression_dttm_utc': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 10:10:00', '2023-01-01 10:15:00', '2023-01-01 10:20:00']),
        'conv_dttm_utc': pd.to_datetime([pd.NaT, '2023-01-01 10:06:00', pd.NaT, '2023-01-01 10:17:00', pd.NaT]),
        'dte': ['2023-01-01'] * 5
    }
    df = pd.DataFrame(data)


# Create the dataset instance
# Note: Further preprocessing like one-hot encoding 'device_type' might be needed
# depending on the model. This dataset returns features as they are in the DataFrame.
# One-hot encode categorical features before passing to the dataset if needed by the model
df = pd.get_dummies(df, columns=['device_type'], drop_first=True) # Example encoding

auction_dataset = AuctionDataset(dataframe=df, target_column='conversion_flag')

# Get the number of samples
print(f"Number of samples: {len(auction_dataset)}")

# Get a single sample
features, target = auction_dataset[0]
print("\nSample 0:")
print("Features:", features)
print("Target:", target)

# Use with DataLoader
data_loader = DataLoader(auction_dataset, batch_size=2, shuffle=True)

print("\nIterating through DataLoader:")
for batch_idx, (batch_features, batch_targets) in enumerate(data_loader):
    print(f"\nBatch {batch_idx}:")
    # Note: batch_features will be a dictionary where each value is a list/tensor of features for the batch
    # You might need to further process this structure depending on your model's input requirements
    print(" Batch Features Structure (Keys):", batch_features.keys())
    print(" Feature 'feature1' batch:", batch_features['feature1'])
    print(" Batch Targets:", batch_targets)
    if batch_idx >= 1: # Show first 2 batches
        break
```

```