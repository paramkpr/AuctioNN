import torch
from torch.utils.data import Dataset
import pandas as pd

class AuctionDataset(Dataset):
    """
    PyTorch Dataset for the cleaned auction data.

    Takes a preprocessed pandas DataFrame, separates features and target,
    and allows iteration over samples.
    """
    def __init__(self, dataframe: pd.DataFrame, target_column: str = 'conversion_flag'):
        """
        Args:
            dataframe (pd.DataFrame): The cleaned and preprocessed DataFrame.
            target_column (str): The name of the target variable column.
        """
        super().__init__()

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        self.target = torch.tensor(dataframe[target_column].values, dtype=torch.float32) # Often float for binary classification loss functions

        # Separate features (X)
        # Drop target and other potentially non-feature columns like IDs or original timestamps
        # Adjust this list based on the final columns in your cleaned DataFrame
        columns_to_drop_for_features = [
            target_column,
            'unique_id',
            'impression_dttm_utc',
            'conv_dttm_utc',
            'dte', # Partition column, likely not a feature
             # Add any other columns that are not features for the model
        ]
        feature_columns = [col for col in dataframe.columns if col not in columns_to_drop_for_features]

        # Store features as a DataFrame for now. Further processing (e.g., encoding, scaling)
        # can happen here or later in the pipeline / model's forward pass.
        self.features = dataframe[feature_columns].copy()

        # Store feature names for potential use later (e.g., mapping back)
        self.feature_names = self.features.columns.tolist()

        print(f"Dataset initialized. Number of samples: {len(self.target)}")
        print(f"Number of features: {len(self.feature_names)}")
        # print(f"Feature names: {self.feature_names}") # Uncomment to see features

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.target)

    def __getitem__(self, idx):
        """
        Retrieves the features and target for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - dict: A dictionary where keys are feature names and values
                        are the corresponding feature values for the sample.
                - torch.Tensor: The target value for the sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get feature row as a dictionary
        # Using .iloc[idx].to_dict() is convenient here
        feature_sample = self.features.iloc[idx].to_dict()

        # Get target value
        target_sample = self.target[idx]

        return feature_sample, target_sample