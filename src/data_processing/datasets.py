import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import warnings

class AuctionDataset(Dataset):
    """
    PyTorch Dataset for the cleaned auction data. Optimized for faster __getitem__.

    Takes a pandas DataFrame slice (train/val/test), converts features to NumPy
    arrays during initialization, separates features and target,
    and applies pre-fitted transformations (encoder, scaler) during __getitem__.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 target_column: str = 'conversion_flag',
                 categorical_encoder: OrdinalEncoder = None,
                 numerical_scaler: StandardScaler = None,
                 categorical_features: list[str] = None,
                 boolean_features: list[str] = None,
                 cyclical_features: list[str] = None,
                 numerical_features_to_scale: list[str] = None
                 ):
        """
        Args:
            dataframe (pd.DataFrame): The DataFrame slice for this dataset (e.g., train_df).
            target_column (str): The name of the target variable column.
            categorical_encoder (OrdinalEncoder): Fitted sklearn OrdinalEncoder.
            numerical_scaler (StandardScaler): Fitted sklearn StandardScaler.
            categorical_features (list): List of names of categorical columns.
            boolean_features (list): List of names of boolean columns.
            cyclical_features (list): List of names of cyclical columns.
            numerical_features_to_scale (list): List of names of the final numerical columns
                                                (after bool/cyclical processing) expected by scaler.
        """
        super().__init__()

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        if not all([categorical_encoder, numerical_scaler, categorical_features,
                    boolean_features, cyclical_features, numerical_features_to_scale]):
            raise ValueError("Must provide all preprocessors and feature lists.")

        self.categorical_encoder = categorical_encoder
        self.numerical_scaler = numerical_scaler
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features
        self.cyclical_features = cyclical_features
        self.numerical_features_to_scale = numerical_features_to_scale

        self.target_np = dataframe[target_column].values.astype(np.float32)

        self.cat_data_np = dataframe[self.categorical_features].values
        self.bool_data_np = dataframe[self.boolean_features].values.astype(np.float32)
        self.cyclical_data_np = dataframe[self.cyclical_features].values.astype(np.float32)

        self.hour_idx = self.cyclical_features.index('impression_hour')
        self.day_idx = self.cyclical_features.index('impression_dayofweek')

        del dataframe

        print(f"Dataset initialized. Features converted to NumPy. Number of samples: {len(self.target_np)}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.target_np)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - categorical_data (torch.LongTensor): Tensor of encoded categorical features.
                - numerical_data (torch.FloatTensor): Tensor of scaled numerical features.
                - target (torch.FloatTensor): The target value for the sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cat_row = self.cat_data_np[idx]
        bool_row = self.bool_data_np[idx]
        cyclical_row = self.cyclical_data_np[idx]
        target_sample = self.target_np[idx]

        # --- 1. Process Categorical Features ---
        cat_values = cat_row.reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            encoded_cats = self.categorical_encoder.transform(cat_values).flatten()

        # Handle unknowns (-1 -> 0, shift others +1)
        encoded_cats[encoded_cats == -1] = 0
        encoded_cats[encoded_cats > -1] += 1
        categorical_data = torch.from_numpy(encoded_cats).long()

        # --- 2. Process Numerical Features (Boolean + Cyclical -> Scale) ---
        processed_numerical = {}
        for i, col_name in enumerate(self.boolean_features):
            processed_numerical[col_name] = bool_row[i]

        hour = cyclical_row[self.hour_idx]
        day = cyclical_row[self.day_idx]
        processed_numerical['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        processed_numerical['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        processed_numerical['day_sin'] = np.sin(2 * np.pi * day / 7.0)
        processed_numerical['day_cos'] = np.cos(2 * np.pi * day / 7.0)

        numerical_values_ordered = [processed_numerical[col] for col in self.numerical_features_to_scale]
        numerical_values_np = np.array(numerical_values_ordered, dtype=np.float32).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scaled_numerical = self.numerical_scaler.transform(numerical_values_np).flatten()

        numerical_data = torch.from_numpy(scaled_numerical)

        target_tensor = torch.tensor(target_sample)

        return categorical_data, numerical_data, target_tensor