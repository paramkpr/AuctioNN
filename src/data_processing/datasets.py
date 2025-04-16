import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

class AuctionDataset(Dataset):
    """
    PyTorch Dataset for the cleaned auction data.

    Takes a pandas DataFrame slice (train/val/test), separates features and target,
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

        self.target = torch.tensor(dataframe[target_column].values, dtype=torch.float32)

        model_feature_columns = self.categorical_features + self.boolean_features + self.cyclical_features

        missing_model_features = [col for col in model_feature_columns if col not in dataframe.columns]
        if missing_model_features:
            raise ValueError(f"DataFrame is missing required model feature columns: {missing_model_features}")

        self.features_df = dataframe[model_feature_columns].copy()

        print(f"Dataset initialized. Number of samples: {len(self.target)}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.target)

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

        sample_features = self.features_df.iloc[idx]
        sample_target = self.target[idx]

        # --- 1. Process Categorical Features ---
        # Create a DataFrame with the expected column names
        cat_df_slice = pd.DataFrame([sample_features[self.categorical_features].values], columns=self.categorical_features)
        # Transform the DataFrame
        encoded_cats = self.categorical_encoder.transform(cat_df_slice).flatten()

        # Handle unknowns (-1 -> 0, shift others +1)
        encoded_cats[encoded_cats == -1] = 0
        encoded_cats[encoded_cats > -1] += 1
        categorical_data = torch.LongTensor(encoded_cats)

        # --- 2. Process Numerical Features (Boolean + Cyclical -> Scale) ---
        processed_numerical = {}
        for col in self.boolean_features:
             processed_numerical[col] = float(sample_features[col])
        # ... (cyclical transformations remain the same) ...
        hour = sample_features['impression_hour']
        day = sample_features['impression_dayofweek']
        processed_numerical['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        processed_numerical['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        processed_numerical['day_sin'] = np.sin(2 * np.pi * day / 7.0)
        processed_numerical['day_cos'] = np.cos(2 * np.pi * day / 7.0)

        # Create a DataFrame slice in the correct order with expected names
        numerical_values_ordered = [processed_numerical[col] for col in self.numerical_features_to_scale]
        num_df_slice = pd.DataFrame([numerical_values_ordered], columns=self.numerical_features_to_scale)

        # Scale the DataFrame
        scaled_numerical = self.numerical_scaler.transform(num_df_slice).flatten()
        numerical_data = torch.FloatTensor(scaled_numerical)

        return categorical_data, numerical_data, sample_target