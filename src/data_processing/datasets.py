import torch
from torch.utils.data import Dataset
import numpy as np
import os

class AuctionDataset(Dataset):
    """
    PyTorch Dataset loading pre-processed NumPy arrays. Optimized for speed.

    Loads data directly from .npy files created by apply_preprocessors_to_split.
    """
    def __init__(self,
                 processed_data_dir: str,
                 split_name: str # e.g., 'train', 'val', 'test'
                 ):
        """
        Args:
            processed_data_dir (str): Directory containing the processed .npy files.
            split_name (str): Name of the split ('train', 'val', 'test') to load.
        """
        super().__init__()

        cat_path = os.path.join(processed_data_dir, f"{split_name}_categorical_data.npy")
        num_path = os.path.join(processed_data_dir, f"{split_name}_numerical_data.npy")
        tgt_path = os.path.join(processed_data_dir, f"{split_name}_target_data.npy")

        print(f"Loading pre-processed data for '{split_name}' split:")
        print(f"  Categorical: {cat_path}")
        print(f"  Numerical:   {num_path}")
        print(f"  Target:      {tgt_path}")

        try:
            # Use memory mapping for potentially large arrays
            self.categorical_data = np.load(cat_path, mmap_mode='r')
            self.numerical_data = np.load(num_path, mmap_mode='r')
            self.target_data = np.load(tgt_path, mmap_mode='r')
        except FileNotFoundError as e:
            print(f"ERROR: Failed to load pre-processed .npy file. "
                  f"Ensure files exist in '{processed_data_dir}'. Details: {e}")
            raise

        # --- Basic Shape Validation ---
        n_samples = len(self.target_data)
        if not (len(self.categorical_data) == n_samples and len(self.numerical_data) == n_samples):
             raise ValueError(f"Loaded NumPy arrays for split '{split_name}' have mismatched lengths.")

        print(f"Dataset initialized for '{split_name}'. Number of samples: {n_samples}")
        print(f"  Categorical shape: {self.categorical_data.shape}")
        print(f"  Numerical shape:   {self.numerical_data.shape}")
        print(f"  Target shape:      {self.target_data.shape}")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # All arrays should have the same length, checked in init
        return len(self.target_data)

    def __getitem__(self, idx):
        """
        Retrieves a pre-processed sample from the loaded NumPy arrays.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - categorical_tensor (torch.LongTensor)
                - numerical_tensor (torch.FloatTensor)
                - target_tensor (torch.FloatTensor)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # --- Get slices directly from memory-mapped arrays ---
        # Indexing mmap arrays reads only the necessary data from disk
        cat_sample = self.categorical_data[idx]
        num_sample = self.numerical_data[idx]
        target_sample = self.target_data[idx]

        # --- Convert to Tensors ---
        # Ensure correct types
        categorical_tensor = torch.from_numpy(cat_sample.astype(np.int64)).long()
        numerical_tensor = torch.from_numpy(num_sample.astype(np.float32)).float()
        target_tensor = torch.tensor(target_sample, dtype=torch.float32) # Target was already float32

        return categorical_tensor, numerical_tensor, target_tensor