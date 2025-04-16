import torch
import torch.nn as nn
from collections import OrderedDict


class AuctionNetwork(nn.Module):
    """
    Neural Network for predicting conversion probability based on impression features.
    
    Architecture:
    - Embedding layers for categorical features
    - Concatenation of embeddings and numerical features
    - Sequential hidden layers (Linear → BatchNorm → ReLU → Dropout)
    - Single output logit (use with BCEWithLogitsLoss)
    """
    def __init__(
        self,
        category_sizes: dict[str, int],
        num_numerical_features: int,
        embedding_dim: int = 32,
        hidden_dims: list[int] = [128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the auction network.
        
        Args:
            category_sizes: Dictionary mapping category names to number of unique values
            num_numerical_features: Number of numerical features in the input
            embedding_dim: Dimension of embedding vectors for categorical features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        # Input validation
        if not category_sizes:
            raise ValueError("category_sizes dictionary cannot be empty")
        if num_numerical_features < 0:
            raise ValueError("num_numerical_features cannot be negative")

        self.category_sizes = category_sizes
        self.num_numerical_features = num_numerical_features

        # Create embedding layers for each categorical feature
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(
                num_embeddings=max(size, 2),
                embedding_dim=embedding_dim,
                padding_idx=0
            ) for name, size in category_sizes.items()
        })
        
        # Calculate input dimension for the first hidden layer
        total_embedding_dim = embedding_dim * len(category_sizes)
        input_dim = total_embedding_dim + num_numerical_features

        # Build sequential hidden layers
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                (f'linear_{i}', nn.Linear(input_dim, hidden_dim)),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_dim)),
                (f'relu_{i}', nn.ReLU()),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            input_dim = hidden_dim  # Update input dim for next layer
            
        self.hidden_layers = nn.Sequential(OrderedDict(layers))
        
        # Output layer produces a single logit
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, categorical_data: torch.LongTensor, numerical_data: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            categorical_data: Tensor of categorical feature indices [batch_size, num_categorical_features]
            numerical_data: Tensor of numerical features [batch_size, num_numerical_features]
            
        Returns:
            Logits tensor [batch_size, 1]
        """
        # Validate input dimensions
        if categorical_data.shape[1] != len(self.category_sizes):
            raise ValueError(f"Expected {len(self.category_sizes)} categorical features, got {categorical_data.shape[1]}")
        if numerical_data.shape[1] != self.num_numerical_features:
            raise ValueError(f"Expected {self.num_numerical_features} numerical features, got {numerical_data.shape[1]}")

        # Process embeddings
        embeddings = []
        for i, name in enumerate(self.category_sizes):
            feature_indices = categorical_data[:, i]
            embedded = self.embedding_layers[name](feature_indices)
            embeddings.append(embedded)
            
        # Concatenate embeddings and numerical features
        x = torch.cat(embeddings + [numerical_data], dim=1)
        
        # Pass through hidden layers
        x = self.hidden_layers(x)
        
        # Output layer
        return self.output_layer(x)