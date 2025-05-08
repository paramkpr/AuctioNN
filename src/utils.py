import torch

from src.models.network import ImpressionConversionNetwork


def constant_pconv_heuristic(
    cat_batch: torch.Tensor, num_batch: torch.Tensor
) -> torch.Tensor:
    """
    A simple heuristic predictor that returns a constant pConv value for every impression.

    Args:
        cat_batch: Batch of categorical feature tensors (unused in this heuristic).
        num_batch: Batch of numerical feature tensors (unused in this heuristic).

    Returns:
        A tensor containing a constant pConv value (e.g., 0.01) for each item in the batch.
    """
    # Get the batch size from one of the input tensors (e.g., categorical)
    batch_size = cat_batch.shape[0]

    # Define the constant pConv value you want to use for the baseline
    constant_value = 0.01  # You can adjust this value

    # Create and return a tensor of the correct size filled with the constant value
    # Ensure the tensor is on the correct device if needed, though for constants CPU is fine.
    # DecisionLoop moves ScriptModules to the device, but not callables.
    # However, since this returns a new tensor, it shouldn't matter here.
    return torch.full((batch_size,), constant_value, dtype=torch.float32)


def load_model(
    ckpt_path: str, device: torch.device, cardinalities: list[int]
) -> torch.nn.Module:
    model = ImpressionConversionNetwork(
        cardinalities, numeric_dim=8, deep_embedding_dim=16
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model
