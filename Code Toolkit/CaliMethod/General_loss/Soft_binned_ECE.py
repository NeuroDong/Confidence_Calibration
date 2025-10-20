import torch
import math
import numpy as np

EPS = 1e-5

def is_softmax_output(vec, atol=1e-6):
    """
    vec: 1D tensor of shape (C,) or 2D tensor (N, C)
    """
    if vec.dim() == 1:
        s = vec.sum().item()
        in_range = torch.all((vec >= -atol) & (vec <= 1 + atol)).item()
        return in_range and abs(s - 1.0) <= atol
    elif vec.dim() == 2:
        s = vec.sum(dim=1)
        in_range = torch.all((vec >= -atol) & (vec <= 1 + atol)).item()
        return in_range and torch.all(torch.abs(s - 1.0) <= atol).item()
    else:
        raise ValueError("Expect 1D or 2D tensor.")

def soft_binning_ece(
        predictions,
        labels,
        soft_binning_bins = 15,
        soft_binning_use_decay=True,
        soft_binning_decay_factor=0.9,
        soft_binning_temp=0.01
):
    """Computes and returns the soft-binned ECE (binned) scalar tensor (PyTorch).

    Soft-binned ECE (binned, L2-norm) is defined in Eq. (11) of:
    https://arxiv.org/abs/2108.00106. This is a softened version of ECE (binned)
    defined in Eq. (6).

    Args:
        predictions: 1D tensor (N,) or (N,C)
        labels: 1D tensor (N,) of incorrect(0)/correct(1) labels.
        soft_binning_bins: number of bins (int).
        soft_binning_use_decay: whether temp is determined by decay factor (bool).
        soft_binning_decay_factor: approximate decay factor between successive bins.
        soft_binning_temp: soft binning temperature (float) when not using decay.

    Returns:
        A 0-dim torch.Tensor containing the soft-binned ECE value.
    """

    # Convert inputs to torch tensors (float) and ensure 1D shape
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.as_tensor(predictions)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    if len(predictions.shape) == 2:
        if is_softmax_output(predictions):
            predictions = predictions.max(dim=1).values
        else:
            predictions = torch.softmax(predictions, dim=1).max(dim=1).values
    

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    if predictions.numel() != labels.numel():
        raise ValueError("predictions and labels must have the same number of elements")

    # Dtype/device handling
    if not predictions.is_floating_point():
        predictions = predictions.to(dtype=torch.get_default_dtype())
    dtype = predictions.dtype
    device = predictions.device
    labels = labels.to(dtype=dtype, device=device)

    B = int(soft_binning_bins)
    if B <= 0:
        raise ValueError("soft_binning_bins must be a positive integer")

    # Bin anchors: midpoints (2i+1)/(2B) for i=0..B-1
    anchors = (torch.arange(B, dtype=dtype, device=device) * 2 + 1) / (2.0 * B)

    # Temperature via decay if requested
    if soft_binning_use_decay:
        soft_binning_temp = 1.0 / (math.log(soft_binning_decay_factor) * B * B)

    # Soft assignment to bins
    diffs = predictions.unsqueeze(1) - anchors.unsqueeze(0)  # [N, B]
    scores = - (diffs ** 2) / soft_binning_temp              # [N, B]
    coeffs = torch.softmax(scores, dim=1)                    # [N, B]

    # Aggregate per-bin statistics
    sum_coeffs_for_bin = coeffs.sum(dim=0)                   # [B]
    denom = torch.clamp(sum_coeffs_for_bin, min=EPS)

    net_bin_confidence = (predictions.unsqueeze(1) * coeffs).sum(dim=0) / denom
    net_bin_accuracy = (labels.unsqueeze(1) * coeffs).sum(dim=0) / denom

    # Bin weights: L1-normalized sum of coeffs per bin
    total = torch.clamp(sum_coeffs_for_bin.sum(), min=EPS)
    bin_weights = sum_coeffs_for_bin / total                  # [B]

    # ECE: sqrt( sum_b w_b * (conf_b - acc_b)^2 )
    ece = torch.sqrt(((net_bin_confidence - net_bin_accuracy) ** 2 * bin_weights).sum())

    return ece

if __name__ == "__main__":
    # Example usage
    n = 1000
    labels = np.random.binomial(1, 0.7, n)
    #preds = np.clip(np.random.normal(0.7, 0.15, n), 0, 1)
    preds = torch.randn(n,2)
    preds = torch.softmax(preds, dim=1)

    ece = soft_binning_ece(
        predictions=preds,
        labels=labels,
        soft_binning_bins=10,
        soft_binning_use_decay=True,
        soft_binning_decay_factor=0.9,
        soft_binning_temp=0.01
    )
    print(f"Soft-binned ECE: {ece.item():.6f}")