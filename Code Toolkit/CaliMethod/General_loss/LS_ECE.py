import torch
import numpy as np

def gaussian_kernel(x, sigma):
    """Compute Gaussian kernel function"""
    return 1/(sigma * np.sqrt(2 * np.pi)) * torch.exp(-torch.square(x) / (2 * sigma ** 2))

def kernel_reg(logits, labels, ts, sigma):
    """Compute kernel regression"""
    total = gaussian_kernel(ts - logits, sigma)
    return (total * labels).sum(dim=0) / total.sum(dim=0)

def logit_smoothed_ece(logits, labels, n_t, sigma):
    """
    Compute logit-smoothed Expected Calibration Error (LS-ECE)
    LS-ECE loss is defined in Eq. (4.1) of https://arxiv.org/abs/2402.10046.
    
    Args:
        logits: Logits tensor of shape (n, 1)
        labels: Labels tensor of shape (n, 1)
        n_t: Number of sampling points
        sigma: Bandwidth parameter for Gaussian kernel
    
    Returns:
        Smoothed ECE value
    """
    # Expect logits to be shape (n, 1) and labels to be shape (n, 1)
    emp_sample = torch.randint(len(logits), (n_t,))
    ts = logits[emp_sample].squeeze(dim=1) + sigma * torch.randn(n_t,)
    ests = kernel_reg(logits, labels, ts, sigma)
    return torch.abs((ests - torch.nn.functional.sigmoid(ts))).mean()

if __name__ == "__main__":
    # Debug example
    torch.manual_seed(42)  # Set random seed for reproducible results
    
    # Generate example data
    n_samples = 1000
    n_t = 500
    sigma = 0.1
    
    # Generate logits and labels
    logits = torch.randn(n_samples, 1)  # Shape (n, 1)
    true_probs = torch.sigmoid(logits)  # Convert to probabilities
    
    # Generate binary classification labels based on probabilities
    labels = torch.bernoulli(true_probs).float()
    
    # Compute smoothed ECE
    ece_value = logit_smoothed_ece(logits, labels, n_t, sigma)
    print(f"\nLogit-smoothed ECE: {ece_value.item():.6f}")
    
    # Test the effect of different sigma values
    print(f"\nECE with different sigma values:")
    sigma_values = [0.05, 0.1, 0.2, 0.3]
    for sigma_val in sigma_values:  
        ece = logit_smoothed_ece(logits, labels, n_t, sigma_val)
        print(f"  sigma={sigma_val}: {ece.item():.6f}")
    
    # Test the effect of different numbers of sampling points
    print(f"\nECE with different numbers of sampling points (sigma=0.1):")
    n_t_values = [100, 500, 1000, 2000]
    for n_t_val in n_t_values:
        ece = logit_smoothed_ece(logits, labels, n_t_val, sigma)
        print(f"  n_t={n_t_val}: {ece.item():.6f}")
    