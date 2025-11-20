import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DualFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        """Computes and returns the Dual Focal Loss tensor (PyTorch).

        Dual Focal Loss is defined in Eq. (4) of https://arxiv.org/abs/2305.13665.

        Reference:
            [1] Wang, Z., et al. (2023). Dual Focal Loss for Calibration.
                arXiv: https://arxiv.org/abs/2305.13665

        Args:
            input: (N, C) predicted logits.
            target: (N,) integer labels in [0, C).

        Returns:
            torch.Tensor containing the Dual Focal Loss.
        """
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()

if __name__ == "__main__":
    # Minimal example to validate DualFocalLoss
    torch.manual_seed(0)
    N, C = 512, 5

    # Synthetic labels and biased logits (towards the true class)
    labels = torch.randint(low=0, high=C, size=(N,))
    logits = torch.randn(N, C)
    logits[torch.arange(N), labels] += 1.5  # encourage correct predictions

    # Convert to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Parameters for Dual Focal Loss
    gamma_values = [0, 1, 2]  # different gamma parameters

    # Test different gamma values
    for gamma in gamma_values:
        loss_fn = DualFocalLoss(gamma=gamma, size_average=True)
        loss = loss_fn(logits, labels)

        with torch.no_grad():
            # Calculate accuracy
            preds = probabilities.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            
            # Calculate confidence (probability at true labels)
            confidences = probabilities[torch.arange(N), labels]
            avg_confidence = confidences.mean().item()
            
            # Calculate entropy for reference
            EPS = 1e-5
            uniform = torch.full((C,), 1.0 / C, dtype=probabilities.dtype, device=probabilities.device)
            log_safe_prob = (1.0 - EPS) * probabilities + EPS * uniform.unsqueeze(0)
            entropies = -(log_safe_prob * torch.log(log_safe_prob)).sum(dim=1)
            mean_entropy = entropies.mean().item()
            
            # Compare with standard cross entropy
            ce_loss = F.cross_entropy(logits, labels).item()

            print(f"DualFocalLoss (gamma={gamma}): {loss.item():.6f}")
            print(f"CrossEntropy Loss: {ce_loss:.6f}")
            print(f"Accuracy: {acc * 100:.2f}%")
            print(f"Average confidence: {avg_confidence:.4f}")
            print(f"Mean entropy: {mean_entropy:.4f} nats; log(C) = {torch.log(torch.tensor(C, dtype=torch.float)).item():.4f} nats")
