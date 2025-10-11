
import torch
import numpy as np
import math

EPS = 1e-5

def get_avuc_loss(
    probabilities,
    labels,
    avuc_stop_prob_gradients,
    avuc_entropy_threshold,
):
  """Computes and returns the AvUC loss tensor (PyTorch).

  AvUC loss is defined in Eq. (3):
  https://arxiv.org/pdf/2012.07923.pdf. The gradient-stopping variant is in
  the appendix of https://arxiv.org/pdf/2108.00106.pdf.

  Args:
    probabilities: (N, C) predicted class probabilities.
    labels: (N,) integer labels in [0, C).
    avuc_stop_prob_gradients: whether to stop gradients through confidences.
    avuc_entropy_threshold: entropy threshold u_th.

  Returns:
    0-dim torch.Tensor: the AvUC loss value.
  """

  # Convert to tensors and unify dtype/device
  if not isinstance(probabilities, torch.Tensor):
    probabilities = torch.as_tensor(probabilities)
  if not isinstance(labels, torch.Tensor):
    labels = torch.as_tensor(labels)

  if probabilities.dim() != 2:
    probabilities = probabilities.reshape(probabilities.shape[0], -1)
  labels = labels.reshape(-1)

  if probabilities.size(0) != labels.size(0):
    raise ValueError("probabilities and labels must have the same batch size")

  if not probabilities.is_floating_point():
    probabilities = probabilities.to(dtype=torch.get_default_dtype())
  dtype = probabilities.dtype
  device = probabilities.device
  labels = labels.to(device=device, dtype=torch.long)

  # Confidences (max prob per sample); optionally stop gradients
  confidences = probabilities.max(dim=1).values  # [N]
  if avuc_stop_prob_gradients:
    confidences = confidences.detach()

  # Accuracies: 1.0 if predicted class equals label, else 0.0
  preds = probabilities.argmax(dim=1)
  accuracies = (preds == labels).to(dtype=dtype)  # [N]

  # Entropy with epsilon-smoothing for log safety
  C = probabilities.size(1)
  uniform = torch.full((C,), 1.0 / C, dtype=dtype, device=device)
  log_safe_prob = (1.0 - EPS) * probabilities + EPS * uniform.unsqueeze(0)
  log_prob = torch.log(log_safe_prob)
  entropies = -(log_safe_prob * log_prob).sum(dim=1)  # [N]

  # Masks for the four regions
  ent_low = entropies < avuc_entropy_threshold
  ent_high = ~ent_low
  acc_pos = accuracies > 0.5
  acc_neg = ~acc_pos

  tanh_ent = torch.tanh(entropies)

  one = torch.tensor(1.0, dtype=dtype, device=device)

  # Weighted sums (vectorized)
  nac_diff = (confidences * (one - tanh_ent) * (ent_low & acc_pos).to(dtype)).sum()
  nau_diff = (confidences * (tanh_ent) * (ent_high & acc_pos).to(dtype)).sum()
  nic_diff = ((one - confidences) * (one - tanh_ent) * (ent_low & acc_neg).to(dtype)).sum()
  niu_diff = ((one - confidences) * (tanh_ent) * (ent_high & acc_neg).to(dtype)).sum()

  denom = torch.clamp(nac_diff + niu_diff, min=EPS)
  avuc_loss = torch.log(one + (nau_diff + nic_diff) / denom)

  return avuc_loss

if __name__ == "__main__":
  # Minimal example to validate get_avuc_loss
  torch.manual_seed(0)
  N, C = 512, 5

  # Create synthetic labels and biased logits (towards the true class)
  labels = torch.randint(low=0, high=C, size=(N,))
  logits = torch.randn(N, C)
  logits[torch.arange(N), labels] += 1.5  # encourage correct predictions

  # Convert to probabilities
  probabilities = torch.softmax(logits, dim=1)

  # Choose an entropy threshold in nats (e.g., 0.5 * ln(C))
  u_th = 0.5 * math.log(C)

  # Compute AvUC loss (with gradient stopping on confidences)
  loss = get_avuc_loss(
      probabilities=probabilities,
      labels=labels,
      avuc_stop_prob_gradients=True,
      avuc_entropy_threshold=u_th,
  )

  # Report basic stats for a quick sanity check
  with torch.no_grad():
    acc = (probabilities.argmax(dim=1) == labels).float().mean().item()
    # Entropy for reference (using the same epsilon smoothing as in the function)
    uniform = torch.full((C,), 1.0 / C, dtype=probabilities.dtype, device=probabilities.device)
    log_safe_prob = (1.0 - EPS) * probabilities + EPS * uniform.unsqueeze(0)
    entropies = -(log_safe_prob * torch.log(log_safe_prob)).sum(dim=1)
    print(f"AvUC loss: {loss.item():.6f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Mean entropy: {entropies.mean().item():.4f} nats; threshold u_th: {u_th:.4f} nats")