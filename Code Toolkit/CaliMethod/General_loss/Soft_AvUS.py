import torch
import math

EPS = 1e-5

def get_soft_avuc_loss(
        probabilities,
        labels,
        soft_avuc_use_deprecated_v0,
        soft_avuc_temp,
        soft_avuc_theta,
):
    """Computes and returns the soft AvUC loss tensor (PyTorch).

    Soft AvUC loss is defined in Eq. (15) of https://arxiv.org/pdf/2108.00106.pdf.

    Args:
        probabilities: (N, C) predicted probabilities.
        labels: (N,) integer labels in [0, C).
        soft_avuc_use_deprecated_v0: whether to use the deprecated v0 formulation.
        soft_avuc_temp: temperature > 0 (T in Eq. 15).
        soft_avuc_theta: threshold in (0,1) (kappa in Eq. 15).

    Returns:
        0-dim torch.Tensor containing the soft AvUC loss.
    """

    # Convert inputs to tensors and unify dtype/device
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

    # Accuracies (1.0 if predicted class equals label)
    preds = probabilities.argmax(dim=1)
    accuracies = (preds == labels).to(dtype=dtype)  # [N]

    # Entropy with epsilon-smoothing (nats)
    C = probabilities.size(1)
    uniform = torch.full((C,), 1.0 / C, dtype=dtype, device=device)
    log_safe_prob = (1.0 - EPS) * probabilities + EPS * uniform.unsqueeze(0)
    log_prob = torch.log(log_safe_prob)
    entropies = -(log_safe_prob * log_prob).sum(dim=1)  # [N]

    entmax = math.log(C)

    def soft_uncertainty(e: torch.Tensor, temp: float, theta: float) -> torch.Tensor:
        # e expected in [0,1]; returns probability of "uncertain" class
        return torch.sigmoid((1.0 / temp) * torch.log(e * (1.0 - theta) / ((1.0 - e) * theta)))

    if soft_avuc_use_deprecated_v0:
        # Softmax over two scores: uncertain vs certain
        xus = -((entropies - entmax) ** 2)
        xcs = -(entropies ** 2)
        qucs = torch.softmax(torch.stack([xus, xcs], dim=1), dim=1)  # [N,2]
        qus = qucs[:, 0]  # uncertain membership
        qcs = qucs[:, 1]  # certain membership
    else:
        eus = entropies / entmax
        qus = soft_uncertainty(eus, temp=soft_avuc_temp, theta=soft_avuc_theta)
        qcs = 1.0 - qus

    tanh_ent = torch.tanh(entropies)
    one = torch.tensor(1.0, dtype=dtype, device=device)

    acc_pos = (accuracies > 0.5)
    acc_neg = ~acc_pos

    # Weighted sums
    nac_diff = (qcs * (one - tanh_ent) * acc_pos.to(dtype)).sum()
    nau_diff = (qus * tanh_ent * acc_pos.to(dtype)).sum()
    nic_diff = (qcs * (one - tanh_ent) * acc_neg.to(dtype)).sum()
    niu_diff = (qus * tanh_ent * acc_neg.to(dtype)).sum()

    denom = torch.clamp(nac_diff + niu_diff, min=EPS)
    avuc_loss = torch.log(one + (nau_diff + nic_diff) / denom)

    return avuc_loss

if __name__ == "__main__":
    # Minimal example to validate get_soft_avuc_loss
    torch.manual_seed(0)
    N, C = 512, 5

    # Synthetic labels and biased logits (towards the true class)
    labels = torch.randint(low=0, high=C, size=(N,))
    logits = torch.randn(N, C)
    logits[torch.arange(N), labels] += 1.5  # encourage correct predictions

    # Convert to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Parameters for the soft AvUC (new formulation)
    T = 0.5      # temperature > 0
    kappa = 0.5  # threshold in (0,1)

    loss_new = get_soft_avuc_loss(
        probabilities=probabilities,
        labels=labels,
        soft_avuc_use_deprecated_v0=False,
        soft_avuc_temp=T,
        soft_avuc_theta=kappa,
    )

    # Deprecated v0 formulation for comparison
    loss_v0 = get_soft_avuc_loss(
        probabilities=probabilities,
        labels=labels,
        soft_avuc_use_deprecated_v0=True,
        soft_avuc_temp=T,
        soft_avuc_theta=kappa,
    )

    with torch.no_grad():
        acc = (probabilities.argmax(dim=1) == labels).float().mean().item()
        # Entropy for reference (using the same epsilon smoothing as in the function)
        uniform = torch.full((C,), 1.0 / C, dtype=probabilities.dtype, device=probabilities.device)
        log_safe_prob = (1.0 - EPS) * probabilities + EPS * uniform.unsqueeze(0)
        entropies = -(log_safe_prob * torch.log(log_safe_prob)).sum(dim=1)
        print(f"Soft-AvUC (new): {loss_new.item():.6f}")
        print(f"Soft-AvUC (v0):  {loss_v0.item():.6f}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Mean entropy: {entropies.mean().item():.4f} nats; log(C) = {math.log(C):.4f} nats")

