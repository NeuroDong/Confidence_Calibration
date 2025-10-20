import torch
import torch.nn as nn
import torch.nn.functional as F


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

class Robust_Sigmoid(torch.autograd.Function):
    """Aiming for a stable sigmoid operator with specified sigma"""

    @staticmethod
    def forward(ctx, input, sigma=1.0, gpu=False):
        """
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        """
        x = input if 1.0 == sigma else sigma * input

        torch_half = torch.cuda.FloatTensor([0.5]) if gpu else torch.FloatTensor([0.5])
        sigmoid_x_pos = torch.where(input > 0, 1.0 / (1.0 + torch.exp(-x)), torch_half)

        exp_x = torch.exp(x)
        sigmoid_x = torch.where(input < 0, exp_x / (1.0 + exp_x), sigmoid_x_pos)

        grad = (
            sigmoid_x * (1.0 - sigmoid_x)
            if 1.0 == sigma
            else sigma * sigmoid_x * (1.0 - sigmoid_x)
        )
        ctx.save_for_backward(grad)

        return sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: backpropagated gradients from upper module(s)
        :return:
        """
        grad = ctx.saved_tensors[0]

        bg = grad_output * grad  # chain rule

        return bg, None, None


# - function: robust_sigmoid-#
robust_sigmoid = Robust_Sigmoid.apply


class DECE(nn.Module):
    """
    Computes DECE loss (differentiable expected calibration error).
    """

    def __init__(self, device = 'cpu', num_bins = 10, t_a = 10., t_b = 1.):
        super(DECE, self).__init__()
        self.device = device
        self.num_bins = num_bins
        self.t_a = t_a
        self.t_b = t_b

    def one_hot(self, indices, depth):
        """
        Returns a one-hot tensor.
        This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        Parameters:
        indices:  a (n_batch, m) Tensor or (m) Tensor.
        depth: a scalar. Represents the depth of the one hot dimension.
        Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
        """
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).to(
            device=self.device
        )
        index = indices.view(indices.size() + torch.Size([1]))
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)

        return encoded_indicies

    def forward(self, input, target):
        '''
        input: 1D tensor (N,C) 
        target: 1D tensor (N,)
        '''
        if not isinstance(input, torch.Tensor):
            input = torch.as_tensor(input)
        if not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target)
        if len(input.shape) == 2:
            if not is_softmax_output(input):
                predicted_probs = torch.softmax(input, dim=1)
            else:
                predicted_probs = input
        else:
            raise ValueError("Expect 2D tensor.")
        confidences = torch.max(predicted_probs, dim=1, keepdim=True)[0] 

        cut_points = torch.linspace(0, 1, self.num_bins + 1)[:-1].to(device=self.device, dtype=confidences.dtype)
        W = torch.reshape(
            torch.linspace(1.0, self.num_bins, self.num_bins).to(device=self.device, dtype=confidences.dtype),
            [1, -1],
        )
        b = torch.cumsum(-cut_points, 0)

        h = torch.matmul(confidences, W) + b
        h = h / self.t_b

        bin_probs = F.softmax(h, dim=1)

        # smoothen the probabilities to avoid zeros
        eps = 1e-6
        bin_probs = bin_probs + eps
        # normalize the probabilities to sum to one across bins
        bin_probs = bin_probs / (1.0 + (self.num_bins + 1) * eps)

        # calculate bin confidences
        bin_confs = torch.div(
            bin_probs.transpose(0, 1).matmul(confidences).view(-1),
            torch.sum(bin_probs, dim=0),
        )
        # all-pairs approach
        batch_pred_diffs = torch.unsqueeze(predicted_probs, dim=2) - torch.unsqueeze(
            predicted_probs, dim=1
        )
        # computing pairwise differences, i.e., Sij or Sxy
        if str(self.device) == "cpu":
            gpu = False
        else:
            gpu = True
        # using {-1.0*} may lead to a poor performance when compared with the above way;
        batch_indicators = robust_sigmoid(
            torch.transpose(batch_pred_diffs, dim0=1, dim1=2), self.t_a, gpu
        )

        # get approximated rank positions, i.e., hat_pi(x)
        ranks_all = torch.sum(batch_indicators, dim=2) + 0.5
        # the ranks go from 1 to C, with 1 being the best rank
        true_ranks = ranks_all[torch.arange(ranks_all.size(0)), target]
        accs = F.relu(2.0 - true_ranks)
        bin_accs = torch.div(
            bin_probs.transpose(0, 1).matmul(accs).view(-1), torch.sum(bin_probs, dim=0)
        )

        # calculate overall ECE for the whole batch
        ece = torch.sum(
            torch.sum(bin_probs, dim=0)
            * torch.abs(bin_accs - bin_confs)
            / bin_probs.shape[0],
            dim=0,
        )
        return ece

if __name__ == "__main__":
    # Quick usage example to validate DECE computes without error.
    # Creates random logits for a small batch and integer targets.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters (keep small for quick test)
    num_bins = 10
    t_a = 10.0
    t_b = 1.0

    # instantiate loss
    dece_loss = DECE(device=device, num_bins=num_bins, t_a=t_a, t_b=t_b)

    # sample batch: batch_size x num_classes
    batch_size = 4
    num_classes = 5

    # random logits and targets
    logits = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)

    # compute loss
    loss = dece_loss(logits, targets)
    print("DECE loss:", loss.item())