'''
Reference: "Calibration of Neural Networks using Splines"
'''

import numpy as np
import torch
import torch.nn.functional as F

def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a

def KS_error(scores, labels):
    # KS stands for Kolmogorov-Smirnov

    indices = np.argsort(scores)
    scores = scores[indices]
    labels = labels[indices]

    # Change to numpy, then this will work
    scores = ensure_numpy (scores)
    labels = ensure_numpy (labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy   = np.cumsum(labels) / nsamples

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    return KS_error_max

if __name__=="__main__":
    np.random.seed(42)
    N = 1000
    probs = np.random.rand(N)  # 模拟预测为正类的概率
    labels = np.random.randint(0, 2, size=N)  # 模拟真实标签

    # 预测类别和 correctness
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(int)
    confidences = np.maximum(probs, 1 - probs)  # 用 max prob 作为置信度

    ks_error = KS_error(confidences, correct)
    print(f"ks_error: {ks_error:.4f}")