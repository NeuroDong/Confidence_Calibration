# -*- coding: utf-8 -*-
# @Time    : 2025/10/01 16:26 
# @File    : validity_plot.py
# Reference paper: "Distribution-Free Calibration Guarantees for Histogram Binning without Sample Splitting"


import numpy as np

def bin_points(scores, bin_edges):
    assert(bin_edges is not None), "Bins have not been defined"
    scores = scores.squeeze()
    assert(scores.max() <= 1), "Maximum score value is > 1!"
    assert(np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)

def get_binned_probabilities_discrete(y, pred_prob, pred_prob_base = None):
    assert(len(np.unique(pred_prob))
           <= (pred_prob.shape[0]/10)), "Predicted probabilities are not sufficiently discrete; using corresponding continuous method"
    bin_edges = np.sort(np.unique(pred_prob))
    true_n_bins = len(bin_edges)
    pi_pred = np.zeros(true_n_bins)
    pi_base = np.zeros(true_n_bins)
    pi_true = np.zeros(true_n_bins)
    n_elem = np.zeros(true_n_bins)
    bin_assignment = bin_points(pred_prob, bin_edges)

    for i in range(true_n_bins):
        bin_idx = (bin_assignment == i)
        assert(sum(bin_idx) > 0), "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_prob[bin_idx].mean()
        if(pred_prob_base is not None):
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()    

    assert(sum(n_elem) == y.size)

    return n_elem, pi_pred, pi_base, pi_true

def nudge(matrix, delta=1e-10):
    return((matrix + np.random.uniform(low=0, high=delta, size=(matrix.shape)))/(1+delta))

def get_uniform_mass_bins(probs, n_bins):
    assert(probs.size >= n_bins), "Fewer points than bins"
    
    probs_sorted = np.sort(probs)

    # split probabilities into groups of approx equal size
    groups = np.array_split(probs_sorted, n_bins)
    bin_edges = list()
    bin_upper_edges = list()

    for cur_group in range(n_bins-1):
        bin_upper_edges += [max(groups[cur_group])]
    bin_upper_edges += [np.inf]

    return np.array(bin_upper_edges)

def bin_points_uniform(x, n_bins):
    x = nudge(x.squeeze())
    bin_upper_edges = get_uniform_mass_bins(x, n_bins)
    return np.sum(x.reshape((-1, 1)) > bin_upper_edges, axis=1)

def get_binned_probabilities_continuous(y, pred_prob, n_bins, pred_prob_base = None):
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = bin_points_uniform(pred_prob, n_bins)
    
    for i in range(n_bins):
        bin_idx = (bin_assignment == i)
        assert(sum(bin_idx) > 0), "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_prob[bin_idx].mean()
        if(pred_prob_base is not None):
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()    

    assert(sum(n_elem) == y.size)

    return n_elem, pi_pred, pi_base, pi_true

def validity_plot_delta(Delta, n_elem, ax, color=None, quiet=False):
    assert(np.shape(Delta) == np.shape(n_elem))
    assert(np.size(Delta) == np.shape(Delta)[0]), "this function makes a validity plot for a single run, use function validity_plot_aggregate for multiple runs"
    if(np.shape(np.shape(Delta))[0] == 1):
        Delta = np.expand_dims(Delta, axis=0)
        n_elem = np.expand_dims(n_elem, axis=0)        
    n_points = sum(n_elem[0,:])

    cdf = lambda x: np.diag((Delta <= x) @ n_elem.T)/n_points

    dx = 0.001
    xs  = np.arange(0, 1.0, dx)
    xmaxind = xs.size - 1
    ys = np.zeros(xs.shape)
    for i in range(xs.size):
        ys[i] = cdf(xs[i])
        if(ys[i] == 1.0):
            xmaxind = i
            break
    ys[xmaxind:] = 1.0
    if(color is not None):
        handle = ax.plot(xs, ys, color=color)
    else:
        handle = ax.plot(xs, ys)

    ax.set_xlim([0, min(xs[xmaxind] + 500*dx, 1.0)])
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$V(\epsilon)$')
    ax.grid('on')
    return handle[0]

def validity_plot(y, pred_prob, ax, color=None, n_bins=15, quiet=False):
    if (len(np.unique(pred_prob)) <= (pred_prob.shape[0]/10)):
        if (not quiet):
            print("Classifier has discrete output. Further binning not done for making validity plot.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_prob)
    else:
        if (not quiet):
            print("Using {:d} adaptive bins for making validity plot.".format(n_bins))
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_prob, n_bins)

    Delta = np.abs(pi_pred - pi_true)
    validity_plot_delta(Delta, n_elem, ax, color)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    n = 1000
    y = np.random.binomial(1, 0.7, n)
    pred_prob = np.clip(np.random.normal(0.7, 0.15, n), 0, 1)

    fig, ax = plt.subplots(1, 1)
    validity_plot(y, pred_prob, ax)
    plt.show()