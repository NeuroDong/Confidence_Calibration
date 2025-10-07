'''
# -*- coding: utf-8 -*-
# @Time    : 2025/10/07 21:00
# @File    : scaling_binning.py
# Reference paper: "Verified Uncertainty Calibration"
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, TypeVar
import bisect

Data = List[Tuple[float, float]]  # List of (predicted_probability, true_label).
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0.
BinnedData = List[Data]  # binned_data[i] contains the data in bin i.
T = TypeVar('T')

def get_platt_scaler(model_probs, labels, get_clf=False):
    clf = LogisticRegression(C=1e10, solver='lbfgs')
    eps = 1e-12
    model_probs = model_probs.astype(dtype=np.float64)
    model_probs = np.expand_dims(model_probs, axis=-1)
    model_probs = np.clip(model_probs, eps, 1 - eps)
    model_probs = np.log(model_probs / (1 - model_probs))
    clf.fit(model_probs, labels)
    def calibrator(probs):
        x = np.array(probs, dtype=np.float64)
        x = np.clip(x, eps, 1 - eps)
        x = np.log(x / (1 - x))
        x = x * clf.coef_[0] + clf.intercept_
        output = 1 / (1 + np.exp(-x))
        return output
    if get_clf:
        return calibrator, clf
    return calibrator

def split(sequence: List[T], parts: int) -> List[List[T]]:
    assert parts <= len(sequence)
    array_splits = np.array_split(sequence, parts)
    splits = [list(l) for l in array_splits]
    assert len(splits) == parts
    return splits

def get_equal_bins(probs: List[float], num_bins: int=10) -> Bins:
    """Get bins that contain approximately an equal number of data points."""
    sorted_probs = sorted(probs)
    if num_bins > len(sorted_probs):
        num_bins = len(sorted_probs)
    binned_data = split(sorted_probs, num_bins)
    bins: Bins = []
    for i in range(len(binned_data) - 1):
        last_prob = binned_data[i][-1]
        next_first_prob = binned_data[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    bins = sorted(list(set(bins)))
    return bins

def get_bin(pred_prob: float, bins: List[float]) -> int:
    """Get the index of the bin that pred_prob belongs in."""
    assert 0.0 <= pred_prob <= 1.0
    assert bins[-1] == 1.0
    return bisect.bisect_left(bins, pred_prob)

def get_histogram_calibrator(model_probs, values, bins):
    binned_values = [[] for _ in range(len(bins))]
    for prob, value in zip(model_probs, values):
        bin_idx = get_bin(prob, bins)
        binned_values[bin_idx].append(float(value))
    def safe_mean(values, bin_idx):
        if len(values) == 0:
            if bin_idx == 0:
                return float(bins[0]) / 2.0
            return float(bins[bin_idx] + bins[bin_idx - 1]) / 2.0
        return np.mean(values)
    bin_means = [safe_mean(values, bidx) for values, bidx in zip(binned_values, range(len(bins)))]
    bin_means = np.array(bin_means)
    def calibrator(probs):
        indices = np.searchsorted(bins, probs)
        return bin_means[indices]
    return calibrator

class PlattBinnerCalibrator:
    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, zs, ys):
        self._platt = get_platt_scaler(zs, ys)
        platt_probs = self._platt(zs)
        bins = get_equal_bins(platt_probs, num_bins=self._num_bins)
        self._discrete_calibrator = get_histogram_calibrator(platt_probs, platt_probs, bins)

    def calibrate(self, zs):
        platt_probs = self._platt(zs)
        return self._discrete_calibrator(platt_probs)
    
if __name__ == "__main__":
    import random
    random.seed(0)
    np.random.seed(0)
    zs = np.array([random.random() for _ in range(1000)])
    ys = np.array([1 if z > 0.5 else 0 for z in zs])
    calibrator = PlattBinnerCalibrator(num_calibration=800, num_bins=10)
    calibrator.train_calibration(zs[:800], ys[:800])
    calibrated = calibrator.calibrate(zs[800:])
    print(calibrated)