# -*- coding: utf-8 -*-
# @Time    : 2025/09/30 16:26 
# @File    : Binomial_Process_Sampling.py
# Reference paper: "Combining Priors with Experience: Confidence Calibration Based on Binomial Process Modeling"

import numpy as np
import mpmath


def logit(s):
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    if isinstance(s,np.ndarray):
        return np.log(s_clipped / (1 - s_clipped))
    else:
        return mpmath.log(s_clipped / (1 - s_clipped))

def sigmoid(s):
    if isinstance(s, np.ndarray):
        return 1/(1+np.exp(-s))
    else:
        return 1/(1+mpmath.exp(-s))

# logflip(x) = log(1 - x), and its inverse
def logflip(s):
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    if isinstance(s, np.ndarray):
        return np.log(1.0 - s_clipped)
    else:
        return mpmath.log(1.0 - s_clipped)

def inv_logflip(t):
    if isinstance(t, np.ndarray):
        p = 1.0 - np.exp(t)
        return p
    else:
        p = 1.0 - mpmath.e**(t)
        return p


class logit_logit:
    '''
    link functions is logit
    transform functions is logit
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self,beta0: float = -0.88, beta1: float = 0.49) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def compute(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logit(x))
        return p


class logflip_logflip:
    """
    link:    logflip^{-1}(t) = 1 - exp(t)
    transform: logflip(s) = log(1 - s)
    Corresponds to D2: inv_logflip(beta0 + beta1 * logflip(s))
    """
    def __init__(self, beta0: float = -0.12, beta1: float = 0.58) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def compute(self, x):
        if isinstance(x, list):
            x = np.array(x)
        t = self.beta0 + self.beta1 * float(logflip(x))
        return float(inv_logflip(t))

class log_log:
    """
    link:     exp(·)   (i.e. log^{-1})
    transform: log(s)
    Corresponds to D3: exp(beta0 + beta1 * log(s))
    """
    def __init__(self, beta0: float = -0.03, beta1: float = 1.27) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def compute(self, x):
        if isinstance(x, list):
            x = np.array(x)
        s = np.clip(x, 1e-10, 1 - 1e-10)
        p = np.exp(self.beta0 + self.beta1 * np.log(s))
        return np.clip(p, 1e-10, 1 - 1e-10)

class logit_logflip:
    """
    link:      logit^{-1} = sigmoid
    transform: logflip(s) = log(1 - s)
    Corresponds to D4: sigmoid(beta0 + beta1 * logflip(s)), parameters in Table 2 are (-0.77, -0.80)
    """
    def __init__(self, beta0: float = -0.77, beta1: float = -0.80) -> None:
        self.beta0 = beta0
        self.beta1 = beta1

    def compute(self, x):
        if isinstance(x, list):
            x = np.array(x)
        t = self.beta0 + self.beta1 * float(logflip(x))
        return 1.0 / (1.0 + np.exp(-t))

def Binomial_process_sampling(sampling_num, D="D1"):
    '''
    ========== D1 ==========
    True calibration curve: logit^{-1}(-0.88 + 0.49*logit(s))
    Confidence distribution: Beta(2.77, 0.04)

    ========== D2 ==========
    True calibration curve: logflip^{-1}(-0.12 + 0.58*logflip(s))
    Confidence distribution: Beta(2.17, 0.03)

    ========== D3 (new) ==========
    True calibration curve: exp(-0.03 + 1.27*log(s))
    Confidence distribution: Beta(1.12, 0.11)

    ========== D4 (new) ==========
    True calibration curve: sigmoid(-0.77 - 0.80*log(1 - s))
    Confidence distribution: Beta(1.13, 0.20)

    ========== D5 ==========
    True calibration curve: logit^{-1}(-0.97 + 0.34*logit(s))
    Confidence distribution: Beta(1.19, 0.22)
    '''
    if D == "D1":
        Ps_fit_fun = logit_logit(beta0=-0.88, beta1=0.49)  # true calibration curve
        confidences = np.random.beta(2.77, 0.04, size=sampling_num)
    elif D == "D2":
        Ps_fit_fun = logflip_logflip(beta0=-0.12, beta1=0.58)
        confidences = np.random.beta(2.17, 0.03, size=sampling_num)
    elif D == "D3": 
        Ps_fit_fun = log_log(beta0=-0.03, beta1=1.27)
        confidences = np.random.beta(1.12, 0.11, size=sampling_num)
    elif D == "D4":
        Ps_fit_fun = logit_logflip(beta0=-0.77, beta1=-0.80)
        confidences = np.random.beta(1.13, 0.20, size=sampling_num)
    elif D == "D5":
        Ps_fit_fun = logit_logit(beta0=-0.97, beta1=0.34)
        confidences = np.random.beta(1.19, 0.22, size=sampling_num)

    confidences = np.sort(confidences)
    sample_list = []
    for i in range(len(confidences)):
        Ps = Ps_fit_fun.compute(confidences[i])
        sample = np.random.binomial(1, Ps)
        sample_list.append(sample)

    ys = np.array(sample_list)
    return confidences, ys

if __name__ == "__main__":
    sampling_num = 10000
    confidences, ys = Binomial_process_sampling(sampling_num)  
    print("confidences:",confidences)
    print("ys:",ys)