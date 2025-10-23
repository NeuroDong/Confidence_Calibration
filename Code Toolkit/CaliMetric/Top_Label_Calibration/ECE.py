import numpy as np

def equal_interval_ece(confidences, correct, num_bins=15, return_bins=False, fill_empty_with=np.nan):
    """
    计算等宽分箱的 Expected Calibration Error (ECE)
    
    参数
    ----
    confidences : (N,) array-like
        每个样本的预测类别的置信度（多分类为 max prob；二分类可用 max(p, 1-p) 或对应预测类的概率）。
    correct : (N,) array-like of bool or {0,1}
        每个样本是否预测正确的指示。
    num_bins : int
        分箱数量。
    return_bins : bool
        是否返回 bin 边界数组。
    fill_empty_with : float
        空 bin 的占位值（用于返回的 bin 平均值列表），可用 0 或 np.nan。
    
    返回
    ----
    ece : float
        期望校准误差。
    (可选) bins : (num_bins+1,) array
        分箱边界。
    bin_confs : list of length num_bins
        各 bin 的平均置信度。
    bin_accs : list of length num_bins
        各 bin 的平均准确率（正确率）。
    bin_counts : list of length num_bins
        各 bin 的样本数。
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)  # 允许 bool 或 0/1

    assert confidences.ndim == 1 and correct.ndim == 1 and confidences.shape[0] == correct.shape[0], "confidences 和 correct 应为同长度的一维数组"

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    N = len(confidences)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(correct[in_bin]))
            ece += (count / N) * abs(avg_acc - avg_conf)

            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
            bin_counts.append(count)
        else:
            bin_confs.append(fill_empty_with)
            bin_accs.append(fill_empty_with)
            bin_counts.append(0)

    if return_bins:
        return ece, bins, bin_confs, bin_accs, bin_counts
    else:
        return ece
    
import numpy as np

def equal_mass_ece(confidences, correct, num_bins=15, return_bins=False, fill_empty_with=np.nan):
    """
    等样本分箱（equal-mass / equal-frequency）的 Expected Calibration Error (ECE)

    参数
    ----
    confidences : (N,) array-like
        每个样本预测类别的置信度（多分类通常取 max prob；二分类可用 max(p, 1-p) 或预测类对应概率）。
    correct : (N,) array-like of bool or {0,1}
        每个样本是否预测正确。
    num_bins : int
        分箱数量。
    fill_empty_with : float
        空 bin 的占位值（用于返回的 bin 平均值列表），可设为 0 或 np.nan。
    return_bins : bool
        是否返回分箱边界（按实际分组计算，而非 np.quantile）。

    返回
    ----
    ece : float
        期望校准误差。
    (可选) bins : (num_bins+1,) array
        实际用于切分的边界（基于排序后的分段端点）。
    bin_confs : list of length num_bins
        各 bin 的平均置信度。
    bin_accs : list of length num_bins
        各 bin 的平均正确率。
    bin_counts : list of length num_bins
        各 bin 的样本数。
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    assert confidences.ndim == 1 and correct.ndim == 1 and len(confidences) == len(correct), \
        "confidences 和 correct 应为同长度一维数组"

    # 处理 NaN：这里直接抛错，也可选择剔除样本
    if np.isnan(confidences).any() or np.isnan(correct).any():
        raise ValueError("confidences/correct 含 NaN，请先清洗。")

    N = len(confidences)
    if N == 0:
        raise ValueError("空输入。")

    # 按置信度排序并等量切分索引
    sorted_idx = np.argsort(confidences)
    bin_indices = np.array_split(sorted_idx, num_bins)

    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []

    # 基于“实际切分”的边界（更可靠：与 bin_indices 一致）
    # 左端为 0 的下界，右端为每个 bin 的最大置信度
    right_edges = []
    for inds in bin_indices:
        if len(inds) > 0:
            right_edges.append(float(confidences[inds].max()))
        else:
            right_edges.append(np.nan)

    # 构造 bins：左闭右闭，第一格含最小值，最后一格含最大值
    # 左端点取排序后最小值；若存在空 bin，其右端点为 nan
    left_edge = float(confidences[sorted_idx[0]])
    bins = np.empty(num_bins + 1, dtype=float)
    bins[0] = left_edge
    bins[1:] = np.array(right_edges, dtype=float)

    for inds in bin_indices:
        count = len(inds)
        if count > 0:
            avg_conf = float(np.mean(confidences[inds]))
            avg_acc  = float(np.mean(correct[inds]))
            w = count / N
            ece += w * abs(avg_acc - avg_conf)

            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
            bin_counts.append(count)
        else:
            bin_confs.append(fill_empty_with)
            bin_accs.append(fill_empty_with)
            bin_counts.append(0)

    if return_bins:
        return ece, bins, bin_confs, bin_accs, bin_counts
    else:
        return ece
    

if __name__=="__main__":
    np.random.seed(42)
    N = 1000
    probs = np.random.rand(N)  # 模拟预测为正类的概率
    labels = np.random.randint(0, 2, size=N)  # 模拟真实标签

    # 预测类别和 correctness
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(int)
    confidences = np.maximum(probs, 1 - probs)  # 用 max prob 作为置信度

    print("=== 等宽分箱 ECE ===")
    ece_interval, bins_i, confs_i, accs_i, counts_i = equal_interval_ece(confidences, correct, num_bins=10, return_bins=True)
    print(f"ECE: {ece_interval:.4f}")
    print(f"Bins: {bins_i}")
    print(f"Bin Confidences: {np.round(confs_i, 3)}")
    print(f"Bin Accuracies: {np.round(accs_i, 3)}")
    print(f"Bin Counts: {counts_i}\n")

    print("=== 等样本分箱 ECE ===")
    ece_mass, bins_m, confs_m, accs_m, counts_m = equal_mass_ece(confidences, correct, num_bins=10, return_bins=True)
    print(f"ECE: {ece_mass:.4f}")
    print(f"Bins: {bins_m}")
    print(f"Bin Confidences: {np.round(confs_m, 3)}")
    print(f"Bin Accuracies: {np.round(accs_m, 3)}")
    print(f"Bin Counts: {counts_m}")

