import numpy as np

def temperature_scaling_grid_search(logits, labels, loss, T_min=0.05, T_max=5.0, num_points=1000):
    """
    Input：
    logits: ndarray [N, C]， logits
    labels: ndarray [N]，True label (integer)
    loss: Function Objects. The input is probs and labels, where the shape of probs is [N,C] and the shape of labels is [C].
    T_min, T_max: Search Scope
    num_points: Grid points
    
    Return：
    best_T: Optimal temperature
    best_loss: Corresponding Calibration Loss
    """
    # 转换为 numpy

    logits = np.array(logits)
    labels = np.array(labels)
    
    Ts = np.linspace(T_min, T_max, num_points)
    
    best_T = None
    best_loss = float('inf')
    
    for T in Ts:
        scaled_logits = logits / T
        # softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # compute loss
        loss_value = loss(probs, labels)
        
        if loss_value < best_loss:
            best_loss = loss_value
            best_T = T

    return best_T, best_loss

if __name__=="__main__":
    # 示例
    logits_val = np.random.randn(500, 10)  # 假设验证集 logits
    labels_val = np.random.randint(0, 10, size=500)

    import sys
    import os
    proj_root = r"c:\Users\admin\Documents\Project\Confidence_Calibration\Code Toolkit"
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from CaliMethod.General_loss import soft_binning_ece, DECE
    from CaliMetric.Multi_Calibration import get_ece_kde

    best_T, best_loss = temperature_scaling_grid_search(logits_val, labels_val, soft_binning_ece)
    print(f"最佳温度 T = {best_T:.4f}, 对应 loss = {best_loss:.4f}")

    dece = DECE()
    best_T, best_loss = temperature_scaling_grid_search(logits_val, labels_val, dece)
    print(f"最佳温度 T = {best_T:.4f}, 对应 loss = {best_loss:.4f}")

    best_T, best_loss = temperature_scaling_grid_search(logits_val, labels_val, get_ece_kde)
    print(f"最佳温度 T = {best_T:.4f}, 对应 loss = {best_loss:.4f}")