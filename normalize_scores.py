import numpy as np

def normalize_scores(scores):
    """
    对分数进行非线性归一化处理（Softmax Scaling）。
    Args:
        scores (numpy.ndarray): 原始分数数组（例如 Fisher Scores 或 Pearson 相似性值）。
    Returns:
        numpy.ndarray: 归一化后的分数。
    """
    mean = np.nanmean(scores)  # 计算均值
    std = np.std(scores)    # 计算标准差
    normalized_scores = 1 / (1 + np.exp(-(scores - mean) / (std)))  # Softmax Scaling
    return normalized_scores