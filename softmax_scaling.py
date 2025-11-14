import numpy as np

# def softmax_scaling(scores):
#     """实现非线性归一化
#     Args:
#         scores (np.ndarray): 原始相似性矩阵
#     Returns:
#         np.ndarray: 归一化到[0,1]的矩阵
#     """
#     mu = np.mean(scores)       # 计算全局均值
#     sigma = np.std(scores)     # 计算全局标准差
#     return 1 / (1 + np.exp(-(scores - mu)/sigma))

def softmax_scaling(scores):
    """
    对分数进行 Softmax 缩放处理。
    Args:
        scores (numpy.ndarray): 原始分数数组（例如特征值、得分等）。
    Returns:
        numpy.ndarray: Softmax 缩放后的分数。
    """
    # 计算每个分数的指数
    exp_scores = np.exp(scores - np.max(scores))  # 减去最大值防止溢出
    return exp_scores / np.sum(exp_scores)