import numpy as np
from scipy.stats import pearsonr
from softmax_scaling import softmax_scaling

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def compute_similarity_matrix(X_train):
#     """
#     计算特征间相似性矩阵。

#     参数：
#         X_train (numpy.ndarray): 训练数据，形状为 (n_samples, n_features)。

#     返回：
#         numpy.ndarray: 特征间的相似性矩阵，形状为 (n_features, n_features)。
#     """
#     # 转置数据集，以便特征在行方向
#     X_transposed = X_train.T

#     # 使用余弦相似性计算相似性矩阵
#     similarity_matrix = cosine_similarity(X_transposed)

#     # 调用 softmax_scaling 进行归一化
#     normalized_matrix = softmax_scaling(similarity_matrix)
    
#     return normalized_matrix

def compute_similarity_matrix(X_train):
    """
    计算特征间 Pearson 相关系数矩阵，并进行 softmax 归一化。

    参数：
        X_train (numpy.ndarray): 训练数据，形状为 (n_samples, n_features)。

    返回：
        numpy.ndarray: 归一化的 Pearson 相似性矩阵，形状为 (n_features, n_features)。
    """
    # 计算标准化数据（z-score）
    X_train_standardized = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

    # 计算 Pearson 相关系数矩阵
    similarity_matrix = np.corrcoef(X_train_standardized, rowvar=False)

    # 调用 softmax_scaling 进行归一化
    # normalized_matrix = softmax_scaling(similarity_matrix)

    return similarity_matrix