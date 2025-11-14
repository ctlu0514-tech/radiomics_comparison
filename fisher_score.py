import numpy as np

def compute_fisher_score(X, y):
    """
    计算每个特征的 Fisher Score。
    Args:
        X (numpy.ndarray): 特征矩阵，形状为 (样本数, 特征数)。
        y (numpy.ndarray): 标签数组，形状为 (样本数, ) 或 (样本数, 1)。
    Returns:
        numpy.ndarray: 每个特征的 Fisher Score。
    """
    # 如果 y 是二维数组，将其转换为一维数组
    if y.ndim == 2:
        y = y.squeeze()  # 或者使用 y.squeeze()

    scores = []  # 用于存储每个特征的 Fisher Score
    classes = np.unique(y)  # 提取所有唯一的类别标签
    for i in range(X.shape[1]):  # 遍历每个特征
        numerator = 0  # 初始化分子：类间差异
        denominator = 0  # 初始化分母：类内差异
        for c in classes:  # 遍历每个类别
            X_c = X[y == c, i]  # 提取类别 c 中特征 i 的所有样本值
            n_c = len(X_c)  # 类别 c 的样本数量
            mean_c = np.mean(X_c)  # 类别 c 中特征 i 的均值
            std_c = np.std(X_c)  # 类别 c 中特征 i 的标准差
            numerator += n_c * (mean_c - np.mean(X[:, i]))**2
            denominator += n_c * (std_c**2)
        scores.append(numerator / (denominator + 1e-10))  # 防止分母为零
    return np.array(scores)

# ------------------- 测试用例 -------------------
if __name__ == "__main__":
    # 测试数据（论文中的例子）
    X_test = np.array([
        [90], [85], [95],  # 学霸组（A类）
        [50], [60], [55]   # 学渣组（B类）
    ])
    y_test = np.array([0, 0, 0, 1, 1, 1])
  
    # 运行计算
    scores = compute_fisher_score(X_test, y_test)
  
    # 验证结果（预期值 18.375）
    print(f"测试特征的Fisher Score: {scores[0]:.3f}")
    # 输出：测试特征的Fisher Score: 18.375

    # 测试无效特征（所有样本值相同）
    X_bad = np.array([[1], [1], [1], [1], [1]])
    y_bad = np.array([0, 0, 1, 1, 1])
    print("无效特征的Fisher Score:", compute_fisher_score(X_bad, y_bad))
    # 输出：无效特征的Fisher Score: [0.]