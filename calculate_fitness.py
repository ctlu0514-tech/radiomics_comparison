import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score  <--- 移除这一行
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# +++ 添加这两个导入 +++
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1):
    """
    计算种群适应度 - 升级版
    ...
    """

    # +++ 定义一个可重用的分割器 +++
    # n_splits=1: 只分割1次 (快 3 倍)
    # test_size=0.3: 70% 训练, 30% 验证
    # random_state=42: 确保*所有*染色体都使用*同*一个分割方案，
    #                这对于公平比较它们之间的适应度至关重要。
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    # 并行计算 (传入 sss)
    return Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr, X, y, similarity_matrix, sss) for chr in chromosomes)


# --- 修改 _process_chromosome 以接收 X, y, sim_matrix 和 sss ---
def _process_chromosome(chromosome, X, y, similarity_matrix, sss):
    """
    (修改了函数签名以接收更多参数)
    """
    chromosome_arr = np.array(chromosome)
    selected_mask = chromosome_arr.astype(bool)
    selected_features = np.where(selected_mask)[0]
  
    if len(selected_features) == 0:
        return 0.0
    
    X_sub = X[:, selected_features]
  
    clf = LogisticRegression(
        solver='liblinear', 
        class_weight='balanced',
        random_state=42, 
        max_iter=1000
    )

    # --- [核心改进：用 StratifiedShuffleSplit 替换 cv=3] ---
    try:
        # 我们不再使用 cross_val_score，而是手动执行这个单一的、分层的分割
        # 'next' 会获取 sss.split 提供的第一个 (也是唯一一个) 分割
        train_idx, test_idx = next(sss.split(X_sub, y))
        
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 在 70% 的数据上训练
        clf.fit(X_train, y_train)
        
        # 在 30% 的数据上评估 AUC
        y_proba = clf.predict_proba(X_test)[:, 1]
        main_score = roc_auc_score(y_test, y_proba)
        
    except ValueError:
        # 极少数情况（如分割后只有一类样本）
        main_score = 0.0
    # --- [修改结束] ---

    # --- 冗余度惩罚 (保持原逻辑) ---
    n = len(selected_features)
    if n < 2:
        denominator = 0.0
    else:
        sub_sim = similarity_matrix[selected_features, :][:, selected_features]
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        total_sim = sub_sim[triu_idx].sum()
        denominator = (2 * total_sim) / (n * (n - 1))
    
    w = 0.9  
    fitness = (w * main_score) + ((1 - w) * (1 - denominator))

    return fitness
