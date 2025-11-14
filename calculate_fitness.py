import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations
import networkx as nx
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed  # 新增并行库导入

def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1):
    """
    参数：
    - chromosomes: 染色体列表 [n_chromosomes, n_features]
    - X: 已标准化的特征矩阵 [n_samples, n_features]
    - y: 目标标签 [n_samples]
    - similarity_matrix: 预计算的全局特征相似性矩阵 [n_features, n_features]
    - n_jobs: 并行任务数（-1使用全部核心）
  
    返回：
    - fitness_values: 适应度值列表 [n_chromosomes]
    """

    # 定义单个染色体处理函数
    def _process_chromosome(chromosome):
        chromosome_arr = np.array(chromosome)
        selected_mask = chromosome_arr.astype(bool)
        selected_features = np.where(selected_mask)[0]
      
        X_sub = X[:, selected_features]
      
        X_train, X_val, y_train, y_val = train_test_split(
            X_sub, y, 
            test_size=0.3, 
            stratify=y, 
            random_state=42  # 保持固定随机种子保证可重复性
        )
      
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        ca = knn.score(X_val, y_val)

        #从全局相似性矩阵中提取仅包含当前染色体选中特征的子矩阵
        sub_sim = similarity_matrix[selected_features, :][:, selected_features]
        #生成上三角索引
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        total_sim = sub_sim[triu_idx].sum()
      
        n = len(selected_features)
        denominator = (2 * total_sim) / (n * (n - 1))
        fitness = ca / (denominator + 1e-9)
        return fitness

    # 使用并行处理替代原有循环
    return Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)

# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import networkx as nx
# from multiprocessing import Pool
# import os
# os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 根据CPU核心数调整

# # 全局缓存字典
# knn_cache = {}

# def calculate_chromosome_fitness(args):
#     chromosome, X_train, y_train, similarity_matrix = args
#     chromosome_array = np.asarray(chromosome).astype(bool)
#     selected_features = np.where(chromosome_array)[0]
#     num_selected = len(selected_features)
  
#     # 计算相似度和
#     if num_selected >= 2:
#         sub_matrix = similarity_matrix[np.ix_(selected_features, selected_features)]
#         sim_sum = (sub_matrix.sum() - np.diag(sub_matrix).sum()) / 2
#     else:
#         sim_sum = 0
  
#     # 计算分类准确率
#     if num_selected == 0:
#         C4 = 0
#     else:
#         selected_tuple = tuple(sorted(selected_features))
#         if selected_tuple in knn_cache:
#             C4 = knn_cache[selected_tuple]
#         else:
#             X_train_selected = X_train[:, chromosome_array]
#             knn = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
#             knn.fit(X_train_selected, y_train)
#             y_pred = knn.predict(X_train_selected)
#             C4 = np.mean(y_pred == y_train)
#             knn_cache[selected_tuple] = C4
  
#     # 计算适应度值
#     if num_selected < 2:
#         fitness = C4
#     else:
#         denominator = ((2 * sim_sum) / (num_selected * (num_selected - 1))) * sim_sum
#         fitness = C4 / denominator if denominator != 0 else 0
  
#     return fitness

# def calculate_fitness(population, X_train, y_train, similarity_matrix):
#     with Pool(processes=min(4, os.cpu_count())) as pool:
#         args = [(chromosome, X_train, y_train, similarity_matrix) for chromosome in population]
#         fitness_values = pool.map(calculate_chromosome_fitness, args)
#     return fitness_values