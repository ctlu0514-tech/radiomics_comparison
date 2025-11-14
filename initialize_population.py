import numpy as np

def initialize_population(num_features, clusters, omega, population_size):
    """
    初始化种群。
    Args:
        num_features (int): 原始特征总数。
        clusters (list): 每个簇包含的特征索引列表。
        omega (int): 每个簇中选取的特征数百分比。
        population_size (int): 种群大小。
    Returns:
        list: 初始化的种群，每个个体是一个染色体（特征选择向量）。
    """
    population = []
    
    for _ in range(population_size):
        chromosome = np.zeros(num_features, dtype=int)  # 初始化染色体为全0
        for cluster_features in clusters:  # 遍历每个簇
            # 在该簇中随机选择 ω% 个特征
            selected_features = np.random.choice(cluster_features, size=int(omega * len(cluster_features)), replace=False)
            chromosome[selected_features] = 1  # 将选中的特征置为1
        population.append(chromosome)
    
    return population
