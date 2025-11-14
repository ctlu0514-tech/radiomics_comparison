import random
from collections import defaultdict
import numpy as np
from calculate_fitness import calculate_fitness  

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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

# 交叉操作函数
def crossover(parent1, parent2, num_features):
    """
    根据特征数量选择单点或双点交叉操作。
    :param parent1: 父染色体1（列表）
    :param parent2: 父染色体2（列表）
    :param num_features: 数据集的特征总数
    :return: 两个子代染色体
    """
    if num_features <= 20:
        # 单点交叉
        crossover_point = random.randint(1, len(parent1) - 2)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    else:
        # 双点交叉
        point1, point2 = sorted(random.sample(range(1, len(parent1) - 1), 2))
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))

    return child1, child2

# 变异操作函数
def mutation(chromosome, id, mutation_rate=0.05):
    """
    变异操作：随机改变染色体中的某些基因（反转0为1，1为0），并输出变异信息。
    :param chromosome: 染色体（列表）
    :param id: 染色体的编号（序号）
    :param mutation_rate: 变异概率
    :return: 变异后的染色体
    """
    mutated_genes = []  # 用于记录发生变异的基因位置

    # 遍历染色体的每个基因
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:  # 根据变异概率判断是否发生变异
            # 反转基因的值：0 -> 1 或 1 -> 0
            chromosome[i] = 1 - chromosome[i]
            mutated_genes.append(i)  # 记录发生变异的基因位置
    
    return chromosome

# 修复操作函数
# def repair(chromosome, clusters, omega, normalized_fisher_scores):
    """
    修复操作：确保染色体中每个簇的特征数量不超过omega，基于Fisher Score概率决定特征的添加或移除。
    :param chromosome: 二进制编码的染色体（列表，0表示未选，1表示选中）
    :param clusters: 簇信息，每个簇为一个特征索引的列表
    :param omega: 每个簇中允许的最大特征数
    :param normalized_fisher_scores: 归一化后的Fisher Score，字典形式 {feature_index: score}
    :return: 修复后的染色体
    """
    repaired_chromosome = chromosome.copy()  # 创建染色体副本

    for cluster in clusters:
        cluster_size = len(cluster)
        # 计算当前簇允许的最大特征数（至少1个）
        omega_cluster = max(1, int(cluster_size * omega))  # 四舍五入取整
      
        # 找到当前簇中已选中的特征
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1 and i in cluster]
        
        # 如果选中特征过多，基于逆Fisher Score概率移除特征
        if len(selected_features) > omega_cluster:
            inverse_scores = [1 / normalized_fisher_scores[feature] for feature in selected_features]
            total_inverse = sum(inverse_scores)
            remove_probabilities = [score / total_inverse for score in inverse_scores]

            # 根据概率移除特征，直到满足数量约束
            num_to_remove = len(selected_features) - omega_cluster
            features_to_remove = np.random.choice(selected_features, size=num_to_remove, replace=False, p=remove_probabilities)
            for feature in features_to_remove:
                repaired_chromosome[feature] = 0  # 将选中的特征标记为未选中（0）
        
        # 如果选中特征不足，基于Fisher Score概率添加特征
        elif len(selected_features) < omega_cluster:
            remaining_features = [feature for feature in cluster if chromosome[feature] == 0]
            num_to_add = omega_cluster - len(selected_features)

            # 正常按概率添加特征
            fisher_scores = [normalized_fisher_scores[feature] for feature in remaining_features]
            total_score = sum(fisher_scores) # 计算所有候选特征的 Fisher Score 总和，用于归一化概率
            add_probabilities = [score / total_score for score in fisher_scores] # 计算特征选择概率
            features_to_add = random.choices(remaining_features, weights=add_probabilities, k=num_to_add)

            for feature in features_to_add:
                repaired_chromosome[feature] = 1  # 将未选中特征标记为选中（1）

    return repaired_chromosome

def repair(chromosome, clusters, omega, normalized_fisher_scores):
    """
    修复操作：确保染色体中每个簇的特征数量恰好等于 omega_cluster，
    并按 Fisher 分数从高到低添加/移除特征。
    
    :param chromosome: 二进制编码的染色体（numpy 数组，0 表示未选，1 表示选中）
    :param clusters: 簇信息，每个簇为一个特征索引的列表
    :param omega: 每个簇允许的最大特征比例（小数，例如 0.2 表示 20%）
    :param normalized_fisher_scores: 归一化后的 Fisher Score，字典形式 {feature_index: score}
    :return: 修复后的染色体（numpy 数组）
    """
    repaired_chromosome = chromosome.copy()

    for cluster in clusters:
        cluster_size = len(cluster)
        # 计算该簇允许的特征数量（至少 1 个）
        omega_cluster = max(1, int(cluster_size * omega))

        # 找出该簇中当前被选中的特征列表
        selected_features = [i for i in cluster if repaired_chromosome[i] == 1]

        # 如果选中的特征数量 > omega_cluster，需要移除一些
        if len(selected_features) > omega_cluster:
            # 按 Fisher 分数从低到高排序，低分先被移除
            selected_features_sorted = sorted(selected_features, key=lambda feat: normalized_fisher_scores[feat])
            # 需要移除的数量
            num_to_remove = len(selected_features) - omega_cluster
            # 取分数最低的 num_to_remove 个特征进行移除
            for feat in selected_features_sorted[:num_to_remove]:
                repaired_chromosome[feat] = 0

        # 如果选中的特征数量 < omega_cluster，需要添加一些
        elif len(selected_features) < omega_cluster:
            # 先找出当前簇中未被选中的候选特征
            remaining_features = [feat for feat in cluster if repaired_chromosome[feat] == 0]
            # 按 Fisher 分数从高到低排序，分数高的优先添加
            remaining_sorted = sorted(remaining_features, key=lambda feat: normalized_fisher_scores[feat], reverse=True)
            # 需要添加的数量
            num_to_add = omega_cluster - len(selected_features)
            # 取分数最高的 num_to_add 个进行添加
            for feat in remaining_sorted[:num_to_add]:
                repaired_chromosome[feat] = 1

        # 如果等于 omega_cluster，则不需要做任何操作

    return repaired_chromosome

# 基于适应度选择最高的染色体函数
def select_top_individuals(population, fitness_values, population_size):
    """
    基于适应度选择最高的指定数量的染色体。
    
    参数：
        population (list): 当前种群，每个元素表示一个个体。
        fitness_values (list): 每个个体的适应度值。
        population_size (int): 要选取的染色体数量。

    返回：
        selected_population (list): 选中的个体列表。
        selected_fitness (list): 选中个体对应的适应度值。
    """
    # 获取适应度排序的索引，从高到低
    sorted_indices = np.argsort(fitness_values)[::-1]

    # 选择前 population_size 个个体
    selected_indices = sorted_indices[:population_size]
    
    selected_population = [population[i] for i in selected_indices]
    selected_fitness = [fitness_values[i] for i in selected_indices]

    return selected_population, selected_fitness
    
# 轮盘赌选择函数
def roulette_wheel_selection(population, fitness_values, select_count):
    """
    基于轮盘赌选择法，从种群中选择指定数量的个体，选择概率与适应度成正比。
    
    参数：
        population (list): 当前种群，每个元素表示一个个体。
        fitness_values (list): 每个个体的适应度值，表示其优劣程度。
        select_count (int): 要从种群中选择的个体数量。
    
    返回：
        selected_population (list): 选中的个体列表。
        selected_fitness (list): 选中个体对应的适应度值。
    """
    # 计算总适应度，用于归一化选择概率
    total_fitness = sum(fitness_values)
    
    # 计算每个个体被选中的概率
    # 此时 selection_probs 中每个值表示该个体被选中的概率，所有概率之和为 1
    selection_probs = [fitness / total_fitness for fitness in fitness_values]  

    # 使用 np.random.choice 选择个体的索引
    selected_indices = np.random.choice(len(population), size=select_count, p=selection_probs)
    
    # 获取被选中的个体及其适应度
    selected_population = [population[i] for i in selected_indices]
    selected_fitness = [fitness_values[i] for i in selected_indices]
    
    return selected_population, selected_fitness

# 交叉变异过程：生成新种群
def perform_crossover_mutation(population, clusters, omega, num_features, normalized_fisher_scores):
    new_population = []
    for i in range(0, len(population), 2):  # 每次取两条染色体进行交叉
        parent1 = population[i]
        parent2 = population[i + 1]
        # 执行交叉操作
        child1, child2 = crossover(parent1, parent2, num_features)

        # 执行变异操作
        child1 = mutation(child1, i, mutation_rate=0.05)  # 传入染色体序号 i
        child2 = mutation(child2, i+1, mutation_rate=0.05)  # 传入染色体序号 i+1

        # 修复操作
        child1 = repair(child1, clusters, omega, normalized_fisher_scores)
        child2 = repair(child2, clusters, omega, normalized_fisher_scores)

        new_population.extend([child1, child2])

    return new_population

# 主遗传算法流程
def genetic_algorithm(population, fitness_values, X_train, y_train, clusters, omega, similarity_matrix, population_size, 
                      num_features, normalized_fisher_scores):
    """
    遗传算法主函数
    """
    generations = 100  # 最大迭代代数
    set_random_seed(42)

    for generation in range(generations):
        
        # Step 2: 进行交叉、变异和修复，生成新种群
        new_population = perform_crossover_mutation(population, clusters, omega, num_features, normalized_fisher_scores)
        
        # Step 3: 计算新种群的适应度值
        new_fitness_values = calculate_fitness(new_population, X_train, y_train, similarity_matrix, n_jobs=10)
        
        # Step 4: 合并新旧种群（不包含精英）
        combined_population = population + new_population
        combined_fitness = fitness_values + new_fitness_values

        # Step 5: 根据适应度选择最高的个体
        selected_population, selected_fitness = select_top_individuals(
            combined_population, combined_fitness, population_size 
        )
        
        # Step 6: 更新种群
        population = selected_population
        fitness_values = selected_fitness

        # 输出当前代的最佳适应度
        print(f"Best fitness value in generation {generation + 1}: {max(selected_fitness)}")

    return population, fitness_values