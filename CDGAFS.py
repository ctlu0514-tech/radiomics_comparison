import pandas as pd
import numpy as np
from collections import defaultdict
import random
from fisher_score import compute_fisher_score
from normalize_scores import normalize_scores
from construct_feature_graph import construct_feature_graph, construct_pearson_only_graph
# from compare_graphs import construct_pearson_only_graph, analyze_and_visualize # (compare_graphs 似乎未提供)
from compute_similarity_matrix import compute_similarity_matrix
from community_detection import iscd_algorithm_auto_k
from calculate_fitness import calculate_fitness 
from genetic_algorithm_utils import initialize_population, genetic_algorithm
from final_subset_selection import final_subset_selection

def cdgafs_feature_selection(X, y, gene_list, theta, omega, population_size, 
                             w_bio_boost=0.5, pre_filter_top_n=None, graph_type='pearson_only'):
    """
    (已修改)
    执行完整的特征选择流程（预筛选、建图、社团检测、GA），
    并返回GA的结果，以及用于方案二的中间数据。

    输入:
        ... (参数不变) ...
        w_bio_boost (float): 乘法融合权重
        graph_type (str): 'fused' (方案二) 或 'pearson_only' (方案一)
        
    返回: (元组)
        selected_original_indices (list): GA选出的最终特征索引
        X_subset (np.ndarray): 预筛选后的特征矩阵 (供方案二使用)
        top_indices (np.ndarray): 预筛选出的特征在原始矩阵中的索引 (供方案二使用)
        gene_list_subset (list): 预筛选后的基因列表 (供方案二使用)
        normalized_fisher_scores (dict/np.ndarray): 归一化的FS (供方案二使用)
    """

    # Step 1: 计算 Fisher Scores 和 预筛选
    print("Step 1: 计算 Fisher Scores 和执行预筛选...")
    fisher_scores = compute_fisher_score(X, y)
    
    original_indices = np.arange(X.shape[1])
    
    if pre_filter_top_n is not None and pre_filter_top_n < X.shape[1]:
        print(f"执行预筛选，选取 Fisher Score 最高的 {pre_filter_top_n} 个特征...")
        top_indices = np.argsort(fisher_scores)[-pre_filter_top_n:]
        
        X_subset = X[:, top_indices]
        fisher_scores_subset = fisher_scores[top_indices]
        gene_list_subset = [gene_list[i] for i in top_indices]
        original_indices = top_indices # 这里的 original_indices 是 GA 流程内部使用的
        
        print(f"预筛选完成，特征数量从 {X.shape[1]} 减少到 {X_subset.shape[1]}")
    else:
        print("未执行预筛选，使用全部特征。")
        top_indices = original_indices # 如果未筛选，top_indices就是全部索引
        X_subset = X
        fisher_scores_subset = fisher_scores
        gene_list_subset = gene_list
    
    normalized_fisher_scores = normalize_scores(fisher_scores_subset)
    print("归一化后的 Fisher Scores (子集):", normalized_fisher_scores[:5]) 

    # Step 2: 构造特征图 (在数据子集上进行)
    print(f"\nStep 2: 构造特征图 (模式: {graph_type})...")
    
    if graph_type == 'fused':
        feature_graph = construct_feature_graph(X_subset, y, gene_list_subset, theta, w_bio_boost=w_bio_boost)
    elif graph_type == 'pearson_only':
        feature_graph = construct_pearson_only_graph(X_subset, theta)
    else:
        raise ValueError("graph_type 必须是 'fused' 或 'pearson_only'")

    # (注意: GA 流程使用 similarity_matrix，而 Centrality 流程使用 feature_graph)
    similarity_matrix = compute_similarity_matrix(X_subset)

    # # Step 3: 社区检测 (使用你恢复的逻辑)
    # print("\nStep 3: 进行社区检测...")
    # partition = iscd_algorithm_auto_k(feature_graph)

    # clusters = defaultdict(list)
    # for node, community in partition.items():
    #     clusters[community].append(node)
    # clusters = [cluster for cluster in clusters.values()]
    # print(f"检测到 {len(clusters)} 个社区。")

    # Step 3: 将所有节点定义为一个社区 (跳过ISCD)
    print("\nStep 3: 跳过社群检测，将所有节点定义为一个大社区...")
    # 获取特征子集中的节点总数（即特征数量）
    num_features_subset = X_subset.shape[1]
    # 创建一个包含所有节点索引的列表
    all_nodes_as_one_community = list(range(num_features_subset))
    # `clusters` 的数据结构是一个列表的列表，所以我们将上面的列表作为唯一的元素
    clusters = [all_nodes_as_one_community]
    partition = {node: 0 for node in range(num_features_subset)}
    print(f"已手动定义 {len(clusters)} 个社群。")
    
    # Step 4: 初始化种群并执行遗传算法
    print("\nStep 4: 执行遗传算法...")
    num_features_subset = X_subset.shape[1]
    population = initialize_population(num_features_subset, clusters, omega, population_size)
    print(f"初始化种群大小: {len(population)}")
    
    fitness_values = calculate_fitness(population, X_subset, y, similarity_matrix, n_jobs=10)
    print(f"初始适应度值示例: {fitness_values[:5]}")

    population, fitness_values = genetic_algorithm(
        population, fitness_values, X_subset, y, clusters, omega, 
        similarity_matrix, population_size, num_features_subset, normalized_fisher_scores
    )
    
    # Step 5: 选择最终特征子集
    print("\nStep 5: 选择最终特征子集...")
    best_chromosome, selected_subset_indices = final_subset_selection(population, fitness_values)
    print(f"从子集中选出了 {len(selected_subset_indices)} 个特征。")
    
    # 将子集中的索引映射回原始数据集中的索引
    selected_original_indices_ga = [original_indices[i] for i in selected_subset_indices]
    
    print(f"\n最终最佳染色体 (在子集上): {best_chromosome}")
    print(f"映射回原始索引: {selected_original_indices_ga[:10]} ...") 
    
    # ========================> 关键修改：返回物料 <========================
    # 返回GA的结果，以及方案二所需的所有预处理数据
    return (
        selected_original_indices_ga, 
        X_subset, 
        top_indices, 
        gene_list_subset, 
        normalized_fisher_scores
    )
    # =================================================================