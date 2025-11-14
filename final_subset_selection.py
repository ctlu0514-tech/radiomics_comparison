import numpy as np

def final_subset_selection(population, fitness_values):
    """
    选择最终一代中适应度最高的染色体及其对应的特征
    :param population: 最终一代的种群
    :param fitness_values: 最终一代的适应度值
    :return: 最优染色体及其选中的特征索引
    """
    # 找到适应度最高的染色体索引
    best_index = np.argmax(fitness_values)
    best_chromosome = population[best_index]  # 最优染色体
    selected_features = [i for i, gene in enumerate(best_chromosome) if gene == 1]  # 选中的特征索引

    return best_chromosome, selected_features