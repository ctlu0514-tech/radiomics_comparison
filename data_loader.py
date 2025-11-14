import pandas as pd
import numpy as np

def load_and_preprocess_data(file_name, label_line_name):
    """
    一个通用的数据加载和预处理函数。
    它负责：读取CSV，移除零方差特征，处理缺失值，归一化。
    
    返回:
        feature_data_full (np.ndarray): 完整的特征矩阵 (样本数, 特征数)
        label_data (np.ndarray): 标签数组
        gene_list_full (list): 完整的基因名列表
    """
    print(f"--- 正在加载和预处理数据: {file_name} ---")
    
    try:
        total_data = pd.read_csv('/data/qh_20T_share_file/lct/CuMiDa/' + file_name)
    except FileNotFoundError:
        print(f"错误：找不到数据文件 '{file_name}'。请检查路径。")
        return None, None, None

    col_name = total_data.columns
    patient_id_col = col_name[0]
    label_col = label_line_name

    feature_columns = [col for col in col_name if col not in [label_col, patient_id_col]]
    
    print(f"原始特征数量: {len(feature_columns)}")
    variances = total_data[feature_columns].var()
    non_zero_var_features = variances[variances > 0].index.tolist()
    print(f"移除常数特征后特征数量: {len(non_zero_var_features)}")
    
    total_data_clean = total_data.dropna(subset=non_zero_var_features)
    labels = total_data_clean[label_col].values

    feature_min = total_data_clean[non_zero_var_features].min()
    feature_max = total_data_clean[non_zero_var_features].max()
    normalized_features = (total_data_clean[non_zero_var_features] - feature_min) / (feature_max - feature_min + 1e-10)
    
    gene_list_full = normalized_features.columns.tolist()
    feature_data_full = normalized_features.values
    label_data = np.array(labels.tolist())
    
    print("--- 数据处理完成 ---")
    return feature_data_full, label_data, gene_list_full

