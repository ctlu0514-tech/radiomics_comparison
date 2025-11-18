# 文件名: run_radiomics_comparison.py
# 描述: 主脚本，用于数据清理、特征选择和调用评估 (已移除所有 try-except)

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys
import radMLBench  # 确保已安装: pip install radMLBench

# 机器学习和 mRMR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE 
import pymrmr

# --- 导入你项目中的核心功能 ---
from CDGAFS import cdgafs_feature_selection
from fisher_score import compute_fisher_score
from run_evaluation import evaluate_model_performance, print_summary_table

# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载与清理模块
# ===================================================================
def clean_radiomics_df(data, label_col_name):
    """
    通用核心函数：接收一个 DataFrame，进行清洗、标准化和标签处理。
    """
    print(f"    - [处理中] 原始数据形状: {data.shape}")

    # 1. 检查标签列
    if label_col_name not in data.columns:
        print(f"!!! 致命错误: 找不到标签列 '{label_col_name}'。")
        sys.exit(1) # 直接退出

    # 2. 处理标签 (转为 0/1)
    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
    else:
        print(f"!!! 致命错误: 标签列必须包含2个唯一的类别。找到: {unique_labels}")
        sys.exit(1)
    
    # 3. 移除无关列 (ID, Diagnostics)
    id_cols = [col for col in data.columns if 'ID' in col or 'id' in col] 
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = id_cols + [label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    
    # 4. 缺失值填充
    if X_df.isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        X_unscaled = imputer.fit_transform(X_df)
    else:
        X_unscaled = X_df.values
    
    # 5. 移除低方差特征
    stds = np.std(X_unscaled, axis=0)
    variance_threshold = 1e-6 
    good_indices = np.where(stds > variance_threshold)[0]
    
    if len(good_indices) == 0:
        print("!!! 错误: 所有特征方差均为0，无法继续。")
        sys.exit(1)

    X_unscaled = X_unscaled[:, good_indices]
    feature_names = [feature_names[i] for i in good_indices]
    
    print(f"    - 清理后剩余特征数: {len(feature_names)}")

    # 6. 标准化 (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    # 重建 DataFrame (为了兼容 mRMR)
    X_df_filled = pd.DataFrame(X_scaled, columns=feature_names) 
    
    return X_scaled, X_df_filled, y, feature_names

def load_and_clean_radiomics(csv_path, label_col_name):
    """
    原函数的封装：专门用于读取 CSV 文件
    """
    print(f"--- 加载本地 CSV: {csv_path} ---")
    # [修改] 直接读取，不做异常捕获
    data = pd.read_csv(csv_path)
    return clean_radiomics_df(data, label_col_name)

def load_radmlbench_data(dataset_name):
    """
    适配器：加载 radMLBench 数据并传入通用的清理函数
    """
    print(f"--- [radMLBench] 加载数据集: {dataset_name} ---")
    
    # [修正] 函数名从 load_dataset 改为 loadData
    # 注意：该库默认会将数据下载到用户目录的 .radMLBench 文件夹下
    df = radMLBench.loadData(dataset_name)
    
    return clean_radiomics_df(df, label_col_name='Target')

# ===================================================================
# 2. 特征选择器模块
# ===================================================================
def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, pruning_method='RFE'):
    print(f"\n--- 正在运行: CDGAFS (剪枝方法: {pruning_method}) ---")
    start_time = time.time()

    # 调用 GA 流程
    (selected_indices, 
     _, _, _, _) = cdgafs_feature_selection(
        X=X, 
        y=y, 
        gene_list=feature_names, 
        theta=THETA, 
        omega=GA_OMEGA, 
        population_size=GA_POPULATION_SIZE,
        w_bio_boost=0.0,
        pre_filter_top_n=None,
        graph_type='pearson_only'
    )
        
    elapsed = time.time() - start_time
    print(f"--- CDGAFS GA 阶段完成。耗时: {elapsed:.2f} 秒。GA 选出 {len(selected_indices)} 个特征。---")
        
    if len(selected_indices) > K_FEATURES:
        if pruning_method == 'RFE':
            print(f"    - [优化] 使用 RFE (递归特征消除) 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
            selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
            
            X_ga_selected = X[:, selected_indices]
            selector.fit(X_ga_selected, y)
            
            rfe_support = selector.support_
            selected_indices = np.array(selected_indices)[rfe_support]
            print(f"    - RFE 剪枝完成。")

        elif pruning_method == 'FISHER':
            print(f"    - [优化] 使用 Fisher Score (过滤法) 从 {len(selected_indices)} 精选到 {K_FEATURES} 个...")
            X_ga_selected = X[:, selected_indices]
            scores_on_subset = compute_fisher_score(X_ga_selected, y)
            top_subset_indices = np.argsort(scores_on_subset)[-K_FEATURES:]
            selected_indices_array = np.array(selected_indices)
            selected_indices = selected_indices_array[top_subset_indices]
            print(f"    - Fisher Score 剪枝完成。")
            
    elif len(selected_indices) == 0:
        print("    - !!! 警告: CDGAFS 未选出任何特征。")
        return []

    return selected_indices

def select_features_mrmr(X_df, y, K_FEATURES):
    print("\n--- 正在运行: mRMR ---")
    start_time = time.time()
    
    X_df_discrete = X_df.copy()
    n_bins = 10
    
    # 离散化
    for col in X_df_discrete.columns:
        # 这里使用 Pandas 的函数，出错也是正常的 Python 错误，不需要 try
        X_df_discrete[col] = pd.qcut(X_df_discrete[col], q=n_bins, labels=False, duplicates='drop')
            
    X_df_discrete['label'] = y
    
    # [修改] 直接调用，不做异常捕获
    selected_feature_names = pymrmr.mRMR(X_df_discrete, 'MIQ', K_FEATURES)
    
    if 'label' in selected_feature_names:
        selected_feature_names.remove('label')

    name_to_index_map = {name: i for i, name in enumerate(X_df.columns)}
    selected_indices = [name_to_index_map[name] for name in selected_feature_names]
    
    print(f"--- mRMR 完成。选出 {len(selected_indices)} 个特征。---")
    return selected_indices


def select_features_lasso(X, y, K_FEATURES):
    print("\n--- 正在运行: LASSO (L1) ---")
    start_time = time.time()
    
    model = LogisticRegression(
        C=1.0,  
        penalty='l1', 
        solver='liblinear', 
        random_state=42, 
        max_iter=1000
    )
    
    model.fit(X, y)
    coefficients = model.coef_[0]

    non_zero_indices = np.where(np.abs(coefficients) > 1e-5)[0]
    
    if len(non_zero_indices) == 0:
        print("!!! LASSO 将所有特征系数都惩罚为0。")
        return []

    sorted_indices = sorted(non_zero_indices, 
                        key=lambda i: np.abs(coefficients[i]), 
                        reverse=True)

    selected_indices = sorted_indices[:K_FEATURES] 

    elapsed = time.time() - start_time
    print(f"--- LASSO 完成。耗时: {elapsed:.2f} 秒。")
    return selected_indices

def select_features_rfe_only(X, y, K_FEATURES):
    print("\n--- 正在运行: RFE-Only (递归特征消除) ---")
    start_time = time.time()

    estimator = LogisticRegression(
        solver='liblinear', 
        class_weight='balanced', 
        random_state=42,
        max_iter=3000 
    )

    selector = RFE(
        estimator, 
        n_features_to_select=K_FEATURES, 
        step=1  
    )

    print(f"    - 正在从 {X.shape[1]} 个特征中精选 {K_FEATURES} 个...")
    selector.fit(X, y) 

    selected_indices = np.where(selector.support_)[0]

    elapsed = time.time() - start_time
    print(f"--- RFE-Only 完成。耗时: {elapsed:.2f} 秒。选出 {len(selected_indices)} 个特征。---")

    return selected_indices

def run_analysis_on_dataset(X_scaled, X_df_filled, y, feature_names, 
                            K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
                            dataset_title):
    """
    通用函数：接收处理好的数据，运行所有特征选择方法并评估。
    [修改] 移除了所有 try-except，出错即崩溃。
    """
    print(f"\n{'-'*20} 正在分析: {dataset_title} {'-'*20}")
    print(f"    - 样本数: {X_scaled.shape[0]}, 特征数: {X_scaled.shape[1]}")
    
    all_selected_indices = {}
    
    # 1. 运行 CDGAFS
    all_selected_indices['CDGAFS'] = select_features_cdafs(
        X_scaled, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
        pruning_method='RFE'
    )

    # 2. 运行 mRMR
    all_selected_indices['mRMR'] = select_features_mrmr(X_df_filled.copy(), y, K_FEATURES)

    # 3. 运行 LASSO
    all_selected_indices['LASSO'] = select_features_lasso(X_scaled, y, K_FEATURES)    
    
    # 4. 运行 RFE-Only
    all_selected_indices['RFE-Only'] = select_features_rfe_only(X_scaled, y, K_FEATURES)

    # 5. 统一评估与打印报表
    print(f"\n>>> {dataset_title} 的最终评估结果 <<<")
    all_results = {}
    for method_name, indices in all_selected_indices.items():
        if indices is None or len(indices) == 0: 
            continue
        
        results = evaluate_model_performance(X_scaled, y, indices)
        all_results[method_name] = results

    if all_results:
        print_summary_table(all_results, all_selected_indices)
    else:
        print(f"警告: {dataset_title} 没有产生任何有效结果。")

# ===================================================================
# 3. 主程序
# ===================================================================
def main():
    # --- 参数设置 ---
    # 本地文件路径
    LOCAL_CSV_PATH = '/data/qh_20T_share_file/lct/CT67/ovarian_features_with_label.csv'
    LOCAL_LABEL_COL = 'label'
    
    K_FEATURES = 50 
    GA_POPULATION_SIZE = 50 
    GA_OMEGA = 0.5
    THETA = 0.9

    # 定义要跑的公开数据集
    public_datasets = ['C4KC-KiTS', 'BraTS-2021']

    print("#"*70)
    print(f"### 开始运行实验：本地数据 + 公开基准测试 (K={K_FEATURES}) ###")
    print("#"*70)
    
    # ==========================================
    # 任务 1: 跑本地数据
    # ==========================================
    print(f"\n\n>>> [任务 1] 加载本地数据... <<<")
    # 这里的加载如果失败会直接报错退出
    local_data = load_and_clean_radiomics(LOCAL_CSV_PATH, LOCAL_LABEL_COL)
    
    # 解包数据
    X, X_df, y, f_names = local_data
    
    # 运行分析
    # run_analysis_on_dataset(X, X_df, y, f_names, 
    #                         K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
    #                         dataset_title="Local Ovarian Data")

    # ==========================================
    # 任务 2: 跑公开数据集 (循环)
    # ==========================================
    for ds_name in public_datasets:
        print(f"\n\n>>> [任务 2] 加载公开数据集: {ds_name}... <<<")
        
        # 这里的加载如果失败会直接报错退出
        rad_data = load_radmlbench_data(ds_name)
        
        X, X_df, y, f_names = rad_data
        
        run_analysis_on_dataset(X, X_df, y, f_names, 
                                K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, 
                                dataset_title=ds_name)

if __name__ == "__main__":
    main()
