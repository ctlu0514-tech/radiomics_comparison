# 文件名: run_radiomics_comparison.py
# 描述: 主脚本，用于数据清理、特征选择和调用评估
# 依赖: run_evaluation.py, CDGAFS.py, fisher_score.py, pymrmr

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys

# 机器学习和 mRMR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pymrmr


# --- 导入你项目中的核心功能 ---
from CDGAFS import cdgafs_feature_selection
from fisher_score import compute_fisher_score
from run_evaluation import evaluate_model_performance, print_summary_table


# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 
# 1. 数据加载与清理模块 (原封不动)
# 
# ===================================================================
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_clean_radiomics(csv_path, label_col_name):
    """
    加载 "dirty" 的放射组学CSV，并将其彻底清理干净。
    - 移除 'diagnostics_' 列
    - 转换标签为 0/1
    - 【新增】移除标准差为0的常数特征
    - 填充 NaN
    - 标准化
    """
    print(f"--- 步骤 1: 加载并清理数据: {csv_path} ---")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"!!! 致命错误: 找不到文件 '{csv_path}'。")
        sys.exit(1)

    if label_col_name not in data.columns:
        print(f"!!! 致命错误: 在CSV中找不到标签列 '{label_col_name}'。")
        sys.exit(1)

    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
        print(f"    - 标签已转换为 0 和 1。 (原始标签 {unique_labels} -> 0/1)")
    else:
        print(f"!!! 致命错误: 标签列必须包含2个唯一的类别。找到: {unique_labels}")
        sys.exit(1)
    
    id_col = 'ID' if 'ID' in data.columns else 'Patient_ID'
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = [id_col, label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    
    print(f"    - 已丢弃 {len(diag_cols)} 个 'diagnostics' 元数据列。")
    print(f"    - 清理后剩余特征数: {len(feature_names)}")

    # 填充 NaN (必须在计算标准差之前完成)
    if X_df.isna().any().any():
        print("    - 发现缺失值 (NaN)，正在使用特征平均值填充...")
        imputer = SimpleImputer(strategy='mean')
        X_unscaled = imputer.fit_transform(X_df)
    else:
        print("    - 未发现缺失值。")
        X_unscaled = X_df.values
    
    # --- 【新增步骤：移除常数特征】 ---
    # 在填充NaN之后，标准化之前
    print("    - 正在检查标准差为0的常数特征...")
    
    # 计算所有特征的标准差 (沿列计算 axis=0)
    stds = np.std(X_unscaled, axis=0)
    
    # 我们设置一个极小的阈值，而不是==0，以防浮点数问题
    variance_threshold = 1e-6 
    
    # 找到标准差 *大于* 阈值的“好”特征的索引
    good_indices = np.where(stds > variance_threshold)[0]
    # 找到标准差 *小于等于* 阈值的“坏”特征的索引
    bad_indices = np.where(stds <= variance_threshold)[0]

    if len(bad_indices) > 0:
        print(f"    - 正在移除 {len(bad_indices)} 个标准差为0 (或极低) 的常数特征。")
        
        # 为了日志清晰，打印出删掉的特征名
        original_feature_names = feature_names
        removed_features = [original_feature_names[i] for i in bad_indices]
        print(f"    - (被移除的特征示例: {removed_features[:5]}...)")
        
        # --- 关键过滤 ---
        # 1. 只保留“好”的数据列
        X_unscaled = X_unscaled[:, good_indices]
        # 2. 只保留“好”的特征名
        feature_names = [original_feature_names[i] for i in good_indices]
        
        print(f"    - 移除后剩余特征数: {len(feature_names)}")
    else:
        print("    - 检查完毕：所有特征都具有方差。")
    # --- 【新增步骤结束】 ---

    print("    - 正在对特征进行 Z-Score 标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    print(f"--- 数据清理完毕 ---")
    
    # X_df_filled 应该是 *未标准化* 的数据，但 *已经过筛选*
    X_df_filled = pd.DataFrame(X_unscaled, columns=feature_names) 
    
    return X_scaled, X_df_filled, y, feature_names


# ===================================================================
# 
# 2. 特征选择器模块 (原封不动)
# 
# ===================================================================

def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA):
    print("\n--- 正在运行: CDGAFS (社区检测 + 遗传算法) ---")
    start_time = time.time()

    
    (selected_indices, 
     _, _, _, _) = cdgafs_feature_selection(
        X=X, 
        y=y, 
        gene_list=feature_names, 
        theta=THETA, 
        omega=GA_OMEGA, 
        population_size=GA_POPULATION_SIZE,
        w_bio_boost=0.0,
        pre_filter_top_n=None,  # <-- 关键：不进行预筛选 (如你所愿)
        graph_type='pearson_only'
    )
        
    elapsed = time.time() - start_time
    print(f"--- CDGAFS 完成。耗时: {elapsed:.2f} 秒。选出 {len(selected_indices)} 个特征。---")
        
    if len(selected_indices) > K_FEATURES * 1.5:
         print(f"    - CDGAFS 选出过多特征({len(selected_indices)})，将使用 Fisher Score 截取最重要的 {K_FEATURES} 个。")
         # 使用 Fisher Score 排序来公平截取
         sub_scores = compute_fisher_score(X[:, selected_indices], y)
         top_sub_indices = np.argsort(sub_scores)[-K_FEATURES:]
         selected_indices = np.array(selected_indices)[top_sub_indices]
             
    elif len(selected_indices) == 0:
        print("    - !!! 警告: CDGAFS 未选出任何特征。")
        return []

    return selected_indices

def select_features_mrmr(X_df, y, K_FEATURES):
    """
    运行 mRMR (封装函数)
    【已修改】 - 增加了对连续数据的分箱（离散化）
    【已修复】 - 增加了对 'label' 键的过滤
    """
    print("\n--- 正在运行: mRMR ---")
    start_time = time.time()
    
    # --- 离散化 ---
    print("    - 正在将连续的放射组学特征离散化 (分箱) 以便 mRMR 计算...")
    X_df_discrete = X_df.copy()
    n_bins = 10
    
    for col in X_df_discrete.columns:
        try:
            X_df_discrete[col] = pd.qcut(X_df_discrete[col], q=n_bins, labels=False, duplicates='drop')
        except Exception as e:
            X_df_discrete[col] = 0
            
    X_df_discrete['label'] = y
    
    try:
        selected_feature_names = pymrmr.mRMR(X_df_discrete, 'MIQ', K_FEATURES)
        
        # --- 【BUG 修复】 ---
        # 1. 检查 pymrmr 是否错误地将 'label' 包含在返回列表中
        if 'label' in selected_feature_names:
            print("    - (mRMR 警告: 'label' 被错误地包含在特征列表中，已自动移除。)")
            selected_feature_names.remove('label')
        # --- 【修复结束】 ---

        # 2. 转换回特征索引
        name_to_index_map = {name: i for i, name in enumerate(X_df.columns)}
        selected_indices = [name_to_index_map[name] for name in selected_feature_names]
        
        elapsed = time.time() - start_time
        print(f"--- mRMR 完成。耗时: {elapsed:.2f} 秒。选出 {len(selected_indices)} 个特征。---")
        
        return selected_indices
        
    except Exception as e:
        print(f"!!! mRMR 运行失败: {e}")
        return []

def select_features_lasso(X, y, K_FEATURES):
    print("\n--- 正在运行: LASSO (L1) ---")
    start_time = time.time()
    
    # --- 修改开始 ---
    # 我们不再使用 LogisticRegressionCV，而是使用 LogisticRegression
    # 我们手动设置 C=1.0 (一个标准的、相对较弱的惩罚)。
    # 你可以尝试 C=1.0, C=5.0, C=10.0
    # C 越大 = 惩罚越弱 = 特征越多
    
    model = LogisticRegression(
        C=1.0,  # <-- 手动设置 C 值
        penalty='l1', 
        solver='liblinear', 
        random_state=42, 
        max_iter=1000
    )
    # --- 修改结束 ---
    
    model.fit(X, y)
    coefficients = model.coef_[0]
    
    non_zero_indices = np.where(np.abs(coefficients) > 1e-5)[0]
        
    if len(non_zero_indices) == 0:
        print("!!! LASSO 将所有特征系数都惩罚为0。")
        print("    - 提示：尝试在 'select_features_lasso' 函数中增大 C 的值 (例如 C=5.0 或 C=10.0)")
        return []
    
    # 你的排序逻辑是完美的
    sorted_indices = sorted(non_zero_indices, 
                            key=lambda i: np.abs(coefficients[i]), 
                            reverse=True)
    
    # 你的截取逻辑也是完美的
    # 如果 C=1.0 产生了 (例如) 80 个特征, 
    # sorted_indices[:K_FEATURES] 将会正确截取前 35 个
    selected_indices = sorted_indices[:K_FEATURES] 
    
    elapsed = time.time() - start_time
    print(f"--- LASSO 完成。耗时: {elapsed:.2f} 秒。")
    print(f"    - L1 (C=1.0) 找到了 {len(non_zero_indices)} 个非零特征。")
    print(f"    - 已按系数大小排序并截取前 {len(selected_indices)} 个。---")
    return selected_indices
    
# ===================================================================
# 
# 3. 主程序 (Main)
# 
# ===================================================================

def main():
    # --- 在这里设置你的参数 ---
    CSV_PATH = '/data/qh_20T_share_file/lct/CT67/ovarian_features_with_label.csv'
    LABEL_COLUMN = 'label'
    K_FEATURES = 20 # 目标特征数
    GA_POPULATION_SIZE = 100 
    GA_OMEGA = 0.03
    THETA = 0.7

    print("#"*70)
    print("### 开始运行放射组学特征选择与评估对比实验 ###")
    print("#"*70)
    
    # 1. 加载和清理数据
    X_scaled, X_df_filled, y, feature_names = load_and_clean_radiomics(CSV_PATH, LABEL_COLUMN)
    if X_scaled is None:
        return
        
    # 2. 运行所有特征选择器
    all_selected_indices = {}
    
    # (方法 1: 你的 CDGAFS)
    all_selected_indices['CDGAFS'] = select_features_cdafs(X_scaled, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA)
    
    # (方法 2: mRMR)
    # all_selected_indices['mRMR'] = select_features_mrmr(X_df_filled.copy(), y, K_FEATURES)

    # (方法 3: LASSO)
    # all_selected_indices['LASSO'] = select_features_lasso(X_scaled, y, K_FEATURES)    
    
    # 3. 评估所有方法的结果
    print("\n" + "#"*70)
    print("### 开始评估所有特征选择方法的性能 ###")
    print(f"### 评估分类器: 逻辑回归 (L2) - 5折交叉验证 ###")
    print("#"*70)
    
    all_results = {}
    
    for method_name, indices in all_selected_indices.items():
        print(f"\n--- (评估) {method_name} ---")
        if not isinstance(indices, (list, np.ndarray)) or len(indices) == 0:
            print("    - !!! 跳过评估，因为该方法未能选出任何特征。")
            continue
            
        print(f"    - 选定特征数: {len(indices)}")
        
        # --- 调用新文件中的函数 ---
        results = evaluate_model_performance(X_scaled, y, indices)
        
        all_results[method_name] = results
        
        # 实时打印结果
        print("    - 评估结果:")
        for metric, value in results.items():
            print(f"        - {metric:<12}: {value:.4f}")

    # 4. 打印最终对比表格 (调用新文件中的函数)
    print_summary_table(all_results, all_selected_indices)

if __name__ == "__main__":
    main()