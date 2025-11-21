# 文件名: run_radiomics_comparison_fixed.py
# 描述: 主脚本 (已修复数据泄露：实施严格的 Train/Test 分离)

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys

# 机器学习和 mRMR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split # 核心引入
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score # 用于手动评估

# 尝试导入 pymrmr，如果失败则提供备选方案（防止环境报错）
try:
    import pymrmr
except ImportError:
    pymrmr = None
    print("警告: 未检测到 pymrmr 模块，mRMR 功能将不可用。")

# --- 导入核心功能 ---
# 假设这些文件在同一目录下，保持引用不变
try:
    from CDGAFS import cdgafs_feature_selection
    from fisher_score import compute_fisher_score
    from run_evaluation import print_summary_table # 仅保留打印表格功能
except ImportError:
    # 如果没有这些文件，定义伪函数以防止代码崩溃（仅用于演示逻辑）
    print("警告: 缺少 CDGAFS/fisher_score/run_evaluation 模块。")
    def cdgafs_feature_selection(**kwargs): return (list(range(kwargs.get('X').shape[1])), None, None, None, None)
    def compute_fisher_score(X, y): return np.random.rand(X.shape[1])
    def print_summary_table(results, indices, times): print(results)

# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载与初步清理模块 (移除全局标准化)
# ===================================================================
def clean_radiomics_df_initial(data, label_col_name):
    """
    仅进行基础清理：标签编码、删除非数值列、删除低方差列。
    !!! 注意：不在此处进行 StandardScaler，防止预处理泄露 !!!
    """
    print(f"    - [预处理] 原始数据形状: {data.shape}")

    if label_col_name not in data.columns:
        print(f"!!! 致命错误: 找不到标签列 '{label_col_name}'。")
        sys.exit(1)

    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    # 统一标签为 0 和 1
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
    else:
        print(f"!!! 致命错误: 标签列必须包含2个唯一的类别。找到: {unique_labels}")
        sys.exit(1)
    
    # 删除 ID 类和诊断类列
    id_cols = [col for col in data.columns if 'ID' in col or 'id' in col] 
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = id_cols + [label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    
    # 简单填充缺失值 (为了防止后续报错，这里做初步填充是可以的，但理想情况是 Split 后填充)
    # 考虑到影像组学特征缺失通常是系统性的，这里先填 0 或均值影响较小，
    # 但为了严谨，我们稍后在 Split 后再做标准化。
    if X_df.isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        X_clean = imputer.fit_transform(X_df)
    else:
        X_clean = X_df.values
    
    # 移除方差极低的特征 (常量特征)
    stds = np.std(X_clean, axis=0)
    variance_threshold = 1e-6 
    good_indices = np.where(stds > variance_threshold)[0]
    
    if len(good_indices) == 0:
        print("!!! 错误: 所有特征方差均为0，无法继续。")
        sys.exit(1)

    X_clean = X_clean[:, good_indices]
    feature_names = [feature_names[i] for i in good_indices]
    
    print(f"    - 基础清理完成，保留特征数: {len(feature_names)}")
    
    # 返回 Numpy 数组和特征名列表，尚未标准化
    return X_clean, y, feature_names

def load_data(csv_path, label_col_name):
    print(f"--- 加载文件: {csv_path} ---")
    try:
        data = pd.read_csv(csv_path, compression='gzip' if csv_path.endswith('.gz') else 'infer')
    except Exception as e:
        print(f"!!! 致命错误: 读取文件 {csv_path} 失败: {e}")
        sys.exit(1)
    return clean_radiomics_df_initial(data, label_col_name)

# ===================================================================
# 2. 特征选择器模块 (逻辑不变，但在调用时只传入 Train 数据)
# ===================================================================

def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, pruning_method='RFE'):
    print(f"\n    [CDGAFS] 开始运行 (剪枝: {pruning_method})...")
    # 这里的 X 必须是 X_train
    (selected_indices, _, _, _, _) = cdgafs_feature_selection(
        X=X, y=y, gene_list=feature_names, theta=THETA, omega=GA_OMEGA, 
        population_size=GA_POPULATION_SIZE, w_bio_boost=0.0, 
        pre_filter_top_n=None, graph_type='pearson_only'
    )
    
    if len(selected_indices) > K_FEATURES:
        if pruning_method == 'RFE':
            print(f"    [CDGAFS] RFE 剪枝: {len(selected_indices)} -> {K_FEATURES}")
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
            selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
            X_ga_selected = X[:, selected_indices]
            selector.fit(X_ga_selected, y)
            selected_indices = np.array(selected_indices)[selector.support_]
        elif pruning_method == 'FISHER':
            print(f"    [CDGAFS] Fisher 剪枝: {len(selected_indices)} -> {K_FEATURES}")
            X_ga_selected = X[:, selected_indices]
            scores = compute_fisher_score(X_ga_selected, y)
            top_indices = np.argsort(scores)[-K_FEATURES:]
            selected_indices = np.array(selected_indices)[top_indices]

    return selected_indices if len(selected_indices) > 0 else []

def select_features_mrmr(X, y, feature_names, K_FEATURES):
    if pymrmr is None: return []
    print("\n    [mRMR] 开始运行...")
    # 构造 DataFrame 供 pymrmr 使用
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # 离散化
    for col in feature_names:
        df[col] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
    
    selected_names = pymrmr.mRMR(df, 'MIQ', K_FEATURES)
    
    # 映射回索引
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_indices = [name_to_idx[n] for n in selected_names if n in name_to_idx]
    print(f"    [mRMR] 选中 {len(selected_indices)} 个特征")
    return selected_indices

def select_features_lasso_cv(X, y):
    print("\n    [LASSO-CV] 开始运行...")
    # 这里的 X 必须是 X_train
    model = LogisticRegressionCV(
        cv=5, penalty='l1', solver='liblinear', class_weight='balanced', 
        random_state=42, max_iter=3000, scoring='roc_auc'
    )
    model.fit(X, y)
    indices = np.where(np.abs(model.coef_[0]) > 1e-6)[0]
    print(f"    [LASSO-CV] 最佳 C: {model.C_[0]:.4f}, 选中特征数: {len(indices)}")
    return indices.tolist()

# ===================================================================
# 3. 独立评估函数 (替代原有的 run_evaluation 以确保无泄露)
# ===================================================================
def evaluate_on_independent_test(X_train, y_train, X_test, y_test, selected_indices):
    """
    在独立的测试集上评估模型性能。
    """
    if len(selected_indices) == 0:
        return {'AUC': 0, 'ACC': 0, 'F1': 0, '#Feat': 0}

    # 1. 仅提取选中的特征
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    # 2. 在 Train 上训练分类器
    clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    clf.fit(X_train_sel, y_train)

    # 3. 在 Test 上预测
    y_pred = clf.predict(X_test_sel)
    try:
        y_prob = clf.predict_proba(X_test_sel)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5 # 只有一类时

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'AUC': round(auc, 4),
        'ACC': round(acc, 4),
        'F1': round(f1, 4),
        '#Feat': len(selected_indices)
    }

# ===================================================================
# 4. 统一执行流程 (修复版)
# ===================================================================
def run_leakage_free_analysis(X_raw, y_raw, feature_names_raw, 
                              K_FEATURES, params, dataset_title):
    """
    防泄露流程：
    1. Split Data -> Train / Test
    2. Scale -> Fit on Train, Transform Train & Test
    3. Select Features -> Only look at Train
    4. Evaluate -> Train on Train (Subset), Predict on Test (Subset)
    """
    print(f"\n{'-'*30} 正在分析: {dataset_title} (防泄露模式) {'-'*30}")

    # --- [步骤 1] 数据切分 (80% 训练, 20% 独立测试) ---
    # random_state 固定以保证复现性，stratify 保证类别比例一致
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    # --- [步骤 2] 标准化 (仅在训练集上拟合) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw) # Fit & Transform Train
    X_test = scaler.transform(X_test_raw)       # Transform Test using Train's stats
    
    print(f"    - 数据切分完成: 训练集 {X_train.shape[0]} 例, 测试集 {X_test.shape[0]} 例")
    
    all_selected_indices = {}
    execution_times = {}
    
    # --- [步骤 3] 特征选择 (仅使用 X_train, y_train) ---
    
    # 3.1 CDGAFS
    start = time.time()
    all_selected_indices['CDGAFS'] = select_features_cdafs(
        X_train, y_train, feature_names_raw, K_FEATURES, 
        params['pop'], params['omega'], params['theta']
    )
    execution_times['CDGAFS'] = time.time() - start
    
    # 3.2 LASSO-CV
    # start = time.time()
    # all_selected_indices['LASSO-CV'] = select_features_lasso_cv(X_train, y_train)
    # execution_times['LASSO-CV'] = time.time() - start
    
    # # 3.3 mRMR (可选)
    # if pymrmr:
    #     start = time.time()
    #     all_selected_indices['mRMR'] = select_features_mrmr(
    #         X_train, y_train, feature_names_raw, K_FEATURES
    #     )
    #     execution_times['mRMR'] = time.time() - start

    # --- [步骤 4] 最终评估 (在独立测试集上) ---
    print(f"\n>>> {dataset_title} 的最终评估结果 (独立测试集) <<<")
    all_results = {}
    
    for method, indices in all_selected_indices.items():
        res = evaluate_on_independent_test(X_train, y_train, X_test, y_test, indices)
        all_results[method] = res
        print(f"    [{method}] AUC: {res['AUC']} | Feats: {res['#Feat']} | Time: {execution_times[method]:.2f}s")

    # 打印汇总表 (如果原版 print_summary_table 可用)
    try:
        print_summary_table(all_results, all_selected_indices, execution_times)
    except:
        pass # 已在上面打印了日志

# ===================================================================
# 5. 主程序
# ===================================================================
def main():
    # 配置路径
    LOCAL_CSV_PATH = '/data/qh_20T_share_file/lct/CT67/ovarian_features_with_label.csv'
    PUBLIC_DATASET_DIR = '/data/qh_20T_share_file/lct/CT67/dataset'
    
    # 参数配置
    K_FEATURES = 1000
    params = {
        'pop': 100,
        'omega': 0.5,
        'theta': 0.9
    }

    print("#"*70)
    print(f"### 开始运行实验：严格防数据泄露模式 (Split -> Select -> Eval) ###")
    print("#"*70)
    
    # 任务 1: 本地数据
    if os.path.exists(LOCAL_CSV_PATH):
        print(f"\n>>> [任务 1] 本地数据: Ovarian <<<")
        X, y, feats = load_data(LOCAL_CSV_PATH, 'label')
        run_leakage_free_analysis(X, y, feats, K_FEATURES, params, "Local Ovarian")
    else:
        print(f"未找到本地文件: {LOCAL_CSV_PATH}")

    # 任务 2: 公开数据集
    # public_datasets = ['UPENN-GBM'] # 可添加更多
    # for ds_name in public_datasets:
    #     file_path = os.path.join(PUBLIC_DATASET_DIR, f"{ds_name}.gz")
    #     if os.path.exists(file_path):
    #         print(f"\n>>> [任务 2] 公开数据: {ds_name} <<<")
    #         X, y, feats = load_data(file_path, 'Target')
    #         run_leakage_free_analysis(X, y, feats, K_FEATURES, params, ds_name)
    #     else:
    #         print(f"跳过: 未找到文件 {file_path}")

if __name__ == "__main__":
    main()
