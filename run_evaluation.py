# 文件名: run_evaluation.py
# 描述: 专门用于模型评估和结果报告

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix

def evaluate_model_performance(X, y, selected_indices):
    """
    使用5折交叉验证和L2逻辑回归，评估所选特征的
    Accuracy, AUC, Sensitivity, Specificity, 和 F1-Macro.
    
    参数:
        X (np.ndarray): 已经标准化后的 *完整* 特征矩阵
        y (np.ndarray): 标签数组 (0 和 1)
        selected_indices (list): 要评估的特征索引列表
    
    返回:
        dict: 包含所有平均指标的字典
    """
    
    if len(selected_indices) == 0:
        print("    - !!! 跳过评估，因为没有选出任何特征。")
        return {"Accuracy": 0, "AUC": 0, "Sensitivity": 0, "Specificity": 0, "F1-Macro": 0}
        
    X_subset = X[:, selected_indices]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {
        "acc": [], "auc": [], "sens": [], "spec": [], "f1": []
    }

    # (用于 specificity)
    # y_test 中必须包含 0 和 1 两个类别才能计算所有指标
    valid_folds = 0 

    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 确保测试集中有两个类别
        if len(np.unique(y_test)) < 2:
            continue # 跳过这个无法评估的折
        
        valid_folds += 1
        
        # 使用L2逻辑回归 (Ridge) 作为分类器
        model = LogisticRegressionCV(
            Cs=10, cv=3, penalty='l2', solver='liblinear', 
            random_state=42, max_iter=1000
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # 类别1的概率
        
        # 计算所有指标
        metrics["acc"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred, average='macro'))
        metrics["auc"].append(roc_auc_score(y_test, y_proba))
        
        # Sensitivity (Recall for positive class 1)
        metrics["sens"].append(recall_score(y_test, y_pred, pos_label=1, average='binary'))
        
        # Specificity (Recall for negative class 0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP)
        metrics["spec"].append(specificity)

    if valid_folds == 0:
        print("    - !!! 评估失败，交叉验证的每个折中都只有一个类别。")
        return {"Accuracy": 0, "AUC": 0, "Sensitivity": 0, "Specificity": 0, "F1-Macro": 0}

    # 计算平均值
    avg_results = {
        "Accuracy": np.mean(metrics["acc"]),
        "AUC": np.mean(metrics["auc"]),
        "Sensitivity": np.mean(metrics["sens"]),
        "Specificity": np.mean(metrics["spec"]),
        "F1-Macro": np.mean(metrics["f1"])
    }
    
    return avg_results

def print_summary_table(all_results, all_selected_indices):
    """
    打印最终的对比表格
    """
    print("\n" + "#"*70)
    print("### 最终实验对比总结 ###")
    print("#"*70)
    
    header = f"{'Method':<12} | {'K':<4} | {'Accuracy':<10} | {'AUC':<10} | {'Sensitivity':<11} | {'Specificity':<11} | {'F1-Macro':<10}"
    print(header)
    print("-" * len(header))
    
    # 按 Accuracy 排序
    sorted_methods = sorted(all_results.items(), key=lambda item: item[1].get('Accuracy', 0), reverse=True)
    
    for method_name, metrics in sorted_methods:
        k = len(all_selected_indices.get(method_name, []))
        print(f"{method_name:<12} | {k:<4} | "
              f"{metrics.get('Accuracy', 0):<10.4f} | "
              f"{metrics.get('AUC', 0):<10.4f} | "
              f"{metrics.get('Sensitivity', 0):<11.4f} | "
              f"{metrics.get('Specificity', 0):<11.4f} | "
              f"{metrics.get('F1-Macro', 0):<10.4f}")
              
    print("#"*70)