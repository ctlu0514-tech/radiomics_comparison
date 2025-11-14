import pandas as pd
import os
import sys

# --- 1. 在这里修改你的文件路径和列名 ---

# (必需) 你的特征文件路径
FEATURES_FILE_PATH = r'/data/qh_20T_share_file/lct/CT67/ovarian_features.csv'

# (必需) 你的标签文件路径
LABEL_FILE_PATH = r'/data/qh_20T_share_file/lct/CT67/output.csv'

# (必需) 特征文件中包含ID的列名 (来自你的上一个脚本)
FEATURES_ID_COLUMN_NAME = 'ID'

# (必需) 标签文件中包含ID的列名
LABEL_ID_COLUMN_NAME = 'CT'

# (必需) 标签文件中包含标签的列名
LABEL_COLUMN_NAME = 'label'

# (推荐) 合并后保存的新文件名
OUTPUT_FILE_PATH = r'/data/qh_20T_share_file/lct/CT67/ovarian_features_with_label.csv'


# --- 2. 主程序 ---

print("--- 脚本开始：合并特征和标签 ---")

# --- 加载数据 ---
try:
    print(f"正在加载特征文件: {FEATURES_FILE_PATH}")
    df_features = pd.read_csv(FEATURES_FILE_PATH)
except FileNotFoundError:
    print(f"!!! 致命错误: 找不到特征文件 '{FEATURES_FILE_PATH}'。请检查路径。")
    sys.exit(1)

try:
    print(f"正在加载标签文件: {LABEL_FILE_PATH}")
    df_labels = pd.read_csv(LABEL_FILE_PATH)
except FileNotFoundError:
    print(f"!!! 致命错误: 找不到标签文件 '{LABEL_FILE_PATH}'。")
    sys.exit(1)

print("\n--- 步骤 1: 准备特征文件 (ovarian_features.csv) ---")
# 检查特征ID列是否存在
if FEATURES_ID_COLUMN_NAME not in df_features.columns:
    print(f"!!! 致命错误: 在 {FEATURES_FILE_PATH} 中找不到 '{FEATURES_ID_COLUMN_NAME}' 列。")
    print(f"    - 实际找到的列: {df_features.columns.tolist()}")
    sys.exit(1)

# 确保ID为字符串类型，以便安全合并
df_features[FEATURES_ID_COLUMN_NAME] = df_features[FEATURES_ID_COLUMN_NAME].astype(str)
print(f"    - 找到了 '{FEATURES_ID_COLUMN_NAME}' 列。示例: {df_features[FEATURES_ID_COLUMN_NAME].iloc[0]}")


print("\n--- 步骤 2: 准备标签文件 (output.csv) ---")
# 检查所需的列是否存在
if LABEL_ID_COLUMN_NAME not in df_labels.columns or LABEL_COLUMN_NAME not in df_labels.columns:
    print(f"!!! 致命错误: 标签文件 {LABEL_FILE_PATH} 中必须包含 '{LABEL_ID_COLUMN_NAME}' 和 '{LABEL_COLUMN_NAME}' 列。")
    print(f"    - 实际找到的列: {df_labels.columns.tolist()}")
    sys.exit(1)

# 1. 提取我们需要的两列: ID 和 标签
df_labels_subset = df_labels[[LABEL_ID_COLUMN_NAME, LABEL_COLUMN_NAME]]
# 2. 确保标签文件中的ID (CT列) 也是字符串，以便匹配
df_labels_subset[LABEL_ID_COLUMN_NAME] = df_labels_subset[LABEL_ID_COLUMN_NAME].astype(str)

print(f"    - 已加载 {len(df_labels_subset)} 条标签记录。")


print("\n--- 步骤 3: 合并两个文件 ---")
# 使用 'left' 合并：保留 features 文件中的所有行，
# 并根据 'ID' 和 'CT' 的匹配，将 'label' 添加进去
df_merged = pd.merge(
    df_features,
    df_labels_subset,
    left_on=FEATURES_ID_COLUMN_NAME,    # 特征文件中的ID列
    right_on=LABEL_ID_COLUMN_NAME,      # 标签文件中的ID列
    how='left'                          # 保留所有特征，即使没有标签
)


print("\n--- 步骤 4: 清理和验证 ---")

# 1. 检查是否有未匹配上的行
missing_count = df_merged[LABEL_COLUMN_NAME].isna().sum()
if missing_count > 0:
    print(f"!!! 警告: 有 {missing_count} 行特征未能匹配到标签。")
    missing_ids = df_merged[df_merged[LABEL_COLUMN_NAME].isna()][FEATURES_ID_COLUMN_NAME].unique()
    print(f"    - 未匹配到的ID (最多显示20个): {missing_ids[:20]}")
    print(f"    - 请检查你的标签文件 {LABEL_FILE_PATH}，确保这些ID存在并且格式一致。")
else:
    print("    - 所有特征行都成功匹配到了标签！")

# 2. 重新整理列的顺序 (ID第一列，label最后一列)
#    删除多余的 'CT' 列
df_final = df_merged.drop([LABEL_ID_COLUMN_NAME], axis=1)

#    获取所有列名
all_cols = df_final.columns.tolist()
#    把 'ID' 和 'label' 抽出来
all_cols.remove(FEATURES_ID_COLUMN_NAME)
all_cols.remove(LABEL_COLUMN_NAME)
#    按你希望的顺序重新组合
final_order = [FEATURES_ID_COLUMN_NAME] + all_cols + [LABEL_COLUMN_NAME]
#    应用新顺序
df_final = df_final[final_order]

print(f"    - 已将 '{LABEL_COLUMN_NAME}' 列添加到最后一列。")


print(f"\n--- 步骤 5: 保存结果 ---")
df_final.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"    - 成功！合并后的文件已保存到: {OUTPUT_FILE_PATH}")
print(f"    - 最终文件包含 {len(df_final)} 行 和 {len(df_final.columns)} 列。")

print("\n--- 脚本执行完毕 ---")