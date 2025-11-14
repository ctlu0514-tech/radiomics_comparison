import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd     # pandas库为2.0.3版本，写入函数为pd.concat()，其他版本可能为pd.append()或者.append()或者其他可能，根据报错修改59行即可
import os
#import numpy as np
#import nibabel as nib
#import nrrd

df = pd.DataFrame()  # 创建一个空的DataFrame用于提取特征的写入

# 特征提取器CT参数设置,已包含重采样至1*1*1mm，插值方式为BSpline
settings = {}
settings['binWidth'] = 25  # 5
settings['sigma'] = [3, 5]
settings['Interpolator'] = sitk.sitkBSpline
settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
settings['voxelArrayShift'] = 1000  # 300
settings['normalize'] = True
settings['normalizeScale'] = 100
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
# 直接启用所有图像转化类别，或者指定使用 LoG 和 Wavelet 滤波器
extractor.enableAllImageTypes()
#extractor.enableImageTypeByName('LoG')
#extractor.enableImageTypeByName('Wavelet')
# 启用所有特征
extractor.enableAllFeatures()
# 手动启用默认禁用的一阶特征和形状特征
extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy','Minimum', '10Percentile', '90Percentile',
                                                 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range',
                                                 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation','RootMeanSquared',
                                                 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 
                                            'Sphericity', 'SphericalDisproportion',  'Maximum3DDiameter', 'Maximum2DDiameterSlice', 
                                            'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 
                                            'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])
print("抽取参数:\n\t", extractor.settings)
print("启用的滤波器:\n\t", extractor.enabledImagetypes)
print("启用的特征:\n\t", extractor.enabledFeatures)
# # 设定图像和分割路径，路径要用双反斜杠\\，图像和分割的文件名应该一一对应
# imagePath = 'G:\\06-22CT174\\Image'
# maskPath = 'G:\\06-22CT174\\Mask'
# # 获取路径下所有病人的图像名和分割名，如3001.nii.gz和3001.seg.nrrd
# patient_list = sorted(os.listdir(imagePath))
# patient_mask_list = sorted(os.listdir(maskPath))
# 图像和掩码的主目录路径
image_path = r'/data/qh_20T_share_file/lct/CT67/22-23CT67/Image'
mask_path = r'/data/qh_20T_share_file/lct/CT67/22-23CT67/Mask'

print("正在搜寻图像和掩码...")
patient_list = []
patient_mask_list = []
patient_id_list = []

# 获取Image文件夹下所有文件名 (e.g., '4001.nii')
# 我们假设Image文件夹里只有图像文件
try:
    image_files = sorted(os.listdir(image_path))
except FileNotFoundError:
    print(f"!!! 错误: 找不到路径 {image_path}")
    print("!!! 请检查第 48 行的 image_path 路径是否正确")
    exit() # 路径错了，直接退出

for image_filename in image_files:
    # 确保我们只处理 .nii 文件 (你也可以改成 .nii.gz)
    if not image_filename.endswith('.nii'):
        print(f"跳过非 .nii 文件: {image_filename}")
        continue

    # 构建图像的完整路径
    image_full_path = os.path.join(image_path, image_filename)

    # --- 关键：构建对应的掩码文件名 ---
    # 从 '4001.nii' 得到 '4001'
    base_name = image_filename.split('.nii')[0]

    # 优先尝试 .seg.nrrd 格式
    mask_filename_nrrd = f"{base_name}.seg.nrrd"
    mask_full_path = os.path.join(mask_path, mask_filename_nrrd)

    # 如果 .seg.nrrd 不存在，再尝试 .nii 格式
    if not os.path.exists(mask_full_path):
        mask_filename_nii = f"{base_name}.nii"
        mask_full_path = os.path.join(mask_path, mask_filename_nii)

    # 检查这个配对的掩码文件是否存在
    if os.path.exists(mask_full_path) and os.path.exists(image_full_path):
        # 如果都存在，才把 *完整路径* 加入列表
        patient_list.append(image_full_path)
        patient_mask_list.append(mask_full_path)
        patient_id_list.append(base_name)
    else:
        if not os.path.exists(mask_full_path):
            print(f"--- 警告: 找到了图像 {image_filename}, 但未找到对应的掩码 {mask_full_path}")
        if not os.path.exists(image_full_path):
             print(f"--- 警告: 掩码存在，但图像路径构建失败 {image_full_path}")

# 检查和打印 (这段和原来一样)
print("Patient List (找到的配对):", patient_list)
print("Patient List(n):", len(patient_list))
print("Patient Mask List (找到的配对):", patient_mask_list)
print("Patient Mask List(n):", len(patient_mask_list))
# 判断两个列表长度是否一致
assert len(patient_list) == len(patient_mask_list), "两个列表长度不一致"
successful_patient_ids = []

# 遍历每个病人的图像和分割进行特征提取
for image_full_path, mask_full_path, patient_id in zip(patient_list, patient_mask_list, patient_id_list):

    print(f"--- 正在提取: {os.path.basename(image_full_path)} ---")
    
    # 提取特征 (直接使用我们生成的完整路径!)
    try:
        # features = extractor.execute(image_full_path, mask_full_path) # 原来的
        features = extractor.execute(image_full_path, mask_full_path, label=1) # 推荐：指定label=1
        
        # 用features_dict存放临时的特征数据
        features_dict = dict()
        # 输出特征
        for key, value in features.items():  
            features_dict[key] = value
        
        # 将临时特征写入df
        df = pd.concat([df, pd.DataFrame([features_dict])], ignore_index=True)
        successful_patient_ids.append(patient_id)
        print(f"--- {os.path.basename(image_full_path)} 特征提取完成 ---")

    except Exception as e:
        print(f"!!!! 提取失败 {os.path.basename(image_full_path)}，错误: {e}")
        print("     图像路径:", image_full_path)
        print("     掩码路径:", mask_full_path)
        # 你可以选择是跳过 (continue) 还是停止 (break)
        continue

# 将特征写入csv文件，需要修改路径，路径要用双反斜杠\\
df.columns = features_dict.keys()
df.insert(0, 'ID', successful_patient_ids)
df.to_csv(r'/data/qh_20T_share_file/lct/CT67/ovarian_features.csv', index=0)
print('Done')
