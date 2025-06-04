import os
import cv2
import numpy as np
from skimage.feature import hog

# 配置
DATA_DIR   = r"H:\lanefollowing\dataset"               # 根目录，里面有 straight/、left/、right/
IMAGE_SIZE = (128, 128)
LABEL_MAP  = {"straight": 0, "left": -1, "right": 1}

all_features = []
all_labels   = []

# 批量遍历每个类别子文件夹
for class_name, class_idx in LABEL_MAP.items():
    class_folder = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_folder):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(class_folder, fname)

        # 1. 读取并预处理
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        edges = cv2.Canny(img, 50, 150)

        # 2. 提取 HOG 特征
        feat = hog(edges,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False)

        all_features.append(feat)
        all_labels.append(class_idx)

# 转 numpy 数组并保存
X = np.array(all_features)  # (N, D)
y = np.array(all_labels)    # (N,)

np.save("X_features1.npy", X)
np.save("y_labels1.npy", y)

print(f"一共处理了 {X.shape[0]} 张图片，特征维度 = {X.shape[1]}")
