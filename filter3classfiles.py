import os
import cv2
import shutil

# 源目录：存放所有抽帧图片的文件夹
SRC_DIR = r"H:\lanefollowing\_out_dataset"
# 目标根目录：里面应该已有 left/ straight/ right 三个子文件夹
DST_ROOT = r"H:\lanefollowing\dataset"

# 确保目标子文件夹存在
for cls in ("left", "straight", "right"):
    os.makedirs(os.path.join(DST_ROOT, cls), exist_ok=True)

# 创建可缩放窗口并设置初始尺寸
win_name = "Label (A: left, W: straight, D: right)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 640, 480)

for fname in sorted(os.listdir(SRC_DIR)):
    src_path = os.path.join(SRC_DIR, fname)
    # 跳过非文件
    if not os.path.isfile(src_path):
        continue
    # 只处理图片后缀
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print("Reading:", src_path)
    img = cv2.imread(src_path)
    # 读取失败时跳过
    if img is None:
        print(f"Cannot read, skipping: {src_path}")
        continue

    # 统一缩放后显示
    disp = cv2.resize(img, (640, 480))
    cv2.imshow(win_name, disp)
    key = cv2.waitKey(0) & 0xFF

    # 根据按键决定移动到哪个子目录
    if key == ord('a'):
        dst_path = os.path.join(DST_ROOT, "left", fname)
    elif key == ord('w'):
        dst_path = os.path.join(DST_ROOT, "straight", fname)
    elif key == ord('d'):
        dst_path = os.path.join(DST_ROOT, "right", fname)
    else:
        # 其他按键跳过，不移动
        print("Skipped:", fname)
        continue

    shutil.move(src_path, dst_path)
    print(f"Moved {fname} → {os.path.basename(dst_path)}")

cv2.destroyAllWindows()

