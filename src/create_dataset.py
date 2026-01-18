# src/create_dataset.py (已修正版)
import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path

# --- 路径管理 (保持不变) ---
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
SAMPLES_DIR = PROJECT_ROOT / 'results' / 'data_samples'

# --- 配置 (保持不变) ---
IMG_SIZE = 32
DIGITS = [3, 7]

def colorize(img_tensor, color):
    img_rgb = img_tensor.repeat(3, 1, 1)
    if color == 'red':
        img_rgb[1, :, :] = 0
        img_rgb[2, :, :] = 0
    elif color == 'green':
        img_rgb[0, :, :] = 0
        img_rgb[2, :, :] = 0
    return img_rgb

# =================================================================================
# === create_causal_mnist 函数已重构，以修复概率分配逻辑 ===
# =================================================================================
def create_causal_mnist(dataset, digits, correlation_prob, is_ood=False):
    filtered_data = []
    causal_labels, spurious_labels = [], []
    digit_A, digit_B = digits[0], digits[1]
    
    for img, label in dataset:
        if label not in digits: continue

        # --- 核心修正：为每个数字独立设置颜色概率 ---
        if label == digit_A: # 当数字是 3
            # is_ood=False (训练集): 90%概率为红
            # is_ood=True (测试集): 10%概率为红
            prob_red = correlation_prob if not is_ood else (1 - correlation_prob)
            color = 'red' if np.random.rand() < prob_red else 'green'

        elif label == digit_B: # 当数字是 7
            # is_ood=False (训练集): 90%概率为绿
            # is_ood=True (测试集): 10%概率为绿
            prob_green = correlation_prob if not is_ood else (1 - correlation_prob)
            color = 'green' if np.random.rand() < prob_green else 'red'
        
        filtered_data.append(colorize(img, color))
        causal_labels.append(0 if label == digit_A else 1)
        spurious_labels.append(0 if color == 'red' else 1)

    return torch.stack(filtered_data), torch.tensor(causal_labels), torch.tensor(spurious_labels)

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print("正在创建 Causal-MNIST 数据集 (已修正逻辑)...")
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    # 1. 创建训练集 (IID - 90% 红3, 90% 绿7)
    train_images, train_zc, train_zs = create_causal_mnist(mnist_train, DIGITS, 0.9, is_ood=False)
    train_path = DATA_DIR / 'train.pt'
    torch.save({'images': train_images, 'causal_labels': train_zc, 'spurious_labels': train_zs}, train_path)
    print(f"训练集创建完成: {len(train_images)} 张图片。保存在 {train_path}")

    # 2. 创建测试集 (OOD - 90% 绿3, 90% 红7)
    test_images, test_zc, test_zs = create_causal_mnist(mnist_test, DIGITS, 0.9, is_ood=True)
    test_path = DATA_DIR / 'test.pt'
    torch.save({'images': test_images, 'causal_labels': test_zc, 'spurious_labels': test_zs}, test_path)
    print(f"测试集创建完成: {len(test_images)} 张图片。保存在 {test_path}")

    # (可选) 保存样本
    to_pil = transforms.ToPILImage()
    for i in range(10):
        img_pil = to_pil(train_images[i])
        zc_label = DIGITS[train_zc[i]]
        zs_label = 'red' if train_zs[i] == 0 else 'green'
        img_pil.save(SAMPLES_DIR / f'train_sample_{i}_digit{zc_label}_{zs_label}.png')
    print(f"数据样本已保存到 {SAMPLES_DIR}")