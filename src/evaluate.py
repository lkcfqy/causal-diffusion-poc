# src/evaluate.py (已更新)
import torch
import yaml
import os
from torchvision.utils import save_image, make_grid
import numpy as np
from pathlib import Path

from models import ConditionalUNet
from diffusion import Diffusion

# --- 路径管理 (保持不变) ---
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent

def get_color_distribution(images):
    # ... (此函数保持不变)
    mean_r = images[:, 0, :, :].mean(dim=[1, 2])
    mean_g = images[:, 1, :, :].mean(dim=[1, 2])
    num_red = (mean_r > mean_g).sum().item()
    total = len(images)
    return {'red': num_red / total * 100, 'green': (total - num_red) / total * 100}

def evaluate_h1_robustness(cfg, model, diffusion):
    # ... (此函数保持不变)
    print(f"\n--- H1: OOD 鲁棒性评估 ({cfg['MODEL_TYPE'].upper()}) ---")
    save_dir = PROJECT_ROOT / 'results' / 'h1_robustness' / cfg['MODEL_TYPE']
    os.makedirs(save_dir, exist_ok=True)
    n_samples = 64

    for digit_idx, digit_val in enumerate(cfg['DIGITS']):
        print(f"正在为数字 '{digit_val}' 生成图像...")
        z_c = torch.tensor([digit_idx] * n_samples, device=cfg['DEVICE'])
        
        if cfg['MODEL_TYPE'] == 'baseline':
            generated_images = diffusion.sample(model, n_samples, z_c)
        else:
            z_s = torch.tensor([0]*(n_samples//2) + [1]*(n_samples//2), device=cfg['DEVICE'])
            generated_images = diffusion.sample(model, n_samples, z_c, z_s)

        grid = make_grid(generated_images, nrow=8)
        save_image(grid, save_dir / f'generated_digit_{digit_val}.png')
        
        dist = get_color_distribution(generated_images)
        print(f"  > 颜色分布: 红色: {dist['red']:.1f}%, 绿色: {dist['green']:.1f}%")
        
    print(f"H1评估图像已保存到 {save_dir}")

# --- SCD的反事实评估函数 (保持不变) ---
def run_single_counterfactual_test(cfg, model, diffusion, data, source_digit, source_color, target_digit):
    # ... (此函数保持不变, 但我们将修改保存目录)
    device = cfg['DEVICE']
    # 【修改】为SCD的结果创建一个专门的子目录
    save_dir = PROJECT_ROOT / 'results' / 'h2_counterfactuals' / 'scd'
    os.makedirs(save_dir, exist_ok=True)

    digit_map = {cfg['DIGITS'][0]: 0, cfg['DIGITS'][1]: 1}
    color_map = {'red': 0, 'green': 1}
    
    source_digit_idx = digit_map[source_digit]
    source_color_idx = color_map[source_color]
    target_digit_idx = digit_map[target_digit]

    try:
        images = data['images'] * 2 - 1
        causal_labels = data['causal_labels']
        spurious_labels = data['spurious_labels']
        idx = np.where((causal_labels.numpy() == source_digit_idx) & (spurious_labels.numpy() == source_color_idx))[0][0]
    except IndexError:
        print(f"警告: 在训练集中未找到 '{source_color} {source_digit}' 的样本。跳过此测试。")
        return
        
    original_image = ((images[idx] + 1) * 0.5).unsqueeze(0)
    
    z_c_target = torch.tensor([target_digit_idx], device=device)
    z_s_kept = torch.tensor([source_color_idx], device=device)
    
    print(f"正在执行反事实: '{source_color} {source_digit}' -> '{source_color} {target_digit}' ...")
    counterfactual_image = diffusion.sample(model, 1, z_c_target, z_s_kept)
    
    comparison_grid = make_grid(torch.cat([original_image.cpu(), counterfactual_image.cpu()]), nrow=2)
    filename = f"cf_{source_color}_{source_digit}_to_{source_color}_{target_digit}.png"
    save_path = save_dir / filename
    save_image(comparison_grid, save_path)
    print(f"  > 结果已保存到: {save_path.name}")


def evaluate_h2_counterfactuals_symmetric(cfg, model, diffusion):
    # ... (此函数保持不变)
    if cfg['MODEL_TYPE'] != 'scd': return
    print("\n--- H2: 对称性反事实可解释性评估 (SCD) ---")
    data = torch.load(PROJECT_ROOT / 'data' / 'train.pt')
    run_single_counterfactual_test(cfg, model, diffusion, data, source_digit=3, source_color='red', target_digit=7)
    run_single_counterfactual_test(cfg, model, diffusion, data, source_digit=3, source_color='green', target_digit=7)
    run_single_counterfactual_test(cfg, model, diffusion, data, source_digit=7, source_color='red', target_digit=3)
    run_single_counterfactual_test(cfg, model, diffusion, data, source_digit=7, source_color='green', target_digit=3)


# =================================================================================
# === 【新增】为 Baseline 模型设计的伪反事实评估函数 ===
# =================================================================================
def run_single_baseline_test(cfg, model, diffusion, data, source_digit, source_color, target_digit):
    """
    为Baseline模型执行单次伪反事实测试并保存结果。
    """
    device = cfg['DEVICE']
    # 为Baseline的结果创建一个专门的子目录
    save_dir = PROJECT_ROOT / 'results' / 'h2_counterfactuals' / 'baseline'
    os.makedirs(save_dir, exist_ok=True)

    # 映射名称到索引
    digit_map = {cfg['DIGITS'][0]: 0, cfg['DIGITS'][1]: 1}
    color_map = {'red': 0, 'green': 1}
    
    source_digit_idx = digit_map[source_digit]
    source_color_idx = color_map[source_color]
    target_digit_idx = digit_map[target_digit]

    # 1. Abduction: 从数据集中找到一个源图像 (与SCD相同)
    images = data['images'] * 2 - 1
    causal_labels = data['causal_labels']
    spurious_labels = data['spurious_labels']
    
    try:
        idx = np.where((causal_labels.numpy() == source_digit_idx) & (spurious_labels.numpy() == source_color_idx))[0][0]
    except IndexError:
        print(f"警告: 在训练集中未找到 '{source_color} {source_digit}' 的样本。跳过此测试。")
        return
        
    original_image = ((images[idx] + 1) * 0.5).unsqueeze(0)
    
    # 2. Action: 定义干预 (关键区别)
    # Baseline模型无法接收颜色(z_s)条件。它只能接收目标数字。
    z_c_target = torch.tensor([target_digit_idx], device=device)
    
    # 3. Prediction: 生成图像
    print(f"正在为Baseline执行任务: 从 '{source_color} {source_digit}' 生成最可能的 '{target_digit}' ...")
    # 注意：这里调用 sample 时没有 z_s 参数
    generated_image = diffusion.sample(model, 1, z_c_target)
    
    # 合并并保存
    comparison_grid = make_grid(torch.cat([original_image.cpu(), generated_image.cpu()]), nrow=2)
    filename = f"baseline_cf_{source_color}_{source_digit}_to_{target_digit}.png"
    save_path = save_dir / filename
    save_image(comparison_grid, save_path)
    print(f"  > 结果已保存到: {save_path.name}")


def evaluate_h2_for_baseline(cfg, model, diffusion):
    """
    对Baseline模型执行完整的对称性伪反事实评估。
    """
    if cfg['MODEL_TYPE'] != 'baseline': return
    
    print("\n--- H2: 伪反事实评估 (BASELINE) ---")
    
    # 加载一次数据以供所有测试使用
    data = torch.load(PROJECT_ROOT / 'data' / 'train.pt')

    # 执行所有四个对称性测试
    run_single_baseline_test(cfg, model, diffusion, data, source_digit=3, source_color='red', target_digit=7)
    run_single_baseline_test(cfg, model, diffusion, data, source_digit=3, source_color='green', target_digit=7)
    run_single_baseline_test(cfg, model, diffusion, data, source_digit=7, source_color='red', target_digit=3)
    run_single_baseline_test(cfg, model, diffusion, data, source_digit=7, source_color='green', target_digit=3)


if __name__ == '__main__':
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 重新加载扩散工具
    diffusion = Diffusion(cfg)

    print("\n开始评估 BASELINE 模型...")
    cfg['MODEL_TYPE'] = 'baseline'
    model_baseline = ConditionalUNet(cfg).to(cfg['DEVICE'])
    model_baseline.load_state_dict(torch.load(PROJECT_ROOT / 'saved_models' / 'baseline_model.pt', map_location=cfg['DEVICE']))
    model_baseline.eval()
    evaluate_h1_robustness(cfg, model_baseline, diffusion)
    # 【新增】调用对Baseline的H2评估
    evaluate_h2_for_baseline(cfg, model_baseline, diffusion)
    
    print("\n开始评估 SCD 模型...")
    cfg['MODEL_TYPE'] = 'scd'
    model_scd = ConditionalUNet(cfg).to(cfg['DEVICE'])
    model_scd.load_state_dict(torch.load(PROJECT_ROOT / 'saved_models' / 'scd_model.pt', map_location=cfg['DEVICE']))
    model_scd.eval()
    evaluate_h1_robustness(cfg, model_scd, diffusion)
    evaluate_h2_counterfactuals_symmetric(cfg, model_scd, diffusion)