# src/train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import yaml
import os
from tqdm import tqdm
from pathlib import Path

from models import ConditionalUNet
from diffusion import Diffusion

# --- 路径管理 (关键修正) ---
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent

def train(cfg):
    device = cfg['DEVICE']
    model_type = cfg['MODEL_TYPE']
    save_dir = PROJECT_ROOT / 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- 开始训练 {model_type.upper()} 模型 ---")
    print(f"使用设备: {device}")

    # 1. 加载数据
    data_path = PROJECT_ROOT / 'data' / 'train.pt'
    data = torch.load(data_path, weights_only=True)
    images = data['images'] * 2 - 1 # 归一化到 [-1, 1]
    dataset = TensorDataset(images, data['causal_labels'], data['spurious_labels'])
    dataloader = DataLoader(dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=4)

    # 2. 初始化模型、扩散和优化器
    model = ConditionalUNet(cfg).to(device)
    diffusion = Diffusion(cfg)
    optimizer = AdamW(model.parameters(), lr=cfg['LEARNING_RATE'])
    
    # 3. 训练循环
    for epoch in range(cfg['EPOCHS']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['EPOCHS']}")
        epoch_loss = 0
        for step, (batch_images, batch_zc, batch_zs) in enumerate(pbar):
            optimizer.zero_grad()
            
            batch_images = batch_images.to(device)
            batch_zc = batch_zc.to(device)
            batch_zs = batch_zs.to(device)

            t = torch.randint(0, cfg['TIMESTEPS'], (batch_images.shape[0],), device=device).long()
            
            noisy_images, noise = diffusion.forward_process(batch_images, t)
            
            if model_type == 'baseline':
                predicted_noise = model(noisy_images, t, batch_zc)
            else: # scd
                predicted_noise = model(noisy_images, t, batch_zc, batch_zs)
            
            loss = F.mse_loss(noise, predicted_noise)

            # 【修正】使用稳定的标准精度进行反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1} 平均损失: {epoch_loss / len(dataloader):.4f}")

    # 4. 保存模型
    model_path = save_dir / f'{model_type}_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == '__main__':
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # --- 训练 Baseline 模型 ---
    cfg['MODEL_TYPE'] = 'baseline'
    train(cfg)

    # --- 训练 SCD 模型 ---
    cfg['MODEL_TYPE'] = 'scd'
    train(cfg)