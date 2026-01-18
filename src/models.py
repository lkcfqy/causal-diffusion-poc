import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================================
# === 新增模块: SelfAttention ===
# =================================================================================
class SelfAttention(nn.Module):
    """
    自注意力模块，帮助模型关注图像的全局结构信息。
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 使用4个头的多头注意力机制
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x 的输入形状: [batch, channels, height, width]
        batch_size, _, size, _ = x.shape
        
        # 1. 将图像展平为序列
        # [batch, channels, H*W] -> [batch, H*W, channels]
        x = x.view(batch_size, self.channels, size * size).swapaxes(1, 2)
        
        # 2. 计算注意力
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        
        # 3. 添加残差连接
        attention_value = attention_value + x
        
        # 4. 通过前馈网络并添加第二个残差连接
        attention_value = self.ff_self(attention_value) + attention_value
        
        # 5. 将序列恢复为图像形状
        # [batch, H*W, channels] -> [batch, channels, H*W] -> [batch, channels, H, W]
        return attention_value.swapaxes(2, 1).view(batch_size, self.channels, size, size)

# =================================================================================
# === 现有模块 (保持不变) ===
# =================================================================================
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.bn2(self.relu(self.conv2(h)))
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# =================================================================================
# === 主模型: ConditionalUNet (已集成 Attention) ===
# =================================================================================
class ConditionalUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_type = cfg['MODEL_TYPE']
        image_channels = 3
        down_channels = (cfg['BASE_CHANNELS'], cfg['BASE_CHANNELS']*2, cfg['BASE_CHANNELS']*4)
        up_channels = (cfg['BASE_CHANNELS']*4, cfg['BASE_CHANNELS']*2, cfg['BASE_CHANNELS'])
        out_dim = image_channels
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.class_emb = nn.Embedding(cfg['NUM_CLASSES'], time_emb_dim)
        if self.model_type == 'scd':
            self.color_emb = nn.Embedding(cfg['NUM_COLORS'], time_emb_dim)

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # 下采样 (Encoder)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])
        
        # === 在U-Net的瓶颈层加入两个 Attention Block ===
        self.attn1 = SelfAttention(down_channels[-1])
        self.attn2 = SelfAttention(down_channels[-1])
        
        # 上采样 (Decoder)
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, z_c, z_s=None):
        # 1. 计算时间嵌入和条件嵌入 (已修复原地操作错误)
        t_embedding = self.time_mlp(timestep)
        
        cond_embedding = self.class_emb(z_c)
        if self.model_type == 'scd':
            if z_s is None:
                raise ValueError("SCD model requires z_s (color) condition.")
            cond_embedding = cond_embedding + self.color_emb(z_s)
        
        t = t_embedding + cond_embedding

        # --- U-Net 前向传播 ---
        x = self.conv0(x)
        residual_inputs = []
        # Encoder
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        # === 在瓶颈层应用 Attention ===
        x = self.attn1(x)
        x = self.attn2(x)
        
        # Decoder
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
            
        return self.output(x)