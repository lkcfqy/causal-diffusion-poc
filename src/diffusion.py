import torch
import torch.nn.functional as F
from tqdm import tqdm

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Helper function to extract correct values from a 1D tensor (vals) 
    based on a batch of indices (t).
    """
    batch_size = t.shape[0]
    # 【关键修正】: 移除了 .cpu() 调用。
    # 现在 t 和 vals 都在同一个 'mps' 设备上，gather 操作可以正常执行。
    out = vals.gather(-1, t) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion:
    def __init__(self, cfg):
        self.timesteps = cfg['TIMESTEPS']
        self.device = cfg['DEVICE']
        
        self.betas = linear_beta_schedule(self.timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_process(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        mean = sqrt_alphas_cumprod_t * x0
        variance = sqrt_one_minus_alphas_cumprod_t * noise
        return mean + variance, noise

    @torch.no_grad()
    def sample_timestep(self, model, x, t, z_c, z_s=None):
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # 根据模型类型决定调用方式
        if model.model_type == 'baseline':
            predicted_noise = model(x, t, z_c)
        else:
            predicted_noise = model(x, t, z_c, z_s)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t.min() == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, batch_size, z_c, z_s=None):
        img_size = 32
        img = torch.randn((batch_size, 3, img_size, img_size), device=self.device)
        
        # 确保 z_c 和 z_s (如果存在) 是正确的形状和设备
        if z_c.shape[0] != batch_size:
            z_c = z_c.repeat(batch_size)
        if z_s is not None and z_s.shape[0] != batch_size:
            z_s = z_s.repeat(batch_size)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps, leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(model, img, t, z_c, z_s)
        
        img = (img + 1) * 0.5
        return img.clamp(0, 1)