# 中文注释优化版20250528_DDP_ArgParse_JupyterDisplay
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt # 确保导入
import math
import json
from datetime import datetime
from torch.amp import autocast, GradScaler
import argparse

# DDP相关导入
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- DDP 辅助函数 (假设这些函数已定义，同前) ---
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    # dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Hack for Kaggle Notebook if dist is already initialized by torchrun
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    else:
        print(f"DDP: Rank {rank}/{world_size} process group already initialized.")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    print(f"DDP: Rank {rank}/{world_size} configured for device {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'} using backend {backend}.")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP: Process group destroyed.")

def is_main_process(rank=None): # Modified to get rank if not passed
    if rank is None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0 # Assume main process if DDP not active/initialized
    return rank == 0

# 设置随机种子 (假设已定义)
def set_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# ... (get_blur_kernel_2d, Blur, UpsampleBlock, DownsampleBlock, ProductImageDataset, Generator, Discriminator, weights_init, create_data_loaders, save_checkpoint, load_checkpoint 都保持不变，这里省略以减少篇幅) ...
# --- 辅助函数和层，用于实现StyleGAN3启发式的抗混叠操作 ---

def get_blur_kernel_2d(size=3, normalize=True, dtype=torch.float32, device='cpu'):
    """
    生成一个2D模糊核 (例如，近似于StyleGAN中使用的FIR滤波器)。
    这里我们使用一个固定的、可分离的二项式滤波器。
    返回: torch.Tensor, shape [size, size]
    """
    k_1d_list = []
    if size == 1:
        k_1d_list = [1.]
    elif size == 3:
        k_1d_list = [1., 2., 1.]
    elif size == 5:
        k_1d_list = [1., 4., 6., 4., 1.]
    else: # 默认或奇数大小，回退到3x3
        # print(f"警告: 不支持的模糊核大小 {size}, 使用3x3替代。") # 在DDP中减少打印
        k_1d_list = [1., 2., 1.]
    
    k_1d = torch.tensor(k_1d_list, dtype=dtype, device=device) # 从列表创建张量
    kernel_2d = torch.outer(k_1d, k_1d)
    if normalize:
        kernel_2d = kernel_2d / torch.sum(kernel_2d)
    return kernel_2d

class Blur(nn.Module):
    """
    一个简单的模糊层，用于在特征图上应用2D模糊。
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        blur_k_2d = get_blur_kernel_2d(kernel_size, normalize=True, dtype=torch.float32)
        # 确保 kernel 是持久化的 buffer
        kernel = blur_k_2d.reshape(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer('blur_kernel', kernel, persistent=False)


    def forward(self, x):
        # 声明 kernel_to_use
        kernel_to_use = self.blur_kernel.to(device=x.device, dtype=x.dtype)
        if x.shape[1] != self.channels: # 动态适应通道数，例如在判别器第一层输入是3通道，但blur初始化可能是其他数量
            # This dynamic part might be tricky if self.channels is used to define the initial blur_kernel's repeat factor
            # For now, assume if channels mismatch, it's likely the input (e.g. RGB image to first layer)
            # and we need a kernel repeated for x.shape[1] (e.g. 3 for RGB)
            blur_k_2d_dyn = get_blur_kernel_2d(self.kernel_size, normalize=True, dtype=x.dtype, device=x.device)
            kernel_to_use = blur_k_2d_dyn.reshape(1, 1, self.kernel_size, self.kernel_size).repeat(x.shape[1], 1, 1, 1)
            # print(f"Blur: Dynamic kernel for input channels {x.shape[1]}, original self.channels {self.channels}")
            return F.conv2d(x, kernel_to_use, padding=self.padding, groups=x.shape[1])
        
        return F.conv2d(x, kernel_to_use, padding=self.padding, groups=self.channels)


class UpsampleBlock(nn.Module):
    """
    上采样块，借鉴StyleGAN3的抗混叠思想。
    操作顺序: 插值上采样 -> 轻微模糊 (抗混叠) -> 卷积 -> 归一化 -> 激活
    """
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_padding=1,
                 blur_kernel_size=3, use_bias=False):
        super().__init__()
        self.interpolate_mode = 'bilinear'
        self.blur = None # 声明 self.blur

        self.apply_blur = blur_kernel_size > 0
        if self.apply_blur:
            self.blur = Blur(in_channels, kernel_size=blur_kernel_size) # Blur on in_channels before conv
        
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=1, padding=conv_padding, bias=use_bias)
        # 使用 SyncBatchNorm 替换 BatchNorm2d 以适应 DDP
        self.norm = nn.SyncBatchNorm(out_channels) if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode, align_corners=False)
        x_blurred = x_upsampled # 默认值
        if self.apply_blur and self.blur is not None: # 检查 self.blur 是否已初始化
            x_blurred = self.blur(x_upsampled)
        
        x_conv = self.conv(x_blurred)
        x_norm = self.norm(x_conv)
        x_activated = self.activation(x_norm)
        return x_activated

class DownsampleBlock(nn.Module):
    """
    下采样块，借鉴StyleGAN3的抗混叠思想。
    操作顺序: 轻微模糊 (抗混叠) -> 卷积 (stride=2实现下采样) -> 归一化 (可选) -> 激活
    """
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_padding=1,
                 blur_kernel_size=3, use_bias=False, use_norm=True):
        super().__init__()
        self.blur = None # 声明 self.blur
        
        self.apply_blur = blur_kernel_size > 0
        if self.apply_blur:
            self.blur = Blur(in_channels, kernel_size=blur_kernel_size)

        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=2, padding=conv_padding, bias=use_bias)
        
        self.use_norm = use_norm
        self.norm = None # 声明 self.norm
        if self.use_norm:
            self.norm = nn.SyncBatchNorm(out_channels) if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_blurred = x # 默认值
        if self.apply_blur and self.blur is not None: # 检查 self.blur
            x_blurred = self.blur(x)
        
        x_conv = self.conv(x_blurred)
        x_norm = x_conv # 默认值
        if self.use_norm and self.norm is not None: # 检查 self.norm
            x_norm = self.norm(x_conv)
        
        x_activated = self.activation(x_norm)
        return x_activated

# --- 数据集定义 ---
class ProductImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, rank=None): # rank for logging if needed
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if is_main_process(rank): # 只在主进程打印，避免DDP下重复打印
                print(f"Rank {rank if rank is not None else 'N/A'}: 找到 {len(self.image_files)} 张图片在 {data_dir}")
        else:
            if is_main_process(rank):
                print(f"Rank {rank if rank is not None else 'N/A'}: 错误: 数据目录 {data_dir} 不存在或不是一个目录。")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = None # 声明 image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 在DDP中，一个进程出错可能导致挂起，错误处理要小心
            # print(f"错误: 无法打开或转换图片 {img_path}: {e}") # 减少打印
            placeholder_size = (256, 256) 
            if self.transform:
                 # 确保 t 被正确处理
                 for t_item in self.transform.transforms:
                     if isinstance(t_item, transforms.Resize):
                         if isinstance(t_item.size, int):
                             placeholder_size = (t_item.size, t_item.size)
                         else:
                             placeholder_size = t_item.size 
                         break
            image = Image.new('RGB', placeholder_size, color='black') 
            if self.transform:
                return self.transform(image)
            else: 
                return transforms.ToTensor()(image)

        if self.transform and image is not None: # 确保 image 不是 None
            image = self.transform(image)
        elif image is None: # 如果 try-except 后 image 仍为 None (理论上不应发生，因为有 placeholder)
             placeholder_size_fallback = (256,256) # 和上面逻辑类似
             if self.transform:
                  for t_item in self.transform.transforms:
                     if isinstance(t_item, transforms.Resize):
                         if isinstance(t_item.size, int):
                             placeholder_size_fallback = (t_item.size, t_item.size)
                         else:
                             placeholder_size_fallback = t_item.size
                         break
             image_fallback = Image.new('RGB', placeholder_size_fallback, color='grey') # 使用不同颜色以区分
             if self.transform:
                 return self.transform(image_fallback)
             else:
                 return transforms.ToTensor()(image_fallback)


        return image # 返回处理后的 image

# --- 模型定义 ---
class Generator(nn.Module):
    """
    生成器网络 - 改进版，采用抗混叠思想的上采样块
    """
    def __init__(self, nz=100, ngf=64, nc=3, blur_kernel_size=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        BN_layer = nn.SyncBatchNorm if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d
        
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            BN_layer(ngf * 8),
            nn.ReLU(True)
        )
        
        self.up1 = UpsampleBlock(ngf * 8, ngf * 4, blur_kernel_size=blur_kernel_size)
        self.up2 = UpsampleBlock(ngf * 4, ngf * 2, blur_kernel_size=blur_kernel_size)
        self.up3 = UpsampleBlock(ngf * 2, ngf, blur_kernel_size=blur_kernel_size)
        self.up4 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size)
        self.up5 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size)
        self.up6 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size)
        self.up7 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size)
        
        self.output_64 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.output_128 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.output_256 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.output_512 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        
    def forward(self, input_noise, target_size=512):
        x = input_noise.view(-1, self.nz, 1, 1)
        
        x = self.initial(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        feat_64 = self.up4(x) 
        
        if target_size == 64:
            return torch.tanh(self.output_64(feat_64))
        
        feat_128 = self.up5(feat_64)
        if target_size == 128:
            return torch.tanh(self.output_128(feat_128))
            
        feat_256 = self.up6(feat_128)
        if target_size == 256:
            return torch.tanh(self.output_256(feat_256))
            
        feat_512 = self.up7(feat_256)
        return torch.tanh(self.output_512(feat_512))

class Discriminator(nn.Module):
    """
    判别器网络 - 改进版，采用抗混叠思想的下采样块
    """
    def __init__(self, nc=3, ndf=64, blur_kernel_size=3):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        self.down1 = DownsampleBlock(nc, ndf, blur_kernel_size=blur_kernel_size, use_norm=False)
        self.down2 = DownsampleBlock(ndf, ndf * 2, blur_kernel_size=blur_kernel_size)
        self.down3 = DownsampleBlock(ndf * 2, ndf * 4, blur_kernel_size=blur_kernel_size)
        self.down4 = DownsampleBlock(ndf * 4, ndf * 8, blur_kernel_size=blur_kernel_size)
        self.down5 = DownsampleBlock(ndf * 8, ndf * 8, blur_kernel_size=blur_kernel_size)
        self.down6 = DownsampleBlock(ndf * 8, ndf * 8, blur_kernel_size=blur_kernel_size)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.final_conv = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
    def forward(self, input_image):
        x = self.down1(input_image)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        if x.shape[2] > 8: 
             x = self.down5(x)
        if x.shape[2] > 4 :
             x = self.down6(x)
        
        x = self.adaptive_pool(x)
        x = self.final_conv(x)
        
        return x.view(-1)

# --- 权重初始化 ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('SyncBatchNorm') != -1: # Include SyncBatchNorm
        if hasattr(m, 'weight') and m.weight is not None: # 确保BN层有weight
             nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# --- 数据加载器创建 ---
def create_data_loaders(data_dir, batch_size=8, num_workers=0, rank=None, world_size=1, max_size=512):
    transforms_dict = {}
    datasets_dict = {}
    dataloaders_dict = {}
    # Filter possible_sizes based on max_size
    possible_sizes = [s for s in [64, 128, 256, 512] if s <= max_size]

    for size_val in possible_sizes: # 重命名 size 变量避免冲突
        current_transform = transforms.Compose([ # 重命名 transform 变量
            transforms.Resize((size_val, size_val)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transforms_dict[size_val] = current_transform
        
        current_dataset = ProductImageDataset(data_dir, transforms_dict[size_val], rank=rank) # 重命名 dataset
        datasets_dict[size_val] = current_dataset
        
        sampler = None
        shuffle_dl = True # Shuffle for non-DDP or if sampler handles it
        if world_size > 1 and dist.is_initialized(): # DDP active
            sampler = DistributedSampler(current_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            shuffle_dl = False # Sampler handles shuffling

        if len(current_dataset) > 0 :
            # drop_last is important for DDP if batch sizes are not perfectly divisible
            # It's also generally good for GANs to have consistent batch sizes
            effective_len_per_gpu = len(current_dataset) // world_size if world_size > 0 else len(current_dataset)
            if effective_len_per_gpu >= batch_size : # batch_size is per-GPU here
                dataloaders_dict[size_val] = DataLoader(current_dataset, batch_size=batch_size, 
                                                    shuffle=shuffle_dl, sampler=sampler,
                                                    num_workers=num_workers, drop_last=True, 
                                                    pin_memory=torch.cuda.is_available())
            else:
                if is_main_process(rank):
                    print(f"警告: {size_val}x{size_val} 数据集在rank {rank if rank is not None else 'N/A'}上的有效样本数 {effective_len_per_gpu} 小于批大小 {batch_size} (且drop_last=True)，无法创建DataLoader。")
                dataloaders_dict[size_val] = None
        else:
            if is_main_process(rank):
                print(f"警告: {size_val}x{size_val} 数据集为空，无法创建DataLoader。")
            dataloaders_dict[size_val] = None
            
    return dataloaders_dict

# --- 检查点保存与加载 (基本同前，省略) ---
def save_checkpoint(netG_module, netD_module, optimizerG, optimizerD, scaler, epoch, phase_info,
                    checkpoint_dir='checkpoints', amp_enabled=False, rank=None):
    if not is_main_process(rank): return
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_data = {
        'generator_state_dict': netG_module.state_dict(),
        'discriminator_state_dict': netD_module.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'epoch': epoch, 'phase_info': phase_info, 'timestamp': datetime.now().isoformat()}
    if amp_enabled and scaler is not None: checkpoint_data['scaler_state_dict'] = scaler.state_dict()
    
    new_checkpoint_filename = f'checkpoint_epoch_{epoch:04d}.pth'
    new_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint_filename)
    torch.save(checkpoint_data, new_checkpoint_path)
    
    latest_json_path = os.path.join(checkpoint_dir, 'latest.json')
    previous_checkpoint_file_to_delete = None
    if os.path.exists(latest_json_path):
        try:
            with open(latest_json_path, 'r', encoding='utf-8') as f: latest_info_old = json.load(f)
            if 'latest_checkpoint' in latest_info_old and \
               os.path.basename(latest_info_old['latest_checkpoint']).startswith('checkpoint_epoch_') and \
               latest_info_old['latest_checkpoint'].endswith('.pth'):
                previous_checkpoint_file_to_delete = latest_info_old['latest_checkpoint']
        except (json.JSONDecodeError, FileNotFoundError):
            current_rank = rank if rank is not None else 'N/A'
            print(f"Rank {current_rank}: 警告: 读取或解析旧的 {latest_json_path} 失败。")

    latest_info_new = {'latest_checkpoint': new_checkpoint_path, 'epoch': epoch, 'phase_info': phase_info, 'timestamp': datetime.now().isoformat()}
    with open(latest_json_path, 'w', encoding='utf-8') as f: json.dump(latest_info_new, f, ensure_ascii=False, indent=2)
    current_rank_str = str(rank) if rank is not None else "N/A"
    print(f"Rank {current_rank_str}: 检查点已保存: {new_checkpoint_path}")

    if previous_checkpoint_file_to_delete and \
       previous_checkpoint_file_to_delete != new_checkpoint_path and \
       os.path.exists(previous_checkpoint_file_to_delete):
        try:
            os.remove(previous_checkpoint_file_to_delete)
            print(f"Rank {current_rank_str}: 已删除旧的检查点文件: {previous_checkpoint_file_to_delete}")
        except OSError as e:
            print(f"Rank {current_rank_str}: 警告: 删除旧检查点文件 {previous_checkpoint_file_to_delete} 失败: {e}")

def load_checkpoint(checkpoint_path, netG_module, netD_module, optimizerG, optimizerD, scaler, device, rank=None):
    current_rank_str = str(rank) if rank is not None else "N/A"
    if not os.path.exists(checkpoint_path):
        if is_main_process(rank): print(f"Rank {current_rank_str}: 检查点文件不存在: {checkpoint_path}")
        return 0, {}
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    netG_module.load_state_dict(checkpoint_data['generator_state_dict'])
    netD_module.load_state_dict(checkpoint_data['discriminator_state_dict'])
    if optimizerG and 'optimizerG_state_dict' in checkpoint_data: optimizerG.load_state_dict(checkpoint_data['optimizerG_state_dict'])
    if optimizerD and 'optimizerD_state_dict' in checkpoint_data: optimizerD.load_state_dict(checkpoint_data['optimizerD_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint_data:
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        if is_main_process(rank): print(f"Rank {current_rank_str}: GradScaler状态已加载。")
    elif scaler is not None and is_main_process(rank): print(f"Rank {current_rank_str}: 警告: 检查点中未找到GradScaler状态。")
    start_epoch = checkpoint_data.get('epoch', 0)
    phase_info = checkpoint_data.get('phase_info', {})
    if is_main_process(rank):
        print(f"Rank {current_rank_str}: 已从检查点加载: {checkpoint_path}")
        print(f"Rank {current_rank_str}: 恢复到全局第 {start_epoch} 个epoch之后，阶段信息: {phase_info}")
    return start_epoch, phase_info

# --- 保存样本图片 (修改版) ---
def save_sample_images(generator_module, epoch, current_size, nz=100, num_samples=4,
                       save_dir_prefix='generated_samples', device='cpu', amp_enabled=False,
                       display_in_jupyter=False, rank=None): # 新增 display_in_jupyter 和 rank
    # 此函数只应由主进程调用以避免文件写入冲突和多次显示
    if not is_main_process(rank):
        return

    generator_module.eval()
    
    # 为了减少显存峰值，可以分批生成，这里演示一次性生成（若 num_samples 较小）
    # 如果 num_samples 很大导致OOM，需要改成循环单张或小批量生成然后拼接
    generated_images_for_grid = []
    if num_samples > 0 :
        try:
            with torch.no_grad():
                # 尝试分批生成以减少单次推理的显存占用
                # 假设单次生成一张图片是安全的
                for _ in range(num_samples):
                    current_noise = torch.randn(1, nz, device=device) # 生成单张图片的噪声
                    amp_dtype = torch.float16 if device.type == 'cuda' and amp_enabled else torch.float32
                    with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        fake_image_single = generator_module(current_noise, target_size=current_size) # 生成单张
                    fake_image_single = (fake_image_single + 1) / 2.0
                    fake_image_single = torch.clamp(fake_image_single, 0, 1)
                    generated_images_for_grid.append(fake_image_single.cpu()) # 移到CPU

                if not generated_images_for_grid:
                    print(f"Rank {rank if rank is not None else 'N/A'}:未能生成任何图片。")
                    generator_module.train()
                    return

                # 将CPU上的图片列表拼接成一个batch
                fake_images_cpu = torch.cat(generated_images_for_grid, dim=0)
                
        except RuntimeError as e: # OOM or other runtime errors during generation
             current_rank_str = str(rank) if rank is not None else "N/A"
             print(f"Rank {current_rank_str}: 在 save_sample_images 中生成图片时发生错误: {e}")
             print(f"Rank {current_rank_str}: 尝试减少 num_save_samples 或 batch_size 来解决OOM。跳过本次样本保存。")
             generator_module.train()
             return # 提前返回，不进行后续的保存和显示

    if not generated_images_for_grid : # 如果上面try-except块后列表仍为空
        generator_module.train()
        return

    save_dir = f'{save_dir_prefix}_{current_size}x{current_size}'
    os.makedirs(save_dir, exist_ok=True)
    
    fake_images_np = fake_images_cpu.float().numpy()

    # 动态计算网格大小
    actual_num_samples = fake_images_np.shape[0]
    if actual_num_samples == 0:
        generator_module.train()
        return

    grid_cols = int(math.sqrt(actual_num_samples))
    grid_rows = math.ceil(actual_num_samples / grid_cols)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2.5, grid_rows * 2.5)) # 调整figsize
    if actual_num_samples == 1: # 如果只有一个样本，axes不是数组
        axes = np.array([axes])
    axes = axes.flatten() # 确保axes是扁平化的数组

    fig.suptitle(f'Epoch {epoch} - Size {current_size}x{current_size}', fontsize=16)
    for i_plt in range(actual_num_samples):
        img_np_permuted = np.transpose(fake_images_np[i_plt], (1, 2, 0))
        axes[i_plt].imshow(img_np_permuted)
        axes[i_plt].axis('off')
    
    # 关闭多余的子图（如果网格大于样本数）
    for i_extra in range(actual_num_samples, grid_rows * grid_cols):
        axes[i_extra].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整以适应标题
    
    # 保存到文件
    save_path = os.path.join(save_dir, f'epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=150)
    
    if display_in_jupyter:
        # 在Jupyter中显示图片
        # 注意：在DDP的非交互式torchrun脚本中，plt.show()可能不起作用或行为异常
        # 但如果是从Jupyter Notebook内部启动的单进程或主进程，则可以显示
        print(f"Rank {rank if rank is not None else 'N/A'}: 正在Jupyter中显示图片 (Epoch: {epoch}, Size: {current_size})...")
        plt.show() # Jupyter通常会自动处理，但显式调用确保
    
    plt.close(fig) # 关闭图像以释放内存
    generator_module.train()


# --- 主训练函数 (大部分同前，注意调用 save_sample_images 时传递 display_in_jupyter 和 rank) ---
def train_gan(args, rank=None, world_size=1, ddp_active=False):
    # ... (设备设置, AMP, 模型初始化, DDP包装, 优化器, scaler, dataloaders, training_phases, schedulers, 检查点加载逻辑 - 基本不变)
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() and ddp_active else \
             (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    current_rank_str = str(rank) if rank is not None else "N/A" # For logging

    amp_enabled = (device.type == 'cuda')
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    if is_main_process(rank):
        print(f"Rank {current_rank_str}: 使用设备: {device}")
        if amp_enabled: print(f"Rank {current_rank_str}: 自动混合精度 (AMP) 已启用，使用数据类型: {amp_dtype}。")
        else: print(f"Rank {current_rank_str}: 自动混合精度 (AMP) 未启用。")

    # 模型初始化
    netG_base = Generator(nz=args.noise_dim, ngf=args.ngf, nc=3, blur_kernel_size=args.blur_kernel_size).to(device)
    netD_base = Discriminator(nc=3, ndf=args.ndf, blur_kernel_size=args.blur_kernel_size).to(device)
    netG_base.apply(weights_init)
    netD_base.apply(weights_init)

    if ddp_active and dist.is_initialized():
        # DDP 包装
        # find_unused_parameters=True can be important for GANs if parts of graph aren't always used
        # However, it adds overhead. Set to False if you are sure all parameters are used or after debugging.
        netG = DDP(netG_base, device_ids=[rank] if device.type == 'cuda' else None, output_device=rank if device.type == 'cuda' else None, find_unused_parameters=True)
        netD = DDP(netD_base, device_ids=[rank] if device.type == 'cuda' else None, output_device=rank if device.type == 'cuda' else None, find_unused_parameters=True)
    else:
        netG = netG_base
        netD = netD_base
    
    netG_module = netG.module if ddp_active and dist.is_initialized() else netG
    netD_module = netD.module if ddp_active and dist.is_initialized() else netD

    criterion = nn.BCEWithLogitsLoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), eps=1e-8)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), eps=1e-8)
    scaler = GradScaler(enabled=amp_enabled)

    dataloaders = create_data_loaders(args.data_dir, args.batch_size, args.num_workers, rank, world_size, args.max_train_size)
    if not any(dataloaders.values()):
        if is_main_process(rank): print(f"Rank {current_rank_str}: 错误: 所有有效尺寸的数据加载器均为空。程序将退出。")
        return None, None

    all_possible_phases = [
        {'size': 64, 'epochs': args.num_epochs_per_phase},
        {'size': 128, 'epochs': args.num_epochs_per_phase},
        {'size': 256, 'epochs': args.num_epochs_per_phase},
        {'size': 512, 'epochs': args.num_epochs_per_phase}
    ]
    training_phases = [p for p in all_possible_phases if p['size'] <= args.max_train_size and dataloaders.get(p['size']) is not None]

    if not training_phases:
        if is_main_process(rank): print(f"Rank {current_rank_str}: 错误: 没有可训练的阶段 (检查max_train_size和数据集)。程序将退出。")
        return None, None

    total_epochs_across_phases = sum(p['epochs'] for p in training_phases)
    if total_epochs_across_phases == 0: # Should be caught by the above check, but as a safeguard
        if is_main_process(rank): print(f"Rank {current_rank_str}: 错误: 总训练epoch数为0。程序将退出。")
        return None, None

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=total_epochs_across_phases, eta_min=args.learning_rate * 0.01)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=total_epochs_across_phases, eta_min=args.learning_rate * 0.01)

    global_start_epoch = 0; current_phase_idx_resume = 0; epoch_in_phase_start_resume = 0
    if args.resume_from_checkpoint:
        latest_json_path = os.path.join(args.checkpoint_dir, 'latest.json')
        if os.path.exists(latest_json_path):
            try:
                with open(latest_json_path, 'r', encoding='utf-8') as f_json: latest_info = json.load(f_json)
                checkpoint_path_to_load = latest_info['latest_checkpoint']
                global_start_epoch, phase_info_loaded = load_checkpoint(
                    checkpoint_path_to_load, netG_module, netD_module, optimizerG, optimizerD, scaler, device, rank)
                if phase_info_loaded:
                    current_phase_idx_resume = phase_info_loaded.get('phase_idx', 0)
                    epoch_in_phase_start_resume = phase_info_loaded.get('epoch_in_phase', 0)
                # Sync schedulers for all processes
                for _ in range(global_start_epoch): schedulerG.step(); schedulerD.step()
            except Exception as e:
                if is_main_process(rank):
                    print(f"Rank {current_rank_str}: 从检查点恢复失败: {e}。将从头开始训练。")
                global_start_epoch = 0; current_phase_idx_resume = 0; epoch_in_phase_start_resume = 0

        elif is_main_process(rank): print(f"Rank {current_rank_str}: 未找到 latest.json，从头开始训练。")

    if ddp_active and dist.is_initialized(): dist.barrier()
    if is_main_process(rank): print(f"Rank {current_rank_str}: 开始训练...")
    current_global_epoch = global_start_epoch # This is number of epochs COMPLETED

    for phase_idx in range(current_phase_idx_resume, len(training_phases)):
        phase = training_phases[phase_idx]
        current_target_size = phase['size']
        phase_total_epochs = phase['epochs']
        current_dataloader = dataloaders.get(current_target_size)

        if ddp_active and dist.is_initialized() and hasattr(current_dataloader.sampler, 'set_epoch'):
            # Use global_epoch + phase_idx as a proxy for a continually increasing epoch number for the sampler
            # This ensures different shuffling if resuming mid-training into a new phase
            current_dataloader.sampler.set_epoch(current_global_epoch + phase_idx)


        if is_main_process(rank):
            print(f"\nRank {current_rank_str}: === 阶段 {phase_idx + 1}/{len(training_phases)}: 训练 {current_target_size}x{current_target_size} 尺寸 ===")
        
        # If resuming, epoch_in_phase_start_resume is the number of epochs completed in this phase.
        # So, the loop should start from this number.
        start_e_in_phase = epoch_in_phase_start_resume if phase_idx == current_phase_idx_resume else 0

        for epoch_in_phase in range(start_e_in_phase, phase_total_epochs):
            epoch_start_time = datetime.now()
            netG.train(); netD.train()
            
            # Iterate over data
            for i_batch, real_images in enumerate(current_dataloader):
                real_images = real_images.to(device, non_blocking=True) # non_blocking for potential perf gain
                batch_actual_size = real_images.size(0) # Per-GPU batch size
                
                real_label_val = 0.9 
                fake_label_val = 0.1
                real_label = torch.full((batch_actual_size,), real_label_val, dtype=torch.float32, device=device)
                fake_label = torch.full((batch_actual_size,), fake_label_val, dtype=torch.float32, device=device)

                # --- 训练判别器 ---
                optimizerD.zero_grad(set_to_none=True)
                
                noise = torch.randn(batch_actual_size, args.noise_dim, device=device)
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    fake_images = netG(noise, target_size=current_target_size)

                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    output_real = netD(real_images)
                    loss_D_real = criterion(output_real, real_label)
                    
                    output_fake_detached = netD(fake_images.detach()) # Detach to avoid G grads flowing here
                    loss_D_fake = criterion(output_fake_detached, fake_label)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                if loss_D is not None: # Should always be not None here
                    scaler.scale(loss_D).backward() # DDP handles gradient sync
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizerD) 
                        torch.nn.utils.clip_grad_norm_(netD.parameters(), args.grad_clip_norm)
                    scaler.step(optimizerD)

                # --- 训练生成器 ---
                optimizerG.zero_grad(set_to_none=True)
                # Re-use fake_images generated earlier, no need to detach this time for G training
                if fake_images is not None: 
                    with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        output_fake_for_G = netD(fake_images) 
                        loss_G = criterion(output_fake_for_G, real_label) # G wants D to think fakes are real
                    
                    if loss_G is not None: # Should always be not None here
                        scaler.scale(loss_G).backward() # DDP handles gradient sync
                        if args.grad_clip_norm > 0:
                            scaler.unscale_(optimizerG)
                            torch.nn.utils.clip_grad_norm_(netG.parameters(), args.grad_clip_norm)
                        scaler.step(optimizerG)

                scaler.update()

                # Logging (only from main process)
                if i_batch % args.log_every_iters == 0 and is_main_process(rank):
                    loss_D_item = loss_D.item() if loss_D is not None else float('nan')
                    loss_G_item = loss_G.item() if loss_G is not None else float('nan')
                    output_real_mean = output_real.mean().item() if output_real is not None else float('nan')
                    output_fake_detached_mean = output_fake_detached.mean().item() if output_fake_detached is not None else float('nan')
                    output_fake_for_G_mean = output_fake_for_G.mean().item() if output_fake_for_G is not None else float('nan')
                    current_lr_G = optimizerG.param_groups[0]['lr']
                    current_lr_D = optimizerD.param_groups[0]['lr']
                    print(f'[R{current_rank_str}][Ph{phase_idx+1}][GEp {current_global_epoch+1}/{total_epochs_across_phases}] '
                          f'[PhEp {epoch_in_phase+1}/{phase_total_epochs}] [{i_batch}/{len(current_dataloader)}] '
                          f'L_D: {loss_D_item:.4f} L_G: {loss_G_item:.4f} '
                          f'D(x): {output_real_mean:.4f} D(G(z)): {output_fake_detached_mean:.4f}(D)/{output_fake_for_G_mean:.4f}(G) '
                          f'LR_G: {current_lr_G:.6f} LR_D: {current_lr_D:.6f} Scale: {scaler.get_scale():.1f}')
            
            # End of an epoch_in_phase
            current_global_epoch += 1 # Increment global epoch count (completed)
            # Schedulers step based on global epochs completed
            schedulerG.step() 
            schedulerD.step()

            # Save samples and checkpoint (only from main process)
            if is_main_process(rank):
                # epoch_in_phase is 0-indexed, so +1 for 1-indexed comparison
                if (epoch_in_phase + 1) % args.save_every_epochs == 0 or (epoch_in_phase + 1) == phase_total_epochs:
                    save_sample_images(netG_module, current_global_epoch, current_target_size, args.noise_dim,
                                       num_samples=args.num_save_samples,
                                       save_dir_prefix=args.sample_save_dir_prefix,
                                       device=device, amp_enabled=amp_enabled,
                                       display_in_jupyter=args.display_jupyter, rank=rank) # Pass display flag
                    
                    # phase_info stores COMPLETED epochs in this phase
                    current_phase_progress_info = {
                        'phase_idx': phase_idx,
                        'epoch_in_phase': epoch_in_phase + 1, 
                        'size': current_target_size
                    }
                    save_checkpoint(netG_module, netD_module, optimizerG, optimizerD, scaler, 
                                    current_global_epoch, current_phase_progress_info, 
                                    checkpoint_dir=args.checkpoint_dir,
                                    amp_enabled=amp_enabled, rank=rank)

                epoch_duration = datetime.now() - epoch_start_time
                print(f"Rank {current_rank_str}: 阶段Epoch {epoch_in_phase+1} 完成，耗时: {epoch_duration}")
            
            if ddp_active and dist.is_initialized(): # Barrier to ensure all processes complete epoch before next phase/sampler update
                dist.barrier()

        # Reset for the next phase, as epoch_in_phase_start_resume is specific to a resumed phase
        epoch_in_phase_start_resume = 0 
        if is_main_process(rank):
            print(f"Rank {current_rank_str}: 阶段 {phase_idx + 1} ({current_target_size}x{current_target_size}) 训练完成。")
    
    if is_main_process(rank):
        print("\n=== 所有训练阶段完成！ ===")
        os.makedirs(args.final_model_dir, exist_ok=True)
        torch.save(netG_module.state_dict(), os.path.join(args.final_model_dir, 'generator_final.pth'))
        torch.save(netD_module.state_dict(), os.path.join(args.final_model_dir, 'discriminator_final.pth'))
        
        # Save training configuration
        config_summary = vars(args).copy() # Save all command line args
        config_summary.update({
            'total_global_epochs_trained': current_global_epoch, # Total epochs completed
            'world_size': world_size,
            'global_batch_size': args.batch_size * world_size if world_size > 0 else args.batch_size,
            'amp_enabled_during_training': amp_enabled,
            'amp_dtype_used': str(amp_dtype) if amp_enabled else 'N/A',
            'ddp_active': ddp_active and dist.is_initialized(),
            'final_timestamp': datetime.now().isoformat()
        })
        with open(os.path.join(args.final_model_dir, 'training_config.json'), 'w', encoding='utf-8') as f_json_out:
            json.dump(config_summary, f_json_out, ensure_ascii=False, indent=2)
        print(f"Rank {current_rank_str}: 最终模型和配置已保存到: {args.final_model_dir}")
    
    return netG_module, netD_module

# --- 生成图片 (基本同前，省略) ---
def generate_images(args, rank=None): # args is the parsed namespace
    if not is_main_process(rank): return

    current_rank_str = str(rank) if rank is not None else "N/A"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use default GPU for generation
    amp_enabled_inference = (device.type == 'cuda') 
    amp_dtype_inference = torch.float16 if amp_enabled_inference else torch.float32

    print(f"Rank {current_rank_str}: 使用设备进行生成: {device}, AMP推理: {'启用' if amp_enabled_inference else '未启用'}, Dtype: {amp_dtype_inference if amp_enabled_inference else 'default'}")
    try:
        # Initialize base model, not DDP wrapped
        netG = Generator(nz=args.noise_dim, ngf=args.ngf, nc=3, blur_kernel_size=args.blur_kernel_size).to(device)
        if not os.path.exists(args.model_path):
            print(f"Rank {current_rank_str}: 错误：模型文件不存在 {args.model_path}")
            return
        
        # state_dict should be from netG.module if saved from DDP training
        state_dict = torch.load(args.model_path, map_location=device)
        netG.load_state_dict(state_dict)
        netG.eval()
        print(f"Rank {current_rank_str}: 成功加载模型: {args.model_path}")

        # Parse sizes_to_generate from string if it's a string
        sizes_to_generate_actual = []
        if isinstance(args.sizes_to_generate, str):
            try:
                sizes_to_generate_actual = [int(s.strip()) for s in args.sizes_to_generate.split(',') if s.strip().isdigit()]
            except ValueError:
                print(f"Rank {current_rank_str}: 错误: sizes_to_generate '{args.sizes_to_generate}' 格式不正确。应为逗号分隔的数字。")
                return
        elif isinstance(args.sizes_to_generate, list): # If it's already a list (e.g. from interactive mode)
            sizes_to_generate_actual = args.sizes_to_generate
        else:
            print(f"Rank {current_rank_str}: 错误: sizes_to_generate 类型无法处理: {type(args.sizes_to_generate)}。")
            return
        
        if not sizes_to_generate_actual:
            print(f"Rank {current_rank_str}: 没有指定要生成的有效尺寸。")
            return


        with torch.no_grad():
            for size_val_gen in sizes_to_generate_actual:
                print(f"\nRank {current_rank_str}: 正在生成 {size_val_gen}x{size_val_gen} 尺寸的图片...")
                save_dir = f'{args.gen_output_dir_prefix}_{size_val_gen}x{size_val_gen}'
                os.makedirs(save_dir, exist_ok=True)
                
                generated_count_current_size = 0
                for i_gen in range(args.num_generate):
                    try:
                        noise = torch.randn(1, args.noise_dim, device=device) # Generate one image's noise at a time
                        fake_image = None 
                        with autocast(device_type=device.type, enabled=amp_enabled_inference, dtype=amp_dtype_inference):
                            fake_image = netG(noise, target_size=size_val_gen)
                        
                        if fake_image is not None: 
                            fake_image = (fake_image + 1) / 2.0
                            fake_image = torch.clamp(fake_image, 0, 1)
                            
                            img_tensor_chw = fake_image[0]
                            img_np_hwc = img_tensor_chw.cpu().float().numpy().transpose(1, 2, 0) 
                            img_pil = Image.fromarray((img_np_hwc * 255).astype(np.uint8))
                            img_path = os.path.join(save_dir, f'generated_{i_gen+1:03d}.png')
                            img_pil.save(img_path)
                            generated_count_current_size +=1
                            if generated_count_current_size % 5 == 0 or generated_count_current_size == args.num_generate:
                                print(f"  Rank {current_rank_str}: 已生成 {generated_count_current_size}/{args.num_generate} 张图片 (尺寸 {size_val_gen}x{size_val_gen})")
                    except RuntimeError as e_gen:
                        print(f"Rank {current_rank_str}: 生成第 {i_gen+1} 张图片 (尺寸 {size_val_gen}x{size_val_gen}) 时发生错误: {e_gen}。跳过此张图片。")
                        continue # Skip to the next image if one fails

                print(f"Rank {current_rank_str}: ✓ {size_val_gen}x{size_val_gen} 图片生成完成 ({generated_count_current_size}/{args.num_generate} 张成功)，保存在 {save_dir} 目录")
        print(f"\nRank {current_rank_str}: 所有图片生成完成！")
    except Exception as e:
        print(f"Rank {current_rank_str}: 生成图片时发生严重错误: {str(e)}")
        import traceback
        traceback.print_exc()


# --- 虚拟数据创建 (同前，省略) ---
def create_dummy_data_if_not_exists(data_dir, num_dummy_images=16, rank=None):
    if not is_main_process(rank): return
    current_rank_str = str(rank) if rank is not None else "N/A"
    if not os.path.exists(data_dir):
        print(f"Rank {current_rank_str}: 数据目录 {data_dir} 不存在，将创建它。")
        os.makedirs(data_dir, exist_ok=True)
    
    img_extensions = ('.png', '.jpg', '.jpeg')
    try:
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(img_extensions)]
        if not image_files:
            print(f"Rank {current_rank_str}: 数据目录 {data_dir} 为空，将创建 {num_dummy_images} 张虚拟PNG图片。")
            for i_dummy in range(num_dummy_images): # 重命名 i
                color = (np.random.randint(0,255), np.random.randint(0,128), np.random.randint(128,255))
                dummy_img = Image.new('RGB', (512,512), color=color) # Create 512x512 dummy images
                dummy_img.save(os.path.join(data_dir, f"dummy_image_{i_dummy+1}.png"))
            print(f"Rank {current_rank_str}: 已创建 {num_dummy_images} 张虚拟图片在 {data_dir}。")
    except OSError as e:
        print(f"Rank {current_rank_str}: 检查或创建虚拟数据时发生OS错误 (可能是权限问题): {e} in {data_dir}")
    except Exception as e:
        print(f"Rank {current_rank_str}: 创建虚拟图片失败: {e}")

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="PyTorch GAN (StyleGAN3-inspired) Training and Generation Script with DDP")
    # 通用参数
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--ngf', type=int, default=64, help='Number of generator features')
    parser.add_argument('--ndf', type=int, default=64, help='Number of discriminator features')

    # 模式选择
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'], help='Operating mode (interactive mode removed)')

    # 训练参数
    parser.add_argument('--data_dir', type=str, default="./product_images", help='Path to training image data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU for training') # Default reduced
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for Adam optimizers')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of the input noise vector Z')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from the latest checkpoint')
    parser.add_argument('--blur_kernel_size', type=int, default=3, choices=[0, 1, 3, 5], help='Size of blur kernel for anti-aliasing (0 or 1 to disable)')
    parser.add_argument('--num_epochs_per_phase', type=int, default=10, help='Number of epochs for each resolution training phase')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes (0 for Windows, 2 for Linux often good)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Directory to save checkpoints')
    parser.add_argument('--sample_save_dir_prefix', type=str, default="generated_samples", help='Prefix for saving sample images directory')
    parser.add_argument('--final_model_dir', type=str, default="final_models_stylegan3_inspired", help='Directory to save final models and config')
    parser.add_argument('--save_every_epochs', type=int, default=5, help='Save samples and checkpoint every N epochs within a phase')
    parser.add_argument('--log_every_iters', type=int, default=50, help='Log training progress every N iterations')
    parser.add_argument('--num_save_samples', type=int, default=4, help='Number of samples to generate and save during training (e.g., 4 for 2x2 grid)')
    parser.add_argument('--max_train_size', type=int, default=512, choices=[64, 128, 256, 512], help='Maximum image size to train up to')
    parser.add_argument('--display_jupyter', action='store_true', help='Display generated sample images in Jupyter Notebook during training (main process only)')


    # 生成参数
    parser.add_argument('--model_path', type=str, default="final_models_stylegan3_inspired/generator_final.pth", help='Path to the trained generator model for generation')
    parser.add_argument('--num_generate', type=int, default=4, help='Number of images to generate in generate mode')
    parser.add_argument('--sizes_to_generate', type=str, default="64,128,256,512", help='Comma-separated list of sizes to generate (e.g., "64,128,256,512")')
    parser.add_argument('--gen_output_dir_prefix', type=str, default="final_generated_stylegan3_inspired", help='Prefix for image generation output directory')
    
    args = parser.parse_args()

    # DDP Rank and World Size determination
    # These are set by torchrun
    rank_str = os.environ.get("RANK")
    world_size_str = os.environ.get("WORLD_SIZE")

    ddp_active = rank_str is not None and world_size_str is not None
    rank = int(rank_str) if ddp_active else 0
    world_size = int(world_size_str) if ddp_active else 1
    
    current_rank_str_log = str(rank) if ddp_active else "N/A (no DDP)"


    if ddp_active:
        setup_ddp(rank, world_size)
        # Broadcast args from rank 0 to all other processes
        args_list = [args if rank == 0 else None] # type: ignore
        dist.broadcast_object_list(args_list, src=0)
        if rank != 0: args = args_list[0] # type: ignore

    set_seed(args.seed, rank) # Set seed for all processes based on global rank for consistency
    
    # Ensure output directories exist (main process responsibility)
    if is_main_process(rank):
        create_dummy_data_if_not_exists(args.data_dir, rank=rank)
        # For prefixes, we need to ensure the base part of the path exists.
        # Example: if prefix is "outputs/samples", "outputs" must exist.
        dirs_to_check_or_create = [args.checkpoint_dir, args.final_model_dir]
        if args.sample_save_dir_prefix:
            dirs_to_check_or_create.append(os.path.dirname(args.sample_save_dir_prefix + "_dummy")) # Add dummy to get dir
        if args.gen_output_dir_prefix:
            dirs_to_check_or_create.append(os.path.dirname(args.gen_output_dir_prefix + "_dummy"))

        for dir_path in dirs_to_check_or_create:
            if dir_path and not os.path.exists(dir_path): # Check if dir_path is not empty
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Rank {current_rank_str_log}: Created directory {dir_path}")
                except OSError as e:
                    print(f"Rank {current_rank_str_log}: Warning, could not create directory {dir_path}: {e}")


    if is_main_process(rank):
        print("=" * 60)
        print(f"🎯 PyTorch GAN - 任务执行 (Rank {current_rank_str_log}/{world_size})")
        print("=" * 60)
        print("\n📋 使用配置:")
        for k_arg, v_arg in vars(args).items(): print(f"   {k_arg}: {v_arg}") # Renamed k,v
        print("-" * 30)

    try:
        if args.mode == 'train':
            if ddp_active and dist.is_initialized(): dist.barrier() # Sync before training starts
            netG_final, netD_final = train_gan(args, rank, world_size, ddp_active)
            
            if is_main_process(rank): print("\n🎉 训练流程完成！")
            
            # Post-training generation (only by main process)
            if is_main_process(rank) and netG_final is not None: 
                final_model_path = os.path.join(args.final_model_dir, 'generator_final.pth')
                if os.path.exists(final_model_path):
                    print("\n🎨 尝试使用最终训练好的模型生成一些测试图片...")
                    
                    # Create a new Namespace for generation to avoid modifying training args
                    gen_args = argparse.Namespace(**vars(args)) 
                    gen_args.model_path = final_model_path
                    gen_args.num_generate = args.num_save_samples # Generate same number as saved during training
                    
                    # Determine sizes to generate based on max_train_size
                    trained_sizes = [s for s in [64, 128, 256, 512] if s <= args.max_train_size]
                    if trained_sizes:
                        gen_args.sizes_to_generate = ",".join(map(str, trained_sizes))
                        generate_images(gen_args, rank)
                    else:
                        print("没有基于max_train_size的有效生成尺寸，跳过最终样本生成。")
                else:
                    print(f"未找到最终模型 {final_model_path}，跳过最终样本生成。")

        elif args.mode == 'generate':
            generate_images(args, rank)
        
    except KeyboardInterrupt:
        if is_main_process(rank): print("\n\n⏸️  操作被用户中断。")
    except Exception as e_global: # Renamed e
        if is_main_process(rank): print(f"\n❌ 发生严重错误: {e_global}")
        import traceback; traceback.print_exc()
    finally:
        if ddp_active and dist.is_initialized():
            cleanup_ddp()
        if is_main_process(rank): print("\n👋 程序执行完毕。")

if __name__ == "__main__":
    # 这部分是为了在Jupyter Notebook中使用 `%%writefile` 后，
    # 如果直接运行该cell（作为脚本），也能获取RANK和WORLD_SIZE。
    # 但通过 `torchrun` 启动时，torchrun会设置这些环境变量。
    # if 'RANK' not in os.environ: os.environ['RANK'] = '0'
    # if 'WORLD_SIZE' not in os.environ: os.environ['WORLD_SIZE'] = '1'
    # if 'MASTER_ADDR' not in os.environ: os.environ['MASTER_ADDR'] = 'localhost'
    # if 'MASTER_PORT' not in os.environ: os.environ['MASTER_PORT'] = '12355' # Arbitrary free port
    main()
