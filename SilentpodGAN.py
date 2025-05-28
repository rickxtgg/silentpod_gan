#中文注释优化版20250528_DDP
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
import json
from datetime import datetime
from torch.amp import autocast, GradScaler

# DDP相关导入
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- DDP 辅助函数 ---
def setup_ddp(rank, world_size):
    """初始化DDP进程组"""
    os.environ['MASTER_ADDR'] = 'localhost' # 通常由torchrun设置，这里为本地测试提供
    os.environ['MASTER_PORT'] = '12355'     # 选择一个未被占用的端口
    
    # 根据后端和环境调整
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank) # local_rank is implicitly rank here
    print(f"DDP: Rank {rank}/{world_size} initialized using backend {backend}.")

def cleanup_ddp():
    """清理DDP进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP: Process group destroyed.")

def is_main_process(rank):
    """检查当前是否为主进程 (rank 0)"""
    return rank == 0

# 设置随机种子以确保结果可复现
def set_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank) # 为不同进程设置略微不同的种子，但基于同一基础
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
        # 为了CUDNN的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    else: 
        print(f"警告: 不支持的模糊核大小 {size}, 使用3x3替代。")
        k_1d_list = [1., 2., 1.]
    
    k_1d = torch.tensor(k_1d_list, dtype=dtype, device=device)
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
        kernel = blur_k_2d.reshape(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer('blur_kernel', kernel, persistent=False)


    def forward(self, x):
        kernel_to_use = None 
        if x.shape[1] != self.channels:
            # 动态适应输入通道数，这在某些情况下有用，但要确保分组卷积的组数正确
            # 对于DDP，如果模型结构在不同rank上因这个逻辑产生差异，可能会有问题。
            # 但这里是根据输入x的shape[1]来repeat，应该是安全的。
            blur_k_2d_dyn = get_blur_kernel_2d(self.kernel_size, normalize=True, dtype=x.dtype, device=x.device)
            kernel_to_use = blur_k_2d_dyn.reshape(1, 1, self.kernel_size, self.kernel_size).repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel_to_use, padding=self.padding, groups=x.shape[1])
        
        kernel_to_use = self.blur_kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, kernel_to_use, padding=self.padding, groups=self.channels)

class UpsampleBlock(nn.Module):
    """
    上采样块，借鉴StyleGAN3的抗混叠思想。
    操作顺序: 插值上采样 -> 轻微模糊 (抗混叠) -> 卷积 -> 归一化 -> 激活
    """
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_padding=1,
                 blur_kernel_size=3, use_bias=False):
        super().__init__()
        self.interpolate_mode = 'bilinear' # StyleGAN3 推荐 'bilinear' 或更复杂的插值
        self.blur = None

        self.apply_blur = blur_kernel_size > 0
        if self.apply_blur:
            self.blur = Blur(in_channels, kernel_size=blur_kernel_size)
        
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=1, padding=conv_padding, bias=use_bias)
        # 使用 SyncBatchNorm 替换 BatchNorm2d 以适应 DDP
        # 注意：SyncBatchNorm 通常在模型较大，batch_size较小时表现更好，但也可能增加通信开销
        # 如果性能下降或遇到问题，可以考虑是否所有BN都需要Sync，或者特定条件下切换回普通BN
        self.norm = nn.SyncBatchNorm(out_channels) if dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode, align_corners=False)
        x_blurred = x_upsampled
        if self.apply_blur and self.blur is not None:
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
        self.blur = None
        
        self.apply_blur = blur_kernel_size > 0
        if self.apply_blur:
            self.blur = Blur(in_channels, kernel_size=blur_kernel_size)

        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=2, padding=conv_padding, bias=use_bias)
        
        self.use_norm = use_norm
        self.norm = None
        if self.use_norm:
            self.norm = nn.SyncBatchNorm(out_channels) if dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_blurred = x
        if self.apply_blur and self.blur is not None:
            x_blurred = self.blur(x)
        
        x_conv = self.conv(x_blurred)
        x_norm = x_conv
        if self.use_norm and self.norm is not None:
            x_norm = self.norm(x_conv)
        
        x_activated = self.activation(x_norm)
        return x_activated

# --- 数据集定义 ---
class ProductImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, rank=0): # rank for logging if needed
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if is_main_process(rank): # 只在主进程打印，避免DDP下重复打印
                print(f"Rank {rank}: 找到 {len(self.image_files)} 张图片在 {data_dir}")
        else:
            if is_main_process(rank):
                print(f"Rank {rank}: 错误: 数据目录 {data_dir} 不存在或不是一个目录。")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = None
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 在DDP中，一个进程出错可能导致挂起，错误处理要小心
            # print(f"错误: 无法打开或转换图片 {img_path}: {e}") # 减少打印
            placeholder_size = (256, 256) 
            if self.transform:
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

        if self.transform and image is not None:
            image = self.transform(image)
        elif image is None: 
             placeholder_size_fallback = (256,256)
             if self.transform:
                  for t_item in self.transform.transforms:
                     if isinstance(t_item, transforms.Resize):
                         if isinstance(t_item.size, int):
                             placeholder_size_fallback = (t_item.size, t_item.size)
                         else:
                             placeholder_size_fallback = t_item.size
                         break
             image_fallback = Image.new('RGB', placeholder_size_fallback, color='grey')
             if self.transform:
                 return self.transform(image_fallback)
             else:
                 return transforms.ToTensor()(image_fallback)
        return image

# --- 模型定义 ---
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, blur_kernel_size=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # BatchNorm -> SyncBatchNorm if DDP
            nn.SyncBatchNorm(ngf * 8) if dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        
        self.up1 = UpsampleBlock(ngf * 8, ngf * 4, blur_kernel_size=blur_kernel_size)
        self.up2 = UpsampleBlock(ngf * 4, ngf * 2, blur_kernel_size=blur_kernel_size)
        self.up3 = UpsampleBlock(ngf * 2, ngf, blur_kernel_size=blur_kernel_size)
        # 根据目标输出尺寸动态添加更多上采样层，支持到512x512
        self.up4 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size) # to 64x64 (feature map size)
        self.up5 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size) # to 128x128
        self.up6 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size) # to 256x256
        self.up7 = UpsampleBlock(ngf, ngf, blur_kernel_size=blur_kernel_size) # to 512x512
        
        # 输出层，不使用BN，通常在GAN的G的最后一层使用Tanh
        self.output_64 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.output_128 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.output_256 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        self.output_512 = nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1)
        
    def forward(self, input_noise, target_size=512):
        x = input_noise.view(-1, self.nz, 1, 1) # Reshape noise
        
        x = self.initial(x) # To 4x4
        x = self.up1(x) # To 8x8
        x = self.up2(x) # To 16x16
        x = self.up3(x) # To 32x32 (features for 64x64 output)
        
        feat_64 = self.up4(x) # To 64x64 (features for 128x128 output, or output for 64x64)
        if target_size == 64:
            return torch.tanh(self.output_64(feat_64))
        
        feat_128 = self.up5(feat_64) # To 128x128
        if target_size == 128:
            return torch.tanh(self.output_128(feat_128))
            
        feat_256 = self.up6(feat_128) # To 256x256
        if target_size == 256:
            return torch.tanh(self.output_256(feat_256))
            
        feat_512 = self.up7(feat_256) # To 512x512
        return torch.tanh(self.output_512(feat_512))

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, blur_kernel_size=3):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        # 判别器的第一层通常不使用BN
        self.down1 = DownsampleBlock(nc, ndf, blur_kernel_size=blur_kernel_size, use_norm=False)
        self.down2 = DownsampleBlock(ndf, ndf * 2, blur_kernel_size=blur_kernel_size)
        self.down3 = DownsampleBlock(ndf * 2, ndf * 4, blur_kernel_size=blur_kernel_size)
        self.down4 = DownsampleBlock(ndf * 4, ndf * 8, blur_kernel_size=blur_kernel_size)
        # 根据输入图像大小动态调整层数
        self.down5 = DownsampleBlock(ndf * 8, ndf * 8, blur_kernel_size=blur_kernel_size) # For inputs >= 128x128 -> feature map to 4x4 (if from 128) or 8x8 (if from 256)
        self.down6 = DownsampleBlock(ndf * 8, ndf * 8, blur_kernel_size=blur_kernel_size) # For inputs >= 256x256 -> feature map to 4x4 (if from 256) or 8x8 (if from 512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Ensure final feature map is 4x4 before final conv
        self.final_conv = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
    def forward(self, input_image):
        # input_image: [N, C, H, W]
        x = self.down1(input_image) # H/2
        x = self.down2(x)           # H/4
        x = self.down3(x)           # H/8
        x = self.down4(x)           # H/16

        # Dynamically use more downsampling layers based on feature map size
        # Assumes H_initial (input_image.shape[2]) is one of [64, 128, 256, 512]
        # After down4, H is H_initial / 16.
        # If H_initial = 64, H_after_down4 = 4.
        # If H_initial = 128, H_after_down4 = 8.
        # If H_initial = 256, H_after_down4 = 16.
        # If H_initial = 512, H_after_down4 = 32.
        
        if x.shape[2] > 8: # Corresponds to input_image > 128x128 (e.g. 256 or 512)
             x = self.down5(x) # H/32
        if x.shape[2] > 4 : # Corresponds to input_image > 64x64 (e.g. 128, 256 or 512, after down5 if applicable)
             x = self.down6(x) # H/64 (if from 256/512) or H/32 (if from 128)
        
        x = self.adaptive_pool(x) # Guarantees 4x4 feature map
        x = self.final_conv(x)    # Output is [N, 1, 1, 1]
        
        return x.view(-1) # Flatten to [N] for BCEWithLogitsLoss

# --- 权重初始化 ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('SyncBatchNorm') != -1: # Include SyncBatchNorm
        if hasattr(m, 'weight') and m.weight is not None:
             nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# --- 数据加载器创建 ---
def create_data_loaders(data_dir, batch_size=8, num_workers=0, rank=0, world_size=1):
    transforms_dict = {}
    datasets_dict = {}
    dataloaders_dict = {}
    samplers_dict = {}

    for size_val in [64, 128, 256, 512]:
        current_transform = transforms.Compose([
            transforms.Resize((size_val, size_val)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transforms_dict[size_val] = current_transform
        
        current_dataset = ProductImageDataset(data_dir, transforms_dict[size_val], rank=rank)
        datasets_dict[size_val] = current_dataset
        
        sampler = None
        shuffle_dl = True # Shuffle for non-DDP or if sampler handles it
        if world_size > 1: # DDP active
            sampler = DistributedSampler(current_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            samplers_dict[size_val] = sampler
            shuffle_dl = False # Sampler handles shuffling

        if len(current_dataset) > 0:
            # drop_last is important for DDP if batch sizes are not perfectly divisible
            # It's also generally good for GANs to have consistent batch sizes
            effective_len = len(current_dataset) // world_size if world_size > 0 else len(current_dataset)
            if effective_len >= batch_size : # batch_size is per-GPU here
                dataloaders_dict[size_val] = DataLoader(current_dataset, batch_size=batch_size, 
                                                    shuffle=shuffle_dl, sampler=sampler,
                                                    num_workers=num_workers, drop_last=True, 
                                                    pin_memory=torch.cuda.is_available())
            else:
                if is_main_process(rank):
                    print(f"警告: {size_val}x{size_val} 数据集在rank {rank}上的有效样本数 {effective_len} 小于批大小 {batch_size} (且drop_last=True)，无法创建DataLoader。")
                dataloaders_dict[size_val] = None
        else:
            if is_main_process(rank):
                print(f"警告: {size_val}x{size_val} 数据集为空，无法创建DataLoader。")
            dataloaders_dict[size_val] = None
            
    return dataloaders_dict

# --- 保存样本图片 (只在主进程执行) ---
def save_sample_images(generator_module, epoch, current_size, nz=100, save_dir_prefix='generated_samples', device='cpu', amp_enabled=False):
    # generator_module should be netG.module if DDP, or netG if not
    generator_module.eval() 
    with torch.no_grad(): 
        fixed_noise = torch.randn(16, nz, device=device) 
        amp_dtype = torch.float16 if device.type == 'cuda' and amp_enabled else torch.float32

        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            fake_images = generator_module(fixed_noise, target_size=current_size)
        
        fake_images = (fake_images + 1) / 2.0 
        fake_images = torch.clamp(fake_images, 0, 1)
        
        save_dir = f'{save_dir_prefix}_{current_size}x{current_size}'
        os.makedirs(save_dir, exist_ok=True)
        
        fake_images_np = fake_images.cpu().float().numpy() 
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'Epoch {epoch} - Size {current_size}x{current_size}', fontsize=16)
        for i_plt in range(16): # Renamed i to i_plt to avoid conflict
            row, col = i_plt // 4, i_plt % 4
            img_np_permuted = np.transpose(fake_images_np[i_plt], (1, 2, 0))
            axes[row, col].imshow(img_np_permuted)
            axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_dir, f'epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig) 
    generator_module.train()

# --- 检查点保存与加载 (只在主进程执行保存，所有进程加载) ---
def save_checkpoint(netG_module, netD_module, optimizerG, optimizerD, scaler, epoch, phase_info, 
                    checkpoint_dir='checkpoints', amp_enabled=False, rank=0):
    # netG_module and netD_module should be the underlying models (e.g., netG.module)
    if not is_main_process(rank): # 只有主进程保存检查点
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'generator_state_dict': netG_module.state_dict(),
        'discriminator_state_dict': netD_module.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'epoch': epoch, # Global epoch
        'phase_info': phase_info, # Info about current training phase
        'timestamp': datetime.now().isoformat()
    }
    if amp_enabled and scaler is not None:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
    
    # 新检查点路径
    new_checkpoint_filename = f'checkpoint_epoch_{epoch:04d}.pth'
    new_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint_filename)
    torch.save(checkpoint_data, new_checkpoint_path)
    
    # 清理旧的检查点文件，只保留最新的
    latest_json_path = os.path.join(checkpoint_dir, 'latest.json')
    previous_checkpoint_file_to_delete = None

    if os.path.exists(latest_json_path):
        try:
            with open(latest_json_path, 'r', encoding='utf-8') as f:
                latest_info_old = json.load(f)
            if 'latest_checkpoint' in latest_info_old:
                # Make sure it's a .pth file as expected
                if os.path.basename(latest_info_old['latest_checkpoint']).startswith('checkpoint_epoch_') and \
                   latest_info_old['latest_checkpoint'].endswith('.pth'):
                    previous_checkpoint_file_to_delete = latest_info_old['latest_checkpoint']
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Rank {rank}: 警告: 读取或解析旧的 {latest_json_path} 失败。可能无法清理旧的检查点。")

    # 更新 latest.json 指向新的检查点
    latest_info_new = {
        'latest_checkpoint': new_checkpoint_path, # Store the full path
        'epoch': epoch,
        'phase_info': phase_info,
        'timestamp': datetime.now().isoformat()
    }
    with open(latest_json_path, 'w', encoding='utf-8') as f:
        json.dump(latest_info_new, f, ensure_ascii=False, indent=2)
    
    print(f"Rank {rank}: 检查点已保存: {new_checkpoint_path}")

    # 删除旧的 .pth 文件 (如果它存在且与新文件不同)
    if previous_checkpoint_file_to_delete and \
       previous_checkpoint_file_to_delete != new_checkpoint_path and \
       os.path.exists(previous_checkpoint_file_to_delete):
        try:
            os.remove(previous_checkpoint_file_to_delete)
            print(f"Rank {rank}: 已删除旧的检查点文件: {previous_checkpoint_file_to_delete}")
        except OSError as e:
            print(f"Rank {rank}: 警告: 删除旧检查点文件 {previous_checkpoint_file_to_delete} 失败: {e}")


def load_checkpoint(checkpoint_path, netG_module, netD_module, optimizerG, optimizerD, scaler, device, rank=0): 
    # netG_module and netD_module are the underlying models
    if not os.path.exists(checkpoint_path):
        if is_main_process(rank):
            print(f"Rank {rank}: 检查点文件不存在: {checkpoint_path}")
        return 0, {}
    
    # 所有进程都从同一个文件加载，确保map_location正确
    checkpoint_data = torch.load(checkpoint_path, map_location=device) 
    
    netG_module.load_state_dict(checkpoint_data['generator_state_dict'])
    netD_module.load_state_dict(checkpoint_data['discriminator_state_dict'])
    
    if optimizerG and 'optimizerG_state_dict' in checkpoint_data:
        optimizerG.load_state_dict(checkpoint_data['optimizerG_state_dict'])
    if optimizerD and 'optimizerD_state_dict' in checkpoint_data:
        optimizerD.load_state_dict(checkpoint_data['optimizerD_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint_data: 
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        if is_main_process(rank): print("Rank {rank}: GradScaler状态已加载。")
    elif scaler is not None and is_main_process(rank):
        print("Rank {rank}: 警告: 检查点中未找到GradScaler状态。将使用新的Scaler。")

    start_epoch = checkpoint_data.get('epoch', 0) # Global epoch completed
    phase_info = checkpoint_data.get('phase_info', {})
    
    if is_main_process(rank):
        print(f"Rank {rank}: 已从检查点加载: {checkpoint_path}")
        print(f"Rank {rank}: 恢复到全局第 {start_epoch} 个epoch之后，阶段信息: {phase_info}")
    
    return start_epoch, phase_info


# --- 主训练函数 ---
def train_gan(data_dir, num_epochs_override=None, batch_size=4, lr=0.0002, nz=100, 
              resume_from_checkpoint=False, blur_kernel_size=3, num_workers=0, grad_clip_norm=1.0,
              rank=0, world_size=1, ddp_active=False):
    
    # DDP: device is the local rank's GPU
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() and ddp_active else \
             torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    amp_enabled = (device.type == 'cuda') 
    amp_dtype = torch.float16 if amp_enabled else torch.float32
    
    if is_main_process(rank):
        print(f"Rank {rank}: 使用设备: {device}")
        if amp_enabled:
            print(f"Rank {rank}: 自动混合精度 (AMP) 已启用，使用数据类型: {amp_dtype}。")
        else:
            print("Rank {rank}: 自动混合精度 (AMP) 未启用。")

    # 模型初始化在各自的设备上
    netG_base = Generator(nz=nz, ngf=64, nc=3, blur_kernel_size=blur_kernel_size).to(device)
    netD_base = Discriminator(nc=3, ndf=64, blur_kernel_size=blur_kernel_size).to(device)
    
    netG_base.apply(weights_init)
    netD_base.apply(weights_init)

    # DDP 包装
    if ddp_active:
        # find_unused_parameters=True can be important for GANs if parts of graph aren't always used
        netG = DDP(netG_base, device_ids=[rank] if device.type == 'cuda' else None, output_device=rank if device.type == 'cuda' else None, find_unused_parameters=True)
        netD = DDP(netD_base, device_ids=[rank] if device.type == 'cuda' else None, output_device=rank if device.type == 'cuda' else None, find_unused_parameters=True)
    else:
        netG = netG_base
        netD = netD_base
    
    # 获取用于保存/加载/评估的模块
    netG_module = netG.module if ddp_active else netG
    netD_module = netD.module if ddp_active else netD

    criterion = nn.BCEWithLogitsLoss()
    
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-8)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-8)

    scaler = GradScaler(enabled=amp_enabled)
    
    dataloaders = create_data_loaders(data_dir, batch_size, num_workers, rank, world_size)
    if not any(dataloaders.values()):
        if is_main_process(rank):
            print("Rank {rank}: 错误: 所有尺寸的数据加载器均为空。程序将退出。")
        return None, None

    training_phases = [
        {'size': 64, 'epochs': 10}, 
        {'size': 128, 'epochs': 10},
        {'size': 256, 'epochs': 10},
        {'size': 512, 'epochs': 10}
    ]
    if num_epochs_override is not None:
        for phase_item in training_phases:
            phase_item['epochs'] = num_epochs_override

    # 计算有效的总epoch数 (只计算有数据加载器的阶段)
    total_epochs_across_phases = sum(p['epochs'] for p in training_phases if dataloaders.get(p['size']) is not None)
    if total_epochs_across_phases == 0 :
        if is_main_process(rank):
            print(f"Rank {rank}: 错误: 没有可训练的阶段。程序将退出。")
        return None, None

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=total_epochs_across_phases, eta_min=lr * 0.01)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=total_epochs_across_phases, eta_min=lr * 0.01)
    
    global_start_epoch = 0 #表示已完成的全局epoch数
    current_phase_idx_resume = 0
    epoch_in_phase_start_resume = 0 #表示当前阶段已完成的epoch数
    phase_info_loaded = {}

    if resume_from_checkpoint:
        latest_json_path = 'checkpoints/latest.json'
        if os.path.exists(latest_json_path): # 所有进程检查，但加载由load_checkpoint处理
            with open(latest_json_path, 'r', encoding='utf-8') as f_json: # 重命名 f
                latest_info = json.load(f_json)
            checkpoint_path_to_load = latest_info['latest_checkpoint']
            
            # 所有进程加载模型和优化器状态
            # netG_module, netD_module 是未包装或 .module 的模型
            global_start_epoch, phase_info_loaded = load_checkpoint(
                checkpoint_path_to_load, netG_module, netD_module, optimizerG, optimizerD, scaler, device, rank
            )
            
            if phase_info_loaded: 
                current_phase_idx_resume = phase_info_loaded.get('phase_idx', 0)
                # epoch_in_phase in phase_info is the epoch *just completed* or about to start if 0
                # If it's 'epoch_in_phase': 1, it means 1 epoch of that phase was done. So next is 1.
                epoch_in_phase_start_resume = phase_info_loaded.get('epoch_in_phase', 0) 

            # 所有进程同步调度器状态
            for _ in range(global_start_epoch): 
                schedulerG.step()
                schedulerD.step()
        elif is_main_process(rank):
            print(f"Rank {rank}: 未找到 latest.json，从头开始训练。")

    if ddp_active: # DDP Barrier: 确保所有进程在开始训练前都已完成加载或初始化
        dist.barrier()

    if is_main_process(rank): print(f"Rank {rank}: 开始训练...")
    current_global_epoch = global_start_epoch # 当前正在进行的全局epoch (0-indexed)
    
    for phase_idx in range(current_phase_idx_resume, len(training_phases)):
        phase = training_phases[phase_idx]
        current_target_size = phase['size']
        phase_total_epochs = phase['epochs']
        
        current_dataloader = dataloaders.get(current_target_size)
        if current_dataloader is None:
            if is_main_process(rank):
                print(f"Rank {rank}: 跳过阶段 {phase_idx + 1} (尺寸 {current_target_size}x{current_target_size}) 因为数据加载器为空。")
            continue
        
        # DDP: set epoch for DistributedSampler, to ensure shuffling varies across epochs
        if ddp_active and hasattr(current_dataloader.sampler, 'set_epoch'):
            current_dataloader.sampler.set_epoch(current_global_epoch) # Use global epoch for consistent shuffling state if resuming

        if is_main_process(rank):
            print(f"\nRank {rank}: === 阶段 {phase_idx + 1}/{len(training_phases)}: 训练 {current_target_size}x{current_target_size} 尺寸 ===")
        
        # 如果是从这个阶段恢复，start_e_in_phase 是已完成的epoch数，所以从它开始
        start_e_in_phase = epoch_in_phase_start_resume if phase_idx == current_phase_idx_resume else 0
        
        for epoch_in_phase in range(start_e_in_phase, phase_total_epochs):
            epoch_start_time = datetime.now()
            netG.train() # Sets DDP-wrapped model to train mode
            netD.train()
            
            for i_batch, real_images in enumerate(current_dataloader):
                real_images = real_images.to(device)
                batch_actual_size = real_images.size(0) # Per-GPU batch size
                
                real_label_val = 0.9 
                fake_label_val = 0.1
                real_label = torch.full((batch_actual_size,), real_label_val, dtype=torch.float32, device=device)
                fake_label = torch.full((batch_actual_size,), fake_label_val, dtype=torch.float32, device=device)

                # --- 训练判别器 ---
                optimizerD.zero_grad(set_to_none=True)
                
                noise = torch.randn(batch_actual_size, nz, device=device)
                fake_images, output_real, loss_D_real, output_fake_detached, loss_D_fake, loss_D = None, None, None, None, None, None

                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    fake_images = netG(noise, target_size=current_target_size) # DDP forward

                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    output_real = netD(real_images) # DDP forward
                    loss_D_real = criterion(output_real, real_label)
                    
                    output_fake_detached = netD(fake_images.detach()) # DDP forward
                    loss_D_fake = criterion(output_fake_detached, fake_label)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                if loss_D is not None:
                    scaler.scale(loss_D).backward() # DDP handles gradient sync
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizerD) 
                        torch.nn.utils.clip_grad_norm_(netD.parameters(), grad_clip_norm)
                    scaler.step(optimizerD)

                # --- 训练生成器 ---
                optimizerG.zero_grad(set_to_none=True)
                output_fake_for_G, loss_G = None, None
                if fake_images is not None: 
                    with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        output_fake_for_G = netD(fake_images) # DDP forward (new graph for G)
                        loss_G = criterion(output_fake_for_G, real_label) 
                    
                    if loss_G is not None:
                        scaler.scale(loss_G).backward() # DDP handles gradient sync
                        if grad_clip_norm > 0:
                            scaler.unscale_(optimizerG)
                            torch.nn.utils.clip_grad_norm_(netG.parameters(), grad_clip_norm)
                        scaler.step(optimizerG)

                scaler.update()

                loss_D_item = loss_D.item() if loss_D is not None else float('nan')
                loss_G_item = loss_G.item() if loss_G is not None else float('nan')
                # For DDP, these means are per-process. Could aggregate if needed for global stats.
                output_real_mean = output_real.mean().item() if output_real is not None else float('nan')
                output_fake_detached_mean = output_fake_detached.mean().item() if output_fake_detached is not None else float('nan')
                output_fake_for_G_mean = output_fake_for_G.mean().item() if output_fake_for_G is not None else float('nan')

                if i_batch % 50 == 0 and is_main_process(rank): # Log only from main process
                    current_lr_G = optimizerG.param_groups[0]['lr']
                    current_lr_D = optimizerD.param_groups[0]['lr']
                    print(f'[R{rank}][Ph{phase_idx+1}][GEp {current_global_epoch+1}/{total_epochs_across_phases}] '
                          f'[PhEp {epoch_in_phase+1}/{phase_total_epochs}] [{i_batch}/{len(current_dataloader)}] '
                          f'L_D: {loss_D_item:.4f} L_G: {loss_G_item:.4f} '
                          f'D(x): {output_real_mean:.4f} '
                          f'D(G(z)): {output_fake_detached_mean:.4f}(D)/{output_fake_for_G_mean:.4f}(G) '
                          f'LR_G: {current_lr_G:.6f} LR_D: {current_lr_D:.6f} Scale: {scaler.get_scale():.1f}')
            
            # End of an epoch_in_phase
            current_global_epoch += 1 
            schedulerG.step() # Step schedulers for all processes
            schedulerD.step()

            # Save samples and checkpoint (only from main process)
            if is_main_process(rank):
                if (epoch_in_phase + 1) % 5 == 0 or (epoch_in_phase + 1) == phase_total_epochs:
                    save_sample_images(netG_module, current_global_epoch, current_target_size, nz, 
                                       device=device, amp_enabled=amp_enabled)
                    
                    current_phase_progress_info = {
                        'phase_idx': phase_idx,
                        'epoch_in_phase': epoch_in_phase + 1, # epoch_in_phase just completed
                        'size': current_target_size
                    }
                    save_checkpoint(netG_module, netD_module, optimizerG, optimizerD, scaler, 
                                    current_global_epoch, current_phase_progress_info, # current_global_epoch is epochs completed
                                    amp_enabled=amp_enabled, rank=rank)

                epoch_duration = datetime.now() - epoch_start_time
                print(f"Rank {rank}: 阶段Epoch {epoch_in_phase+1} 完成，耗时: {epoch_duration}")
            
            if ddp_active: # Barrier to ensure all processes complete epoch before next phase/sampler update
                dist.barrier()


        epoch_in_phase_start_resume = 0 # Reset for the next phase
        if is_main_process(rank):
            print(f"Rank {rank}: 阶段 {phase_idx + 1} ({current_target_size}x{current_target_size}) 训练完成。")
    
    if is_main_process(rank):
        print("\n=== 所有训练阶段完成！ ===")
        final_model_dir = 'final_models_stylegan3_inspired'
        os.makedirs(final_model_dir, exist_ok=True)
        torch.save(netG_module.state_dict(), os.path.join(final_model_dir, 'generator_final.pth'))
        torch.save(netD_module.state_dict(), os.path.join(final_model_dir, 'discriminator_final.pth'))
        
        config_summary = {
            'total_global_epochs_trained': current_global_epoch,
            'batch_size_per_gpu': batch_size,
            'world_size': world_size,
            'global_batch_size': batch_size * world_size,
            'initial_learning_rate': lr,
            'noise_dim': nz,
            'blur_kernel_size_for_antialiasing': blur_kernel_size,
            'training_phases_config': training_phases,
            'amp_enabled_during_training': amp_enabled,
            'amp_dtype_used': str(amp_dtype) if amp_enabled else 'N/A',
            'ddp_active': ddp_active
        }
        with open(os.path.join(final_model_dir, 'training_config.json'), 'w', encoding='utf-8') as f_json_out: # Renamed f
            json.dump(config_summary, f_json_out, ensure_ascii=False, indent=2)
        print(f"Rank {rank}: 最终模型已保存到: {final_model_dir}")
    
    return netG_module, netD_module

# --- 生成图片 (通常在单进程/主进程执行) ---
def generate_images(model_path, num_images=10, sizes_to_generate=[64, 128, 256, 512], 
                    nz=100, blur_kernel_size=3, rank=0): # rank for logging
    if not is_main_process(rank): # Generation only on main process
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use default GPU for generation
    amp_enabled_inference = (device.type == 'cuda') 
    amp_dtype_inference = torch.float16 if amp_enabled_inference else torch.float32

    print(f"Rank {rank}: 使用设备进行生成: {device}, AMP推理: {'启用' if amp_enabled_inference else '未启用'}, Dtype: {amp_dtype_inference if amp_enabled_inference else 'default'}")
    try:
        # Initialize base model, not DDP wrapped
        netG = Generator(nz=nz, ngf=64, nc=3, blur_kernel_size=blur_kernel_size).to(device)
        if not os.path.exists(model_path):
            print(f"Rank {rank}: 错误：模型文件不存在 {model_path}")
            return
        
        # state_dict should be from netG.module if saved from DDP training
        state_dict = torch.load(model_path, map_location=device)
        netG.load_state_dict(state_dict)
        netG.eval()
        print(f"Rank {rank}: 成功加载模型: {model_path}")

        with torch.no_grad():
            for size_val_gen in sizes_to_generate:
                print(f"\nRank {rank}: 正在生成 {size_val_gen}x{size_val_gen} 尺寸的图片...")
                save_dir = f'final_generated_stylegan3_inspired_{size_val_gen}x{size_val_gen}'
                os.makedirs(save_dir, exist_ok=True)
                
                for i_gen in range(num_images):
                    noise = torch.randn(1, nz, device=device)
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
                        if (i_gen + 1) % 5 == 0 or (i_gen + 1) == num_images:
                            print(f"  Rank {rank}: 已生成 {i_gen+1}/{num_images} 张图片 (尺寸 {size_val_gen}x{size_val_gen})")
                print(f"Rank {rank}: ✓ {size_val_gen}x{size_val_gen} 图片生成完成，保存在 {save_dir} 目录")
        print(f"\nRank {rank}: 所有图片生成完成！")
    except Exception as e:
        print(f"Rank {rank}: 生成图片时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# --- 交互模式和主函数 ---

def create_dummy_data_if_not_exists(data_dir, num_dummy_images=16, rank=0):
    if not is_main_process(rank): # Only main process creates data
        return

    if not os.path.exists(data_dir):
        print(f"Rank {rank}: 数据目录 {data_dir} 不存在，将创建它。")
        os.makedirs(data_dir, exist_ok=True)
    
    img_extensions = ('.png', '.jpg', '.jpeg')
    try:
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(img_extensions)]
        if not image_files:
            print(f"Rank {rank}: 数据目录 {data_dir} 为空，将创建 {num_dummy_images} 张虚拟PNG图片。")
            for i_dummy in range(num_dummy_images):
                color = (np.random.randint(0,255), np.random.randint(0,128), np.random.randint(128,255))
                dummy_img = Image.new('RGB', (512,512), color=color)
                dummy_img.save(os.path.join(data_dir, f"dummy_image_{i_dummy+1}.png"))
            print(f"Rank {rank}: 已创建 {num_dummy_images} 张虚拟图片在 {data_dir}。")
    except OSError as e:
        print(f"Rank {rank}: 检查或创建虚拟数据时发生OS错误 (可能是权限问题): {e} in {data_dir}")
    except Exception as e:
        print(f"Rank {rank}: 创建虚拟图片失败: {e}")


def configure_training_params(resume=False): # Assumed to be run by main process only if interactive
    print(f"\n{'⏯️  配置恢复训练参数' if resume else '🚀 配置新训练参数'}")
    
    default_data_dir = "./product_images"
    # create_dummy_data_if_not_exists is called by main process later if needed

    data_dir_input_val = input(f"数据目录路径 (默认: {default_data_dir}): ").strip()
    data_dir = data_dir_input_val or default_data_dir
    
    # This validation loop should ideally be before DDP setup if interactive
    img_extensions = ('.png', '.jpg', '.jpeg')
    while True:
        # This function is called only by rank 0 if interactive.
        # So, create_dummy_data_if_not_exists will also be called by rank 0.
        create_dummy_data_if_not_exists(data_dir, rank=0) # Pass rank explicitly

        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            try:
                if any(f.lower().endswith(img_extensions) for f in os.listdir(data_dir)):
                    break 
                else:
                    print(f"错误: 目录 '{data_dir}' 不包含PNG/JPG/JPEG图片。已尝试创建虚拟数据。")
            except OSError as e:
                print(f"错误: 无法访问目录 '{data_dir}' 中的文件 (权限问题?): {e}")
        else:
            print(f"错误: 目录 '{data_dir}' 不存在或不是一个目录。已尝试创建默认目录和虚拟数据。")

        data_dir_input_val = input(f"请重新输入有效的数据目录路径 (或按回车使用默认 '{default_data_dir}'): ").strip()
        data_dir = data_dir_input_val or default_data_dir


    batch_size_str = input("每GPU批次大小 (默认: 2, 如果GPU内存允许可尝试4-8): ").strip() or "2"
    batch_size = int(batch_size_str)
    learning_rate = float(input("学习率 (默认: 0.0002): ").strip() or 0.0002)
    noise_dim = int(input("噪声向量维度 (默认: 100): ").strip() or 100)
    blur_k_size = int(input("抗混叠模糊核大小 (0禁用, 3或5推荐, 默认3): ").strip() or 3)
    num_epochs_per_phase_str = input("每个阶段的Epoch数 (可选, 默认按代码内设置): ").strip()
    num_epochs_per_phase = int(num_epochs_per_phase_str) if num_epochs_per_phase_str else None
    num_workers_str = input("DataLoader工作进程数 (默认: 0 for win, 2 for linux, CPU允许可设更高): ").strip()
    default_workers = 2 if os.name != 'nt' else 0 # Better default for Linux
    num_workers = int(num_workers_str) if num_workers_str else default_workers
    grad_clip_str = input(f"梯度裁剪范数 (默认1.0, 输入0禁用): ").strip() or "1.0"
    grad_clip_norm = float(grad_clip_str)

    return {
        'mode': 'train',
        'data_dir': data_dir,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'noise_dim': noise_dim,
        'resume': resume,
        'blur_kernel_size': blur_k_size,
        'num_epochs_per_phase': num_epochs_per_phase,
        'num_workers': num_workers,
        'grad_clip_norm': grad_clip_norm
    }

def configure_generation_params(): # Assumed to be run by main process only
    print("\n🎨 配置图片生成参数")
    default_model = "final_models_stylegan3_inspired/generator_final.pth"
    model_path = input(f"模型文件路径 (默认: {default_model}): ").strip() or default_model
    while not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在。")
        model_path = input(f"请重新输入有效的模型文件路径 (默认: {default_model}): ").strip() or default_model

    num_generate = int(input("生成图片数量 (默认: 4): ").strip() or 4)
    noise_dim = int(input("噪声向量维度 (默认: 100, 需与训练时一致): ").strip() or 100)
    blur_k_size = int(input("抗混叠模糊核大小 (用于模型初始化, 需与训练时一致, 默认3): ").strip() or 3)
    
    print("\n选择要生成的图片尺寸 (可多选，用逗号分隔，例如 1,3):")
    print("1. 64x64")
    print("2. 128x128")
    print("3. 256x256")
    print("4. 512x512")
    print("5. 所有尺寸 (64, 128, 256, 512)")
    
    size_map_single = {'1': 64, '2': 128, '3': 256, '4': 512}
    all_sizes_list = [64, 128, 256, 512]
    sizes_to_generate_list = []
    
    while True:
        choices_str = input("请选择 (默认: 5): ").strip() or '5'
        if choices_str == '5':
            sizes_to_generate_list = all_sizes_list
            break
        try:
            selected_indices = [s.strip() for s in choices_str.split(',')]
            sizes_to_generate_list = []
            valid_choices = True
            for idx_str in selected_indices:
                if idx_str in size_map_single:
                    sizes_to_generate_list.append(size_map_single[idx_str])
                else:
                    valid_choices = False
                    break
            if valid_choices and sizes_to_generate_list:
                sizes_to_generate_list = sorted(list(set(sizes_to_generate_list)))
                break
            else:
                print("❌ 无效选择或未选择任何尺寸。")
        except Exception as e:
            print(f"❌ 输入格式错误: {e}")

    return {
        'mode': 'generate',
        'model_path': model_path,
        'num_generate': num_generate,
        'noise_dim': noise_dim,
        'sizes_to_generate': sizes_to_generate_list,
        'blur_kernel_size': blur_k_size
    }

def interactive_mode(): # Assumed to be run by main process only
    print("\n" + "=" * 60)
    print("🎯 PyTorch GAN (StyleGAN3启发式优化 & DDP) - 交互模式")
    print("=" * 60)
    print("\n请选择运行模式:")
    print("1. 🚀 开始新的训练")
    print("2. ⏯️  从检查点恢复训练")
    print("3. 🎨 仅生成图片（需要已训练的模型）")
    print("0. 🚪 退出程序")
    
    config_choice = None
    while True:
        choice = input("\n请输入选择 (0-3): ").strip()
        if choice == '0': 
            config_choice = None
            break
        if choice == '1': 
            config_choice = configure_training_params(resume=False)
            break
        if choice == '2': 
            config_choice = configure_training_params(resume=True)
            break
        if choice == '3': 
            config_choice = configure_generation_params()
            break
        print("❌ 无效选择，请输入0-3之间的数字")
    return config_choice


def main():
    # DDP setup: RANK and WORLD_SIZE are set by torchrun
    # LOCAL_RANK is also set by torchrun for node-local rank
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # For single node, local_rank is often the same as rank.
    # If using torch.cuda.set_device, local_rank is the key.
    # torchrun sets LOCAL_RANK, if not, assume it's rank for single node.
    local_rank = int(os.environ.get("LOCAL_RANK", rank)) 

    ddp_active = world_size > 1
    if ddp_active:
        setup_ddp(rank, world_size) # rank here is global rank
    
    set_seed(42, rank) # Set seed for all processes

    config = None
    if is_main_process(rank): # Interactive mode only for main process
        if not ddp_active: # If not DDP, run interactive mode fully
            print("\n🎯 启动交互模式 (单进程)...")
            config = interactive_mode()
        else: # If DDP, main process can still take input, but needs to broadcast
              # For simplicity here, if DDP, we skip interactive and use defaults
              # This part would need a proper CLI arg parser for DDP
            print(f"Rank {rank}: DDP模式激活 (world_size={world_size}). "
                  "跳过交互式配置，使用默认训练参数。")
            default_data_dir = "./product_images"
            create_dummy_data_if_not_exists(default_data_dir, rank=rank) # Main process creates
            
            config = { # Example default config for DDP training
                'mode': 'train', 'data_dir': default_data_dir, 'batch_size': 2,
                'learning_rate': 0.0002, 'noise_dim': 100, 'resume': False,
                'blur_kernel_size': 3, 'num_epochs_per_phase': 2, # Shorter for demo
                'num_workers': 2 if os.name != 'nt' else 0, 'grad_clip_norm': 1.0
            }
    
    if ddp_active:
        # Broadcast config from rank 0 to all other processes
        # This requires config to be picklable.
        config_list = [config] # Put config in a list to use broadcast_object_list
        dist.broadcast_object_list(config_list, src=0)
        if not is_main_process(rank): # Other processes get config from broadcast
            config = config_list[0]

    if config is None: # If main process chose to exit, or broadcast failed (though simple dict should be fine)
        if is_main_process(rank): print("👋 程序退出。")
        if ddp_active: cleanup_ddp()
        return

    if is_main_process(rank):
        print("=" * 60)
        print(f"🎯 PyTorch GAN - 任务执行 (Rank {rank}/{world_size})")
        print("=" * 60)

    try:
        if config['mode'] == 'generate':
            if is_main_process(rank): # Generation only on main process
                print("\n🎨 生成配置:")
                for k_conf, v_conf in config.items(): print(f"   {k_conf}: {v_conf}")
                print("-" * 30)
            # Pass rank for logging, generate_images itself has a main_process guard
            generate_images(
                model_path=config['model_path'],
                num_images=config['num_generate'],
                sizes_to_generate=config['sizes_to_generate'],
                nz=config['noise_dim'],
                blur_kernel_size=config['blur_kernel_size'],
                rank=rank 
            )
        elif config['mode'] == 'train':
            if is_main_process(rank):
                print("\n📋 训练配置:")
                for k_conf, v_conf in config.items(): print(f"   {k_conf}: {v_conf}")
                print("-" * 30)
            
            # Barrier before training to ensure all configs are set and data (dummy) might be created
            if ddp_active:
                dist.barrier()

            netG_final, netD_final = train_gan( 
                data_dir=config['data_dir'],
                num_epochs_override=config.get('num_epochs_per_phase'),
                batch_size=config['batch_size'],
                lr=config['learning_rate'],
                nz=config['noise_dim'],
                resume_from_checkpoint=config['resume'],
                blur_kernel_size=config['blur_kernel_size'],
                num_workers=config['num_workers'],
                grad_clip_norm=config['grad_clip_norm'],
                rank=rank, world_size=world_size, ddp_active=ddp_active
            )
            
            if is_main_process(rank): print("\n🎉 训练流程完成！")
            
            if is_main_process(rank): # Final sample generation by main process
                final_model_path = f'final_models_stylegan3_inspired/generator_final.pth'
                if os.path.exists(final_model_path):
                    print("\n🎨 尝试使用最终训练好的模型生成一些测试图片...")
                    generate_images(
                        model_path=final_model_path, 
                        num_images=4, 
                        sizes_to_generate=[64, 128, 256, 512], 
                        nz=config['noise_dim'], 
                        blur_kernel_size=config['blur_kernel_size'],
                        rank=rank
                    )
                else:
                    print(f"未找到最终模型 {final_model_path}，跳过最终样本生成。")

    except KeyboardInterrupt:
        if is_main_process(rank): print("\n\n⏸️  操作被用户中断。")
    except Exception as e:
        if is_main_process(rank): print(f"\n❌ 发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ddp_active:
            cleanup_ddp()
        if is_main_process(rank): print("\n👋 程序执行完毕。")

if __name__ == "__main__":
    main()
