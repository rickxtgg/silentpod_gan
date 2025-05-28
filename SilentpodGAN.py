#中文注释优化版20250528_DDP_ArgParse
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
import argparse # 新增 argparse

# DDP相关导入
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- DDP 辅助函数 ---
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    print(f"DDP: Rank {rank}/{world_size} initialized using backend {backend}.")

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP: Process group destroyed.")

def is_main_process(rank):
    return rank == 0

# 设置随机种子
def set_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 辅助函数和层 ---
def get_blur_kernel_2d(size=3, normalize=True, dtype=torch.float32, device='cpu'):
    k_1d_list = []
    if size == 1: k_1d_list = [1.]
    elif size == 3: k_1d_list = [1., 2., 1.]
    elif size == 5: k_1d_list = [1., 4., 6., 4., 1.]
    else:
        print(f"警告: 不支持的模糊核大小 {size}, 使用3x3替代。")
        k_1d_list = [1., 2., 1.]
    k_1d = torch.tensor(k_1d_list, dtype=dtype, device=device)
    kernel_2d = torch.outer(k_1d, k_1d)
    if normalize: kernel_2d = kernel_2d / torch.sum(kernel_2d)
    return kernel_2d

class Blur(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        blur_k_2d = get_blur_kernel_2d(kernel_size, normalize=True, dtype=torch.float32)
        kernel = blur_k_2d.reshape(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer('blur_kernel', kernel, persistent=False)

    def forward(self, x):
        kernel_to_use = self.blur_kernel.to(device=x.device, dtype=x.dtype)
        if x.shape[1] != self.channels: # 动态适应通道数，例如在判别器第一层
            blur_k_2d_dyn = get_blur_kernel_2d(self.kernel_size, normalize=True, dtype=x.dtype, device=x.device)
            kernel_to_use = blur_k_2d_dyn.reshape(1, 1, self.kernel_size, self.kernel_size).repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel_to_use, padding=self.padding, groups=x.shape[1])
        return F.conv2d(x, kernel_to_use, padding=self.padding, groups=self.channels)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_padding=1,
                 blur_kernel_size=3, use_bias=False):
        super().__init__()
        self.interpolate_mode = 'bilinear'
        self.blur = None
        self.apply_blur = blur_kernel_size > 0
        if self.apply_blur:
            self.blur = Blur(in_channels, kernel_size=blur_kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=1, padding=conv_padding, bias=use_bias)
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
    def __init__(self, data_dir, transform=None, rank=0):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if is_main_process(rank):
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
            placeholder_size = (256, 256)
            if self.transform:
                for t_item in self.transform.transforms:
                    if isinstance(t_item, transforms.Resize):
                        if isinstance(t_item.size, int): placeholder_size = (t_item.size, t_item.size)
                        else: placeholder_size = t_item.size
                        break
            image = Image.new('RGB', placeholder_size, color='black')
            if self.transform: return self.transform(image)
            else: return transforms.ToTensor()(image)

        if self.transform and image is not None:
            image = self.transform(image)
        elif image is None:
            placeholder_size_fallback = (256,256)
            if self.transform:
                for t_item in self.transform.transforms:
                    if isinstance(t_item, transforms.Resize):
                        if isinstance(t_item.size, int): placeholder_size_fallback = (t_item.size, t_item.size)
                        else: placeholder_size_fallback = t_item.size
                        break
            image_fallback = Image.new('RGB', placeholder_size_fallback, color='grey')
            if self.transform: return self.transform(image_fallback)
            else: return transforms.ToTensor()(image_fallback)
        return image

# --- 模型定义 ---
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, blur_kernel_size=3):
        super(Generator, self).__init__()
        self.nz = nz; self.ngf = ngf; self.nc = nc
        BN_layer = nn.SyncBatchNorm if dist.is_initialized() and dist.get_world_size() > 1 else nn.BatchNorm2d

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            BN_layer(ngf * 8), nn.ReLU(True))
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
        x = self.initial(x); x = self.up1(x); x = self.up2(x); x = self.up3(x)
        feat_64 = self.up4(x)
        if target_size == 64: return torch.tanh(self.output_64(feat_64))
        feat_128 = self.up5(feat_64)
        if target_size == 128: return torch.tanh(self.output_128(feat_128))
        feat_256 = self.up6(feat_128)
        if target_size == 256: return torch.tanh(self.output_256(feat_256))
        feat_512 = self.up7(feat_256)
        return torch.tanh(self.output_512(feat_512))

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, blur_kernel_size=3):
        super(Discriminator, self).__init__()
        self.nc = nc; self.ndf = ndf
        self.down1 = DownsampleBlock(nc, ndf, blur_kernel_size=blur_kernel_size, use_norm=False)
        self.down2 = DownsampleBlock(ndf, ndf * 2, blur_kernel_size=blur_kernel_size)
        self.down3 = DownsampleBlock(ndf * 2, ndf * 4, blur_kernel_size=blur_kernel_size)
        self.down4 = DownsampleBlock(ndf * 4, ndf * 8, blur_kernel_size=blur_kernel_size)
        self.down5 = DownsampleBlock(ndf * 8, ndf * 8, blur_kernel_size=blur_kernel_size)
        self.down6 = DownsampleBlock(ndf * 8, ndf * 8, blur_kernel_size=blur_kernel_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.final_conv = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input_image):
        x = self.down1(input_image); x = self.down2(x); x = self.down3(x); x = self.down4(x)
        if x.shape[2] > 8: x = self.down5(x)
        if x.shape[2] > 4: x = self.down6(x)
        x = self.adaptive_pool(x); x = self.final_conv(x)
        return x.view(-1)

# --- 权重初始化 ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('SyncBatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias.data, 0.0)

# --- 数据加载器创建 ---
def create_data_loaders(data_dir, batch_size=8, num_workers=0, rank=0, world_size=1, max_size=512):
    transforms_dict = {}; datasets_dict = {}; dataloaders_dict = {}
    possible_sizes = [s for s in [64, 128, 256, 512] if s <= max_size] # 只考虑小于等于max_size的尺寸

    for size_val in possible_sizes:
        current_transform = transforms.Compose([
            transforms.Resize((size_val, size_val)), transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        transforms_dict[size_val] = current_transform
        current_dataset = ProductImageDataset(data_dir, transforms_dict[size_val], rank=rank)
        datasets_dict[size_val] = current_dataset
        sampler = DistributedSampler(current_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        shuffle_dl = sampler is None

        if len(current_dataset) > 0:
            effective_len = len(current_dataset) // world_size if world_size > 0 else len(current_dataset)
            if effective_len >= batch_size:
                dataloaders_dict[size_val] = DataLoader(current_dataset, batch_size=batch_size,
                                                        shuffle=shuffle_dl, sampler=sampler,
                                                        num_workers=num_workers, drop_last=True,
                                                        pin_memory=torch.cuda.is_available())
            else:
                if is_main_process(rank): print(f"警告: {size_val}x{size_val} 数据集在rank {rank}上有效样本数 {effective_len} 小于批大小 {batch_size}，不创建DataLoader。")
                dataloaders_dict[size_val] = None
        else:
            if is_main_process(rank): print(f"警告: {size_val}x{size_val} 数据集为空，不创建DataLoader。")
            dataloaders_dict[size_val] = None
    return dataloaders_dict

# --- 保存样本图片 ---
def save_sample_images(generator_module, epoch, current_size, nz=100, num_samples=4,
                       save_dir_prefix='generated_samples', device='cpu', amp_enabled=False):
    generator_module.eval()
    with torch.no_grad():
        fixed_noise = torch.randn(num_samples, nz, device=device)
        amp_dtype = torch.float16 if device.type == 'cuda' and amp_enabled else torch.float32
        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            fake_images = generator_module(fixed_noise, target_size=current_size)
        fake_images = (fake_images + 1) / 2.0; fake_images = torch.clamp(fake_images, 0, 1)
        save_dir = f'{save_dir_prefix}_{current_size}x{current_size}'; os.makedirs(save_dir, exist_ok=True)
        fake_images_np = fake_images.cpu().float().numpy()

        grid_size = int(math.sqrt(num_samples))
        if grid_size * grid_size != num_samples: # Handle non-square number of samples
            grid_cols = grid_size + 1 if grid_size * (grid_size+1) >= num_samples else grid_size
            grid_rows = math.ceil(num_samples / grid_cols)
        else:
            grid_rows, grid_cols = grid_size, grid_size

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
        axes = axes.flatten() # Ensure axes is always a flat array
        fig.suptitle(f'Epoch {epoch} - Size {current_size}x{current_size}', fontsize=16)
        for i_plt in range(num_samples):
            img_np_permuted = np.transpose(fake_images_np[i_plt], (1, 2, 0))
            axes[i_plt].imshow(img_np_permuted)
            axes[i_plt].axis('off')
        for i_plt_extra in range(num_samples, grid_rows * grid_cols): # Turn off extra subplots
            axes[i_plt_extra].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_dir, f'epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=150); plt.close(fig)
    generator_module.train()

# --- 检查点保存与加载 ---
def save_checkpoint(netG_module, netD_module, optimizerG, optimizerD, scaler, epoch, phase_info,
                    checkpoint_dir='checkpoints', amp_enabled=False, rank=0):
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
            print(f"Rank {rank}: 警告: 读取或解析旧的 {latest_json_path} 失败。")

    latest_info_new = {'latest_checkpoint': new_checkpoint_path, 'epoch': epoch, 'phase_info': phase_info, 'timestamp': datetime.now().isoformat()}
    with open(latest_json_path, 'w', encoding='utf-8') as f: json.dump(latest_info_new, f, ensure_ascii=False, indent=2)
    print(f"Rank {rank}: 检查点已保存: {new_checkpoint_path}")

    if previous_checkpoint_file_to_delete and \
       previous_checkpoint_file_to_delete != new_checkpoint_path and \
       os.path.exists(previous_checkpoint_file_to_delete):
        try:
            os.remove(previous_checkpoint_file_to_delete)
            print(f"Rank {rank}: 已删除旧的检查点文件: {previous_checkpoint_file_to_delete}")
        except OSError as e:
            print(f"Rank {rank}: 警告: 删除旧检查点文件 {previous_checkpoint_file_to_delete} 失败: {e}")

def load_checkpoint(checkpoint_path, netG_module, netD_module, optimizerG, optimizerD, scaler, device, rank=0):
    if not os.path.exists(checkpoint_path):
        if is_main_process(rank): print(f"Rank {rank}: 检查点文件不存在: {checkpoint_path}")
        return 0, {}
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    netG_module.load_state_dict(checkpoint_data['generator_state_dict'])
    netD_module.load_state_dict(checkpoint_data['discriminator_state_dict'])
    if optimizerG and 'optimizerG_state_dict' in checkpoint_data: optimizerG.load_state_dict(checkpoint_data['optimizerG_state_dict'])
    if optimizerD and 'optimizerD_state_dict' in checkpoint_data: optimizerD.load_state_dict(checkpoint_data['optimizerD_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint_data:
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        if is_main_process(rank): print(f"Rank {rank}: GradScaler状态已加载。")
    elif scaler is not None and is_main_process(rank): print("Rank {rank}: 警告: 检查点中未找到GradScaler状态。")
    start_epoch = checkpoint_data.get('epoch', 0)
    phase_info = checkpoint_data.get('phase_info', {})
    if is_main_process(rank):
        print(f"Rank {rank}: 已从检查点加载: {checkpoint_path}")
        print(f"Rank {rank}: 恢复到全局第 {start_epoch} 个epoch之后，阶段信息: {phase_info}")
    return start_epoch, phase_info

# --- 主训练函数 ---
def train_gan(args, rank=0, world_size=1, ddp_active=False): # args is the parsed namespace
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() and ddp_active else \
             torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == 'cuda')
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    if is_main_process(rank):
        print(f"Rank {rank}: 使用设备: {device}")
        if amp_enabled: print(f"Rank {rank}: 自动混合精度 (AMP) 已启用，使用数据类型: {amp_dtype}。")
        else: print(f"Rank {rank}: 自动混合精度 (AMP) 未启用。")

    netG_base = Generator(nz=args.noise_dim, ngf=64, nc=3, blur_kernel_size=args.blur_kernel_size).to(device)
    netD_base = Discriminator(nc=3, ndf=64, blur_kernel_size=args.blur_kernel_size).to(device)
    netG_base.apply(weights_init); netD_base.apply(weights_init)

    if ddp_active:
        netG = DDP(netG_base, device_ids=[rank] if device.type == 'cuda' else None, find_unused_parameters=True)
        netD = DDP(netD_base, device_ids=[rank] if device.type == 'cuda' else None, find_unused_parameters=True)
    else:
        netG = netG_base; netD = netD_base
    netG_module = netG.module if ddp_active else netG
    netD_module = netD.module if ddp_active else netD

    criterion = nn.BCEWithLogitsLoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), eps=1e-8)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), eps=1e-8)
    scaler = GradScaler(enabled=amp_enabled)

    dataloaders = create_data_loaders(args.data_dir, args.batch_size, args.num_workers, rank, world_size, args.max_train_size)
    if not any(dataloaders.values()):
        if is_main_process(rank): print(f"Rank {rank}: 错误: 所有有效尺寸的数据加载器均为空。程序将退出。")
        return None, None

    all_possible_phases = [
        {'size': 64, 'epochs': args.num_epochs_per_phase},
        {'size': 128, 'epochs': args.num_epochs_per_phase},
        {'size': 256, 'epochs': args.num_epochs_per_phase},
        {'size': 512, 'epochs': args.num_epochs_per_phase}
    ]
    # Filter phases based on max_train_size and available dataloaders
    training_phases = [p for p in all_possible_phases if p['size'] <= args.max_train_size and dataloaders.get(p['size']) is not None]

    if not training_phases:
        if is_main_process(rank): print(f"Rank {rank}: 错误: 没有可训练的阶段 (检查max_train_size和数据集)。程序将退出。")
        return None, None

    total_epochs_across_phases = sum(p['epochs'] for p in training_phases)
    if total_epochs_across_phases == 0:
        if is_main_process(rank): print(f"Rank {rank}: 错误: 总训练epoch数为0。程序将退出。")
        return None, None

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=total_epochs_across_phases, eta_min=args.learning_rate * 0.01)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=total_epochs_across_phases, eta_min=args.learning_rate * 0.01)

    global_start_epoch = 0; current_phase_idx_resume = 0; epoch_in_phase_start_resume = 0
    if args.resume_from_checkpoint:
        latest_json_path = os.path.join(args.checkpoint_dir, 'latest.json')
        if os.path.exists(latest_json_path):
            with open(latest_json_path, 'r', encoding='utf-8') as f_json: latest_info = json.load(f_json)
            checkpoint_path_to_load = latest_info['latest_checkpoint']
            global_start_epoch, phase_info_loaded = load_checkpoint(
                checkpoint_path_to_load, netG_module, netD_module, optimizerG, optimizerD, scaler, device, rank)
            if phase_info_loaded:
                current_phase_idx_resume = phase_info_loaded.get('phase_idx', 0)
                epoch_in_phase_start_resume = phase_info_loaded.get('epoch_in_phase', 0)
            for _ in range(global_start_epoch): schedulerG.step(); schedulerD.step()
        elif is_main_process(rank): print(f"Rank {rank}: 未找到 latest.json，从头开始训练。")

    if ddp_active: dist.barrier()
    if is_main_process(rank): print(f"Rank {rank}: 开始训练...")
    current_global_epoch = global_start_epoch

    for phase_idx in range(current_phase_idx_resume, len(training_phases)):
        phase = training_phases[phase_idx]
        current_target_size = phase['size']
        phase_total_epochs = phase['epochs']
        current_dataloader = dataloaders.get(current_target_size) # Already filtered, should exist

        if ddp_active and hasattr(current_dataloader.sampler, 'set_epoch'):
            current_dataloader.sampler.set_epoch(current_global_epoch)

        if is_main_process(rank):
            print(f"\nRank {rank}: === 阶段 {phase_idx + 1}/{len(training_phases)}: 训练 {current_target_size}x{current_target_size} 尺寸 ===")
        start_e_in_phase = epoch_in_phase_start_resume if phase_idx == current_phase_idx_resume else 0

        for epoch_in_phase in range(start_e_in_phase, phase_total_epochs):
            epoch_start_time = datetime.now()
            netG.train(); netD.train()
            for i_batch, real_images in enumerate(current_dataloader):
                real_images = real_images.to(device)
                batch_actual_size = real_images.size(0)
                real_label = torch.full((batch_actual_size,), 0.9, dtype=torch.float32, device=device)
                fake_label = torch.full((batch_actual_size,), 0.1, dtype=torch.float32, device=device)

                optimizerD.zero_grad(set_to_none=True)
                noise = torch.randn(batch_actual_size, args.noise_dim, device=device)
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    fake_images = netG(noise, target_size=current_target_size)
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    output_real = netD(real_images)
                    loss_D_real = criterion(output_real, real_label)
                    output_fake_detached = netD(fake_images.detach())
                    loss_D_fake = criterion(output_fake_detached, fake_label)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                if loss_D is not None:
                    scaler.scale(loss_D).backward()
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizerD)
                        torch.nn.utils.clip_grad_norm_(netD.parameters(), args.grad_clip_norm)
                    scaler.step(optimizerD)

                optimizerG.zero_grad(set_to_none=True)
                if fake_images is not None:
                    with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        output_fake_for_G = netD(fake_images)
                        loss_G = criterion(output_fake_for_G, real_label)
                    if loss_G is not None:
                        scaler.scale(loss_G).backward()
                        if args.grad_clip_norm > 0:
                            scaler.unscale_(optimizerG)
                            torch.nn.utils.clip_grad_norm_(netG.parameters(), args.grad_clip_norm)
                        scaler.step(optimizerG)
                scaler.update()

                if i_batch % 50 == 0 and is_main_process(rank):
                    loss_D_item = loss_D.item() if loss_D is not None else float('nan')
                    loss_G_item = loss_G.item() if loss_G is not None else float('nan')
                    output_real_mean = output_real.mean().item() if output_real is not None else float('nan')
                    output_fake_detached_mean = output_fake_detached.mean().item() if output_fake_detached is not None else float('nan')
                    output_fake_for_G_mean = output_fake_for_G.mean().item() if output_fake_for_G is not None else float('nan')
                    current_lr_G = optimizerG.param_groups[0]['lr']
                    current_lr_D = optimizerD.param_groups[0]['lr']
                    print(f'[R{rank}][Ph{phase_idx+1}][GEp {current_global_epoch+1}/{total_epochs_across_phases}] '
                          f'[PhEp {epoch_in_phase+1}/{phase_total_epochs}] [{i_batch}/{len(current_dataloader)}] '
                          f'L_D: {loss_D_item:.4f} L_G: {loss_G_item:.4f} '
                          f'D(x): {output_real_mean:.4f} D(G(z)): {output_fake_detached_mean:.4f}(D)/{output_fake_for_G_mean:.4f}(G) '
                          f'LR_G: {current_lr_G:.6f} LR_D: {current_lr_D:.6f} Scale: {scaler.get_scale():.1f}')

            current_global_epoch += 1
            schedulerG.step(); schedulerD.step()
            if is_main_process(rank):
                if (epoch_in_phase + 1) % args.save_every_epochs == 0 or (epoch_in_phase + 1) == phase_total_epochs:
                    save_sample_images(netG_module, current_global_epoch, current_target_size, args.noise_dim,
                                       num_samples=args.num_save_samples,
                                       save_dir_prefix=args.sample_save_dir_prefix,
                                       device=device, amp_enabled=amp_enabled)
                    current_phase_progress_info = {
                        'phase_idx': phase_idx, 'epoch_in_phase': epoch_in_phase + 1, 'size': current_target_size}
                    save_checkpoint(netG_module, netD_module, optimizerG, optimizerD, scaler,
                                    current_global_epoch, current_phase_progress_info,
                                    checkpoint_dir=args.checkpoint_dir,
                                    amp_enabled=amp_enabled, rank=rank)
                epoch_duration = datetime.now() - epoch_start_time
                print(f"Rank {rank}: 阶段Epoch {epoch_in_phase+1} 完成，耗时: {epoch_duration}")
            if ddp_active: dist.barrier()
        epoch_in_phase_start_resume = 0
        if is_main_process(rank): print(f"Rank {rank}: 阶段 {phase_idx + 1} ({current_target_size}x{current_target_size}) 训练完成。")

    if is_main_process(rank):
        print("\n=== 所有训练阶段完成！ ===")
        os.makedirs(args.final_model_dir, exist_ok=True)
        torch.save(netG_module.state_dict(), os.path.join(args.final_model_dir, 'generator_final.pth'))
        torch.save(netD_module.state_dict(), os.path.join(args.final_model_dir, 'discriminator_final.pth'))
        config_summary = vars(args) # Save all command line args
        config_summary.update({
            'total_global_epochs_trained': current_global_epoch,
            'world_size': world_size,
            'global_batch_size': args.batch_size * world_size if world_size > 0 else args.batch_size,
            'amp_enabled_during_training': amp_enabled,
            'amp_dtype_used': str(amp_dtype) if amp_enabled else 'N/A',
            'ddp_active': ddp_active,
            'final_timestamp': datetime.now().isoformat()
        })
        with open(os.path.join(args.final_model_dir, 'training_config.json'), 'w', encoding='utf-8') as f_json_out:
            json.dump(config_summary, f_json_out, ensure_ascii=False, indent=2)
        print(f"Rank {rank}: 最终模型和配置已保存到: {args.final_model_dir}")
    return netG_module, netD_module

# --- 生成图片 ---
def generate_images(args, rank=0): # args is the parsed namespace
    if not is_main_process(rank): return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled_inference = (device.type == 'cuda')
    amp_dtype_inference = torch.float16 if amp_enabled_inference else torch.float32

    print(f"Rank {rank}: 使用设备进行生成: {device}, AMP推理: {'启用' if amp_enabled_inference else '未启用'}, Dtype: {amp_dtype_inference if amp_enabled_inference else 'default'}")
    try:
        netG = Generator(nz=args.noise_dim, ngf=64, nc=3, blur_kernel_size=args.blur_kernel_size).to(device)
        if not os.path.exists(args.model_path):
            print(f"Rank {rank}: 错误：模型文件不存在 {args.model_path}"); return
        state_dict = torch.load(args.model_path, map_location=device)
        netG.load_state_dict(state_dict); netG.eval()
        print(f"Rank {rank}: 成功加载模型: {args.model_path}")

        sizes_to_generate_actual = [int(s) for s in args.sizes_to_generate.split(',')] if isinstance(args.sizes_to_generate, str) else args.sizes_to_generate

        with torch.no_grad():
            for size_val_gen in sizes_to_generate_actual:
                print(f"\nRank {rank}: 正在生成 {size_val_gen}x{size_val_gen} 尺寸的图片...")
                save_dir = f'{args.gen_output_dir_prefix}_{size_val_gen}x{size_val_gen}'; os.makedirs(save_dir, exist_ok=True)
                for i_gen in range(args.num_generate):
                    noise = torch.randn(1, args.noise_dim, device=device) # Generate one image at a time
                    with autocast(device_type=device.type, enabled=amp_enabled_inference, dtype=amp_dtype_inference):
                        fake_image = netG(noise, target_size=size_val_gen)
                    if fake_image is not None:
                        fake_image = (fake_image + 1) / 2.0; fake_image = torch.clamp(fake_image, 0, 1)
                        img_tensor_chw = fake_image[0]
                        img_np_hwc = img_tensor_chw.cpu().float().numpy().transpose(1, 2, 0)
                        img_pil = Image.fromarray((img_np_hwc * 255).astype(np.uint8))
                        img_path = os.path.join(save_dir, f'generated_{i_gen+1:03d}.png')
                        img_pil.save(img_path)
                        if (i_gen + 1) % 5 == 0 or (i_gen + 1) == args.num_generate:
                            print(f"  Rank {rank}: 已生成 {i_gen+1}/{args.num_generate} 张图片 (尺寸 {size_val_gen}x{size_val_gen})")
                print(f"Rank {rank}: ✓ {size_val_gen}x{size_val_gen} 图片生成完成，保存在 {save_dir} 目录")
        print(f"\nRank {rank}: 所有图片生成完成！")
    except Exception as e:
        print(f"Rank {rank}: 生成图片时发生错误: {str(e)}"); import traceback; traceback.print_exc()

# --- 虚拟数据创建 ---
def create_dummy_data_if_not_exists(data_dir, num_dummy_images=16, rank=0):
    if not is_main_process(rank): return
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
    except OSError as e: print(f"Rank {rank}: 检查或创建虚拟数据时发生OS错误: {e} in {data_dir}")
    except Exception as e: print(f"Rank {rank}: 创建虚拟图片失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch GAN (StyleGAN3-inspired) Training and Generation Script with DDP")
    # 通用参数
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    # 模式选择
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate', 'interactive'], help='Operating mode')

    # 训练参数
    parser.add_argument('--data_dir', type=str, default="./product_images", help='Path to training image data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for Adam optimizers')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of the input noise vector Z')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from the latest checkpoint')
    parser.add_argument('--blur_kernel_size', type=int, default=3, choices=[0, 1, 3, 5], help='Size of blur kernel for anti-aliasing (0 or 1 to disable)')
    parser.add_argument('--num_epochs_per_phase', type=int, default=10, help='Number of epochs for each resolution training phase')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Directory to save checkpoints')
    parser.add_argument('--sample_save_dir_prefix', type=str, default="generated_samples", help='Prefix for saving sample images directory')
    parser.add_argument('--final_model_dir', type=str, default="final_models_stylegan3_inspired", help='Directory to save final models and config')
    parser.add_argument('--save_every_epochs', type=int, default=5, help='Save samples and checkpoint every N epochs within a phase')
    parser.add_argument('--num_save_samples', type=int, default=4, help='Number of samples to generate and save during training (e.g., 4 for 2x2 grid)')
    parser.add_argument('--max_train_size', type=int, default=512, choices=[64, 128, 256, 512], help='Maximum image size to train up to')


    # 生成参数
    parser.add_argument('--model_path', type=str, default="final_models_stylegan3_inspired/generator_final.pth", help='Path to the trained generator model for generation')
    parser.add_argument('--num_generate', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--sizes_to_generate', type=str, default="64,128,256,512", help='Comma-separated list of sizes to generate (e.g., "64,128,256,512")')
    parser.add_argument('--gen_output_dir_prefix', type=str, default="final_generated_stylegan3_inspired", help='Prefix for image generation output directory')
    
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_active = world_size > 1

    if ddp_active:
        setup_ddp(rank, world_size)
        # Broadcast args from rank 0 to all other processes
        # argparse ensures args is picklable if defaults are standard types
        args_list = [args if rank == 0 else None]
        dist.broadcast_object_list(args_list, src=0)
        if rank != 0: args = args_list[0]

    set_seed(args.seed, rank)
    
    # 确保目录存在（主要由主进程操作）
    if is_main_process(rank):
        create_dummy_data_if_not_exists(args.data_dir, rank=rank) # 主进程创建虚拟数据
        for dir_path in [args.checkpoint_dir, args.sample_save_dir_prefix.split('_')[0], args.final_model_dir, args.gen_output_dir_prefix.split('_')[0]]:
             # Basic check for prefix part of directory name
            actual_dir_to_make = dir_path
            if args.sample_save_dir_prefix.startswith(dir_path) and dir_path != args.sample_save_dir_prefix :
                 actual_dir_to_make = os.path.dirname(args.sample_save_dir_prefix) if os.path.dirname(args.sample_save_dir_prefix) else "."
            elif args.gen_output_dir_prefix.startswith(dir_path) and dir_path != args.gen_output_dir_prefix:
                 actual_dir_to_make = os.path.dirname(args.gen_output_dir_prefix) if os.path.dirname(args.gen_output_dir_prefix) else "."

            if actual_dir_to_make and not os.path.exists(actual_dir_to_make):
                try:
                    os.makedirs(actual_dir_to_make, exist_ok=True)
                except OSError as e:
                    print(f"Rank {rank}: Warning, could not create directory {actual_dir_to_make}: {e}")


    if is_main_process(rank):
        print("=" * 60)
        print(f"🎯 PyTorch GAN - 任务执行 (Rank {rank}/{world_size})")
        print("=" * 60)
        print("\n📋 使用配置:")
        for k, v in vars(args).items(): print(f"   {k}: {v}")
        print("-" * 30)

    try:
        if args.mode == 'train':
            if ddp_active: dist.barrier() # Sync before training starts
            netG_final, netD_final = train_gan(args, rank, world_size, ddp_active)
            if is_main_process(rank): print("\n🎉 训练流程完成！")
            if is_main_process(rank) and netG_final is not None: # Check if training was successful
                if os.path.exists(os.path.join(args.final_model_dir, 'generator_final.pth')):
                    print("\n🎨 尝试使用最终训练好的模型生成一些测试图片...")
                    # Create a temporary args for generation if needed or reuse
                    gen_args = argparse.Namespace(**vars(args)) # Copy current args
                    gen_args.model_path = os.path.join(args.final_model_dir, 'generator_final.pth')
                    # gen_args.num_generate = 4 # or use args.num_save_samples
                    # gen_args.sizes_to_generate = "64,128" # Example: generate a subset
                    if args.max_train_size < 256 : # If trained only up to small sizes, generate those
                         gen_args.sizes_to_generate = ",".join(map(str, [s for s in [64,128] if s <= args.max_train_size]))
                    else:
                         gen_args.sizes_to_generate = ",".join(map(str, [s for s in [64,128,256] if s <= args.max_train_size]))


                    if gen_args.sizes_to_generate: # only generate if there are sizes to generate
                        generate_images(gen_args, rank)
                    else:
                        print("没有符合最大训练尺寸的默认生成尺寸，跳过最终样本生成。")
                else:
                    print(f"未找到最终模型，跳过最终样本生成。")

        elif args.mode == 'generate':
            generate_images(args, rank)
        
        elif args.mode == 'interactive':
            if is_main_process(rank):
                print("交互模式当前在此版本中已由argparse替代。请直接使用命令行参数运行。")
                parser.print_help()
            else:
                pass # Non-main processes do nothing in this dummy interactive mode

    except KeyboardInterrupt:
        if is_main_process(rank): print("\n\n⏸️  操作被用户中断。")
    except Exception as e:
        if is_main_process(rank): print(f"\n❌ 发生严重错误: {e}")
        import traceback; traceback.print_exc()
    finally:
        if ddp_active: cleanup_ddp()
        if is_main_process(rank): print("\n👋 程序执行完毕。")

if __name__ == "__main__":
    main()
