# -*- coding: utf-8 -*-
"""
GAN训练器 - 支持emoji符号和Unicode字符 + DDP分布式训练
使用UTF-8编码确保所有字符正确显示和保存
支持单机多卡和多机多卡的分布式训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.utils.data import random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.amp import GradScaler, autocast  # 使用新的API
from IPython.display import HTML
import os
import time
import logging
import logging.config
import argparse

def setup_utf8_environment():
    """设置UTF-8环境，确保emoji符号能正确显示"""
    import sys
    
    # 为Windows设置UTF-8环境
    if os.name == 'nt':  # Windows系统
        try:
            # 设置控制台代码页为UTF-8
            os.system('chcp 65001 > nul')
            
            # 重新配置标准输出和错误输出
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f"警告：无法设置UTF-8环境: {e}")
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def safe_emoji(emoji_text, fallback_text):
    """安全的emoji显示函数，在不支持的环境中使用替代文本"""
    try:
        # 测试是否能编码emoji
        emoji_text.encode('utf-8')
        return emoji_text
    except UnicodeEncodeError:
        return fallback_text

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 初始化进程组
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

# 在模块加载时立即设置UTF-8环境
setup_utf8_environment()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    """批量数据整理函数，确保数据在正确的设备上，支持分布式训练"""
    images, labels = zip(*batch)
    # 在分布式训练中使用当前设备
    if dist.is_initialized():
        device = torch.cuda.current_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    images = torch.stack([img.to(device) for img in images])
    # 将标签转换为tensor并移动到设备
    labels = torch.tensor(labels, device=device)
    return images, labels

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 返回tensor类型的标签以保持一致性
        return image, torch.tensor(0)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_layers, base_channels):
        super(Generator, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False))
            else:
                layers.append(nn.ConvTranspose2d(base_channels * (8 // (2 ** (i - 1))), base_channels * (8 // (2 ** i)), 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(base_channels * (8 // (2 ** i))))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)


    def forward(self, input):
        return self.main(input.to(torch.float32))# 将权重类型设置为单精度浮点数

class Discriminator(nn.Module):
    def __init__(self, num_layers, base_channels):
        super(Discriminator, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(3, base_channels, 4, 2, 1, bias=False))
            else:
                layers.append(nn.Conv2d(base_channels * (2 ** (i - 1)), base_channels * (2 ** i), 4, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(base_channels * (2 ** i)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(base_channels * (2 ** (num_layers - 1)), 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input.to(torch.float32))

class GANTrainer:
    def __init__(self, dataset_path, latent_dim=100, lr_G=0.0002, lr_D=0.0002, betas=(0.5, 0.999), batch_size=128, image_size=64,
                 epochs=50,start_epoch=0,patience=5, min_delta=0.0001, num_layers=4, base_channels=64, load_models=False,gradient_accumulation_steps=1, validation_frequency=10):
        
        # 设置分布式训练
        self.is_distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        
        self.dataset_path = dataset_path
        self.latent_dim = latent_dim
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.betas = betas
        self.batch_size = batch_size
        self.image_size = image_size
        self.epochs = epochs
        self.start_epoch=start_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_frequency = validation_frequency  # 新增验证频率参数，与patience语义分离
        self.load_models = load_models  # 保存加载模型标志
        self.best_val_loss = float('inf')  # 设置一个初始值，例如正无穷大

        # 设置设备
        if self.is_distributed:
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 统一Logger管理 - 只在主进程创建logger
        if is_main_process():
            self.logger = self._setup_logger()
            self.logger.debug(f'{safe_emoji("💻", "[DEVICE]")} 当前使用的设备是：{self.device}')
            if self.is_distributed:
                self.logger.info(f'{safe_emoji("🌐", "[DDP]")} 分布式训练模式：Rank {self.rank}/{self.world_size}, Local Rank: {self.local_rank}')
        else:
            self.logger = None

        self.generator = Generator(latent_dim, num_layers, base_channels).to(self.device)
        self.discriminator = Discriminator(num_layers, base_channels).to(self.device)

        # 在分布式训练中同步BN层
        if self.is_distributed:
            self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=self.lr_G, betas=betas)
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_D, betas=betas)

        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.epochs)
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=self.epochs)

        # 为DDP设置GradScaler
        if self.is_distributed:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = self.load_dataset()
        self.train_dataloader, self.val_dataloader = self.split_dataset()
        
        # 包装模型为DDP
        if self.is_distributed:
            self.generator = DDP(self.generator, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        self.train_epoch_dir = 'train_epoch'  # Specify the directory for saving files

        # 只在主进程创建目录
        if is_main_process() and not os.path.exists(self.train_epoch_dir):
            os.makedirs(self.train_epoch_dir)

    def _setup_logger(self):
        """统一设置Logger配置"""
        module_name = 'YPS'
        logger = logging.getLogger(f'{__name__}')
        
        # 如果logger已经有handler，先清空避免重复
        if logger.handlers:
            logger.handlers.clear()
            
        logger.setLevel(logging.DEBUG)

        # 确保目录存在
        if not os.path.exists('train_epoch'):
            os.makedirs('train_epoch')

        # 使用UTF-8编码来支持emoji符号
        fh = logging.FileHandler(
            os.path.join('train_epoch', f"{module_name}_训练log.txt"), 
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def safe_log(self, level, message):
        """安全的日志记录方法，只在主进程且logger存在时记录"""
        if is_main_process() and self.logger:
            getattr(self.logger, level)(message)

    def _load_models_unified(self):
        """统一的模型加载方法，按优先级尝试不同的加载方式，支持DDP"""
        if not self.load_models:
            if is_main_process() and self.logger:
                self.logger.debug("未启用模型加载，使用随机初始化的模型")
            return 0
            
        # 优先级1: 尝试加载完整的checkpoint
        checkpoint_path = os.path.join(self.train_epoch_dir, f'checkpoint_epoch_{self.start_epoch}to{self.epochs}.pth')
        if os.path.isfile(checkpoint_path):
            try:
                # 在GPU上加载checkpoint
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_distributed else None
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
                
                # 获取实际的模型（剥离DDP包装）
                generator_model = self.generator.module if self.is_distributed else self.generator
                discriminator_model = self.discriminator.module if self.is_distributed else self.discriminator
                
                generator_model.load_state_dict(checkpoint['generator_state_dict'])
                discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
                self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
                
                recovered_epoch = checkpoint['epoch']
                if is_main_process() and self.logger:
                    self.logger.info(f"{safe_emoji('✅', '[SUCCESS]')} 成功从checkpoint恢复训练状态，上次训练到第{recovered_epoch}个epoch")
                    self.logger.debug("生成器结构：\n%s", generator_model)
                    self.logger.debug("判别器结构：\n%s", discriminator_model)
                    self.logger.debug("生成器参数数量： %s", sum(p.numel() for p in generator_model.parameters()))
                    self.logger.debug("判别器参数数量： %s", sum(p.numel() for p in discriminator_model.parameters()))
                
                # 同步所有进程
                if self.is_distributed:
                    dist.barrier()
                
                return recovered_epoch + 1  # 返回下一个要训练的epoch
                
            except Exception as e:
                if is_main_process() and self.logger:
                    self.logger.warning(f"{safe_emoji('⚠️', '[WARNING]')} Checkpoint加载失败：{e}")
        
        # 优先级2: 尝试加载简单的模型文件
        generator_path = f'generator_simple_epoch_{self.start_epoch}to{self.epochs}.pth'
        discriminator_path = f'discriminator_simple_epoch_{self.start_epoch}to{self.epochs}.pth'
        
        if os.path.isfile(generator_path) and os.path.isfile(discriminator_path):
            try:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_distributed else None
                
                # 获取实际的模型（剥离DDP包装）
                generator_model = self.generator.module if self.is_distributed else self.generator
                discriminator_model = self.discriminator.module if self.is_distributed else self.discriminator
                
                generator_model.load_state_dict(torch.load(generator_path, map_location=map_location))
                discriminator_model.load_state_dict(torch.load(discriminator_path, map_location=map_location))
                
                if is_main_process() and self.logger:
                    self.logger.info(f"{safe_emoji('✅', '[SUCCESS]')} 成功加载简单模型文件，从第{self.start_epoch}个epoch开始训练")
                    self.logger.debug("生成器结构：\n%s", generator_model)
                    self.logger.debug("判别器结构：\n%s", discriminator_model)
                
                # 同步所有进程
                if self.is_distributed:
                    dist.barrier()
                
                return self.start_epoch
                
            except Exception as e:
                if is_main_process() and self.logger:
                    self.logger.warning(f"{safe_emoji('⚠️', '[WARNING]')} 简单模型加载失败：{e}")
        
        # 优先级3: 寻找其他可用的checkpoint文件
        checkpoint_pattern = os.path.join(self.train_epoch_dir, 'checkpoint_epoch_*.pth')
        import glob
        checkpoint_files = glob.glob(checkpoint_pattern)
        if checkpoint_files:
            # 选择最新的checkpoint文件
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            try:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_distributed else None
                checkpoint = torch.load(latest_checkpoint, map_location=map_location)
                
                # 获取实际的模型（剥离DDP包装）
                generator_model = self.generator.module if self.is_distributed else self.generator
                discriminator_model = self.discriminator.module if self.is_distributed else self.discriminator
                
                generator_model.load_state_dict(checkpoint['generator_state_dict'])
                discriminator_model.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
                self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
                
                recovered_epoch = checkpoint['epoch']
                if is_main_process() and self.logger:
                    self.logger.info(f"{safe_emoji('✅', '[SUCCESS]')} 找到并加载了其他checkpoint文件：{os.path.basename(latest_checkpoint)}")
                    self.logger.info(f"恢复到第{recovered_epoch}个epoch")
                
                # 同步所有进程
                if self.is_distributed:
                    dist.barrier()
                
                return recovered_epoch + 1
                
            except Exception as e:
                if is_main_process() and self.logger:
                    self.logger.warning(f"{safe_emoji('⚠️', '[WARNING]')} 其他checkpoint加载失败：{e}")
        
        # 所有加载方式都失败
        if is_main_process() and self.logger:
            self.logger.info(f"{safe_emoji('ℹ️', '[INFO]')} 未找到可用的预训练模型，将从头开始训练")
        
        # 同步所有进程
        if self.is_distributed:
            dist.barrier()
        
        return 0

    def save_state(self, epoch):
        """保存训练状态，支持DDP模型"""
        # 在分布式训练中，先同步所有进程
        if self.is_distributed:
            dist.barrier()
        
        if not is_main_process():
            return
            
        checkpoint_path = os.path.join(self.train_epoch_dir, f'checkpoint_epoch_{self.start_epoch}to{self.epochs}.pth')
        
        # 获取实际的模型状态（剥离DDP包装）
        generator_state = self.generator.module.state_dict() if self.is_distributed else self.generator.state_dict()
        discriminator_state = self.discriminator.module.state_dict() if self.is_distributed else self.discriminator.state_dict()
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator_state,
            'discriminator_state_dict': discriminator_state,
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
        }, checkpoint_path)

    def load_state(self):
        # 这个方法现在已经被_load_models_unified替代，保留是为了兼容性
        checkpoint_path = os.path.join(self.train_epoch_dir, f'checkpoint_epoch_{self.start_epoch}to{self.epochs}.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            return checkpoint['epoch']
        else:
            return None

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = MyDataset(root_dir=self.dataset_path, transform=transform)
        return dataset

    def split_dataset(self):
        """数据集分割，支持分布式采样"""
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        
        if self.is_distributed:
            # 分布式采样器
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                sampler=train_sampler,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=True
            )
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                sampler=val_sampler,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=True
            )
        else:
            # 非分布式训练
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=0,
                collate_fn=collate_fn
            )
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=0,
                collate_fn=collate_fn
            )
        
        return train_dataloader, val_dataloader

    def visualize_model_output(self, num_images=5):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim, 1, 1).to(self.device)
            fake_images = self.generator(z)
            fake_images = fake_images.cpu().detach()
            fig, axs = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
            for i, img in enumerate(fake_images):
                axs[i].imshow(img.permute(1, 2, 0))
                axs[i].axis('off')
            plt.show()
            plt.close()

    def visualize_weights(self):
        for name, param in self.generator.named_parameters():
            if 'weight' in name:
                plt.figure(figsize=(15, 5))
                plt.title(f'{name} weights')
                plt.hist(param.data.cpu().numpy().flatten(), bins=100)
                plt.show()
                plt.close()

        for name, param in self.discriminator.named_parameters():
            if 'weight' in name:
                plt.figure(figsize=(15, 5))
                plt.title(f'{name} weights')
                plt.hist(param.data.cpu().numpy().flatten(), bins=100)
                plt.show()
                plt.close()

    def train(self):

        start_time = time.time()
        if is_main_process() and self.logger:
            self.logger.debug(f'{safe_emoji("🚀", "[START]")} 开始训练的时间为：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

        # 创建存储生成图片的目录（只在主进程）
        if is_main_process() and not os.path.exists(os.path.join(self.train_epoch_dir, 'fake_images_new')):
            os.makedirs(os.path.join(self.train_epoch_dir, 'fake_images_new'))

        real_label = 1.0
        fake_label = 0.0
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        early_stopping_counter = 0

        if is_main_process() and self.logger:
            self.logger.debug(f'{safe_emoji("🔥", "")}{"<" * 30}开始训练{">" * 30}{safe_emoji("🔥", "")}')
        
        self.generator.train()
        self.discriminator.train()
        
        # 使用统一的模型加载方法
        start_epoch = self._load_models_unified()
        self.start_epoch = start_epoch
        
        # 在分布式训练中设置epoch（用于DistributedSampler）
        if self.is_distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(start_epoch)
        
        try:
            for epoch in range(start_epoch, self.epochs):
                # 为分布式采样器设置epoch
                if self.is_distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                
                for i, data in enumerate(self.train_dataloader, 0):
                    real_cpu = data[0].to(self.device)
                    b_size = real_cpu.size(0)
                    
                    ############################
                    # (1) 训练判别器：最大化 log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # 在梯度累积开始时清零梯度
                    if i % self.gradient_accumulation_steps == 0:
                        self.discriminator.zero_grad()
                    
                    # 训练真实数据
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        output = self.discriminator(real_cpu).view(-1)
                        errD_real = self.criterion(output, label)
                    
                    # 混合精度训练的正确流程
                    self.scaler.scale(errD_real).backward()
                    D_x = output.mean().item()

                    # 训练生成的假数据
                    noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        fake = self.generator(noise)
                        label.fill_(fake_label)
                        output = self.discriminator(fake.detach()).view(-1)
                        errD_fake = self.criterion(output, label)
                    
                    self.scaler.scale(errD_fake).backward()
                    D_G_z1 = output.mean().item()
                    errD = errD_real + errD_fake
                    
                    # 梯度累积处理 - 在累积步数完成时更新参数
                    if (i + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer_D)
                        self.scaler.update()

                    ############################
                    # (2) 训练生成器：最大化 log(D(G(z)))
                    ###########################
                    # 在梯度累积开始时清零梯度
                    if i % self.gradient_accumulation_steps == 0:
                        self.generator.zero_grad()
                    
                    label.fill_(real_label)  # 生成器希望判别器认为假数据是真的
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        output = self.discriminator(fake).view(-1)
                        errG = self.criterion(output, label)
                    
                    self.scaler.scale(errG).backward()
                    D_G_z2 = output.mean().item()

                    # 梯度累积处理 - 在累积步数完成时更新参数
                    if (i + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer_G)
                        self.scaler.update()

                    # 只在主进程记录日志
                    if i % 50 == 0 and is_main_process() and self.logger:
                        self.logger.debug('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                                     % (epoch, self.epochs, i, len(self.train_dataloader),
                                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                    # 在主进程中记录损失
                    if is_main_process():
                        G_losses.append(errG.item())
                        D_losses.append(errD.item())

                    if ((iters % 500 == 0) or ((epoch == self.epochs - 1) and (i == len(self.train_dataloader) - 1))) and is_main_process():
                        with torch.no_grad():
                            fake = self.generator(fixed_noise.to(torch.float32)).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    iters += 1

                # 同步所有进程
                if self.is_distributed:
                    dist.barrier()

                #在训练循环中，每个epoch结束时，调用save_state方法来保存状态：
                self.save_state(epoch)

                # 在每个epoch结束时保存生成的图像（只在主进程）
                if is_main_process():
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    save_image(fake.data,os.path.join(self.train_epoch_dir, f'fake_images_new/fake_images_epoch_{epoch}_Loss_D_{errD.item():.4f}_Loss_G_{errG.item():.4f}_D_x_{D_x}_D_x_{(D_G_z1/D_G_z2):.4f}.png'), normalize=True)

                # 在训练循环结束后，添加以下代码以打印生成器和判别器的损失并可视化生成的图像
                self.generator.eval()
                self.discriminator.eval()

                if epoch % 50==0 and is_main_process():
                    # 将生成的图像显示在控制台
                    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
                    plt.axis('off')
                    #plt.show()

                    # 绘制生成器和判别器的损失曲线
                    plt.figure(figsize=(10, 5))
                    plt.title(f"Generator and Discriminator Loss After Epoch {epoch}")
                    plt.plot(G_losses, label="Generator Loss")
                    plt.plot(D_losses, label="Discriminator Loss")
                    plt.xlabel("Iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    #plt.show()

                # 修正验证逻辑：使用validation_frequency而不是patience作为验证频率
                if epoch % self.validation_frequency == 0:
                    self.generator.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_i, val_data in enumerate(self.val_dataloader, 0):
                            real_cpu = val_data[0].to(self.device)
                            b_size = real_cpu.size(0)
                            label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                            output = self.discriminator(real_cpu).view(-1)
                            errD_real = self.criterion(output, label)

                            noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
                            fake = self.generator(noise)
                            label.fill_(fake_label)
                            output = self.discriminator(fake.detach()).view(-1)
                            errD_fake = self.criterion(output, label)
                            errD = errD_real + errD_fake

                            label.fill_(real_label)
                            output = self.discriminator(fake).view(-1)
                            errG = self.criterion(output, label)

                            val_loss += errG.item() + errD.item()

                    val_loss /= len(self.val_dataloader)
                    
                    # 在分布式训练中同步验证损失
                    if self.is_distributed:
                        val_loss_tensor = torch.tensor(val_loss, device=self.device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                        val_loss = val_loss_tensor.item() / self.world_size
                    
                    if is_main_process() and self.logger:
                        self.logger.debug(f'训练到第{epoch}个周期后的验证损失为: {val_loss}')

                    # 早停逻辑，确保所有进程同步
                    should_stop = False
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        early_stopping_counter = 0
                        self.generator.train()
                        self.discriminator.train()

                        # 只在主进程保存模型
                        if is_main_process():
                            # 获取实际的模型状态（剥离DDP包装）
                            generator_state = self.generator.module.state_dict() if self.is_distributed else self.generator.state_dict()
                            discriminator_state = self.discriminator.module.state_dict() if self.is_distributed else self.discriminator.state_dict()
                            
                            torch.save({
                                'epoch': epoch,
                                'generator_state_dict': generator_state,
                                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                                'scheduler_G_state_dict': self.scheduler_G.state_dict(),
                                'best_val_loss': self.best_val_loss,
                                'val_loss': val_loss,
                                'early_stopping_counter': early_stopping_counter,
                            }, os.path.join(self.train_epoch_dir, f'generator_all_epoch_{start_epoch}to{self.epochs}.pth'))

                            torch.save({
                                'epoch': epoch,
                                'discriminator_state_dict': discriminator_state,
                                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                                'scheduler_D_state_dict': self.scheduler_D.state_dict(),
                                'best_val_loss': self.best_val_loss,
                                'val_loss': val_loss,
                                'early_stopping_counter': early_stopping_counter,
                            }, os.path.join(self.train_epoch_dir, f'discriminator_all_epoch_{start_epoch}to{self.epochs}.pth'))
                    else:
                        early_stopping_counter += 1
                        if is_main_process() and self.logger:
                            self.logger.debug(f'训练到第{epoch}个周期后早停次数+1，当前早停次数为：{early_stopping_counter}.因为验证损失没有改善，或者改善的幅度小于min_delta')
                        if early_stopping_counter >= self.patience:
                            should_stop = True
                    
                    # 在分布式训练中同步早停决定和计数器
                    if self.is_distributed:
                        # 同步should_stop
                        should_stop_tensor = torch.tensor(int(should_stop), device=self.device)
                        dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
                        should_stop = bool(should_stop_tensor.item())
                        
                        # 同步early_stopping_counter
                        counter_tensor = torch.tensor(early_stopping_counter, device=self.device)
                        dist.all_reduce(counter_tensor, op=dist.ReduceOp.MAX)
                        early_stopping_counter = counter_tensor.item()
                    
                    if should_stop:
                        if is_main_process() and self.logger:
                            self.logger.debug(f'训练到第{epoch}个周期后停止训练，因为早停次数大于等于耐心值：{self.patience}且验证损失没有改善，或者改善的幅度小于min_delta')
                        break

        except KeyboardInterrupt:
            if is_main_process() and self.logger:
                self.logger.info(f'\n{safe_emoji("🛑", "[STOP]")} 用户手动停止训练 (Epoch {epoch}) {safe_emoji("🛑", "[STOP]")}')
                self.logger.info(f'{safe_emoji("💾", "[SAVE]")} 正在保存当前训练状态...')
            # 保存当前状态
            self.save_state(epoch)
            # 保存简单模型（只在主进程）
            if is_main_process():
                generator_state = self.generator.module.state_dict() if self.is_distributed else self.generator.state_dict()
                discriminator_state = self.discriminator.module.state_dict() if self.is_distributed else self.discriminator.state_dict()
                torch.save(generator_state, os.path.join(self.train_epoch_dir,f'generator_simple_epoch_{start_epoch}to{epoch}_interrupted.pth'))
                torch.save(discriminator_state, os.path.join(self.train_epoch_dir,f'discriminator_simple_epoch_{start_epoch}to{epoch}_interrupted.pth'))
                self.logger.info(f'{safe_emoji("✅", "[SUCCESS]")} 训练状态已保存！可以通过设置start_epoch={epoch+1}来恢复训练')
            return G_losses, D_losses, img_list

        if is_main_process() and self.logger:
            self.logger.debug(f'{safe_emoji("🎉", "[COMPLETE]")} 训练结束！')

        # 保存模型（只在主进程）
        if is_main_process():
            generator_state = self.generator.module.state_dict() if self.is_distributed else self.generator.state_dict()
            discriminator_state = self.discriminator.module.state_dict() if self.is_distributed else self.discriminator.state_dict()
            torch.save(generator_state, os.path.join(self.train_epoch_dir,f'generator_simple_epoch_{start_epoch}to{self.epochs}.pth'))
            torch.save(discriminator_state, os.path.join(self.train_epoch_dir,f'discriminator_simple_epoch_{start_epoch}to{self.epochs}.pth'))

        # 绘制损失曲线（只在主进程）
        if is_main_process():
            plt.figure(figsize=(10, 5))
            plt.title(f"Generator and Discriminator Loss During Training Between {start_epoch}To{self.epochs} Epoch \n训练期间生成器和判别器损耗")
            plt.plot(G_losses, label="G生成器")
            plt.plot(D_losses, label="D判别器")
            plt.xlabel("iterations迭代")
            plt.ylabel("Loss损耗")
            plt.legend()
            plt.show()

            # 动态展示生成图片的过程
            # 在创建动画之前设置embed_limit
            plt.rcParams['animation.embed_limit'] = 30.0  # 或者设置适合你需求的值

            fig = plt.figure(figsize=(8, 8))
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
            ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
            HTML(ani.to_jshtml())

            #另外保存生成图片的过程动画为html文件

            '''
            它首先使用to_jshtml方法将动画转换为一个HTML字符串，然后将这个字符串写入到一个名为animation.html的文件中。
            你可以在任何浏览器中打开这个文件来查看动画。
            '''

            html = ani.to_jshtml()
            with open(os.path.join(self.train_epoch_dir,f'animation_epoch_{self.start_epoch}to{self.epochs}.html'), 'w') as f:
                f.write(html)

            # 可视化模型输出
            self.visualize_model_output()
            # 可视化权重
            self.visualize_weights()

            self.logger.debug(f'{safe_emoji("🔥", "")}{"<" * 30}训练结束{">" * 30}{safe_emoji("🔥", "")}')

        # 计算耗时（只在主进程）
        if is_main_process() and self.logger:
            end_time = time.time()
            self.logger.debug(f'{safe_emoji("🏁", "[END]")} 结束训练的时间为：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            diff_time = end_time - start_time
            hours = int(diff_time // 3600)
            minutes = int((diff_time % 3600) // 60)
            seconds = int(diff_time % 60)
            self.logger.debug(f'{safe_emoji("⏱️", "[TIME]")} 训练总耗时：{hours}小时{minutes}分钟{seconds}秒')
        return G_losses, D_losses, img_list


if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='GAN训练器 - 支持分布式训练')
    parser.add_argument('--train_dir', type=str, default='./dataset', 
                       help='训练数据集路径')
    parser.add_argument('--epochs', type=int, default=2000, 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, 
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0002, 
                       help='学习率')
    parser.add_argument('--lr_g', type=float, default=None, 
                       help='生成器学习率（如果不指定，使用learning_rate）')
    parser.add_argument('--lr_d', type=float, default=None, 
                       help='判别器学习率（如果不指定，使用learning_rate）')
    parser.add_argument('--latent_dim', type=int, default=100, 
                       help='潜在空间维度')
    parser.add_argument('--image_size', type=int, default=64, 
                       help='图像大小')
    parser.add_argument('--num_layers', type=int, default=4, 
                       help='网络层数')
    parser.add_argument('--base_channels', type=int, default=64, 
                       help='基础通道数')
    parser.add_argument('--patience', type=int, default=500, 
                       help='早停耐心值')
    parser.add_argument('--validation_frequency', type=int, default=10, 
                       help='验证频率（每N个epoch验证一次）')
    parser.add_argument('--load_models', action='store_true', 
                       help='是否加载预训练模型')
    parser.add_argument('--start_epoch', type=int, default=0, 
                       help='起始训练轮数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='梯度累积步数')
    
    args = parser.parse_args()
    
    # 处理学习率参数
    lr_g = args.lr_g if args.lr_g is not None else args.learning_rate
    lr_d = args.lr_d if args.lr_d is not None else args.learning_rate
    
    # 也支持从环境变量获取数据集路径（向后兼容）
    dataset_path = os.getenv('DATASET_PATH', args.train_dir)
    
    # 在主进程中打印配置信息
    if is_main_process():
        print(f"🚀 启动GAN训练器")
        print(f"📁 数据集路径: {dataset_path}")
        print(f"🔄 训练轮数: {args.epochs}")
        print(f"📦 批次大小: {args.batch_size}")
        print(f"📊 学习率 - 生成器: {lr_g}, 判别器: {lr_d}")
        print(f"🖼️  图像尺寸: {args.image_size}x{args.image_size}")
        print(f"🏗️  网络层数: {args.num_layers}, 基础通道数: {args.base_channels}")
        print(f"⏰ 早停耐心值: {args.patience}, 验证频率: {args.validation_frequency}")
        print(f"🔄 加载预训练模型: {args.load_models}")
        if args.start_epoch > 0:
            print(f"▶️  从第 {args.start_epoch} 轮开始训练")
    
    trainer = GANTrainer(
        dataset_path=dataset_path,
        latent_dim=args.latent_dim,
        lr_G=lr_g,
        lr_D=lr_d,
        batch_size=args.batch_size,
        image_size=args.image_size,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        patience=args.patience,
        num_layers=args.num_layers,
        base_channels=args.base_channels,
        load_models=args.load_models,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_frequency=args.validation_frequency
    )
    
    try:
        trainer.train()
    except Exception as e:
        if is_main_process():
            print(f"❌ 训练过程中发生错误: {e}")
        raise
    finally:
        # 清理分布式训练环境
        cleanup_distributed()
        if is_main_process():
            print("🧹 分布式训练环境已清理")
