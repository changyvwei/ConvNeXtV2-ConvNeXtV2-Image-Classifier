import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def get_transforms(cfg, train=True):
    """根据配置获取基础或增强的数据变换。"""
    img_size = cfg['img_size']
    mean = cfg['mean']
    std = cfg['std']

    if train:
        # 训练集数据增强
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
        ]

        # 如果指定，添加 RandAugment
        if cfg.get('randaugment_num_ops', 0) > 0 and cfg.get('randaugment_magnitude', 0) > 0:
            transform_list.append(transforms.RandAugment(num_ops=cfg['randaugment_num_ops'],
                                                         magnitude=cfg['randaugment_magnitude']))
            logger.info(f"使用 RandAugment (操作数={cfg['randaugment_num_ops']}, 强度={cfg['randaugment_magnitude']})")

        # 添加可选的随机高斯模糊
        if cfg.get('use_random_blur', False):
            blur_prob = cfg.get('random_blur_prob', 0.5)
            transform_list.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=blur_prob))
            logger.info(f"使用随机高斯模糊 (概率 p={blur_prob})")

        # 基础变换：转为 Tensor 并归一化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transforms.Compose(transform_list)
    else:
        # 验证集/测试集数据变换 (通常只有 Resize, ToTensor, Normalize)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_dataloaders(cfg):
    """创建训练和验证 DataLoader。"""
    data_dir = cfg['data_dir']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    pin_memory = cfg.get('pin_memory', True)  # 如果未指定，默认为 True

    train_transform = get_transforms(cfg, train=True)
    val_transform = get_transforms(cfg, train=False)

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    # 检查路径是否存在
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"训练目录未找到: {train_path}")
    if not os.path.isdir(val_path):
        raise FileNotFoundError(f"验证目录未找到: {val_path}")

    logger.info(f"从以下路径加载训练数据: {train_path}")
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)

    logger.info(f"从以下路径加载验证数据: {val_path}")
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    # 确保类别一致性
    if train_dataset.classes != val_dataset.classes:
        logger.warning("训练集和验证集之间的类别不匹配！")
        logger.warning(f"训练集类别: {train_dataset.classes}")
        logger.warning(f"验证集类别: {val_dataset.classes}")
        # 根据需要考虑引发错误或处理此情况

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    logger.info(f"找到 {num_classes} 个类别: {class_names}")

    # 检查配置中的 num_classes 是否与数据集匹配
    if 'num_classes' in cfg and cfg['num_classes'] != num_classes:
        logger.warning(f"配置中的类别数 ({cfg['num_classes']}) 与数据集 ({num_classes}) 不匹配。将使用数据集中的值。")
        # 可选地更新 cfg 或引发错误
        cfg['num_classes'] = num_classes  # 使用从数据集中读取到的类别数

    # 创建训练 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 训练时丢弃最后一个不完整的批次
    )
    logger.info(f"训练 DataLoader 已创建: 批次大小={batch_size}, 工作进程数={num_workers}, 锁页内存={pin_memory}")

    # 创建验证 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # 如果内存允许，验证时可以使用更大的批次大小
        shuffle=False,  # 验证时不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    logger.info(f"验证 DataLoader 已创建: 批次大小={batch_size}, 工作进程数={num_workers}, 锁页内存={pin_memory}")

    return train_loader, val_loader, num_classes, class_names

