import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """
    创建一个带有线性预热 (warmup) 的余弦学习率调度器。
    """

    def lr_lambda(current_epoch):
        # 线性预热阶段
        if current_epoch < warmup_epochs:
            # 确保 warmup_epochs > 0 以避免除以零
            return float(current_epoch + 1) / float(max(1, warmup_epochs))

        # 余弦衰减阶段
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        # 限制 progress 在 [0, 1] 之间，以防 current_epoch 超过 total_epochs
        progress = max(0.0, min(1.0, progress))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        # 最终学习率在初始 LR 和最小 LR 之间插值
        final_lr_multiplier = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
        return final_lr_multiplier

    logger.info(f"使用 Cosine 调度器，预热轮数 {warmup_epochs}，总轮数 {total_epochs}，最小学习率比例 {min_lr_ratio}。")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_reduce_on_plateau_schedule(optimizer, cfg):
    """
    创建一个 ReduceLROnPlateau 调度器。
    """
    patience = cfg.get('scheduler_patience', 5)
    factor = cfg.get('scheduler_factor', 0.5)
    min_lr = cfg.get('scheduler_min_lr', 1e-6)
    mode = cfg.get('scheduler_mode', 'min')  # 监控验证损失 ('min') 或准确率 ('max')

    logger.info(
        f"使用 ReduceLROnPlateau 调度器: 模式='{mode}', 因子={factor}, 耐心={patience}, 最小学习率={min_lr:.2e}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=True  # 记录学习率变化
    )


def build_scheduler(optimizer, cfg):
    """根据配置构建学习率调度器。"""
    scheduler_name = cfg.get('scheduler_name', 'cosine_warmup').lower()
    total_epochs = cfg['epochs']

    if scheduler_name == 'cosine_warmup':
        warmup_epochs = cfg.get('warmup_epochs', 5)
        min_lr_ratio = cfg.get('min_lr_ratio', 0.01)
        return get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_ratio)
    elif scheduler_name == 'reduce_on_plateau':
        return get_reduce_on_plateau_schedule(optimizer, cfg)
    elif scheduler_name == 'none' or scheduler_name is None:
        logger.info("不使用学习率调度器。")
        return None  # 如果不需要调度器，则返回 None
    else:
        logger.warning(f"调度器 '{scheduler_name}' 未被识别。默认使用 CosineWarmup。")
        warmup_epochs = cfg.get('warmup_epochs', 5)
        min_lr_ratio = cfg.get('min_lr_ratio', 0.01)
        return get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_ratio)

