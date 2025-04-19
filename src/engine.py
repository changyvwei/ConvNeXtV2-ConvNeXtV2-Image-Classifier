import torch
import torch.nn as nn
from tqdm import tqdm
import time
import logging
from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup

logger = logging.getLogger(__name__)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg,
                    mixup_fn=None, ema=None):
    """执行一个训练轮次 (epoch)。"""
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    # correct = 0 # 使用 Mixup/SoftTargetCE 时，准确率计算可能具有误导性
    total = 0
    start_time = time.time()

    # AMP Scaler (自动混合精度梯度缩放器)
    use_amp = cfg.get('use_amp', False)
    # 确保 scaler 在正确的设备上创建 ('cuda' 如果 use_amp 为 True)
    scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device=scaler_device, enabled=use_amp)

    num_batches = len(train_loader)
    progress_bar = tqdm(train_loader, desc=f"轮次 {epoch + 1} 训练", ncols=100, leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # 将数据移动到指定设备 (使用 non_blocking 可能加速)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # 如果启用，应用 Mixup/CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
            # 注意：标准的准确率计算在这里没有意义。
            # 损失使用 SoftTargetCrossEntropy 计算。

        optimizer.zero_grad()  # 清除之前的梯度

        # 自动混合精度上下文
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失

        # 缩放损失并执行反向传播
        scaler.scale(loss).backward()

        # 可选：梯度裁剪 (有助于防止梯度爆炸)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 执行优化器步骤
        scaler.step(optimizer)
        # 更新 scaler 以供下一次迭代使用
        scaler.update()

        # 累加损失 (乘以批次大小以获得总损失)
        running_loss += loss.item() * inputs.shape[0]
        # 累加样本总数 (即使使用 Mixup，targets.size(0) 也是正确的)
        total += targets.size(0)

        # 更新进度条描述 (可选)
        if batch_idx % 20 == 0:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 如果启用，更新 EMA 模型
        if ema:
            ema.update(model)

    epoch_time = time.time() - start_time
    # 计算整个轮次的平均损失
    avg_loss = running_loss / total

    # 准确率计算 (仅在未使用 Mixup 或 targets 是硬标签时可靠)
    # 对于 SoftTargetCrossEntropy，准确率信息量较少，应关注损失。
    # 如果需要在 Mixup 下计算准确率，可以比较 argmax(outputs) vs argmax(targets)，但需要谨慎解释。
    # acc = 100. * correct / total if total > 0 and mixup_fn is None else 0.0

    logger.info(f"轮次 {epoch + 1} 训练总结: 平均损失={avg_loss:.4f}, 时间={epoch_time:.2f}s")

    # 返回平均损失。由于 Mixup/SoftTargetCE，省略了准确率。
    return avg_loss


def validate(model, val_loader, criterion, device, return_preds=False):
    """在验证集上执行验证。"""
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    all_preds = []  # 存储所有预测结果
    all_labels = []  # 存储所有真实标签

    progress_bar = tqdm(val_loader, desc="验证中", ncols=100, leave=False)

    with torch.no_grad():  # 验证时禁用梯度计算
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 验证时通常不需要 AMP autocast，除非特定层需要
            outputs = model(inputs)

            # 即使训练使用 SoftTarget，验证也使用标准的 CrossEntropyLoss
            loss = criterion(outputs, targets)

            running_loss += loss.item()  # 累加批次损失
            _, predicted = outputs.max(1)  # 获取最大对数概率的索引 (预测类别)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # 累加正确预测的数量

            # 如果需要，存储预测和标签以计算混淆矩阵等指标
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(val_loader)  # 每个批次的平均损失
    accuracy = 100. * correct / total  # 计算准确率

    logger.info(f"验证总结: 平均损失={avg_loss:.4f}, 准确率={accuracy:.2f}%, 时间={epoch_time:.2f}s")

    if return_preds:
        return avg_loss, accuracy, all_preds, all_labels
    else:
        return avg_loss, accuracy


def setup_loss_and_mixup(cfg, num_classes):
    """根据配置设置损失函数和 Mixup。"""
    use_mixup = cfg.get('use_mixup', False)

    if use_mixup:
        # Mixup/Cutmix 参数
        mixup_args = {
            'mixup_alpha': cfg.get('mixup_alpha', 0.8),
            'cutmix_alpha': cfg.get('cutmix_alpha', 1.0),
            'cutmix_minmax': None,
            'prob': cfg.get('mixup_prob', 1.0),  # 通常设为 1.0 以总是应用
            'switch_prob': cfg.get('mixup_switch_prob', 0.5),
            'mode': 'batch',
            'label_smoothing': cfg.get('label_smoothing', 0.1),
            'num_classes': num_classes
        }
        mixup_fn = Mixup(**mixup_args)
        # 启用 Mixup 时使用 SoftTargetCrossEntropy
        train_criterion = SoftTargetCrossEntropy()
        # 验证时仍使用标准的 CrossEntropyLoss
        val_criterion = nn.CrossEntropyLoss()
        logger.info(f"使用 Mixup/Cutmix，参数: {mixup_args}")
        logger.info("训练损失使用 SoftTargetCrossEntropy，验证损失使用 CrossEntropyLoss。")
    else:
        mixup_fn = None
        # 禁用 Mixup 时，训练和验证都使用标准的 CrossEntropyLoss
        train_criterion = nn.CrossEntropyLoss()
        val_criterion = nn.CrossEntropyLoss()
        logger.info("Mixup 已禁用。训练和验证均使用 CrossEntropyLoss。")

    return train_criterion, val_criterion, mixup_fn

