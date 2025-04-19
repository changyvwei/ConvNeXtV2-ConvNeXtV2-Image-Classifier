import random
import numpy as np
import torch
import os
import copy
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.metrics import confusion_matrix

# 配置基础日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(log_dir, run_name):
    """设置日志记录到文件和控制台。"""
    log_filename = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)

    # 移除现有的 handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件 handler
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')  # 指定编码
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))  # 控制台使用更简洁的格式
    logger.addHandler(console_handler)

    logger.info(f"日志记录已初始化。日志文件: {log_filepath}")
    return logger


def set_seed(seed):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 以下两个设置可能会减慢训练速度，如果不需要可以注释掉
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logger.info(f"已设置随机种子为 {seed}")


class ModelEMA:
    """ 模型指数移动平均 V2
    来源: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    保持模型 state_dict 中所有内容（参数和缓冲区）的移动平均值。
    这旨在实现类似 TensorFlow 的 ExponentialMovingAverage 功能：
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    权重的平滑版本对于某些训练方案的良好性能是必需的。
    这个类对它在模型初始化、GPU 分配和分布式训练包装器序列中的初始化位置很敏感。
    """

    def __init__(self, model, decay=0.9999, device=None):
        # 创建 EMA 模型
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # 如果设置，可以在与模型不同的设备上执行 EMA
        if self.device:
            self.ema.to(self.device)
        self.ema_has_module = hasattr(self.ema, 'module')
        # 修正 EMA：如果模型以 DP/DDP 包装器开始，调整网络
        if self.ema_has_module:
            self.ema = self.ema.module
        for p in self.ema.parameters():
            p.requires_grad_(False)
        logger.info(f"已初始化 ModelEMA，衰减率为 {decay}")

    def update(self, model):
        # 正确处理用 DP/DDP 包装的模型
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k, ema_v in esd.items():
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(self.device)
                if needs_module:
                    # 如果模型被包装但 EMA 没有，则调整键名
                    k = 'module.' + k
                # 更新 EMA 参数
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

    def state_dict(self):
        # 返回 EMA 模型的 state_dict
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        # 加载 state_dict 到 EMA 模型
        self.ema.load_state_dict(state_dict)


class EarlyStopping:
    """如果在给定的耐心值后验证准确率没有提高，则提前停止训练。"""

    def __init__(self, patience=10, verbose=True, delta=0, checkpoint_dir='checkpoints', run_name='experiment'):
        """
        参数:
            patience (int): 验证准确率上次提高后等待的轮数。
                            默认值: 10
            verbose (bool): 如果为 True，则为每次验证准确率提高打印一条消息。
                            默认值: True
            delta (float): 被监控量视为改进所需的最小变化量。
                           默认值: 0
            checkpoint_dir (str): 保存最佳模型检查点的目录。
            run_name (str): 当前运行的名称，用于检查点文件名。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.best_model_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_best_model.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"已初始化 EarlyStopping，耐心值为 {patience}")

    def __call__(self, val_acc, model, optimizer, scheduler, epoch, ema=None):
        """调用早停的步骤。"""
        if val_acc > self.best_acc + self.delta:
            if self.verbose:
                logger.info(f'验证准确率提升 ({self.best_acc:.4f} --> {val_acc:.4f})。正在保存模型...')
            self.save_checkpoint(val_acc, model, optimizer, scheduler, epoch, ema)
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'EarlyStopping 计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning("触发早停。")

    def save_checkpoint(self, val_acc, model, optimizer, scheduler, epoch, ema=None):
        """保存模型检查点。"""
        state = {
            'epoch': epoch + 1,  # 保存下一个 epoch 的编号
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # 保存 scheduler 状态
            'best_acc': val_acc,
            'ema_state_dict': ema.state_dict() if ema else None  # 保存 EMA 状态
        }
        torch.save(state, self.best_model_path)
        logger.info(f"最佳模型已保存至 {self.best_model_path} (轮次 {epoch + 1})")


def load_checkpoint(model, optimizer, scheduler, ema=None, checkpoint_dir=None, resume_path=None):
    """加载最新或指定的检查点。"""
    start_epoch = 0
    best_acc = 0.0
    checkpoint_to_load = None

    if resume_path and os.path.isfile(resume_path):
        logger.info(f"尝试加载指定的检查点: {resume_path}")
        checkpoint_to_load = resume_path
    elif checkpoint_dir and os.path.isdir(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            # 按修改时间或（如果文件名中包含）epoch 编号排序
            try:
                # 尝试按文件名中的 epoch 数字排序 (例如 best_model_epoch_10.pth)
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0,
                                      reverse=True)
            except:
                # 如果无法按 epoch 排序，则按修改时间排序
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)

            latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
            logger.info(f"在 {checkpoint_dir} 中找到检查点。尝试加载最新的: {latest_checkpoint_path}")
            checkpoint_to_load = latest_checkpoint_path
        else:
            logger.warning(f"在目录中未找到检查点文件 (.pth): {checkpoint_dir}")
    else:
        logger.info("未指定或找到检查点目录，或 resume_path 无效。从头开始训练。")
        return start_epoch, best_acc  # 如果未找到/指定检查点，则返回默认值

    if checkpoint_to_load:
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location='cpu')  # 首先加载到 CPU

            # 加载模型状态 - 处理 DataParallel/DistributedDataParallel 前缀
            state_dict = checkpoint['model_state_dict']
            # 检查 state_dict 是否来自 DataParallel/DDP 包装的模型
            is_wrapped = any(key.startswith('module.') for key in state_dict.keys())

            if is_wrapped and not hasattr(model, 'module'):
                # 如果检查点已包装但当前模型未包装 -> 移除 'module.' 前缀
                logger.info("从包装的模型检查点加载到未包装的模型，移除 'module.' 前缀。")
                state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
            elif not is_wrapped and hasattr(model, 'module'):
                # 如果检查点未包装但当前模型已包装 -> 添加 'module.' 前缀
                logger.info("从未包装的模型检查点加载到包装的模型，添加 'module.' 前缀。")
                state_dict = {'module.' + k: v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)

            if 'optimizer_state_dict' in checkpoint and optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                logger.warning("检查点中未找到优化器状态或未提供优化器。")

            if 'scheduler_state_dict' in checkpoint and scheduler:
                # 在加载前检查调度器状态字典是否兼容
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"无法加载调度器状态字典，可能由于不兼容: {e}。重置调度器。")

            if ema is not None:
                if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                    try:
                        # 使用 ema 对象的 load_state_dict 方法
                        ema.load_state_dict(checkpoint['ema_state_dict'])
                        logger.info("已从检查点加载 EMA 状态。")
                    except Exception as e:
                        logger.warning(f"无法加载 EMA 状态字典: {e}。重新初始化 EMA。")
                        # 根据需要重新初始化 EMA 或进行处理
                else:
                    logger.warning("检查点中未找到 EMA 状态字典。EMA 权重将不会恢复。")

            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            logger.info(f"成功从 {checkpoint_to_load} 加载检查点。从轮次 {start_epoch} 继续，最佳准确率: {best_acc:.4f}")

        except Exception as e:
            logger.error(f"从 {checkpoint_to_load} 加载检查点失败: {e}。从头开始训练。")
            start_epoch = 0
            best_acc = 0.0
    else:
        logger.info("未加载检查点。从头开始训练。")

    return start_epoch, best_acc


def save_confusion_matrix(writer, true_labels, pred_labels, epoch, class_names=None):
    """计算混淆矩阵并将其保存到 TensorBoard。"""
    try:
        cm = confusion_matrix(true_labels, pred_labels)
        fig, ax = plt.subplots(figsize=(10, 8))  # 根据需要调整大小
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names if class_names else 'auto',
                    yticklabels=class_names if class_names else 'auto')
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title(f'混淆矩阵 - 轮次 {epoch + 1}')

        # 将绘图保存到缓冲区
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        # 从缓冲区读取图像 - 使用 plt.imread 正确处理 PNG
        image = plt.imread(buf, format='png')
        buf.close()
        plt.close(fig)  # 关闭图形以释放内存

        # 将图像添加到 TensorBoard (确保 dataformats='HWC')
        writer.add_image('验证/混淆矩阵', image, epoch, dataformats='HWC')
        logger.debug(f"已保存轮次 {epoch + 1} 的混淆矩阵")
    except Exception as e:
        logger.error(f"为轮次 {epoch + 1} 生成/保存混淆矩阵时出错: {e}")


def save_augmented_images(writer, train_loader, epoch, cfg):
    """将一批增强后的图像保存到 TensorBoard。"""
    try:
        images, _ = next(iter(train_loader))
        # 反归一化以进行可视化
        mean = torch.tensor(cfg['mean']).view(3, 1, 1)
        std = torch.tensor(cfg['std']).view(3, 1, 1)
        images_unnormalized = images * std + mean
        images_unnormalized = torch.clamp(images_unnormalized, 0, 1)  # 将值限制在 [0, 1]

        # 将图像添加到 TensorBoard (取一个子集，例如前 16 张)
        writer.add_images('训练/增强图像样本', images_unnormalized[:16], epoch)
        logger.debug(f"已保存轮次 {epoch + 1} 的增强图像样本")
    except Exception as e:
        logger.error(f"为轮次 {epoch + 1} 保存增强图像时出错: {e}")

