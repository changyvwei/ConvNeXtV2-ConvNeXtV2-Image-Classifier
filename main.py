import torch
import os
import yaml
import argparse
import logging
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 导入项目模块
from src.data_setup import get_dataloaders
from src.model_builder import build_model, build_optimizer
from src.scheduler import build_scheduler
from src.engine import train_one_epoch, validate, setup_loss_and_mixup
from src.utils import (
    set_seed, ModelEMA, EarlyStopping, load_checkpoint, setup_logging,
    save_confusion_matrix, save_augmented_images
)


def main(cfg):
    """主训练和验证函数。"""

    # --- 设置 ---
    # 如果未在命令行或配置中指定 run_name，则自动生成一个
    run_name = cfg.get('run_name', f"{cfg['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir = cfg['log_dir']
    checkpoint_dir = cfg['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置日志记录
    logger = setup_logging(log_dir, run_name)
    logger.info("开始训练运行...")
    logger.info(f"配置:\n{yaml.dump(cfg, indent=2, allow_unicode=True)}")  # 记录配置 (允许 Unicode)

    # 设置随机种子以保证可复现性
    set_seed(cfg['seed'])

    # 设置设备 (GPU 或 CPU)
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # 为潜在的速度提升启用 cudnn benchmark
        logger.info(f"使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("使用 CPU 设备。")
        if cfg['device'] == 'cuda':
            logger.warning("指定了 CUDA 但不可用。回退到 CPU。")
    cfg['effective_device'] = device  # 存储实际使用的设备

    # --- 数据加载 ---
    try:
        train_loader, val_loader, num_classes_data, class_names = get_dataloaders(cfg)
        # 如果需要，用来自数据的实际类别数更新配置
        if cfg['num_classes'] != num_classes_data:
            logger.warning(
                f"配置中的类别数 ({cfg['num_classes']}) 与数据集 ({num_classes_data}) 不符。将覆盖为数据集中的值。")
            cfg['num_classes'] = num_classes_data
    except FileNotFoundError as e:
        logger.error(f"数据加载失败: {e}")
        return  # 如果找不到数据则退出

    # --- 模型、损失函数、优化器、调度器 ---
    model = build_model(cfg).to(device)
    train_criterion, val_criterion, mixup_fn = setup_loss_and_mixup(cfg, cfg['num_classes'])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- EMA 设置 ---
    ema = ModelEMA(model, decay=cfg['ema_decay'], device=device) if cfg.get('use_ema', False) else None
    if ema: logger.info("模型 EMA 已启用。")

    # --- 检查点加载 ---
    start_epoch, best_acc = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        checkpoint_dir=cfg['checkpoint_dir'],
        resume_path=cfg.get('resume_path')  # 使用 get 获取可选的 resume_path
    )
    cfg['start_epoch'] = start_epoch  # 用实际的开始轮次更新配置
    cfg['best_acc'] = best_acc  # 用加载的最佳准确率更新配置

    # --- 早停设置 ---
    early_stopper = None
    if cfg.get('use_early_stopping', False):
        early_stopper = EarlyStopping(
            patience=cfg['early_stopping_patience'],
            verbose=True,
            checkpoint_dir=cfg['checkpoint_dir'],
            run_name=run_name  # 传递 run_name 以保持最佳模型命名一致
        )
        logger.info("早停已启用。")

    # --- TensorBoard 和 CSV 日志 ---
    tensorboard_log_dir = os.path.join(log_dir, cfg.get('tensorboard_subdir', run_name))
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard 日志将保存到: {tensorboard_log_dir}")

    csv_path = os.path.join(log_dir, f"{run_name}_epoch_metrics.csv")
    # 使用 utf-8 编码打开 CSV 文件
    csv_file = open(csv_path, "w", newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # 写入 CSV 文件头 (中文)
    csv_writer.writerow(["轮次", "训练损失", "训练准确率", "验证损失", "验证准确率", "学习率"])
    logger.info(f"轮次指标 CSV 日志将保存到: {csv_path}")

    # --- 训练循环 ---
    logger.info(f"从轮次 {start_epoch + 1} 开始训练，共 {cfg['epochs']} 轮")
    for epoch in range(start_epoch, cfg['epochs']):
        logger.info(f"\n--- 轮次 {epoch + 1}/{cfg['epochs']} ---")

        # 训练一个轮次
        train_loss = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device, epoch, cfg, mixup_fn, ema
        )

        # 决定使用哪个模型进行验证 (EMA 或原始模型)
        val_model = ema.ema if ema else model

        # 验证
        val_loss, val_acc, preds, labels = validate(
            val_model, val_loader, val_criterion, device, return_preds=True
        )

        # 记录指标到 TensorBoard
        writer.add_scalar('损失/训练', train_loss, epoch)  # 来自 train_one_epoch 的训练损失
        writer.add_scalar('损失/验证', val_loss, epoch)
        writer.add_scalar('准确率/验证', val_acc, epoch)

        # 记录学习率
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        for i, lr in enumerate(current_lrs):
            writer.add_scalar(f'学习率/组_{i}', lr, epoch)
        lr_log_str = ", ".join([f"组 {i}: {lr:.2e}" for i, lr in enumerate(current_lrs)])
        logger.info(f"当前学习率: {lr_log_str}")

        # 记录指标到 CSV (使用验证准确率，训练损失)
        # 注意：由于 mixup，训练准确率不直接从 train_one_epoch 记录
        # 如果需要，可以在训练集上运行 validate()，但这会增加开销。
        # 记录第一个学习率组的学习率
        csv_writer.writerow(
            [epoch + 1, f"{train_loss:.4f}", "N/A", f"{val_loss:.4f}", f"{val_acc:.2f}", f"{current_lrs[0]:.2e}"])
        csv_file.flush()  # 确保数据写入磁盘

        # 可选：定期保存增强图像样本
        if (epoch + 1) % cfg.get('save_augmented_images_interval', 10) == 0:
            save_augmented_images(writer, train_loader, epoch, cfg)

        # 可选：定期保存混淆矩阵
        if (epoch + 1) % cfg.get('save_confusion_matrix_interval', 5) == 0:
            save_confusion_matrix(writer, labels, preds, epoch, class_names)

        # 调度器步骤 (如果是 ReduceLROnPlateau，则基于验证指标)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # 根据验证损失调整
        elif scheduler is not None:
            scheduler.step()  # 对于其他调度器（如 CosineAnnealingLR），每轮调整一次

        # 早停检查并保存最佳模型
        if early_stopper:
            # 使用早停机制来保存最佳模型
            early_stopper(val_acc, model, optimizer, scheduler, epoch, ema)
            if early_stopper.early_stop:
                break  # 如果触发早停，则退出训练循环
        elif val_acc > best_acc:  # 如果没有早停，则手动保存最佳模型
            best_acc = val_acc
            logger.info(f"新的最佳准确率: {best_acc:.2f}%。正在保存模型...")
            save_path = os.path.join(checkpoint_dir, f"{run_name}_best_manual.pth")
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'ema_state_dict': ema.state_dict() if ema else None,
                'config': cfg  # 将配置与检查点一起保存
            }
            torch.save(state, save_path)
            logger.info(f"最佳模型已手动保存至 {save_path}")

    # --- 训练结束 ---
    logger.info("训练完成。")
    if early_stopper and early_stopper.early_stop:
        logger.info(f"在轮次 {epoch + 1} 提前停止。最佳验证准确率: {early_stopper.best_acc:.2f}%")
    else:
        # 如果训练完成（未早停），记录最终的最佳准确率
        final_best_acc = early_stopper.best_acc if early_stopper else best_acc
        logger.info(f"已完成 {cfg['epochs']} 轮训练。达到的最佳验证准确率: {final_best_acc:.2f}%")

    # 关闭资源
    writer.close()
    csv_file.close()
    logger.info("TensorBoard writer 和 CSV logger 已关闭。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像分类训练脚本")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置 YAML 文件的路径 (默认: config.yaml)')
    # 允许通过命令行覆盖特定的配置值
    parser.add_argument('--data-dir', type=str, help='覆盖数据集目录路径')
    parser.add_argument('--model-name', type=str, help='覆盖模型名称')
    parser.add_argument('--epochs', type=int, help='覆盖训练轮数')
    parser.add_argument('--batch-size', type=int, help='覆盖批次大小')
    parser.add_argument('--lr', type=float, help='覆盖基础学习率')
    parser.add_argument('--device', type=str, help='覆盖设备 (例如 "cpu", "cuda")')
    parser.add_argument('--resume-path', type=str, help='覆盖用于恢复训练的检查点路径')
    parser.add_argument('--run-name', type=str, help='为本次运行指定一个名称')

    args = parser.parse_args()

    # 从 YAML 文件加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:  # 指定编码
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析配置文件 '{args.config}' 时出错: {e}")
        exit(1)

    # 如果提供了命令行参数，则覆盖配置
    if args.data_dir: config['data_dir'] = args.data_dir
    if args.model_name: config['model_name'] = args.model_name
    if args.epochs: config['epochs'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.lr: config['lr'] = args.lr
    if args.device: config['device'] = args.device
    if args.resume_path: config['resume_path'] = args.resume_path
    if args.run_name: config['run_name'] = args.run_name

    # 启动主训练过程
    main(config)

