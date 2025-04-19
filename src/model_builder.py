import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


def build_model(cfg):
    """根据配置构建图像分类模型。"""
    model_name = cfg['model_name']
    num_classes = cfg['num_classes']
    pretrained = cfg.get('pretrained', True)
    drop_rate = cfg.get('drop_rate', 0.0)
    global_pool = cfg.get('global_pool', 'avg')  # 默认为 'avg' 池化

    logger.info(f"构建模型: {model_name}，类别数: {num_classes}")
    logger.info(f"是否预训练: {pretrained}, Dropout 比率: {drop_rate}, 全局池化: {global_pool}")

    # 创建基础模型，不带最终分类器
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # 移除默认分类器
        drop_rate=drop_rate,
        global_pool=global_pool
    )

    # 从基础模型获取特征数量
    num_features = model.num_features

    # 定义自定义头部 (示例使用类似 ConvNeXt 的结构)
    # 您可能需要根据具体的模型架构调整此部分
    # 这个头部结构受原始脚本启发，但更健壮
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),    # 池化到 1x1
        nn.Flatten(),                    # 将池化后的特征展平为 [B, model.num_features]
        nn.LayerNorm(model.num_features),  # LayerNorm 增强收敛与稳定性
        nn.Linear(model.num_features, 512),  # 降维+投影
        nn.GELU(),                         # 激活函数，优于 ReLU
        nn.Dropout(0.15),                   # 防止过拟合
        nn.Linear(512, num_classes)        # 输出层
    )
    logger.info(
        f"模型头部已创建: LayerNorm -> Linear({num_features}, 512) -> GELU -> Dropout -> Linear(512, {num_classes})")

    # 计算并记录参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"模型可训练参数量: {trainable_params:,}")

    return model


def build_optimizer(model, cfg):
    """构建优化器，支持可选的分层学习率。"""
    optimizer_name = cfg.get('optimizer_name', 'AdamW').lower()
    lr = cfg['lr']  # 基础学习率
    weight_decay = cfg.get('weight_decay', 0.0)

    # --- 分层学习率设置 (针对类 ConvNeXt 模型的示例) ---
    # 这部分假设了特定的命名约定 (stem, stages.0, stages.1/2, head)
    # 如果使用不同的模型架构，您可能需要调整此部分。
    param_groups = [
        {'params': [], 'lr': lr * cfg.get('lr_mult_stem', 1.0), 'name': 'stem'},
        {'params': [], 'lr': lr * cfg.get('lr_mult_stage0', 1.0), 'name': 'stage0'},
        {'params': [], 'lr': lr * cfg.get('lr_mult_stageN', 1.0), 'name': 'stageN'},  # 中间 stage
        {'params': [], 'lr': lr * cfg.get('lr_mult_head', 1.0), 'name': 'head'}
    ]
    other_params = []  # 不属于已定义组的参数

    logger.info("为优化器分配参数组...")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # 跳过不需要梯度的参数

        assigned = False
        if 'stem.' in n:
            param_groups[0]['params'].append(p)
            assigned = True
        # 检查 'stages' 属性是否存在，以避免在没有 stages 的模型上出错
        elif hasattr(model, 'stages') and isinstance(model.stages, nn.ModuleList):
            if len(model.stages) > 0 and f'stages.0.' in n:
                param_groups[1]['params'].append(p)
                assigned = True
            elif len(model.stages) > 2 and (f'stages.1.' in n or f'stages.2.' in n):  # 假设 stageN 适用于 stage 1 和 2
                param_groups[2]['params'].append(p)
                assigned = True
        elif 'head.' in n:  # 假设分类头命名为 'head'
            param_groups[3]['params'].append(p)
            assigned = True

        if not assigned:
            # 将其他参数分配给默认组 (例如 'stageN' 或新的 'other' 组)
            # 这里使用 stageN 组作为后备
            logger.debug(f"参数 '{n}' 未匹配特定组，分配到 'stageN' 组")
            param_groups[2]['params'].append(p)
            # 或者创建一个单独的组:
            # other_params.append(p)

    # 如果 'other' 组有参数，则添加它
    # if other_params:
    #     param_groups.append({'params': other_params, 'lr': lr, 'name': 'other'})

    # 过滤掉没有参数的组
    param_groups = [pg for pg in param_groups if pg['params']]

    # 记录参数组及其学习率
    logger.info("优化器参数组:")
    for pg in param_groups:
        logger.info(f"  - 组名: {pg['name']}, 学习率: {pg['lr']:.2e}, 参数量: {sum(p.numel() for p in pg['params']):,}")

    # --- 优化器选择 ---
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        logger.info(f"使用 AdamW 优化器，基础学习率={lr:.2e}, 权重衰减={weight_decay}, 并启用分层学习率。")
    # 如果需要，在此处添加其他优化器 (例如 SGD)
    # elif optimizer_name == 'sgd':
    #     optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=cfg.get('momentum', 0.9), weight_decay=weight_decay)
    #     logger.info(f"使用 SGD 优化器，基础学习率={lr:.2e}, momentum={cfg.get('momentum', 0.9)}, 权重衰减={weight_decay}, 并启用分层学习率。")
    else:
        logger.warning(f"优化器 '{optimizer_name}' 未被识别或实现。默认使用 AdamW。")
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    return optimizer

