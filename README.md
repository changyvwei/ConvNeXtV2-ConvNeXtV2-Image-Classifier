# **🚀 图像分类训练框架**

本仓库提供了一个灵活的框架，用于使用 PyTorch 和 timm 库训练图像分类模型。它旨在简化配置、保证可复现性并易于扩展。

## **✨ 功能特性**

* **可配置:** 使用 YAML 文件 (config.yaml) 管理所有超参数和设置。  
* **模块化:** 代码被组织成逻辑模块 (src/)，分别处理数据、模型、训练引擎、工具函数等。  
* **timm** 集成: 利用 PyTorch Image Models (timm) 库，轻松访问最先进的架构和预训练权重。  
* **数据增强:** 包括标准增强、RandAugment、Mixup/CutMix (可选)。  
* **优化器与调度器:** 支持 AdamW (含分层学习率) 和带有预热 (warmup) 的余弦退火调度器 (易于扩展)。  
* **训练实用工具:**  
  * 自动混合精度 (AMP) 支持。  
  * 模型权重的指数移动平均 (EMA)。  
  * 基于验证准确率的早停 (Early Stopping)。  
  * 检查点保存 (最新和最佳)。  
  * 从检查点恢复训练。  
* **日志记录:**  
  * 控制台日志。  
  * 文件日志。  
  * TensorBoard 日志 (损失、准确率、学习率、混淆矩阵、增强图像样本)。  
  * CSV 文件记录每个轮次的指标。  
* **命令行接口:** 允许通过命令行参数覆盖关键配置。

## **📁 项目结构说明**

github-image-classifier/  
├── main.py             \# 运行训练的主脚本  
├── config.yaml         \# 配置文件 (编辑此文件)  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── data\_setup.py   \# 数据加载与变换  
│   ├── engine.py       \# 训练与验证循环  
│   ├── model\_builder.py \# 模型定义与优化器  
│   ├── utils.py        \# 工具函数 (种子, EMA, 早停, 日志, 检查点)  
│   └── scheduler.py    \# 学习率调度器函数  
├── requirements.txt    \# 项目依赖  
├── README.md           \# 本文档 
├── logs/               \# 默认日志目录 (自动创建)  
└── checkpoints/        \# 默认模型检查点目录 (自动创建)

## **🚀 快速开始**

### **1\. 安装依赖**

pip install \-r requirements.txt

*注意: 如果使用 GPU，请确保你的 PyTorch 版本与 CUDA 版本匹配。*

### **2\. 准备数据集**

* 将你的数据集组织在一个主数据文件夹中，内含 train 和 val 子目录。  
* 每个子目录 (train, val) 应包含以你的类别命名的子文件夹，其中包含相应的图像。  
  \<你的数据目录\>/  
  ├── train/  
  │   ├── 类别A/  
  │   │   ├── img1.jpg  
  │   │   └── ...  
  │   └── 类别B/  
  │       ├── img100.jpg  
  │       └── ...  
  └── val/  
      ├── 类别A/  
      │   ├── img50.jpg  
      │   └── ...  
      └── 类别B/  
          ├── img200.jpg  
          └── ...

* 更新 config.yaml 文件中的 data\_dir 路径，使其指向 \<你的数据目录\>。

### **3\. 配置训练**

编辑 config.yaml 文件，设置你想要的模型 (model\_name)、超参数 (学习率、批次大小、轮数)、类别数 (num\_classes)、路径以及其他训练选项。

## **🏃 如何使用**

**使用默认配置开始训练:**

python main.py

**使用指定的配置文件开始训练:**

python main.py \--config path/to/your/custom\_config.yaml

**通过命令行覆盖特定参数:**

\# 使用 50 个轮次和 64 的批次大小进行训练  
python main.py \--epochs 50 \--batch-size 64

\# 使用不同的模型和学习率  
python main.py \--model-name resnet50 \--lr 0.001

\# 为日志/检查点指定运行名称  
python main.py \--run-name 我的实验运行\_1

\# 从指定的检查点恢复训练  
python main.py \--resume-path checkpoints/你的检查点.pth

## **📊 监控训练**

* **控制台:** 训练进度和日志将打印到控制台。  

* **日志文件:** 详细日志保存在 logs/ 目录中。  

* **TensorBoard:** 跟踪指标、查看混淆矩阵和图像样本：  
  tensorboard \--logdir logs/

  然后在你的网页浏览器中访问 http://localhost:6006。  

* **CSV 文件:** 每个轮次的指标保存在 logs/ 目录中的 CSV 文件里。

## **🛠️ 自定义**

* **模型:** 在 config.yaml 中更改 model\_name 为 timm 中可用的任何模型。如果模型的分层学习率命名约定显著不同，你可能需要调整 src/model\_builder.py 中的 build\_optimizer 函数。  
* **数据增强:** 修改 src/data\_setup.py 中的 get\_transforms 函数或在 config.yaml 中调整增强参数。  
* **损失函数/优化器/调度器:** 在 src/engine.py, src/model\_builder.py, 和 src/scheduler.py 中添加新选项或修改现有选项。

## **🙏 贡献**

欢迎贡献！请随时提交 Pull Request 或开启 Issue。
个人邮箱：1165325394@qq.com
