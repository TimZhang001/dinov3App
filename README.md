# DINOv3 Classification Framework

基于 DINOv3 的图像分类框架，支持自定义数据集的线性探测和微调训练。

## 项目概述

本项目提供了一个面向对象的分类框架，封装了 DINOv3 预训练模型，支持：

- **多种模型架构**：ViT (vits16, vits16plus, vitb16) 和 ConvNeXt (tiny, small, base, large)
- **灵活的训练策略**：冻结骨干网络的线性探测或端到端微调
- **完整的评估工具**：混淆矩阵、分类报告、可视化
- **便捷的推理接口**：单张图片或批量图片预测

## 目录结构

```
dinov3App/
├── dinov3/                      # DINOv3 官方代码库
├── dinov3_classifier/           # 分类框架
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── model.py                # 模型定义
│   ├── data.py                 # 数据集类
│   ├── trainer.py              # 训练器
│   ├── evaluator.py            # 评估器
│   └── predictor.py            # 推理器
├── pretrained/
│   ├── backbone/               # 预训练骨干网络权重
│   └── adapter/                # 预训练适配器权重
├── dataset/                    # 数据集目录（符号链接）
├── train.py                    # 训练入口
├── evaluate.py                 # 评估入口
├── inference.py                # 推理入口
├── run_train.sh               # 训练脚本示例
├── requirements.txt            # 依赖列表
└── README.md
```

## 安装

### 环境要求

- Python >= 3.11
- PyTorch >= 2.0.0
- CUDA 支持（推荐）

### 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 conda
conda env create -f dinov3/conda.yaml
conda activate dinov3
```

### 下载预训练权重

预训练权重需从 Meta AI 官方获取：[DINOv3 下载页面](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)

下载后放置于 `pretrained/backbone/` 目录。

## 支持的模型

### ViT 模型

| 模型 | 参数量 | 特征维度 |
|------|--------|----------|
| vits16 | 21M | 384 |
| vits16plus | 29M | 384 |
| vitb16 | 86M | 768 |

### ConvNeXt 模型

| 模型 | 参数量 | 特征维度 |
|------|--------|----------|
| convnext_tiny | 29M | 768 |
| convnext_small | 50M | 768 |
| convnext_base | 89M | 1024 |
| convnext_large | 198M | 1536 |

## 数据集格式

数据集应按以下结构组织：

```
dataset/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_1/
    └── ...
```

## 使用方法

### 训练

```bash
# 基础训练（线性探测）
python train.py \
    --data-dir dataset/classification/abnormal26/abnormal_dataset \
    --model-type vits16 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001

# 微调训练（解冻骨干网络）
python train.py \
    --data-dir dataset/classification/abnormal26/abnormal_dataset \
    --model-type convnext_base \
    --unfreeze-backbone \
    --batch-size 32 \
    --lr 0.0001

# 使用脚本
./run_train.sh
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | 必填 | 数据集目录路径 |
| `--model-type` | vits16 | 模型架构类型 |
| `--output-dir` | outputs/classifier | 输出目录 |
| `--batch-size` | 32 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--epochs` | 100 | 训练轮数 |
| `--resize` | 256 | 图像尺寸 |
| `--unfreeze-backbone` | False | 解冻骨干网络 |
| `--resume` | None | 恢复训练的检查点路径 |

### 评估

```bash
python evaluate.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --data-dir dataset/classification/abnormal26/abnormal_dataset \
    --split test \
    --output-dir eval_results
```

评估结果包含：
- `metrics.json`：详细指标
- `confusion_matrix.png`：混淆矩阵可视化
- `per_class_metrics.png`：各类别指标图表
- `predictions.npz`：预测结果

### 推理

```bash
# 单张图片预测
python inference.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --top-k 5

# 批量预测
python inference.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --image-dir path/to/images/ \
    --output predictions.json
```

## 输出文件

训练完成后，输出目录结构：

```
outputs/classifier_vits16/
├── config.json              # 训练配置
├── history.json             # 训练历史
├── checkpoints/
│   ├── best_model.pth       # 最佳模型
│   ├── final_model.pth      # 最终模型
│   └── checkpoint_epoch_*.pth  # 周期性保存
```

## 代码示例

### Python API 使用

```python
from dinov3_classifier import Config, ClassifierTrainer
from dinov3_classifier.evaluator import ClassifierEvaluator
from dinov3_classifier.predictor import ClassifierPredictor

# 训练
config = Config(
    data_dir="path/to/dataset",
    model_type="vits16",
    epochs=100,
    batch_size=64,
)
trainer = ClassifierTrainer(config)
trainer.train()

# 评估
evaluator = ClassifierEvaluator(
    checkpoint_path="outputs/classifier/checkpoints/best_model.pth",
    weights_path="pretrained/backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
evaluator.run_full_evaluation("path/to/dataset", split="test")

# 推理
predictor = ClassifierPredictor(
    checkpoint_path="outputs/classifier/checkpoints/best_model.pth",
    weights_path="pretrained/backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
result = predictor.predict_single("path/to/image.jpg")
print(f"预测类别: {result['predicted_class']}, 置信度: {result['confidence']:.4f}")
```

## 参考文献

- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Hugging Face DINOv3](https://huggingface.co/docs/transformers/model_doc/dinov3)

## 许可证

本项目代码遵循 MIT 许可证。DINOv3 模型权重遵循其[原始许可证](dinov3/LICENSE.md)。
