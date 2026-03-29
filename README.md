# MoS2 U-Net Segmentation

基于 PyTorch 的二维材料（MoS2）显微图像语义分割项目。包含两个对比实验：

- **Unet**：基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 的官方实现
- **Unet_Improve**：改进版 UNet，引入 Focal Loss、前景感知采样等优化策略

## 分割类别

| 类别 | 像素值 | 颜色 |
|------|--------|------|
| Background | 0 | 黑色 |
| Monolayer | 1 | 红色 |
| Fewlayer | 2 | 绿色 |
| Multilayer | 3 | 蓝色 |

## 实验结果

### 测试集指标对比

| 指标 | Unet | Unet_Improve | Δ |
|------|------|-------------|---|
| **mIoU** | 0.7779 | **0.8085** | +0.0306 |
| **mIoU_fg** | 0.7195 | **0.7551** | **+0.0356** |
| **Pixel Acc** | 0.9529 | **0.9695** | +0.0166 |

### 各类 IoU

| 类别 | Unet | Unet_Improve | Δ |
|------|------|-------------|---|
| Background | 0.9529 | **0.9689** | +0.0160 |
| Monolayer | 0.5224 | **0.5594** | +0.0370 |
| Fewlayer | 0.6978 | **0.7370** | +0.0392 |
| Multilayer | 0.9383 | **0.9687** | +0.0304 |

## 目录结构

```text
.
├── configs/
│   └── mos2_unet_v5_main_only.yaml       # Unet_Improve 配置
├── src/unet_project/                      # Unet_Improve 模型代码
│   ├── config.py                          # 配置解析
│   ├── data.py                            # 数据加载与增强
│   ├── engine.py                          # 训练/验证引擎
│   ├── losses.py                          # 损失函数 (Focal + Dice)
│   ├── metrics.py                         # 评估指标 (mIoU, Dice)
│   ├── model.py                           # UNet 模型定义
│   └── utils.py                           # 工具函数
├── unet_official/                         # 官方 UNet 代码
│   ├── unet/                              # 官方模型定义
│   ├── train_mos2.py                      # 适配 MoS2 的训练脚本
│   ├── evaluate_mos2.py                   # mIoU 评估模块
│   ├── test_evaluate.py                   # 测试集推理
│   ├── plot_training.py                   # 训练曲线绘制
│   └── utils/mos2_dataset.py             # MoS2 数据集加载
├── outputs/mos2_unet_v5_main_only/        # Unet_Improve 训练输出
│   ├── plots/                             # 训练曲线图
│   ├── data/                              # 训练数据 CSV
│   └── test_results/                      # 测试评估结果
├── splits/
│   └── mos2_main_only_split.json          # 数据划分 (train/val/test)
├── train.py                               # Unet_Improve 训练入口
├── predict.py                             # 推理脚本
└── requirements.txt
```

## 训练配置对比

| 配置项 | Unet | Unet_Improve |
|--------|------|-------------|
| 模型 | Pytorch-UNet (base_ch=64) | Custom UNet (base_ch=32) |
| 参数量 | ~31M | ~7.8M |
| 损失函数 | CE + Dice | Focal (γ=2) + Dice |
| 优化器 | RMSprop (lr=1e-5) | AdamW (lr=2.5e-4) |
| 学习率调度 | ReduceLROnPlateau | CosineAnnealing |
| 数据增强 | 随机裁剪 + 翻转旋转 | 前景感知采样 + 翻转旋转 |
| 混合精度 | 否 | AMP ✅ |
| 早停 | 否 | patience=12 |

## 环境安装

```bash
pip install -r requirements.txt
```

需要 PyTorch >= 2.0 + CUDA。

## 训练

### Unet_Improve

```bash
python train.py --config configs/mos2_unet_v5_main_only.yaml
```

### Unet (官方)

```bash
cd unet_official
python train_mos2.py --epochs 100 --batch-size 3 --crop-size 896
```

## 推理

对单张图像推理：

```bash
python predict.py \
  --config configs/mos2_unet_v5_main_only.yaml \
  --checkpoint outputs/mos2_unet_v5_main_only/checkpoints/best.pth \
  --input "Mos2_data/ori/MoS2/m1.jpg" \
  --output-dir outputs/predict_demo
```

## 数据说明

- 数据集 `Mos2_data/` 未包含在仓库中（体积过大）
- 数据划分文件 `splits/mos2_main_only_split.json` 已保存，确保实验可复现
- 划分比例：train 80% / val 10% / test 10%（137/17/18 张）

## 建议参数

默认配置适用于单卡 RTX 3090 (24GB)。如果显存不足，优先降低：

1. `batch_size`
2. `crop_size`
3. `base_channels`
