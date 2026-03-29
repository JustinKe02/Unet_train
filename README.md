# MoS2 U-Net Segmentation

这是一个基于 PyTorch 的二维材料图像语义分割项目，已经按当前仓库里的 `Mos2_data` 和 `MoS2(supplement)` 数据集完成适配，可直接训练 4 类分割模型：

- `background`
- `monolayer`
- `fewlayer`
- `multilayer`

项目默认使用单卡 RTX 3090、混合精度训练、随机裁剪训练和滑窗验证/推理，适合当前这批高分辨率显微图像。

## 目录结构

```text
.
├── configs/
│   ├── mos2_unet.yaml
│   ├── mos2_unet_v2.yaml
│   ├── mos2_unet_v3_ce_focal.yaml
│   ├── mos2_unet_v4_focal_dice.yaml
│   └── mos2_unet_v5_main_only.yaml
├── src/unet_project/
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── train.py
├── predict.py
└── requirements.txt
```

## 数据说明

项目会自动扫描以下目录并匹配同名图像和 mask：

- `Mos2_data/ori/MoS2` -> `Mos2_data/mask`
- `MoS2(supplement)/ori` -> `MoS2(supplement)/mask_generated`

自动处理内容包括：

- 忽略无效扩展名文件，例如 `Zone.Identifier`
- 识别 `.jpg/.jpeg/.png/.tif/.tiff`
- 首次训练时自动生成可复现数据划分文件 `splits/mos2_split.json`

## 环境安装

```bash
pip install -r requirements.txt
```

当前机器已经具备 `torch 2.5.1 + cu124` 和 `torchvision 0.20.1`，通常不需要额外安装 CUDA。

## 开始训练

基线训练：

```bash
python train.py --config configs/mos2_unet.yaml
```

第二轮优化训练：

```bash
python train.py --config configs/mos2_unet_v2.yaml
```

`CE + Focal` 对照训练：

```bash
python train.py --config configs/mos2_unet_v3_ce_focal.yaml
```

`Focal(alpha) + Dice` 训练：

```bash
python train.py --config configs/mos2_unet_v4_focal_dice.yaml
```

仅使用 `Mos2_data` 主数据集训练：

```bash
python train.py --config configs/mos2_unet_v5_main_only.yaml
```

输出默认保存到：

```text
outputs/mos2_unet/
```

主要产物：

- `checkpoints/best.pth`
- `checkpoints/latest.pth`
- `history.json`
- `test_metrics.json`
- `splits/mos2_split.json`

## 常用训练命令

快速 smoke test：

```bash
python train.py \
  --config configs/mos2_unet_v2.yaml \
  --epochs 1 \
  --batch-size 1 \
  --crop-size 256 \
  --num-workers 0 \
  --train-repeat 1 \
  --output-dir outputs/smoke_test
```

只做验证：

```bash
python train.py \
  --config configs/mos2_unet.yaml \
  --eval-only \
  --checkpoint outputs/mos2_unet/checkpoints/best.pth \
  --split val
```

## 推理

对单张图像推理：

```bash
python predict.py \
  --config configs/mos2_unet.yaml \
  --checkpoint outputs/mos2_unet/checkpoints/best.pth \
  --input "Mos2_data/ori/MoS2/m1.jpg" \
  --output-dir outputs/predict_demo
```

对整个目录推理：

```bash
python predict.py \
  --config configs/mos2_unet.yaml \
  --checkpoint outputs/mos2_unet/checkpoints/best.pth \
  --input "MoS2(supplement)/ori" \
  --output-dir outputs/predict_supplement
```

推理输出包括：

- 原始类别 mask
- 调色板彩色 mask
- 叠加可视化 overlay

## 训练策略

- 模型：标准 U-Net
- 输入：RGB 图像
- 类别数：4
- 训练：前景感知裁剪 + 少数类偏置裁剪 + 翻转 + 90 度旋转
- 验证/测试：全图滑窗推理
- 基线损失：`CrossEntropy + Dice`
- 第二轮损失：`Weighted CE + Focal + Tversky`
- 指标：`mIoU / Dice / Pixel Accuracy`
- 训练控制：`best checkpoint + top-k checkpoint + early stopping`
- 加速：AMP 混合精度、`pin_memory`、多进程 DataLoader

## 第二轮优化重点

- 使用 [mos2_unet_v2.yaml](/root/autodl-tmp/Unet/configs/mos2_unet_v2.yaml) 作为第二轮训练配置。
- 对 `monolayer` 和 `fewlayer` 做偏置裁剪，减少大量纯背景或易样本 patch。
- 用 `Focal + Tversky` 强化难分类和少数类边界。
- 开启早停，避免验证集已经平台后继续空转到固定轮数。
- 保存 top-k checkpoint，方便后续对比不同轮次预测质量。

## CE + Focal 配置

- 使用 [mos2_unet_v3_ce_focal.yaml](/root/autodl-tmp/Unet/configs/mos2_unet_v3_ce_focal.yaml) 时，损失只包含 `CrossEntropy + Focal`。
- 当前设置为：
  - `ce_weight: 0.5`
  - `focal_weight: 0.5`
  - `focal_gamma: 2.0`
- `Dice` 和 `Tversky` 权重被设为 `0.0`，不会参与训练。

## Focal(alpha) + Dice 配置

- 使用 [mos2_unet_v4_focal_dice.yaml](/root/autodl-tmp/Unet/configs/mos2_unet_v4_focal_dice.yaml) 时，损失为 `Focal(alpha) + Dice`。
- 当前设置为：
  - `focal_weight: 0.6`
  - `focal_gamma: 2.0`
  - `focal_alpha: [0.2, 1.8, 2.2, 0.6]`
  - `dice_weight: 0.4`
- `use_class_weights` 被关闭，避免与显式 `alpha` 重复加权。

## Main-Only 训练

- 使用 [mos2_unet_v5_main_only.yaml](/root/autodl-tmp/Unet/configs/mos2_unet_v5_main_only.yaml) 时，只会扫描 [Mos2_data](/root/autodl-tmp/Unet/Mos2_data)。
- 会单独生成 [mos2_main_only_split.json](/root/autodl-tmp/Unet/splits/mos2_main_only_split.json)，不会覆盖之前包含 supplement 的 split。
- 当前沿用 `Focal(alpha) + Dice` 作为损失组合。

## 建议参数

当前默认配置已经偏向 3090 单卡：

- `crop_size: 768`
- `batch_size: 4`
- `num_workers: 8`

如果显存紧张，优先降低：

1. `batch_size`
2. `crop_size`
3. `base_channels`

## 后续可扩展方向

- 加入 `DeepLabV3+` / `UNet++`
- 引入更强的数据增强
- 统计每类单独 F1/IoU 曲线
- 增加 TensorBoard 或 WandB 日志
