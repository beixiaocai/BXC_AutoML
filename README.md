# ytstudio

**语言 / Language：** [简体中文](README.md) | [English](README_en.md)

> 宇图瑞视（yuturuishi）开源 AI 算法训练框架集合 —— 覆盖目标检测、图像 / 视频 / 音频分类、人脸识别、车牌识别等任务，并提供瑞芯微（Rockchip）边缘设备的模型转换（onnx → rknn）全链路。

| | |
|---|---|
| 官网 | https://www.yuturuishi.com |
| 微信 | yuturuishi |
| Gitee | https://gitee.com/yuturuishi/ytstudio |
| GitHub | https://github.com/yuturuishi/ytstudio |

## 简介

ytstudio 是一套面向 AI 视觉与音频算法的训练与部署工具集合。它以**模块化**方式组织：每个任务类型对应一个独立训练模块，配套免费公开数据集与示例脚本，开发者可快速完成「数据准备 → 模型训练 → 边缘部署」的完整流程。对于需要落地到瑞芯微硬件的场景，使用 `onnx2rknn` 即可将训练产出的 onnx 模型转换为 rknn 格式，直接在边缘端加速推理。

## 功能模块

| 模块 | 任务类型 | 说明 | 状态 |
|---|---|---|---|
| Train_DFINE | 目标检测（端到端） | 基于 D-FINE（Apache-2.0）的端到端无 NMS 检测框架，支持导出 ONNX / OpenVINO / TensorRT | 内置 |
| Train_yolo8 / Train_yolo11 / Train_yolo26 | 目标检测 | YOLO 官方训练框架；**Train_yolo26 原生支持导出 rknn（瑞芯微推荐）** | 内置 |
| Train_rk_yolo5 / Train_rk_yolo8 / Train_rk_yolo11 | 目标检测（边缘） | 早期瑞芯微专用 YOLO 魔改训练框架，与官方版割裂、难跟随升级 | **不建议使用**（已移至 `old/`） |
| onnx2rknn | 模型转换 | 在 x86 / arm 上将 onnx 模型转换为 rknn 模型（ResNet / XcFaceNet / DeepSort 等仍用本模块；YOLO 推荐 Train_yolo26 原生导出，无需本模块） | 内置 |
| Train_ResNet | 图像分类 | 基于 ResNet 的图片分类算法训练框架 | 内置 |
| Train_AudioNet | 音频分类 | 音频片段分类算法训练框架 | 已迁移独立仓库 |
| Train_VideoNet | 视频分类 | 视频片段分类算法训练框架 | 已迁移独立仓库 |
| Train_XcFaceNet | 人脸识别 | 人脸特征提取算法训练框架 | 已迁移独立仓库 |
| Train_PaddleOCR_Plate | 车牌识别 | 基于 PaddleOCR 与 PP-OCRv4 训练商用级车牌识别模型 | 已迁移独立仓库 |

## 瑞芯微（RK）设备部署：YOLO 推荐路线

**训练 yolo 模型用于瑞芯微设备，推荐直接使用 `Train_yolo26`**。ultralytics 8.4.x 已原生支持 `format="rknn"` 导出，训练与转换在同一框架内完成，无需魔改源码：

```
Train_yolo26 训练 → yolo export model=best.pt format=rknn name=rk3576 → 板端 rknn-toolkit2 转 .rknn
```

- 完整分步指南（PC 导出适配 ONNX、板端环境搭建、int8 校准集要求、踩坑清单）：见 `Train_yolo26/export_rknn.md`
- 已实测：RK3576 真机推理（fp16 13fps / int8 25fps @640），RK3588 / RK3568 转换通过；yolo8 / yolo11 同样适用该导出流程
- ⚠️ **推理端版本要求：rebekah v5.304 及以上**。该方案转换出的 rknn 为新输出格式（单输出 `(1, 84, 8400)`），**rebekah v5.304 以下版本无法使用**，部署前请先升级 rebekah
- `Train_rk_yolo5 / Train_rk_yolo8 / Train_rk_yolo11` 是早期通过修改 ultralytics 源码（改 Focus、删 DFL、锚点外置）实现的边缘方案，与官方版本割裂、难以跟随升级，**不建议新项目使用**，仅作历史保留（已移至 `old/` 目录）

## 项目结构

ytstudio 仓库按任务类型组织为多个独立训练模块：

```
ytstudio/
├── Train_DFINE/          # D-FINE 端到端无 NMS 检测训练框架（Apache-2.0）
├── Train_yolo8/  Train_yolo11/  Train_yolo26/    # YOLO 官方训练框架（目标检测）
├── old/                   # 历史保留：Train_rk_yolo5/8/11 早期瑞芯微魔改方案（不建议使用）
├── onnx2rknn/              # onnx → rknn 模型转换工具
├── Train_ResNet/          # 图像分类训练框架
├── Train_AudioNet/        # 音频分类（已迁移独立仓库）
├── Train_VideoNet/        # 视频分类（已迁移独立仓库）
├── Train_XcFaceNet/       # 人脸识别（已迁移独立仓库）
└── Train_PaddleOCR_Plate/ # 车牌识别（已迁移独立仓库）
```

## Train_DFINE 专项说明

`Train_DFINE` 基于 [Peterande/D-FINE](https://github.com/Peterande/D-FINE)（Apache-2.0）进行本地化改造，提供「自有数据集训练 → 导出 ONNX → 转 OpenVINO / TensorRT」的完整可复现链路。详细用法见模块内 `README.md`。

- **开源协议（License）**：D-FINE 核心框架为 **Apache-2.0**，可免费用于学习、研究与商业用途，无 AGPL 网络服务条款约束；与 YOLO（Ultralytics AGPL-3.0）相比，商用闭源分发更友好。本模块自有训练代码继承 ytstudio 的 MIT；**用户自训练权重对外标注「自有」**。
- **用途（Usage）**：面向高精度通用目标检测，从自有数据集训练专属检测模型，导出为 ONNX / OpenVINO / TensorRT，部署到服务器 GPU 或 x86 边缘设备。
- **能力（Capabilities）**
  - 端到端检测，**无需 NMS** 后处理，部署链路更简洁；
  - 5 档官方 COCO 预训练权重（n / s / m / l / x），支持微调或直接从头训练；
  - 支持 COCO 格式自定义数据集，并内置 YOLO → COCO 批量转换工具；
  - 一键导出 ONNX（动态 batch，opset 17）→ OpenVINO IR（静态 fp32，目标运行时 2024.6）→ TensorRT engine（静态 fp32）；
  - 已在 13 个安防 / 行为分析场景数据集上完成训练验证（火焰烟雾、老鼠、粉尘、打架、学生状态、睡岗、手机、攀爬、安全帽、密集人群等）。
- **商业价值（Commercial Value）**：Apache-2.0 可免费商用与闭源分发；端到端无 NMS 简化推理后处理；高精度检测适配安防监控、工业质检、行为分析等精度敏感的商业场景；导出的 OpenVINO / TensorRT 模型可无缝接入 rebekah 等商用分析系统。

## 技术栈

PyTorch + CUDA | Ultralytics YOLO | OpenCV / PIL / NumPy | RKNN Toolkit / ONNX Runtime | Python

## 环境要求

- **Python** 3.8 及以上
- **训练**：PyTorch + CUDA（具体版本见各模块 README 的环境说明）
- **边缘部署**：Rockchip 模型转换环境（见 `onnx2rknn` 模块 README，提供多种安装方式）

## 快速开始

1. 下载训练数据集（见下方「免费训练数据集」）。
2. 进入对应训练模块目录，按该模块的 README 配置数据集路径并启动训练。
3. 如需部署到瑞芯微设备：YOLO 检测模型推荐用 `Train_yolo26` 训练，并通过 ultralytics 原生导出 rknn（见 `Train_yolo26/export_rknn.md`）；其他模型可训练产出 onnx 后，使用 `onnx2rknn` 转换为 rknn 格式。注意：YOLO 的 rknn 模型需 rebekah v5.304+ 才能推理，旧版本请先升级。

> 各内置模块均自带独立 README，包含该模块的配置与训练细节，请以模块 README 为准。

## 子模块与独立仓库

部分训练框架已拆分为独立仓库进行维护，点击跳转：

- **AudioNet**（音频分类）：[Gitee](https://gitee.com/yuturuishi/AudioNet) · [GitHub](https://github.com/yuturuishi/AudioNet)
- **ResNet**（图像分类）：[Gitee](https://gitee.com/yuturuishi/ResNet) · [GitHub](https://github.com/yuturuishi/ResNet)
- **VideoNet**（视频分类）：[Gitee](https://gitee.com/yuturuishi/VideoNet) · [GitHub](https://github.com/yuturuishi/VideoNet)
- **XcFaceNet**（人脸识别）：[Gitee](https://gitee.com/yuturuishi/XcFaceNet) · [GitHub](https://github.com/yuturuishi/XcFaceNet)
- **PaddleOCR_Plate**（车牌识别）：[Gitee](https://gitee.com/yuturuishi/PaddleOCR_Plate) · [GitHub](https://github.com/yuturuishi/PaddleOCR_Plate)

## 样本采集与标注

推荐使用 **[ytlabel](https://gitee.com/yuturuishi/ytlabel)**（[GitHub](https://github.com/yuturuishi/ytlabel)）—— 集样本采集（视频抽帧）与标注于一体，支持图片 / 视频 / LabelMe 数据集导入、矩形 / 多边形标注，可直接导出 YOLO 格式数据集（含 train / val / test 切分与 `data.yaml`）。

也可使用 [labelme](https://pan.quark.cn/s/7e6accce2a3e) 标注后导入 ytlabel 导出数据集。

## 免费训练数据集

- 训练数据集 - 夸克网盘下载地址：https://pan.quark.cn/s/5dcc2f724bcc
- 检测密集人群数据集 20250625
- 检测明厨亮灶数据集 20250625
- 检测攀爬数据集 20250624
- 检测抽烟数据集 20241012
- 检测打架数据集 20241012
- 检测反光衣数据集 20241013
- 检测粉尘数据集 20241013
- 检测火焰烟雾数据集 20241012
- 检测人头和安全帽数据集 20241013
- 检测人体 5 动作 - 站着 - 摔倒 - 坐 - 深蹲 - 跑数据集 20241012
- 检测学生 3 状态数据集 20241022
- 检测睡岗数据集 20241210
- 检测手机数据集 20251206
- 猫狗 2 分类数据集 - 夸克网盘下载地址 https://pan.quark.cn/s/982dd16cb29d
- 车型 9 分类数据集 - 夸克网盘下载地址 https://pan.quark.cn/s/f698d0e99a4b
- 打鼾 + 不打鼾语音识别 2 分类数据集 - 夸克网盘下载地址：https://pan.quark.cn/s/4d83dabff0a6

## 开源协议

本项目基于 [MIT License](LICENSE) 开源，可免费用于学习、研究与商业用途。

## 第三方依赖与许可说明

ytstudio 的**自有代码**以 MIT License 发布，可自由用于学习、研究与商业用途。

本项目涉及的主要第三方依赖及其许可：

| 依赖 | 用途 | 许可 |
|---|---|---|
| PyTorch | 训练框架基础 | BSD |
| Ultralytics（YOLO 训练框架） | YOLO 模型训练 | AGPL-3.0 |
| D-FINE（检测框架） | 端到端目标检测训练与推理 | Apache-2.0 |
| OpenCV / PIL / NumPy 等 | 图像与数值处理 | MIT / BSD 等宽松许可 |
| RKNN Toolkit / ONNX Runtime | 模型转换与推理 | Apache-2.0 / MIT |

- **Ultralytics（AGPL-3.0）**：使用其训练模型、并导出为 ONNX / RKNN 后在自有硬件上部署，不受 AGPL 网络服务条款约束；但若将 Ultralytics 软件本身以闭源产品、私有部署、SaaS、边缘/嵌入式设备等方式对外提供，须遵守 Ultralytics 许可条款或购买其 Enterprise License。ytstudio 的 MIT 许可不覆盖该部分义务。
- **D-FINE（Apache-2.0）**：与 YOLO（Ultralytics AGPL-3.0）不同，D-FINE 采用 Apache-2.0 协议，可免费用于商业用途及闭源分发，无需担忧 AGPL 网络服务条款。`Train_DFINE` 基于 Peterande/D-FINE 改造，D-FINE 框架本身的许可义务以 Apache-2.0 为准；用户自训练权重对外标注「自有」。
- **模型与数据集**：仓库所引训练数据集、预训练权重由其提供方负责许可；Ultralytics 预训练权重（`.pt`）受 AGPL / 企业许可约束。
- **网络请求**：ytstudio 为本地训练工具，不向任何外部服务器回传用户数据或运行信息。
- **免责**：使用者应自行确保所用模型、数据集与部署环境的合法性。ytstudio 按「现状」提供，不作任何担保。
