# old（历史保留模块，不建议使用）

本目录存放早期瑞芯微（Rockchip）边缘部署方案：`Train_rk_yolo5/`、`Train_rk_yolo8/`、`Train_rk_yolo11/`。三者都是**魔改 ultralytics 官方源码**的 fork——把官方模型中"NPU 跑不了 / 跑不快 / 量化不友好"的部分从图里拆出来挪到 CPU 侧，使剩下的网络能顺利转 rknn 并在 NPU 上高效推理。

> **为什么不建议再用**：魔改散落在 nn 模块与导出链路中，与官方版本割裂，官方每次升级都无法直接合并，训练/转换/后处理三段链路要分开维护。ultralytics 8.4.x 已原生支持 `format="rknn"` 导出，同样效果不再需要魔改（见文末推荐路线）。

---

## Train_rk_yolo5 —— RK 版 YOLOv5

基于官方 yolov5 **v7.0**（commit `915bbf29`）改造，详细说明见模块内 `org_README_rkopt.md`。

| # | 魔改点 | 具体做法 | 为什么要改 |
|---|--------|----------|-----------|
| 1 | **Focus 层优化** | 官方 Focus 用 stride-2 切片（slice）下采样；改为数学等价的标准卷积实现 | 切片类算子在 rknn-toolkit 上支持不佳、NPU 执行效率低，换成标准卷积后性能更好且结果不变 |
| 2 | **SPPF 块优化** | 官方 SPPF 的 MaxPool 串联结构改写为等效实现 | 同上，规避 NPU 上低效/不支持的算子组合，结果不变 |
| 3 | **检测头分离（去图内后处理）** | 输出节点改为 3 个尺度的原始分支（reg + cls），官方图内的 decode（anchor 解算、sigmoid、NMS 前处理）整体移出模型 | 图内后处理含 NMS/TopK 等算子，**对 int8 量化极不友好**（量化误差直接摧毁框精度），且 NPU 无法执行 NMS |
| 4 | **锚点外置** | 导出时生成 `RK_anchors.txt`，anchor 常量从模型内挪到 CPU 后处理代码中 | 减少图内常量与算子；锚点配合外置 decode 使用 |
| 5 | **SiLU → ReLU**（可选） | 训练新模型时把激活函数从 SiLU 换成 ReLU | SiLU = x·sigmoid(x)，int8 量化下误差比 ReLU 大且算子开销高。**注意：这会改变推理结果**，只适用于重新训练的模型 |

导出命令：`python export.py --rknpu --weight yolov5s.pt`（必须带 `--rknpu`，支持 detect/seg），后处理参考 airockchip/rknn_model_zoo。

## Train_rk_yolo8 —— RK 版 YOLOv8

基于 airockchip 同源方案改造（与官方 ultralytics_yolov8 fork 一脉），检测头输出 3 尺度分支张量。

| # | 魔改点 | 具体做法 | 为什么要改 |
|---|--------|----------|-----------|
| 1 | **检测头分离（去图内后处理）** | 输出节点改为 3 尺度 × (reg 64 通道 + cls nc 通道) 的原始张量（共 6 个输出；部分版本每分支再加 1 通道 score-sum，共 9 个输出），官方图内的 dist2bbox 解算与 NMS 前处理移出模型 | 同 yolo5：图内后处理量化不友好 + NPU 跑不了 NMS。C++ 推理侧 `RKNPUDetectYolo8.cpp` 按此格式解析（6/9 输出两种布局都支持） |
| 2 | **删除 DFL 结构** | 官方检测头末端的 DFL（分布式焦点损失解码：16 通道分布 → softmax → 加权求和 → 框坐标）从图中删除，改在 CPU 后处理中完成（`dfl_len = 输出通道/4`） | DFL 的 softmax/reshape/矩阵乘在 NPU 上**执行慢且 per-tensor int8 量化误差大**，挪到 CPU 后 NPU 只跑高效率卷积主干，整体更快更稳 |

## Train_rk_yolo11 —— RK 版 YOLOv11

基于官方 ultralytics（commit `50497218`）改造，详细说明见模块内 `RKOPT_README.md` / `RKOPT_README.zh-CN.md`。

| # | 魔改点 | 具体做法 | 为什么要改 |
|---|--------|----------|-----------|
| 1 | **检测头分离（去图内后处理）** | 同 yolo8：输出节点改造，decode 与 NMS 前处理移出模型，CPU 侧后处理参考 rknn_model_zoo | 图内后处理对量化不友好 |
| 2 | **删除 DFL 结构** | 同 yolo8：DFL 移到 CPU | DFL 在 NPU 上慢 |
| 3 | **新增 score-sum 输出分支** | 每个尺度的 cls 分支额外输出 1 通道的类分数求和（80 类 → 1 通道），共 9 输出 | CPU 后处理先用 sum 快速筛掉大量空网格，只对少量候选做完整 DFL 解算 + argmax，**大幅降低 CPU 后处理耗时** |

---

## 共同代价（也是被淘汰的原因）

1. **源码 fork 维护成本高**：魔改集中在模型定义与导出链路，官方每次升级都无法直接合并，只能停在旧版本（yolo5 停在 v7.0、yolo11 停在 2024 年的 commit）
2. **训练/转换/后处理三段割裂**：模型输出格式是自定义的，必须配套专用后处理（CPU 侧 DFL 解算、锚点解算、score-sum 过滤），多端（python/C++）都要各写一份
3. **换模型代系就要重新魔改一遍**：5/8/11 各改各的，没有通用性

## 推荐路线（已实测，无需魔改）

ultralytics 8.4.x 原生支持 `format="rknn"` 导出：适配由官方导出器自动完成（opset 折叠、int8 坐标归一化保护等），DFL 留在图内由 NPU 执行，输出统一为 `(1, 84, 8400)` 单输出，后处理只需一次类内 NMS。

瑞芯微设备训练 yolo 模型，请直接使用 **[Train_yolo26](../Train_yolo26/README.md)**：

```
Train_yolo26 训练 → yolo export model=best.pt format=rknn name=rk3576 → 板端 rknn-toolkit2 转 .rknn
```

分步转换指南见 **[Train_yolo26/export_rknn.md](../Train_yolo26/export_rknn.md)**（已实测 RK3576 真机：fp16 13fps / int8 25fps @640；RK3588 / RK3568 转换通过）。
