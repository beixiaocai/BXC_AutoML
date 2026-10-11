# YOLO26 → RKNN 转换指南（RK3576 / RK3588 / RK3568 实测）

> 无需魔改源码 fork（Train_rk_yolo5/8/11 那套"改 Focus、删 DFL、锚点外置"的旧方案不需要）。
> ultralytics 8.4.159 已原生支持 `format="rknn"` 导出，整个流程两步：
> **PC 导出"RKNN 适配 ONNX" → Linux 端（板子或 x86）用 rknn-toolkit2 转 .rknn**
>
> 以下步骤均来自 2026-10 实测（RK3576 真机验证通过，rk3588/rk3568 转换通过）。

---

## 0. 先了解：RKNN 版模型的形态

| 项目 | 说明 |
|------|------|
| 输出 | `(1, 84, 8400)`，84 = 4(xywh) + 80(类分数，已过 sigmoid)。**rknn 格式强制禁 end2end**（NPU 跑不动图内 NMS 的 TopK/Gather），所以不能用 (1,300,6) 的 end2end 版 |
| NMS | 外置：host 侧做类内 NMS（C++ 参考 `RKNPUDetectYolo26` 的 process_single_output 分支） |
| 预处理 | 输入 uint8 RGB NHWC 0~255，`mean=0 / std=255`（即 NPU 内部除 255），letterbox 到 640 |
| fp16 坐标 | 输出 0~640 像素坐标，直接用 |
| int8 坐标 | 图内加了 `_NormalizeCoords`（xywh÷640，保护 per-tensor 量化的类分数精度）→ **输出 0~1，需 ×640**。判定依据：`output_attrs[0].type == RKNN_TENSOR_INT8` |

---

## 1. PC 端：导出 RKNN 适配 ONNX（Windows 可做）

环境：Train_yolo26 的 venv（装法见同目录 README.md），无需额外装包。

```bash
# fp16 用（产物 q16 源）
yolo export model=yolo26n.pt format=rknn name=rk3576 quantize=16

# int8 用（产物 q8 源，需校准数据集）
yolo export model=yolo26n.pt format=rknn name=rk3576 quantize=8 data=xxx/data.yaml
```

要点：

- `name` 是目标芯片：rk2118 / rk3562 / rk3566 / **rk3568 / rk3576 / rk3588** / rv1126b（rv1103/rv1106 只支持 int8）
- **Windows 上最后一步装 rknn-toolkit2 必然失败**（该库只有 Linux x86_64 的 wheel）——这是正常的，**适配 ONNX 在失败前已经生成落盘**（opset19 + onnxslim 常量折叠，368 节点；相比常规 opset12 的 402 节点少了动态 Shape 子图）
- **适配 ONNX 与目标平台无关**（平台只在转换时 config 指定）→ 一份 ONNX 可复用转 rk3576/3588/3568，全部实测成功
- int8 的归一化版 ONNX 是临时文件，导出结束会被自动删除；需要截存可 monkeypatch `ultralytics.utils.export.rknn.onnx2rknn`，在函数入口把 `onnx_file` 复制一份
- 实验结论：常规 opset12 ONNX 直接转 rknn 也能成功（与适配源仅差 3KB），但精度未对拍，**正式使用推荐适配源**

---

## 2. 板端环境搭建（RK3576 实测，Debian 12 / Python 3.11 aarch64）

板端 venv 的坑最多，按顺序来：

1. **建 venv**：板子 py3.11 无 ensurepip，先 `pip3 install --break-system-packages virtualenv`，再 `python3 -m virtualenv venv`
2. **装 rknn-toolkit2 必须用 PC 离线下 wheel**：直接在板上 `pip install rknn-toolkit2` 会触发源码编译 onnxoptimizer==0.3.8（无 cp311 aarch64 wheel，永远编不完，记得 pkill）。PC 上执行：
   ```bash
   pip download rknn-toolkit2==2.3.2 --platform manylinux2014_aarch64 --python-version 3.11 --only-binary=:all: -d wheels/
   ```
   上传到板子后：`pip install --no-deps wheels/*.whl`
3. **补依赖**：
   - `torch==2.2.0` aarch64 CPU wheel（约 190MB，rknn_utils 硬 import，必须装）
   - `numpy==1.26.4`、`opencv-python-headless`
   - `setuptools<81`（venv 自带 84 已移除 pkg_resources，rknn 依赖它）
   - onnxoptimizer 全系列无 aarch64 wheel，但 rknn 代码零引用（grep 验证过），**跳过即可**
4. 板上运行时：`librknnrt.so` 2.3.2 + NPU 驱动 0.9.7

> 替代路径：有 x86 Linux 机器或 Docker 的话，PC 上直接 `pip install rknn-toolkit2==2.3.2` 一步转换到位，不用碰板子。

---

## 3. 板端转换（convert_rknn.py 核心逻辑）

与 ultralytics 内置的 onnx2rknn 完全一致：

```python
from rknn.api import RKNN

rknn = RKNN(verbose=False)
# 换平台只改 target_platform 一个词
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3576")
rknn.load_onnx(model="yolo26n_rknn_q16_src.onnx")
# fp16：do_quantization=False；int8：True 且必须给 dataset（校准图清单，每行一张图的绝对路径）
rknn.build(do_quantization=False, rknn_batch_size=1)
rknn.export_rknn("yolo26n_coco_rk3576_fp16.rknn")
```

耗时参考：fp16 约 30s，int8 约 43s（RK3576 板上）。

---

## 4. 板端验证

- 推理脚本要点：letterbox 640 + RGB + uint8 NHWC；输出 (1,84,8400)；对 80 类做类内 NMS
- **期望结果（bus.jpg）**：fp16 / int8 均检出 5 目标（bus ~0.89 + 4 person），与 PC onnxruntime 基线一致
- **性能（RK3576 实测，warmup10+100，C++ 与 python 同量级）**：

| 模型 | CORE_0 | CORE_0_1 | 说明 |
|------|--------|----------|------|
| fp16 | 103.7ms / 9.6fps | **76.7ms / 13.0fps** | |
| int8 | 52.9ms / 18.9fps | **39.6ms / 25.2fps** | NPU 比 CPU 后端（ORT/OV 600ms+）快 15 倍以上 |

- **core mask 限制（RK3576 实测）**：`rknn_set_core_mask(CORE_0_1_2)` 报 `max core mask = 3`，显式三核不可用；多核只能 CORE_0_1（C++ deviceIndex 配 12 或 0），python 侧 `NPU_CORE_ALL` 是驱动单独放行的特殊值可用
- 板上静态 shape rknn 报 `Query dynamic range failed` 警告属正常，可忽略
- **RKNN 模拟器仅支持 x86 主机**：板上 `init_runtime()`（软件仿真）必失败；跨 rk3588/rk3568 的无真机验证只能 PC 装 x86 版 toolkit 跑模拟器

---

## 5. 校准集要求（int8 质量的决定因素）

- **100~200 张真实部署场景图起步**，覆盖类别/光照/尺度；无需标注，只要原图
- 本目录现有 int8 产物当时仅用 2 张图（bus.jpg + zidane.jpg）校准——**只够验证流程**：主框结果正常，但存在低分框漂移风险；正式部署请换足量真实校准集重转（同项目 pp-yoloe 用单图校准的教训：5 个框分数全部饱和卡在 0.791）
- 校准图预处理口径须与推理一致（toolkit 按 config 的 mean/std 处理，此处即原图直读）

---

## 6. rknn_export/ 产物清单

| 文件 | 平台 | 大小 |
|------|------|------|
| yolo26n_coco_rk3576_fp16.rknn | RK3576 | 10.04 MB |
| yolo26n_coco_rk3576_int8.rknn | RK3576 | 7.88 MB |
| yolo26n_coco_rk3588_fp16.rknn | RK3588 | 7.10 MB |
| yolo26n_coco_rk3588_int8.rknn | RK3588 | 4.60 MB |
| yolo26n_coco_rk3568_fp16.rknn | RK3568 | 6.49 MB |
| yolo26n_coco_rk3568_int8.rknn | RK3568 | 3.96 MB |

> 各平台 rknn 大小不同属正常（内部权重布局不同）。板端测试环境保留在 RK3576 的 `/userdata/test_yolo26_rknn/`（venv + convert_rknn.py + infer_test.py + calib 图）。

---

## 7. yolo8 / yolo11 同样适用此流程

ultralytics 原生 rknn 导出对所有检测模型通用（yolo8/11/26 均可），命令相同：

```bash
yolo export model=yolo11n.pt format=rknn name=rk3588 quantize=16
yolo export model=yolov8n.pt  format=rknn name=rk3588 quantize=16
```

唯一差异在 C++ 推理侧：库内 `RKNPUDetectYolo8.cpp` 目前只认旧 fork 的 6/9 输出格式（去 DFL + score-sum 分支），而新导出是单输出 (1,84,8400)——需把 `RKNPUDetectYolo26.cpp` 已验证的单输出后处理分支移植过去（布局完全一致：84 = 4+nc、8400 anchors、int8 坐标 ×640）。

---

## 8. 踩坑速查

| 症状 | 原因 / 处理 |
|------|------------|
| Windows 导出报 rknn-toolkit2 安装失败 | 正常，适配 ONNX 已生成；转换挪到 Linux/板端 |
| 板上 pip 装 rknn-toolkit2 卡死几小时 | 在编译 onnxoptimizer，pkill；改用 PC 离线 wheel + `--no-deps` |
| 板上 import rknn 报 No module named pkg_resources | venv setuptools≥81，补装 `setuptools<81` |
| int8 结果分数饱和 / 低分框漂移 | 校准集太少太单一，换 100~200 张真实图重转 |
| rknn_set_core_mask 报 unavailable core mask | RK3576 三核组合不可用，改 CORE_0_1 |
| 板上跑模拟器 init_runtime 失败 | RKNN 模拟器仅 x86，板端只能真机推理 |
| SFTP 回传 rknn 打不开 | 大文件可能静默截断，回传后 stat 对比大小，不一致重传 |
| 装新包后训练/导出崩 `Failed to initialize NumPy` | numpy 被顶到 2.x，钉回 1.26.4（见同目录 README 注意①） |
