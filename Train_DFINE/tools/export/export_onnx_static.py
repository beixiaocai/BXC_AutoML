# D-FINE 动态 ONNX → 静态 ONNX 批量转换（部署链路第二步：export_onnx.py 之后）
# 源：models/d-fine-onnx-opset17-dyn-fp32/*_dyn.onnx（n/x/l/m/s 五个模型）
# 目标：models/d-fine-onnx-opset17-static-fp32/*_static.onnx
# 固化：images batch=1、orig_target_sizes [1,2]、输出 [1,300]/[1,300,4]/[1,300]
# 验证：每个静态模型 ORT 推理成功，且与动态模型输出 diff≈0
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
import cv2
from onnx import shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
SRC_DIR = os.path.join(PROJ, "models", "d-fine-onnx-opset17-dyn-fp32")
DST_DIR = os.path.join(PROJ, "models", "d-fine-onnx-opset17-static-fp32")
IMG = os.path.join(HERE, "bus.jpg")
os.makedirs(DST_DIR, exist_ok=True)


def preprocess(path, size=640):
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=resized.dtype)
    canvas[:nh, :nw] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0)
    tsz = np.array([[nh, nw]], dtype=np.int64)
    return blob, tsz


def fix_batch(vi, value=1):
    """把输入第 0 维的符号/动态维度固化为 value"""
    dims = vi.type.tensor_type.shape.dim
    if len(dims) and (dims[0].dim_param or dims[0].dim_value in (0, None)):
        dims[0].ClearField("dim_param")
        dims[0].dim_value = value
    return vi


def convert(src, dst, blob, tsz):
    m = onnx.load(src)
    for i in m.graph.input:
        fix_batch(i)
    # 形状推断（输出可能仍带符号，由 ORT 实际运行确定）
    m = shape_inference.infer_shapes(m)

    # 用推断后的模型先跑一次拿真实输出 shape（输出符号维度不影响 ORT 运行）
    tmp = dst + ".tmp"
    onnx.save(m, tmp)
    s = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    outs = s.run(None, {"images": blob, "orig_target_sizes": tsz})
    shapes = {o.name: tuple(np.asarray(v).shape) for o, v in zip(s.get_outputs(), outs)}

    # 固化输出维度
    for o in m.graph.output:
        sh = shapes[o.name]
        tt = o.type.tensor_type
        del tt.shape.dim[:]
        for dv in sh:
            d = tt.shape.dim.add()
            d.dim_value = dv

    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, dst)
    os.remove(tmp)

    # 终验：静态 vs 动态对拍 + 推理可用
    s_dyn = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    s_sta = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    o_dyn = s_dyn.run(None, {"images": blob, "orig_target_sizes": tsz})
    o_sta = s_sta.run(None, {"images": blob, "orig_target_sizes": tsz})
    diffs = {}
    for n, a, b in zip(["labels", "boxes", "scores"], o_dyn, o_sta):
        diffs[n] = float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
    scores = np.asarray(o_sta[2]).reshape(-1)
    ndet = int((scores > 0.5).sum())
    top = round(float(sorted(scores, reverse=True)[0]), 4)
    in_dims = [[d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
               for i in m.graph.input]
    out_dims = [[d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
                for o in m.graph.output]
    print("%-28s in=%s out=%s | diff labels=%.3g boxes=%.3g scores=%.3g | det@0.5=%d top=%.4f" % (
        os.path.basename(dst), in_dims, out_dims,
        diffs["labels"], diffs["boxes"], diffs["scores"], ndet, top))
    return max(diffs.values())


def main():
    blob, tsz = preprocess(IMG)
    srcs = sorted(glob.glob(os.path.join(SRC_DIR, "*.onnx")))
    print("found %d dynamic models" % len(srcs))
    worst = 0.0
    for src in srcs:
        name = os.path.basename(src).replace("_dyn.onnx", "_static.onnx")
        dst = os.path.join(DST_DIR, name)
        worst = max(worst, convert(src, dst, blob, tsz))
    print("ALL DONE, worst diff = %.6g" % worst)


if __name__ == "__main__":
    main()
