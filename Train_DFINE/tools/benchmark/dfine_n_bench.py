# D-FINE n 模型 CPU 推理性能对比：onnxruntime vs openvino
# 方法论与 Train_PaddleDet_ppyoloe 的 ppyoloe_s_bench.py 一致：
# 同一张图、同一预处理（等比 resize + 114 底 + /255），warmup 10 次后循环 100 次计时
import statistics
import time

import cv2
import numpy as np

ROOT = r"D:\project\ytstudio\Train_DFINE"
ONNX = ROOT + r"\models\d-fine-onnx-opset17-dyn-fp32\dfine_n_coco_opset17_fp32_dyn.onnx"
OV = ROOT + r"\models\d-fine-openvino2024.6-static-fp32\dfine_n_coco_ov2024.6_fp32_static.xml"
IMG = ROOT + r"\tools\export\bus.jpg"
SIZE = 640
WARMUP, ITERS = 10, 100


def preprocess(path):
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = min(SIZE / w, SIZE / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=resized.dtype)
    canvas[:nh, :nw] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0)
    target_size = np.array([[nh, nw]], dtype=np.int64)
    return blob, target_size


def bench_ort(blob, target_size):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    feed = {"images": blob, "orig_target_sizes": target_size}
    for _ in range(WARMUP):
        outs = sess.run(None, feed)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1000)
    print_stats("onnxruntime", ts)
    return outs


def bench_ov(blob, target_size):
    from openvino import Core
    core = Core()
    model = core.compile_model(OV, "CPU")
    req = model.create_infer_request()
    inputs = {"images": blob, "orig_target_sizes": target_size}
    for _ in range(WARMUP):
        req.infer(inputs)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        req.infer(inputs)
        ts.append((time.perf_counter() - t0) * 1000)
    print_stats("openvino", ts)
    outs = [req.get_tensor(n).data for n in ("labels", "boxes", "scores")]
    return outs


def print_stats(name, ts):
    print("%-12s mean=%.2fms median=%.2f min=%.2f max=%.2f std=%.2f fps=%.1f" % (
        name, statistics.mean(ts), statistics.median(ts), min(ts), max(ts),
        statistics.pstdev(ts), 1000 / statistics.mean(ts)))


def main():
    blob, target_size = preprocess(IMG)
    print("input: images %s, orig_target_sizes %s" % (blob.shape, target_size.tolist()))

    print("--- onnxruntime ---")
    ort_outs = bench_ort(blob, target_size)
    print("--- openvino ---")
    ov_outs = bench_ov(blob, target_size)

    # 精度对比：labels/boxes/scores（顺序 ort: labels,boxes,scores）
    names = ["labels", "boxes", "scores"]
    for n, a, b in zip(names, ort_outs, ov_outs):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            print("%s shape mismatch: %s vs %s" % (n, a.shape, b.shape))
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print("%-7s max_abs_diff=%.6g mean_abs_diff=%.6g" % (n, d.max(), d.mean()))

    scores = np.asarray(ort_outs[2]).reshape(-1)
    print("detections@0.5:", int((scores > 0.5).sum()),
          "top5:", [round(float(s), 4) for s in sorted(scores, reverse=True)[:5]])


if __name__ == "__main__":
    main()
