# D-FINE n 静态模型性能对比：onnxruntime(静态 ONNX) vs openvino(静态 IR)
# 与 dfine_n_bench.py 同方法论：同一张图、同一预处理（等比 resize + 114 底 + /255），warmup 10 + 100 次计时
import statistics
import time

import cv2
import numpy as np

ROOT = r"D:\project\ytstudio\Train_DFINE"
STATIC_ONNX = ROOT + r"\models\dfine_n_coco_static.onnx"
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


def stats(name, ts):
    print("%-28s mean=%.2fms median=%.2f min=%.2f max=%.2f std=%.2f fps=%.1f" % (
        name, statistics.mean(ts), statistics.median(ts), min(ts), max(ts),
        statistics.pstdev(ts), 1000 / statistics.mean(ts)))


def main():
    blob, target_size = preprocess(IMG)
    feed = {"images": blob, "orig_target_sizes": target_size}

    import onnxruntime as ort
    sess = ort.InferenceSession(STATIC_ONNX, providers=["CPUExecutionProvider"])
    for _ in range(WARMUP):
        ort_outs = sess.run(None, feed)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1000)
    stats("ort-static", ts)

    from openvino import Core
    core = Core()
    model = core.compile_model(OV, "CPU")
    req = model.create_infer_request()
    for _ in range(WARMUP):
        req.infer(feed)
    ts2 = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        req.infer(feed)
        ts2.append((time.perf_counter() - t0) * 1000)
    stats("openvino-static", ts2)
    ov_outs = [req.get_tensor(n).data for n in ("labels", "boxes", "scores")]

    for n, a, b in zip(("labels", "boxes", "scores"), ort_outs, ov_outs):
        a, b = np.asarray(a), np.asarray(b)
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print("%-7s max_abs_diff=%.6g mean_abs_diff=%.6g" % (n, d.max(), d.mean()))

    s = np.asarray(ort_outs[2]).reshape(-1)
    print("detections@0.5:", int((s > 0.5).sum()),
          "top5:", [round(float(x), 4) for x in sorted(s, reverse=True)[:5]])
    print("STATIC BENCH DONE")


if __name__ == "__main__":
    main()
