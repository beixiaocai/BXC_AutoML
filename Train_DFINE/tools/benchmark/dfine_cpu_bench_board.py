# RK3576 板端 CPU 推理 D-FINE n 性能对比：onnxruntime vs openvino
# 与 PC 端 dfine_n_bench.py 同方法论：同一张图、同一预处理（等比 resize + 114 底 + /255），
# warmup 10 次后循环 100 次计时，并做双后端精度对拍
import paramiko

HOST, USER, PWD = "192.168.1.15", "root", "sunshine"
ONNX = r"D:\project\ytstudio\Train_DFINE\models\d-fine-onnx-opset17-dyn-fp32\dfine_n_coco_opset17_fp32_dyn.onnx"

script = r'''
import statistics, time
import numpy as np, cv2

ONNX = "/tmp/rk_bench/dfine_n_coco_opset17_fp32_dyn.onnx"
IMG = "/tmp/rk_bench/bus.jpg"
SIZE, WARMUP, ITERS = 640, 10, 100

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
    print("%-12s mean=%.2fms median=%.2f min=%.2f max=%.2f std=%.2f fps=%.1f" % (
        name, statistics.mean(ts), statistics.median(ts), min(ts), max(ts),
        statistics.pstdev(ts), 1000 / statistics.mean(ts)))

blob, target_size = preprocess(IMG)
print("input: images", blob.shape, "orig_target_sizes", target_size.tolist())

# ---- onnxruntime ----
import onnxruntime as ort
print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
feed = {"images": blob, "orig_target_sizes": target_size}
for _ in range(WARMUP):
    ort_outs = sess.run(None, feed)
ts = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    sess.run(None, feed)
    ts.append((time.perf_counter() - t0) * 1000)
stats("onnxruntime", ts)

# ---- openvino ----
import openvino as ov
print("openvino", ov.__version__)
core = ov.Core()
model = core.compile_model(ONNX, "CPU")
req = model.create_infer_request()
inputs = {"images": blob, "orig_target_sizes": target_size}
for _ in range(WARMUP):
    req.infer(inputs)
ts2 = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    req.infer(inputs)
    ts2.append((time.perf_counter() - t0) * 1000)
stats("openvino", ts2)
ov_outs = [req.get_tensor(n).data for n in ("labels", "boxes", "scores")]

# ---- 精度对拍（ort 顺序: labels, boxes, scores）----
for n, a, b in zip(("labels", "boxes", "scores"), ort_outs, ov_outs):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        print("%s shape mismatch %s vs %s" % (n, a.shape, b.shape)); continue
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print("%-7s max_abs_diff=%.6g mean_abs_diff=%.6g" % (n, d.max(), d.mean()))

s = np.asarray(ort_outs[2]).reshape(-1)
print("detections@0.5:", int((s > 0.5).sum()),
      "top5:", [round(float(x), 4) for x in sorted(s, reverse=True)[:5]])
print("CPU BENCH DONE")
'''

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username=USER, password=PWD, timeout=15)
sftp = cli.open_sftp()
sftp.put(ONNX, "/tmp/rk_bench/dfine_n_coco_opset17_fp32_dyn.onnx")
with sftp.open("/tmp/rk_bench/dfine_cpu_bench.py", "w") as f:
    f.write(script)
sftp.close()
print("uploaded")

si, so, se = cli.exec_command(
    "cd /tmp/rk_bench && timeout 1800 ./venv/bin/python dfine_cpu_bench.py 2>&1 | "
    "grep -vE 'Warning|pkg_resources|it/s|^\\|' | tail -30",
    timeout=1830)
print((so.read() + se.read()).decode(errors="replace"))
cli.close()
