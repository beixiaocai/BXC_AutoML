# RK3576 板端 CPU：静态 ONNX(onnxruntime) vs 静态(openvino, 板端从静态 onnx 直接编译) 对比
# 与 PC 端 dfine_static_bench.py 同口径：warmup 10 + 100 次计时
import paramiko

HOST, USER, PWD = "192.168.1.15", "root", "sunshine"
ONNX = r"D:\project\ytstudio\Train_DFINE\models\dfine_n_coco_static.onnx"

script = r'''
import statistics, time
import numpy as np, cv2

STATIC_ONNX = "/tmp/rk_bench/dfine_n_coco_static.onnx"
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
    print("%-28s mean=%.2fms median=%.2f min=%.2f max=%.2f std=%.2f fps=%.1f" % (
        name, statistics.mean(ts), statistics.median(ts), min(ts), max(ts),
        statistics.pstdev(ts), 1000 / statistics.mean(ts)))

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

import openvino as ov
core = ov.Core()
model = core.compile_model(STATIC_ONNX, "CPU")
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
'''

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username=USER, password=PWD, timeout=15)
sftp = cli.open_sftp()
sftp.put(ONNX, "/tmp/rk_bench/dfine_n_coco_static.onnx")
with sftp.open("/tmp/rk_bench/dfine_static_bench.py", "w") as f:
    f.write(script)
sftp.close()
print("uploaded")

si, so, se = cli.exec_command(
    "cd /tmp/rk_bench && timeout 2400 ./venv/bin/python dfine_static_bench.py 2>&1 | "
    "grep -vE 'Warning|pkg_resources|it/s|^\\|' | tail -30",
    timeout=2430)
print((so.read() + se.read()).decode(errors="replace"))
cli.close()
