# 分析 D-FINE n ONNX：输入输出、算子统计、NMS 子图、orig_target_sizes 消费链
import onnx
from collections import Counter, deque

MODEL = r"D:\project\ytstudio\Train_DFINE\models\d-fine-onnx-opset17-dyn-fp32\dfine_n_coco_opset17_fp32_dyn.onnx"
m = onnx.load(MODEL)
g = m.graph
print("ir_version:", m.ir_version, "opset:", [(o.domain, o.version) for o in m.opset_import])
print("inputs:")
for i in g.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
    print("  ", i.name, i.type.tensor_type.elem_type, dims)
print("outputs:")
for o in g.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
    print("  ", o.name, o.type.tensor_type.elem_type, dims)

ops = Counter(n.op_type for n in g.node)
print("node count:", len(g.node))
print("op stats:", dict(ops.most_common()))

# 关注的后处理算子
special = [n for n in g.node if n.op_type in (
    "NonMaxSuppression", "TopK", "Softmax", "Gather", "ScatterND", "Resize",
    "PiecewiseAgreementCheck", "BatchMulticlassNMS")]
print("special nodes:", [(n.op_type, n.name) for n in special][:30])

# orig_target_sizes 消费链
def consumers(graph):
    cons = {}
    for n in graph.node:
        for t in n.input:
            cons.setdefault(t, []).append(n)
    return cons

cons = consumers(g)

def trace(start):
    seen, q = set(), deque()
    for n in cons.get(start, []):
        q.append(n)
    while q:
        n = q.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        for t in n.output:
            for c in cons.get(t, []):
                q.append(c)
    return [n for n in g.node if id(n) in seen]

ots = trace("orig_target_sizes")
print("orig_target_sizes downstream nodes:", len(ots))
print(Counter(n.op_type for n in ots))
for n in ots:
    print("  ", n.op_type, n.name, "<-", list(n.input)[:3])

# 反向：输出 boxes/scores 的生产链规模
producers = {}
for n in g.node:
    for t in n.output:
        producers[t] = n

def upstream(start):
    seen, q = set(), deque()
    n = producers.get(start)
    if n:
        q.append(n)
    while q:
        n = q.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        for t in n.input:
            p = producers.get(t)
            if p:
                q.append(p)
    return [n for n in g.node if id(n) in seen]

for out in ("labels", "boxes", "scores"):
    if out in producers or any(t == out for n in g.node for t in n.output):
        up = upstream(out)
        print(f"output {out}: upstream {len(up)} nodes, ops={dict(Counter(n.op_type for n in up).most_common(10))}")
