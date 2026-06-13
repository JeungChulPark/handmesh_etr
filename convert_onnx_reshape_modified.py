import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference

SRC = "pretrain/100_opt.onnx"
DST = "pretrain/100_senete_fixed.onnx"

model = onnx.load(SRC)
model = shape_inference.infer_shapes(model)

# value_info / input / output shape map 만들기
shape_map = {}

def read_shape(v):
    try:
        dims = []
        for d in v.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append(None)
        return dims
    except Exception:
        return None

for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
    shape_map[v.name] = read_shape(v)

new_initializers = []

for node in model.graph.node:
    if node.op_type != "Reshape":
        continue
    if "/senet" not in node.name:
        continue

    input_name = node.input[0]
    in_shape = shape_map.get(input_name, None)

    if not in_shape or len(in_shape) != 4:
        print(f"skip {node.name}: no 4D inferred shape for {input_name}, got {in_shape}")
        continue

    n, c, h, w = in_shape
    if n is None:
        n = 1
    if c is None:
        raise RuntimeError(f"Cannot infer channel dim for {node.name}: {in_shape}")

    target = np.array([n, c], dtype=np.int64)
    new_const_name = node.name + "_shape_const"

    init = numpy_helper.from_array(target, name=new_const_name)
    new_initializers.append(init)

    old_shape_input = node.input[1]
    node.input[1] = new_const_name

    print(f"patched {node.name}: {old_shape_input} -> {new_const_name}, target={target.tolist()}")

model.graph.initializer.extend(new_initializers)

onnx.checker.check_model(model)
onnx.save(model, DST)
print("saved:", DST)