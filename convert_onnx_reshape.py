import onnx
from onnx import numpy_helper

model = onnx.load("pretrain/100_opt.onnx")

print("=== Reshape nodes ===")
for node in model.graph.node:
    if node.op_type == "Reshape":
        print("name:", node.name)
        print("inputs:", list(node.input))
        print("outputs:", list(node.output))
        print()

print("=== Initializers related to senet1 reshape ===")
for init in model.graph.initializer:
    if "senet1" in init.name or "1727" in init.name or "Constant_output_0" in init.name:
        arr = numpy_helper.to_array(init)
        print(init.name, arr, arr.shape)