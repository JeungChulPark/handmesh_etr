import yaml
import os
import torch
import numpy as np
import argparse
import time
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

from models.mobrecon_ds import LargeModel_Extra
#from datasets.freihand_ty import Freihand

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image


def convert(model, onnx_save_path, tflite_save_path):

    # convert to .onnx
    input_image = torch.ones((1, 3, 256, 256))
    b = model(input_image)

    torch.onnx.export(
        model,
        input_image,
        onnx_save_path,
        export_params=True,
        verbose=False,
        opset_version=11,
        input_names=["input0"],
        output_names=["output0"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # convert to .pb
    onnx_model = onnx.load(onnx_save_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("model_tf")

    # convert to .tflite
    converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # 기본 TFLite 연산
        tf.lite.OpsSet.SELECT_TF_OPS,  # Flex ops 활성화
    ]
    tflite_model = converter.convert()

    # 저장
    with open(tflite_save_path, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    model = LargeModel_Extra(None)
    pth_path = "pretrain/100.pt"
    model.load_state_dict(torch.load(pth_path), strict=False)
    model.eval()

    onnx_save_path = "pretrain/100.onnx"
    tflite_save_path = "pretrain/img256_extra5.tflite"

    convert(model, onnx_save_path, tflite_save_path)
