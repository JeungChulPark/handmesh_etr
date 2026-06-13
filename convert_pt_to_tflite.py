import torch
import onnx
import tensorflow as tf
from models.mobrecon_ds import LargeModel_Extra

import argparse, os, shutil, subprocess, sys

def convert(model, onnx_save_path, tflite_save_path):
    # input_image = torch.ones(1, 3, 256, 256)  # 입력 크기에 맞추세요
    # b = model(input_image)

    # torch.onnx.export(
    #     model, input_image, onnx_save_path,
    #     opset_version=17,  # 16~19 권장
    #     input_names=["input0"], output_names=["output0"],
    #     dynamic_axes={"input0": {0: "batch"}, "output0": {0: "batch"}},
    #     do_constant_folding=True,
    # )
    # print("Saved: model.onnx")
    
    onnx_model = onnx.load(onnx_save_path)

def run(cmd: list):
    print(">", " ".join(cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("[STDOUT]")
    print(p.stdout)
    print("[STDERR]")
    print(p.stderr)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {p.returncode}\n"
            f"STDERR:\n{p.stderr}"
        )

def check_onnx(onnx_path: str):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"✓ ONNX check passed: {onnx_path}")
    
def simplify_onnx(onnx_path: str, sim_path: str):
    cmd = [
        sys.executable, "-m", "onnxsim",
        onnx_path, sim_path,
    ]
    run(cmd)

def make_savedmodel(onnx_path: str, saved_dir: str):
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    # onnx2tf CLI 호출 (Python API보다 CLI가 Windows에서 가장 안정적)
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", saved_dir,
        "--output_signaturedefs",
    ]
    run(cmd)

def to_tflite(saved_dir: str, tflite_path: str, fp16=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    if fp16:
        converter.target_spec.supported_types = [tf.float16]

    tflite = converter.convert()
    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite)
    print(f"✓ TFLite saved: {tflite_path}")
    
def main():
    onnx_path = "pretrain/100.onnx"
    sim_onnx_path = "pretrain/100_sim.onnx"
    saved_dir = "pretrain/model_tf"
    tflite_path = "pretrain/img256_extra3.tflite"

    check_onnx(onnx_path)
    simplify_onnx(onnx_path, sim_onnx_path)
    check_onnx(sim_onnx_path)

    make_savedmodel(sim_onnx_path, saved_dir)
    to_tflite(saved_dir, tflite_path, fp16=False)

    print("✅ All done.")
    
if __name__ == "__main__":
    # model = LargeModel_Extra(None)
    # pth_path = "pretrain/100.pt"
    # model.load_state_dict(torch.load(pth_path), strict=False)
    # model.eval()
    
    # onnx_save_path = "pretrain/100.onnx"
    # tflite_save_path = "pretrain/img256_extra3.tflite"

    # convert(model, onnx_save_path, tflite_save_path)
    main()
    