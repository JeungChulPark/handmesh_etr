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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}")
    
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

def to_tflite(saved_dir: str, tflite_path: str, fp16=False, int8=False, rep_dir=None):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    # 기본 최적화
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        if not rep_dir:
            raise ValueError("--int8 를 쓰려면 --rep-dir 로 대표 데이터 폴더를 지정하세요.")
        import numpy as np
        from PIL import Image

        def representative_dataset():
            imgs = [os.path.join(rep_dir, f) for f in os.listdir(rep_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            # 100~200장 정도면 충분 (필요시 조정)
            for p in imgs[:200]:
                img = Image.open(p).convert("RGB")
                arr = np.asarray(img, dtype=np.float32)
                # 필요한 전처리(정규화/리사이즈/채널순서) 여기에 맞추세요
                arr = arr / 255.0
                arr = arr[None, ...]  # [1,H,W,C] 가정. 모델 입력에 맞게 수정
                yield [arr]

        converter.representative_dataset = representative_dataset
        # INT8 전용 Op만 쓰고 싶으면 아래 주석 해제
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
    else:
        # FP32/FP16 경로에서는 Flex-ops 허용해 변환 성공률↑
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
    onnx_path = "pretrain/100.onnx"             # ONNX 입력
    saved_dir = "pretrain/model_tf"             # SavedModel 출력
    tflite_path = "pretrain/img256_extra3.tflite" # TFLite 출력

    make_savedmodel(onnx_path, saved_dir)
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
    