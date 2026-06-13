from __future__ import print_function
import os
import argparse
import torch
import onnx

# handmesh 모델 import
from models.mobrecon_ds import LargeModel_Extra
# 만약 실제 클래스명이 handmesh_etr라면 예:
# from models.handmesh_etr import handmesh_etr


parser = argparse.ArgumentParser(description='Export handmesh_etr to ONNX')
parser.add_argument(
    '-m', '--trained_model',
    default='./pretrain/100.pt',
    type=str,
    help='Trained model file path'
)
parser.add_argument(
    '--input_size',
    default=256,
    type=int,
    help='input image size'
)
parser.add_argument(
    '--cpu',
    action='store_true',
    default=True,
    help='Use cpu inference'
)
parser.add_argument(
    '--output_onnx',
    default='./pretrain/handmesh_etr.onnx',
    type=str,
    help='output onnx file path'
)

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys: {}'.format(len(missing_keys)))
    print('Unused checkpoint keys: {}'.format(len(unused_pretrained_keys)))
    print('Used keys: {}'.format(len(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=True):
    print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu:
        ckpt = torch.load(pretrained_path, map_location='cpu')
    else:
        device = torch.cuda.current_device()
        ckpt = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    # checkpoint 구조 대응
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            pretrained_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            pretrained_dict = ckpt['state_dict']
        else:
            pretrained_dict = ckpt
    else:
        pretrained_dict = ckpt

    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)

    model.load_state_dict(pretrained_dict, strict=True)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # -----------------------------
    # model
    # -----------------------------
    # 현재 대화 기준으로는 LargeModel_Extra 사용
    net = LargeModel_Extra(None)

    # 실제 프로젝트 클래스명이 handmesh_etr라면 예:
    # net = handmesh_etr()

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    print('Finished loading model!')
    print(net)

    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # -----------------------------
    # export ONNX
    # -----------------------------
    output_onnx = args.output_onnx
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))

    input_names = ["input0"]
    output_names = ["identity"]

    inputs = torch.randn(1, 3, args.input_size, args.input_size).to(device)

    torch.onnx.export(
        net,
        inputs,
        output_onnx,
        export_params=True,
        verbose=False,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None
    )

    print("ONNX export finished:", output_onnx)