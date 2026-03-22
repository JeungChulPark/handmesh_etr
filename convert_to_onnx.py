from __future__ import print_function
import argparse
import torch
import torch.nn as nn

# handmesh 모델 import
from models.mobrecon_ds import LargeModel_Extra
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


class LandmarkWrapper(nn.Module):
    """
    Export wrapper for ONNX.
    - Uses dict['keypoints']
    - keypoints shape: [B, 21, 3]
    - xy are normalized using hand bbox
    - z is kept raw
    """
    def __init__(self, model, target_scale=0.35):
        super().__init__()
        self.model = model
        self.target_scale = target_scale

    def forward(self, x):
        y = self.model(x)

        if isinstance(y, dict):
            y = y["keypoints"]
        elif isinstance(y, (tuple, list)):
            y = y[0]

        if not torch.is_tensor(y):
            raise TypeError(f"Unsupported model output type: {type(y)}")

        raw_xy = y[..., :2]   # [B,21,2]
        raw_z  = y[..., 2:]   # [B,21,1]

        min_xy = raw_xy.amin(dim=1, keepdim=True)   # [B,1,2]
        max_xy = raw_xy.amax(dim=1, keepdim=True)   # [B,1,2]

        center_xy = (min_xy + max_xy) * 0.5
        range_xy = (max_xy - min_xy).amax(dim=-1, keepdim=True)  # [B,1,1]
        range_xy = torch.clamp(range_xy, min=1e-6)

        xy = (raw_xy - center_xy) / range_xy * self.target_scale + 0.5
        xy = torch.clamp(xy, 0.0, 1.0)

        out = torch.cat([xy, raw_z], dim=-1)
        return out


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # -----------------------------
    # model
    # -----------------------------
    net = LargeModel_Extra(None)
    # net = handmesh_etr()

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    print('Finished loading model!')
    print(net)

    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # -----------------------------
    # debug original model output
    # -----------------------------
    with torch.no_grad():
        test_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        output = net(test_input)

        print("\n===== MODEL OUTPUT DEBUG =====")
        if isinstance(output, dict):
            print("Output type: dict")
            for k, v in output.items():
                if torch.is_tensor(v):
                    print(f"[{k}] shape={tuple(v.shape)}, min={float(v.min())}, max={float(v.max())}")
                else:
                    print(f"[{k}] type={type(v)}")
        elif isinstance(output, (list, tuple)):
            print("Output type: list/tuple")
            for i, v in enumerate(output):
                if torch.is_tensor(v):
                    print(f"[{i}] shape={tuple(v.shape)}, min={float(v.min())}, max={float(v.max())}")
                else:
                    print(f"[{i}] type={type(v)}")
        elif torch.is_tensor(output):
            print("Output type: tensor")
            print(f"shape={tuple(output.shape)}, min={float(output.min())}, max={float(output.max())}")
        else:
            print("Unknown output type:", type(output))
        print("================================\n")

    # -----------------------------
    # wrap model for export
    # -----------------------------
    export_model = LandmarkWrapper(net, target_scale=0.5).to(device)
    export_model.eval()
    print('Using LandmarkWrapper: output = keypoints, xy=sigmoid(xy), z=raw')

    # -----------------------------
    # export ONNX
    # -----------------------------
    output_onnx = args.output_onnx
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))

    input_names = ["input0"]
    output_names = ["Identity"]

    inputs = torch.randn(1, 3, args.input_size, args.input_size).to(device)

    with torch.no_grad():
        test_out = export_model(inputs)
        print("Debug output shape:", tuple(test_out.shape))
        print("Debug output min:", float(test_out.min()))
        print("Debug output max:", float(test_out.max()))

    torch.onnx.export(
        export_model,
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