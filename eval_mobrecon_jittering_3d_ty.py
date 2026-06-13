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

import onnx

from models.mobrecon_ds import LargeModel_Extra
from datasets.freihand_ty import Freihand

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image


from utils import *


def cam2pixel_torch(joints, K):
    x = joints[..., 0] / joints[..., 2] * K[:, 0, 0].unsqueeze(1) + K[:, 0, 2].unsqueeze(1)
    y = joints[..., 1] / joints[..., 2] * K[:, 1, 1].unsqueeze(1) + K[:, 1, 2].unsqueeze(1)
    return torch.stack((x, y), 2)

def to_onnx(model, onnx_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # convert to .onnx
    input_image = torch.ones((1, 3, 256, 256)).to(device)
    
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

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current cuda device ", torch.cuda.device_count())

    result_image_dir = "result_images"
    os.makedirs(result_image_dir, exist_ok=True)

    # checkpoint
    ckpt_dir = "pretrain"
    # ckpt_dir = os.path.join(ckpt_dir, 'etri_new_batch_50_epoch100')
    ckpt = torch.load(os.path.join(ckpt_dir, "100.pt"))
    model = LargeModel_Extra(None)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)

    to_onnx(model, "pretrain/200.onnx")
    
    # dataset
    eval_dataset = Freihand(mode="eval")
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)

    test_epoch_mpjpe = 0
    image_idx = 0
    with torch.no_grad():
        for i, item_dict in enumerate(tqdm(eval_dataloader)):
            img = item_dict["img"].float().to(device)
            original_image = item_dict["ori_image"]
            kps3d = item_dict["align_joint"].float().to(device)  # normalize in [0, 1]
            bs = img.shape[0]
            output_dict = model(
                img,
            )  # iter_loss = iter_loss.item()

            pred_xy = (
                cam2pixel_torch(
                    output_dict["keypoints"] + item_dict["root"][:, None, :].float().to(device),
                    item_dict["cam"].float().to(device),
                )
                / 224.0
            )

            # keypoints_2d = projectPoints(output_dict["keypoints"].detach().cpu()[0], item_dict["K"][0])
            # root_xy = item_dict["xy"][0][0].numpy()
            # keypoints_2d_absolute = keypoints_2d + root_xy
            # print(keypoints_2d_absolute)

            for i in range(len(output_dict["keypoints"])):
                pred_2d_vis = draw_joint2D(original_image, pred_xy, idx=i)
                pred_2d_vis = to_pil_image(pred_2d_vis)

                pred_2d_vis.save(result_image_dir + f"/{image_idx}.png")
                image_idx += 1

            mpjpe = torch.sqrt(((output_dict["keypoints"] - kps3d) ** 2).sum(dim=-1)).mean() * 1000.0
            test_epoch_mpjpe += mpjpe * bs

    print(f"MPJPE: {test_epoch_mpjpe / len(eval_dataset)}")


if __name__ == "__main__":
    main()
