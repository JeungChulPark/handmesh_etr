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

from models.mobrecon_ds import LargeModel, LargeModel_Prev, LargeModel_Extra
from datasets.hanco_ty import HanCo_ETRI_jitter

from torch.utils.data import DataLoader, random_split

# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

import wandb

from utils import *


def cam2pixel_torch(joints, K):
    x = joints[..., 0] / joints[..., 2] * K[:, 0, 0].unsqueeze(1) + K[:, 0, 2].unsqueeze(1)
    y = joints[..., 1] / joints[..., 2] * K[:, 1, 1].unsqueeze(1) + K[:, 1, 2].unsqueeze(1)
    return torch.stack((x, y), 2)


def main(args, log_every=500):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current cuda device ", torch.cuda.device_count())

    # checkpoint
    ckpt_dir = "mobrecon_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_dir, args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)

    # tensorboard
    log_dir = "tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, args.exp)
    os.makedirs(log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir)
    # print("Tensorboard log dir: {log_dir}")

    # limit: No. of samples that will be loaded during execution
    train_dataset = HanCo_ETRI_jitter(limit=5e6)  # 5e6

    len_dataset = int(len(train_dataset) * 0.95)  # 300,000 * 0.2 => 60000
    left_len = len(train_dataset) - len_dataset
    train_dataset, test_dataset = random_split(train_dataset, [len_dataset, left_len])
    test_dataset.mode = "test"  # change mode

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False
    )

    model = LargeModel_Extra(cfg)
    model.to(device)

    l2_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), amsgrad=False, eps=0.00000001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.9)

    train_epoch = range(0, args.epoch)
    for epoch in tqdm(train_epoch, leave=True, position=1):
        iter = 0
        epoch_loss = 0
        epoch_mpjpe = 0

        model.train()

        for i, item_dict in enumerate(tqdm(train_dataloader, leave=False, position=2)):
            steps = len(train_dataloader) * epoch + i
            iter_loss = 0.0
            optimizer.zero_grad()

            img = item_dict["image"].float().to(device)
            kps3d = item_dict["keypoints3D"].float().to(device)  # normalize in [0, 1]
            xy = item_dict["keypoints2D"].float().to(device)  # normalize in [0, 1]

            output_dict = model(img)  # iter_loss = iter_loss.item()

            # 3D loss
            loss_3d = l2_loss(output_dict["keypoints"], kps3d)

            # 2D projection loss
            pred_xy = (
                cam2pixel_torch(
                    output_dict["keypoints"] + item_dict["root"][:, None, :].float().to(device),
                    item_dict["cam"].float().to(device),
                )
                / 256.0
            )
            loss_2d = l2_loss(pred_xy, xy)

            iter_loss = loss_3d + loss_2d
            iter_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += iter_loss.item()
            mpjpe = torch.sqrt(((output_dict["keypoints"] - kps3d) ** 2).sum(dim=-1)).mean() * 1000.0
            epoch_mpjpe += mpjpe

            iter += 1

            if steps % log_every == 0:
                log_dict = {
                    "loss": iter_loss.item(),
                    "loss_3d": loss_3d.item(),
                    "loss_2d": loss_2d.item(),
                    "mpjpe": mpjpe,
                }

                # Visualize
                gt_2d = draw_joint2D(img, xy.detach().cpu(), idx=0)
                pred_2d_vis = draw_joint2D(img, pred_xy.detach().cpu(), idx=0)

                log_dict["gt"] = wandb.Image(to_pil_image(gt_2d))
                log_dict["pred_25d"] = wandb.Image(to_pil_image(pred_2d_vis))

                wandb.log(log_dict, step=steps)

        log_dict = {"train_loss": epoch_loss / iter, "train_mpjpe": epoch_mpjpe / iter}
        wandb.log(log_dict, step=steps)

        if (epoch + 1) % 2 == 0:
            save_dir = os.path.join(ckpt_dir, f"{epoch+1}.pt")
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, save_dir)

        test_iter = 0
        test_epoch_mpjpe = 0

        model.eval()
        with torch.no_grad():
            for i, item_dict in enumerate(tqdm(test_dataloader, leave=False, position=2)):
                img = item_dict["image"].float().to(device)
                kps3d = item_dict["keypoints3D"].float().to(device)  # normalize in [0, 1]
                output_dict = model(
                    img,
                )  # iter_loss = iter_loss.item()
                mpjpe = torch.sqrt(((output_dict["keypoints"] - kps3d) ** 2).sum(dim=-1)).mean() * 1000.0
                test_epoch_mpjpe += mpjpe

                test_iter += 1

        log_dict = {"test_mpjpe": test_epoch_mpjpe / test_iter}

        # Visualize
        xy = item_dict["keypoints2D"].float()
        pred_xy = (
            cam2pixel_torch(
                output_dict["keypoints"] + item_dict["root"][:, None, :].float().to(device),
                item_dict["cam"].float().to(device),
            )
            / 256.0
        )
        gt_2d = draw_joint2D(img, xy, idx=0)
        pred_2d_vis = draw_joint2D(img, pred_xy.detach().cpu(), idx=0)

        log_dict["test_gt"] = wandb.Image(to_pil_image(gt_2d))
        log_dict["test_pred_25d"] = wandb.Image(to_pil_image(pred_2d_vis))

        wandb.log(log_dict, step=steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="Save experiement name")
    parser.add_argument("--batch", default=250, type=int, dest="batch_size")
    parser.add_argument("--epoch", default=200, type=int, dest="epoch")
    parser.add_argument("--cfg", default="./configs.yaml")

    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    #    device = setup_runtime(args)

    run = wandb.init(project="OXR_mobrecon", name=args.exp, job_type="train")
    wandb.run.log_code(
        root=".",
        include_fn=lambda p: any(
            p.endswith(ext) for ext in (".py", ".json", ".yaml", ".md", ".txt.", ".gin")
        ),
        exclude_fn=lambda p: any(s in p for s in ("output", "tmp", "wandb", ".git", ".vscode")),
    )

    main(args)  # throw exp name

    wandb.finish()
    time.sleep(3)
