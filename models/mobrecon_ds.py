# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
* @file mobrecon_ds.py
* @author chenxingyu (chenxy.sean@gmail.com)
* @brief MobRecon model
* @version 0.1
* @date 2022-04-28
*
* @copyright Copyright (c) 2022 chenxingyu
*
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch.nn as nn
from models.densestack import *
from models.modules import Joint3DDecoder, joint3DDecoder_additional

import torchvision.transforms as T


class LargeModel(nn.Module):
    def __init__(self, cfg):
        super(LargeModel, self).__init__()
        self.cfg = cfg
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.backbone = DenseStack_Backbone_super_large(latent_size=self.latent_size, kpts_num=21)
        self.resizer = T.Resize(cfg.IMG_SIZE)

    def forward(self, x):
        x = self.resizer(x)

        pred25d = self.backbone(x)

        return {"keypoints3D": pred25d}


class LargeModel_onnx(nn.Module):
    def __init__(self, cfg):
        super(LargeModel_onnx, self).__init__()
        self.cfg = cfg
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.backbone = DenseStack_Backbone_super_large(latent_size=self.latent_size, kpts_num=21)

    def forward(self, x):
        pred25d = self.backbone(x)

        return pred25d


class LargeModel_Prev(nn.Module):
    def __init__(self, cfg):
        super(LargeModel_Prev, self).__init__()
        self.cfg = cfg
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.backbone = DenseStack_Backbone_prev(latent_size=self.latent_size, kpts_num=21)
        self.resizer = T.Resize(cfg.IMG_SIZE)

    def add_jit(self, joint, scale=0.1):
        B, _ = joint.size()
        noise_per_part = scale * (torch.rand(B, 6).cuda() - 0.5)

        # Add root noise
        joint[0] = noise_per_part[0]

        parts_info = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
        for idx, part in enumerate(["thumb", "index", "middle", "ring", "little"]):
            parts = parts_info[idx]

            noise_part = noise_per_part[idx + 1]
            noise_parts = [noise_part * 0.1, noise_part * 0.2, noise_part * 0.5, noise_part * 0.8]
            parts = joint[parts] + noise_parts

        return joint

    def forward(self, x, prev_joints=None):
        x = self.resizer(x)

        pred25d = self.backbone(x, prev_joints)

        return {"keypoints3D": pred25d}


class LargeModel_Prev_onnx(nn.Module):
    def __init__(self, cfg):
        super(LargeModel_Prev_onnx, self).__init__()
        self.cfg = cfg
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.backbone = DenseStack_Backbone_prev(latent_size=self.latent_size, kpts_num=21)

    def add_jit(self, joint, scale=0.1):
        B, _ = joint.size()
        noise_per_part = scale * (torch.rand(B, 6).cuda() - 0.5)

        # Add root noise
        joint[0] = noise_per_part[0]

        parts_info = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
        for idx, part in enumerate(["thumb", "index", "middle", "ring", "little"]):
            parts = parts_info[idx]

            noise_part = noise_per_part[idx + 1]
            noise_parts = [noise_part * 0.1, noise_part * 0.2, noise_part * 0.5, noise_part * 0.8]
            parts = joint[parts] + noise_parts

        return joint

    def forward(self, x, prev_joints=None):
        pred25d = self.backbone(x, prev_joints)

        return pred25d


class LargeModel_Extra(nn.Module):
    def __init__(self, cfg):
        super(LargeModel_Extra, self).__init__()
        self.cfg = cfg
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.backbone = DenseStack_Backbone_like_prev(latent_size=self.latent_size, kpts_num=21)

    def forward(self, x):

        pred25d = self.backbone(x)

        return {"keypoints": pred25d}


class LargeModel_Extra_onnx(nn.Module):
    def __init__(self):
        super(LargeModel_Extra_onnx, self).__init__()
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.backbone = DenseStack_Backbone_like_prev(latent_size=self.latent_size, kpts_num=21)

    def forward(self, x):

        pred25d = self.backbone(x)

        return pred25d


class LargeModel_Extra_RGBD(nn.Module):
    """RGB-D variant of LargeModel_Extra: early-fusion 4-channel input.

    Identical to LargeModel_Extra except the backbone stem accepts `in_chans`
    channels (RGB + 1 depth = 4). The depth channel is the metric depth produced
    by `datasets/depth_synth.py` (or a real sensor), normalised to ~[-1, 1].

    The number of input channels is read from cfg.MODEL.IN_CHANS (default 4) so
    the same class also serves an RGB-only ablation (IN_CHANS=3).
    """

    def __init__(self, cfg):
        super(LargeModel_Extra_RGBD, self).__init__()
        self.cfg = cfg
        self.latent_size = 1024  # 256 --> 1024   240 --> 900
        self.in_chans = int(cfg.MODEL.get("IN_CHANS", 4)) if hasattr(cfg, "MODEL") else 4
        self.backbone = DenseStack_Backbone_like_prev(
            latent_size=self.latent_size, kpts_num=21, in_chans=self.in_chans
        )

    def forward(self, x):
        pred25d = self.backbone(x)
        return {"keypoints": pred25d}


class LargeModel_Extra_RGBD_onnx(nn.Module):
    """ONNX-export twin of LargeModel_Extra_RGBD (returns the raw tensor)."""

    def __init__(self, in_chans=4):
        super(LargeModel_Extra_RGBD_onnx, self).__init__()
        self.latent_size = 1024
        self.backbone = DenseStack_Backbone_like_prev(
            latent_size=self.latent_size, kpts_num=21, in_chans=in_chans
        )

    def forward(self, x):
        return self.backbone(x)


class SmallModel(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(SmallModel, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Backnone_small(latent_size=1024)  # img size = 224 --> 784  256 --> 1024
        # self.backbone = DenseStack_Backbone_middle(latent_size=1024) # img size = 224 --> 784  256 --> 1024
        self.act = nn.Sigmoid()
        self.resizer = T.Resize(cfg.IMG_SIZE)

    def forward(self, x):
        x = self.resizer(x)

        pred_joint = self.backbone(x)

        return {"keypoints3D": pred_joint}


class MobRecon_DS(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS, self).__init__()
        self.cfg = cfg
        self.latent = 1024
        self.backbone = DenseStack_Backnone_big(latent_size=self.latent, kpts_num=21)

        self.decoder3d = Joint3DDecoder(self.latent, uv_channels=128, out_channels=[128, 256, 512])  # 256
        # self.decoder3d = Joint3DDecoder(self.latent_size, uv_channels=64, out_channels=[64, 128, 256])

        self.resizer = T.Resize(cfg.IMG_SIZE)

    def forward(self, x):
        x = self.resizer(x)

        latent, pred25d = self.backbone(x)
        pred3d = self.decoder3d(pred25d, latent)

        return {
            "keypoints": pred25d,
            "keypoints3D": pred3d,
        }


# middle layer: 2D
# final layer: 2.5D
# additional input: previous joints -> 0-padding or noisy
class MobRecon_DS_additional(nn.Module):
    def __init__(
        self,
    ):
        super(MobRecon_DS_additional, self).__init__()
        self.latent = 1024
        self.backbone = DenseStack_Backnone_big(
            latent_size=self.latent, kpts_num=21, output_dim=2
        )  # output_dim -> out dimension

        self.decoder3d = joint3DDecoder_additional(
            self.latent, uv_channels=128, out_channels=[128, 256, 512]
        )  # 256
        # self.decoder3d = Joint3DDecoder(self.latent_size, uv_channels=64, out_channels=[64, 128, 256]

    def forward(self, x, additional=None):
        latent, pred2d = self.backbone(x)
        pred3d = self.decoder3d(pred2d, latent, additional)

        return {
            "keypoints": pred2d,
            "keypoints3D": pred3d,
        }
