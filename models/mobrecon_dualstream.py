# Copyright (c) 2026. Dual-stream RGB-D hand pose (depth-native redesign).

"""Dual-Stream RGB-D MobRecon — depth-native architecture.

This is a SEPARATE model from the early-fusion `LargeModel_Extra_RGBD`
(`models/mobrecon_ds.py`), deliberately kept apart so the two can be compared
head-to-head. Nothing in `mobrecon_ds.py` / `densestack.py` is modified.

Motivation (see docs/RGBD_DUAL_STREAM_DESIGN.md): the early-fusion model predicts
only root-relative joints and recovers the absolute hand position from a single
wrist-pixel depth backprojection *outside* the network. Monocular RGB is
position/scale-ambiguous, so global translation must come from depth — and it
should be *learned* with supervision, not a fragile heuristic.

Architecture:

    RGB  [B,3,H,W] ─► RGBPoseBackbone (DenseStack, unchanged) ─► 21x3 rel joints
                                                              └─► pose latent (GAP)
    Depth[B,1,H,W] ─► DepthEncoder (DW-sep convs) ─► depth descriptor (GAP)
       + depth_med (raw-depth median, abs distance)   ┐
       + gate (fraction of valid depth pixels)        ┴─► TranslationHead ─► root(x,y,z), scale s

    absolute joints = s * rel_joints + root          (learned; replaces heuristic)

All ops are depthwise-separable conv + linear (same family as the backbone) so
the Core ML / iPhone export path carries no new conversion risk.
"""

import torch
import torch.nn as nn

from models.densestack import DenseStack_Backbone_like_prev
from models.modules import conv_layer, mobile_unit


class RGBPoseBackbone(DenseStack_Backbone_like_prev):
    """DenseStack backbone that also returns a pooled pose latent.

    Subclasses the existing backbone (no edit to densestack.py) and re-implements
    `forward` so it returns `(rel_joints[B,21,3], pose_latent[B,latent_size])`.
    The body is identical to the parent's forward up to the final joint head; the
    only addition is the global-average-pooled `latent` used by the TranslationHead.
    """

    def forward(self, x):
        pre_out = self.pre_layer(x)
        pre_out_reorg = self.reorg(pre_out)
        thrink = self.thrink(pre_out_reorg)

        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)

        latent = self.mid_proj(stack2_mid)  # [B, latent_size, h, w]
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        x = self.de_layer_conv(latent)
        x = x.view(x.shape[0], x.shape[1], -1)
        uv_reg = self.uv_linear(uv_reg)

        x = self.cat_conv_final(x)
        x = self.cat_act(x)
        for conv, act, conv2, norm in self.final_conv:
            x = conv(x)
            x = act(x)
            x = conv2(x)
            x = norm(x)

        x = self.latent_linear(x).permute(0, 2, 1)
        x = x + uv_reg
        for layer in self.prev_linear:
            res_x = x
            x = layer(x) + res_x
        rel_joints = self.final_linear1(x)  # [B, 21, 3] root-relative joints

        pose_latent = latent.mean(dim=(2, 3))  # [B, latent_size]
        return rel_joints, pose_latent


class DepthEncoder(nn.Module):
    """Lightweight depthwise-separable encoder for the (low-res, noisy) depth map.

    Kept tiny (~0.1M params) and separate from the RGB stream so the sensor-
    specific depth domain (ZED stereo / iPhone LiDAR) does not perturb the RGB
    backbone. Returns a global-average-pooled descriptor vector.
    """

    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            conv_layer(1, 16, ks=3, stride=2, padding=1),   # H -> H/2
            mobile_unit(16, 32),
            conv_layer(32, 32, ks=3, stride=2, padding=1),  # H/2 -> H/4
            mobile_unit(32, 64),
            conv_layer(64, 64, ks=3, stride=2, padding=1),  # H/4 -> H/8
            mobile_unit(64, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, depth):           # depth: [B, 1, H, W]
        f = self.net(depth)
        return f.mean(dim=(2, 3))        # [B, out_dim]


class TranslationHead(nn.Module):
    """Predict global root translation (x,y,z) and a metric scale s.

    Inputs:
        pose_latent : [B, P]  RGB pose latent (lateral cue + RGB-prior fallback)
        depth_desc  : [B, D]  depth descriptor (hand shape/extent in depth)
        depth_med   : [B, 1]  raw-depth median over the hand region = absolute
                              camera distance. This is the ONLY absolute-distance
                              signal (the depth channel fed to the net is median-
                              centred and carries no absolute z). At inference it
                              is the median of the raw sensor depth.
        gate        : [B, 1]  fraction of valid depth pixels in [0,1]; lets the
                              head fall back to an RGB-conditioned prior when depth
                              drops out (LiDAR holes / out of range).

    Outputs:
        root  : [B, 3]
        scale : [B, 1]  centred at 1.0 (s = 1 + 0.2*tanh(.)), a gentle correction.
    """

    def __init__(self, pose_dim, depth_dim, hidden=256):
        super().__init__()
        in_dim = pose_dim + depth_dim + 2  # + depth_med + gate
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.root_head = nn.Linear(hidden, 3)
        self.scale_head = nn.Linear(hidden, 1)
        # Init the scale branch near zero so s starts ~1.0 (identity).
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, pose_latent, depth_desc, depth_med, gate):
        h = torch.cat([pose_latent, depth_desc, depth_med, gate], dim=1)
        h = self.mlp(h)
        root = self.root_head(h)
        scale = 1.0 + 0.2 * torch.tanh(self.scale_head(h))
        return root, scale


class MobRecon_DualStream(nn.Module):
    """Dual-stream RGB-D hand pose model.

    Accepts the SAME 4-channel `[B,4,H,W]` tensor the early-fusion dataset emits
    (RGB on [0:3], median-centred depth on [3]) so both models train on identical
    data for a fair comparison; the channels are split internally. `depth_med`
    (raw-depth median) is passed separately because the normalised depth channel
    deliberately discards absolute distance.
    """

    def __init__(self, cfg=None, latent_size=1024, depth_dim=128, pose_in_chans=3):
        super().__init__()
        self.cfg = cfg
        self.latent_size = latent_size
        # pose_in_chans=3 -> RGB-only pose branch (original dual-stream).
        # pose_in_chans=4 -> HYBRID: the pose backbone ALSO sees depth (early
        #   fusion like the baseline), so depth aids pose, not just translation.
        #   Depth is still encoded separately for the TranslationHead.
        self.pose_in_chans = pose_in_chans
        self.rgb_backbone = RGBPoseBackbone(latent_size=latent_size, kpts_num=21, in_chans=pose_in_chans)
        self.depth_encoder = DepthEncoder(out_dim=depth_dim)
        self.translation_head = TranslationHead(pose_dim=latent_size, depth_dim=depth_dim)

    def forward(self, img, depth_med=None):
        depth = img[:, 3:4]
        pose_in = img if self.pose_in_chans == 4 else img[:, :3]

        rel_joints, pose_latent = self.rgb_backbone(pose_in)   # [B,21,3], [B,P]
        depth_desc = self.depth_encoder(depth)             # [B,D]

        # gate = fraction of non-background depth pixels (background == 0).
        gate = (depth.abs() > 1e-6).float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        if depth_med is None:
            depth_med = torch.zeros(img.shape[0], 1, device=img.device, dtype=img.dtype)
        elif depth_med.dim() == 1:
            depth_med = depth_med.unsqueeze(1)

        root, scale = self.translation_head(pose_latent, depth_desc, depth_med, gate)
        abs_joints = scale.unsqueeze(1) * rel_joints + root.unsqueeze(1)

        return {
            "keypoints": rel_joints,        # root-relative (same key as baseline)
            "root": root,                   # [B,3] learned global translation
            "scale": scale,                 # [B,1] metric scale
            "keypoints_abs": abs_joints,    # [B,21,3] absolute camera-space joints
        }


class MobRecon_DualStream_onnx(MobRecon_DualStream):
    """Export twin: returns raw tensors (rel, root, scale, abs) instead of a dict."""

    def forward(self, img, depth_med=None):
        out = super().forward(img, depth_med)
        return out["keypoints"], out["root"], out["scale"], out["keypoints_abs"]
