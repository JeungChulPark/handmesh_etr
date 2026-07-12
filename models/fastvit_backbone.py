"""FastViT hand-pose backbone — prototype drop-in for LargeModel_Extra.

Swaps the DenseStack encoder for Apple's FastViT (timm), keeping the exact I/O
contract of the MobRecon backbone so it plugs into the existing train/eval/ONNX
pipeline:

    input : crop [B, in_chans, 256, 256]   (3 = RGB, 4 = RGB-D early fusion)
    output: {"keypoints": [B, 21, 3]}        wrist-relative 3D (== align_joint space)

FastViT = RepMixer conv-mixer stages + a final self-attention stage, structurally
reparameterised at inference (pure conv, Core ML / iPhone friendly). Encoder is
pretrained on ImageNet; a light MLP head regresses the 21x3 joints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FastViTHandBackbone(nn.Module):
    def __init__(self, variant="fastvit_t8", in_chans=3, n_joints=21,
                 pretrained=True, hidden=512):
        super().__init__()
        self.in_chans = in_chans
        self.n = n_joints
        self.encoder = timm.create_model(
            variant, pretrained=pretrained, num_classes=0, global_pool="avg",
            in_chans=in_chans)
        d = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, n_joints * 3))
        # ImageNet normalisation for the RGB channels (FastViT pretrained expects it).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.in_chans == 3:
            x = (x - self.mean) / self.std
        elif self.in_chans == 4:                       # RGB-D: norm RGB, leave depth
            rgb = (x[:, :3] - self.mean) / self.std
            x = torch.cat([rgb, x[:, 3:4]], dim=1)
        feat = self.encoder(x)                          # [B, d]
        p = self.head(feat).view(-1, self.n, 3)         # [B, 21, 3]
        return {"keypoints": p}


class FastViTHandSoftArgmax(nn.Module):
    """FastViT encoder + a 2.5D soft-argmax head (replaces the pooled-MLP head).

    Each joint gets its OWN spatial heatmap; soft-argmax localises it (u,v) and the
    heatmap attention-pools a per-joint feature. A shared per-joint MLP maps
    [feature, u, v] -> metric wrist-relative 3D. This is far more sample-efficient
    and stable than regressing 21x3 from one global-pooled vector (which overfit).

    ONNX-friendly: conv, softmax, weighted-sum, linear only.
    """
    def __init__(self, variant="fastvit_t8", in_chans=3, n_joints=21,
                 pretrained=True, feat_dim=128):
        super().__init__()
        self.in_chans = in_chans
        self.n = n_joints
        self.encoder = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(1, 3),
            in_chans=in_chans)
        # channels of the two SELECTED maps (dummy forward — feature_info lists all stages)
        with torch.no_grad():
            ex = self.encoder(torch.zeros(1, in_chans, 256, 256))
        chs = [f.shape[1] for f in ex]                           # e.g. [96, 384]
        # fuse hi-res (32x32) + deep-context (8x8 upsampled) -> feat_dim @ 32x32
        self.fuse = nn.Sequential(
            nn.Conv2d(chs[0] + chs[1], feat_dim, 3, 1, 1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(True))
        self.hm = nn.Conv2d(feat_dim, n_joints, 1)                # per-joint heatmaps
        self.mlp = nn.Sequential(                                 # shared per-joint (over B*J)
            nn.Linear(feat_dim + 2, 256), nn.LayerNorm(256), nn.ReLU(True),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(True),
            nn.Linear(128, 3))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.in_chans == 3:
            x = (x - self.mean) / self.std
        elif self.in_chans == 4:
            rgb = (x[:, :3] - self.mean) / self.std
            x = torch.cat([rgb, x[:, 3:4]], dim=1)
        hi, deep = self.encoder(x)                                # [B,96,32,32], [B,384,8,8]
        deep = F.interpolate(deep, size=hi.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse(torch.cat([hi, deep], dim=1))           # [B, Cf, H, W]
        B, Cf, H, W = feat.shape

        hm = self.hm(feat).view(B, self.n, H * W)                # [B,J,HW]
        prob = torch.softmax(hm, dim=-1).view(B, self.n, H, W)   # spatial softmax

        # soft-argmax over a normalised [-1,1] grid -> (u,v) per joint
        gx = torch.linspace(-1, 1, W, device=feat.device).view(1, 1, 1, W)
        gy = torch.linspace(-1, 1, H, device=feat.device).view(1, 1, H, 1)
        u = (prob * gx).sum(dim=(2, 3))                          # [B,J]
        v = (prob * gy).sum(dim=(2, 3))
        uv = torch.stack([u, v], dim=-1)                         # [B,J,2]

        # heatmap-weighted (attention) pooling of the feature per joint
        fj = torch.einsum("bjhw,bchw->bjc", prob, feat)          # [B,J,Cf]

        p = self.mlp(torch.cat([fj, uv], dim=-1)).view(B, self.n, 3)
        return {"keypoints": p, "uv": uv}
