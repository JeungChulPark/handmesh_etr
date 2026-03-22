# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file modules.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief Modules composing MobRecon
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""

import torch.nn as nn
import torch
from conv.spiralconv import SpiralConv
import torch.nn.functional as F

# Basic modules

class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super().__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        # sequence.append(nn.Hardtanh(0,4))
        sequence.append(nn.Tanh())

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super().__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN'xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0)//3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, meshconv=SpiralConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out

class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x


class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)
    def self_attn(self, x):
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x):
        BS, V, f = x.shape
        if f != self.f_dim:
            x = x.permute(0, 2, 1)

        x = x + self.self_attn(self.layer_norm(x))
        x = self.ff(x)

        if f != self.f_dim:
            x = x.permute(0, 2, 1)
        return x




# Advanced modules
class Joint3DDecoder(nn.Module):
    def __init__(self, latent_size, uv_channels, out_channels):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Joint3DDecoder, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.uv_channels = uv_channels

        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, 
        bn=False, relu=False)
        self.uv_linear = nn.Linear(3, self.uv_channels)
        self.upsample_1 = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)
        self.upsample_2 = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)
        self.upsample_3 = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)

        self.cat_conv1 = nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-1], 3, 1, 1)
        self.cat_conv2 = nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-1], 3, 1, 1)
        self.cat_conv3 = nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-1], 3, 1, 1)
        self.cat_act = nn.SiLU()
        self.cat_conv_final = nn.Conv1d(out_channels[-1] * 3, out_channels[-1], 1, 1)

        self.final_conv = nn.ModuleList([])
        for step in range(len(out_channels) - 1):
            self.final_conv.append(
                nn.ModuleList([
                    nn.Conv1d(out_channels[-1 - step], out_channels[-2 - step], 1, 1),
                    nn.SiLU(),
                    nn.Conv1d(out_channels[-2 - step], out_channels[-2 - step], 1, 1),
                    nn.GroupNorm(8, out_channels[-2 - step]),
                    SelfAttn(out_channels[-2 - step]),
                ])
            )
        
        self.final_linear1 = nn.Sequential(
            nn.Linear(out_channels[0], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def index(self, uv, feat):
        import pdb; pdb.set_trace()
        uv = uv.unsqueeze(2)  # [B, N, 1, 3]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        x = self.de_layer_conv(x)
        x_1 = self.index(uv[..., :2], x)
        x_2 = self.index(uv[..., 0::2], x)
        x_3 = self.index(uv[..., 1:], x)
        
        uv_feat = self.uv_linear(uv).permute(0, 2, 1) # [B, 21, 64]?
        
        x_1 = torch.bmm(x_1, self.upsample_1.repeat(x.size(0), 1, 1).to(x.device))
        x_2 = torch.bmm(x_2, self.upsample_2.repeat(x.size(0), 1, 1).to(x.device))
        x_3 = torch.bmm(x_3, self.upsample_3.repeat(x.size(0), 1, 1).to(x.device))

        x_1 = torch.cat([x_1, uv_feat], dim=1)
        x_1 = self.cat_conv1(x_1)
        x_2 = torch.cat([x_2, uv_feat], dim=1)
        x_2 = self.cat_conv2(x_2)
        x_3 = torch.cat([x_3, uv_feat], dim=1)
        x_3 = self.cat_conv3(x_3)
        x =  torch.cat([x_1, x_2, x_3], dim=1)
        
        x = self.cat_conv_final(x)
        x = self.cat_act(x)


        # import pdb; pdb.set_trace()
        for i, (conv, act, conv2, norm, attn) in enumerate(self.final_conv):
            x = conv(x)
            x = act(x)
            x = conv2(x)
            x = norm(x)
            # x = attn(x) + x

        x = x.permute(0, 2, 1)
        x = self.final_linear1(x)
        return x


# Get 2D and previous 2.5D
class joint3DDecoder_additional(nn.Module):
    def __init__(self, latent_size, uv_channels, out_channels):
        super().__init__()

        self.latent_size = latent_size
        self.out_channels = out_channels
        self.uv_channels = uv_channels

        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, 
        bn=False, relu=False)
        self.upsample = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)

        # additional embedding
        self.additional_embedding = nn.Conv1d(3, self.uv_channels, 1)
        self.additional_layers = nn.Sequential(
            nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-2], 1),
            nn.SiLU(),
            # nn.GroupNorm(84, out_channels[-2]),
            nn.BatchNorm1d(out_channels[-2]),
            nn.Dropout(0.2),
            nn.Conv1d(out_channels[-2], out_channels[-2], 1),
            nn.SiLU(),
            nn.BatchNorm1d(out_channels[-2]),
            # nn.GroupNorm(8, out_channels[-1]),
            nn.Dropout(0.2),
            nn.Conv1d(out_channels[-2], out_channels[-1], 1),
        )

        self.final_conv = nn.ModuleList([])
        for step in range(len(out_channels) - 1):
            self.final_conv.append(
                nn.ModuleList([
                    nn.Conv1d(out_channels[-1 - step], out_channels[-2 - step], 1, 1),
                    nn.SiLU(),
                    nn.Conv1d(out_channels[-2 - step], out_channels[-2 - step], 1, 1),
                    # nn.GroupNorm(8, out_channels[-2 - step]),
                    nn.BatchNorm1d(out_channels[-2 - step]),
                    SelfAttn(out_channels[-2 - step]),
                ])
            )
        
        self.final_linear1 = nn.Sequential(
            nn.Linear(out_channels[0], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def index(self, uv, feat):
        # import pdb; pdb.set_trace()
        uv = uv.unsqueeze(2)  # [B, N, 1, 3]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    # uv: 2D joints
    # x: latent feature
    # additional: previous 2.5D -> [B, 21, 3]
    def forward(self, uv, x, additional):
        x = self.de_layer_conv(x) # latent feature
        
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x_sampled = self.index(uv, x)
        
        x_adj = torch.bmm(x_sampled, self.upsample.repeat(x.size(0), 1, 1).to(x.device)) # [21, 21] -> adjacency

        # adding additional information
        # import pdb; pdb.set_trace()
        x_cat = torch.cat((x_adj, self.additional_embedding(additional.transpose(1,2))), dim=1)
        x = self.additional_layers(x_cat)

        # import pdb; pdb.set_trace()
        for i, (conv, act, conv2, norm, attn) in enumerate(self.final_conv):
            x = conv(x)
            x = act(x)
            x = conv2(x)
            x = norm(x)
            # x = attn(x) + x

        x = x.permute(0, 2, 1)
        x = self.final_linear1(x)
        return x

class Joint3DDecoder_small(nn.Module):
    def __init__(self, latent_size, uv_channels, out_channels):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Joint3DDecoder_small, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.uv_channels = uv_channels

        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, 
        bn=False, relu=False)
        self.uv_linear = nn.Linear(3, self.uv_channels)
        self.upsample_1 = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)
        self.upsample_2 = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)
        self.upsample_3 = nn.Parameter(torch.ones([21, 21])*0.01, requires_grad=True)

        self.cat_conv1 = nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-1], 3, 1, 1)
        self.cat_conv2 = nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-1], 3, 1, 1)
        self.cat_conv3 = nn.Conv1d(out_channels[-1] + uv_channels, out_channels[-1], 3, 1, 1)
        self.cat_act = nn.SiLU()
        self.cat_conv_final = nn.Conv1d(out_channels[-1] * 3, out_channels[-1], 1, 1)

        self.final_conv = nn.ModuleList([])
        for step in range(len(out_channels) - 1):
            self.final_conv.append(
                nn.ModuleList([
                    nn.Conv1d(out_channels[-1 - step], out_channels[-2 - step], 1, 1),
                    nn.SiLU(),
                    nn.Conv1d(out_channels[-2 - step], out_channels[-2 - step], 1, 1),
                    nn.GroupNorm(8, out_channels[-2 - step]),
                    SelfAttn(out_channels[-2 - step]),
                ])
            )
        
        self.final_linear1 = nn.Sequential(
            nn.Linear(out_channels[0], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def index(self, uv, feat):
        uv = uv.unsqueeze(2)  # [B, N, 1, 3]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]


    def forward(self, uv, x):
        x = self.de_layer_conv(x)
        x_1 = self.index(uv[..., :2], x)
        x_2 = self.index(uv[..., 0::2], x)
        x_3 = self.index(uv[..., 1:], x)
        
        uv_feat = self.uv_linear(uv).permute(0, 2, 1) # [B, 21, 64]?
        
        x_1 = torch.bmm(x_1, self.upsample_1.repeat(x.size(0), 1, 1).to(x.device))
        x_2 = torch.bmm(x_2, self.upsample_2.repeat(x.size(0), 1, 1).to(x.device))
        x_3 = torch.bmm(x_3, self.upsample_3.repeat(x.size(0), 1, 1).to(x.device))

        x_1 = torch.cat([x_1, uv_feat], dim=1)
        x_1 = self.cat_conv1(x_1)
        x_2 = torch.cat([x_2, uv_feat], dim=1)
        x_2 = self.cat_conv2(x_2)
        x_3 = torch.cat([x_3, uv_feat], dim=1)
        x_3 = self.cat_conv3(x_3)
        x =  torch.cat([x_1, x_2, x_3], dim=1)
        
        x = self.cat_conv_final(x)
        x = self.cat_act(x)


        # import pdb; pdb.set_trace()
        for i, (conv, act, conv2, norm, attn) in enumerate(self.final_conv):
            x = conv(x)
            x = act(x)
            # x = conv2(x)
            x = norm(x)
            # x = attn(x) + x

        x = x.permute(0, 2, 1)
        x = self.final_linear1(x)
        return x



# Advanced modules
class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, bn=False, relu=False)
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel])*0.01, requires_grad=True)


    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        x = self.index(x, uv).permute(0, 2, 1)
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)

        return pred

