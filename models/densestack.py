# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
* @file densestack.py
* @author chenxingyu (chenxy.sean@gmail.com)
* @brief DenseStack
* @version 0.1
* @date 2022-04-28
*
* @copyright Copyright (c) 2022 chenxingyu
*
"""

import torch
import torch.nn as nn
from models.modules import conv_layer, mobile_unit, linear_layer, Reorg
import os


class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in // 4)
        self.conv2 = mobile_unit(channel_in * 5 // 4, channel_in // 4)
        self.conv3 = mobile_unit(channel_in * 6 // 4, channel_in // 4)
        self.conv4 = mobile_unit(channel_in * 7 // 4, channel_in // 4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3), dim=1)
        out4 = self.conv4(comb3)
        comb4 = torch.cat((comb3, out4), dim=1)
        return comb4


class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in // 2)
        self.conv2 = mobile_unit(channel_in * 3 // 2, channel_in // 2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), dim=1)
        return comb2


class DenseBlock3(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock3, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in)
        self.conv2 = mobile_unit(channel_in * 2, channel_in)
        self.conv3 = mobile_unit(channel_in * 3, channel_in)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3), dim=1)
        return comb3


class DenseBlock2_noExpand(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_noExpand, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in * 3 // 4)
        self.conv2 = mobile_unit(channel_in * 7 // 4, channel_in // 4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((out1, out2), dim=1)
        return comb2


class SenetBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel, size):
        super(SenetBlock, self).__init__()
        self.size = size
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel = channel
        self.fc1 = linear_layer(self.channel, min(self.channel // 2, 256))
        self.fc2 = linear_layer(min(self.channel // 2, 256), self.channel, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_out = x
        pool = self.globalAvgPool(x)
        pool = pool.view(pool.size(0), -1)
        fc1 = self.fc1(pool)
        out = self.fc2(fc1)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out * original_out


class DenseStack(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel):
        super(DenseStack, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel * 4)
        self.senet3 = SenetBlock(input_channel * 8, 8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel * 8)
        self.dense5 = DenseBlock2_noExpand(input_channel * 8)
        self.thrink1 = nn.Sequential(
            mobile_unit(input_channel * 8, input_channel * 4, num3x3=1),
            mobile_unit(input_channel * 4, input_channel * 4, num3x3=2),
        )
        self.senet4 = SenetBlock(input_channel * 4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink2 = nn.Sequential(
            mobile_unit(input_channel * 4, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=2),
        )
        self.senet5 = SenetBlock(input_channel * 2, 8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink3 = nn.Sequential(
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, output_channel, num3x3=2),
        )
        self.senet6 = SenetBlock(output_channel, 16)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.dense3(d2))
        u1 = self.upsample1(self.senet4(self.thrink1(d3)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.upsample3(self.senet6(self.thrink3(us2)))
        return u3


class DenseStack2(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel, final_upsample=True, ret_mid=False):
        super(DenseStack2, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel * 4)
        self.senet3 = SenetBlock(input_channel * 8, 8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel * 8)
        self.dense5 = DenseBlock2_noExpand(input_channel * 8)
        self.thrink1 = nn.Sequential(
            mobile_unit(input_channel * 8, input_channel * 4, num3x3=1),
            mobile_unit(input_channel * 4, input_channel * 4, num3x3=2),
        )
        self.senet4 = SenetBlock(input_channel * 4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink2 = nn.Sequential(
            mobile_unit(input_channel * 4, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=2),
        )
        self.senet5 = SenetBlock(input_channel * 2, 8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink3 = nn.Sequential(
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, output_channel, num3x3=2),
        )
        self.senet6 = SenetBlock(output_channel, 16)
        self.final_upsample = final_upsample
        if self.final_upsample:
            self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ret_mid = ret_mid

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.senet3(self.dense3(d2)))
        d4 = self.dense5(self.dense4(d3))
        u1 = self.upsample1(self.senet4(self.thrink1(d4)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.senet6(self.thrink3(us2))
        if self.final_upsample:
            u3 = self.upsample3(u3)
        if self.ret_mid:
            return u3, u2, u1, d4
        else:
            return u3, d4


class DenseStack3(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel):
        super(DenseStack3, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        # self.dense4 = DenseBlock2_noExpand(input_channel*4)
        # self.dense5 = DenseBlock2_noExpand(input_channel*4)
        self.thrink2 = nn.Sequential(
            mobile_unit(input_channel * 4, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=2),
        )

        self.senet5 = SenetBlock(input_channel * 2, 8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink3 = nn.Sequential(
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, output_channel, num3x3=2),
        )
        self.senet6 = SenetBlock(output_channel, 16)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        u2 = self.upsample2(self.senet5(self.thrink2(d2)))
        us2 = d1 + u2
        u3 = self.upsample3(self.senet6(self.thrink3(us2)))
        return u3


class DenseStack4(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel, final_upsample=True, ret_mid=False):
        super(DenseStack4, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel * 4)
        # self.dense5 = DenseBlock2_noExpand(input_channel*4)
        self.thrink2 = nn.Sequential(
            mobile_unit(input_channel * 4, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=2),
        )
        self.senet5 = SenetBlock(input_channel * 2, 8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink3 = nn.Sequential(
            mobile_unit(input_channel * 2, input_channel * 2, num3x3=1),
            mobile_unit(input_channel * 2, output_channel, num3x3=2),
        )
        self.senet6 = SenetBlock(output_channel, 16)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.dense4(d2)
        u2 = self.upsample2(self.senet5(self.thrink2(d3)))
        us2 = d1 + u2
        return self.senet6(self.thrink3(us2))


class DenseStack_Backnone_small(nn.Module):
    def __init__(self, input_channel=64, out_channel=21, latent_size=256, kpts_num=21):
        super(DenseStack_Backnone_small, self).__init__()
        self.pre_layer = nn.Sequential(
            conv_layer(3, input_channel // 4, 3, 2, 1), mobile_unit(input_channel // 4, input_channel // 4)
        )
        self.thrink = conv_layer(input_channel, input_channel)
        self.dense_stack1 = DenseStack3(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)

        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack4(input_channel, out_channel, final_upsample=False)

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(
            linear_layer(latent_size, 32, bn=False), linear_layer(32, 3, bn=False, relu=False)
        )
        self.reorg = conv_layer(input_channel // 4, input_channel, 3, 2, 1)  # Reorg()

    def forward(self, x):
        pre_out = self.pre_layer(x)

        pre_out_reorg = self.reorg(pre_out)
        thrink = self.thrink(pre_out_reorg)
        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, pre_out_reorg), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out = self.dense_stack2(thrink2)

        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        return uv_reg


class DenseStack_Backbone_middle(nn.Module):
    def __init__(self, input_channel=64, out_channel=21, latent_size=256, kpts_num=21, pretrain=False):
        super(DenseStack_Backbone_middle, self).__init__()
        self.pre_layer = nn.Sequential(
            conv_layer(3, input_channel // 2, 3, 2, 1), mobile_unit(input_channel // 2, input_channel)
        )
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)

        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(
            linear_layer(latent_size, 128, bn=False),
            linear_layer(128, 64, bn=False),
            linear_layer(64, 3, bn=False, relu=False),
        )

        self.reorg = conv_layer(input_channel, input_channel * 4, 3, 2, 1)  # Reorg()
        if pretrain:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            weight = torch.load(os.path.join(cur_dir, "../misc/densestack.pth"))
            self.load_state_dict(weight, strict=False)
            print("Load pre-trained weight: densestack.pth")

        self.act = nn.Sigmoid()

    def forward(self, x):
        pre_out = self.pre_layer(x)  # [1, 128, 128, 128]

        pre_out_reorg = self.reorg(pre_out)  # [1, 512, 64, 64]
        thrink = self.thrink(pre_out_reorg)

        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)
        #
        # latent = self.mid_proj(stack2_mid)
        # import pdb; pdb.set_trace()
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        return self.act(uv_reg)


class DenseStack_Backnone_big(nn.Module):
    def __init__(
        self, input_channel=128, out_channel=24, latent_size=256, kpts_num=21, pretrain=True, output_dim=3
    ):
        super(DenseStack_Backnone_big, self).__init__()

        self.pre_layer = nn.Sequential(
            conv_layer(3, input_channel // 2, 3, 2, 1), mobile_unit(input_channel // 2, input_channel)
        )
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)
        self.mid_proj = conv_layer(1024, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)
        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(
            linear_layer(latent_size, 128, bn=False),
            linear_layer(128, 64, bn=False),
            linear_layer(64, output_dim, bn=False, relu=False),
        )
        self.reorg = Reorg()  # conv_layer(input_channel, input_channel * 4, 3, 2, 1) # Reorg()
        if pretrain:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            weight = torch.load(os.path.join(cur_dir, "../misc/densestack.pth"))
            self.load_state_dict(weight, strict=False)
            print("Load pre-trained weight: densestack.pth")

        self.act = nn.Sigmoid()

    def forward(self, x):
        pre_out = self.pre_layer(x)  # [1, 128, 128, 128]

        pre_out_reorg = self.reorg(pre_out)  # [1, 512, 64, 64]
        thrink = self.thrink(pre_out_reorg)

        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)

        latent = self.mid_proj(stack2_mid)
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        return latent, self.act(uv_reg)


class DenseStack_Backbone_super_large(nn.Module):
    def __init__(self, input_channel=64, out_channel=24, latent_size=256, kpts_num=21, pretrain=False):
        super(DenseStack_Backbone_super_large, self).__init__()
        self.pre_layer = nn.Sequential(
            conv_layer(3, input_channel // 2, 3, 2, 1), mobile_unit(input_channel // 2, input_channel)
        )
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)

        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(
            linear_layer(latent_size, 128, bn=False),
            linear_layer(128, 64, bn=False),
            linear_layer(64, 3, bn=False, relu=False),
        )

        self.reorg = conv_layer(input_channel, input_channel * 4, 3, 2, 1)  # Reorg()
        self.mid_proj = conv_layer(512, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)

        self.latent_size = latent_size
        out_channels = [128, 256, 512]
        uv_channels = 128

        self.de_layer_conv = conv_layer(self.latent_size, out_channels[-1], 1, bn=False, relu=False)
        self.uv_linear = nn.Linear(3, uv_channels)

        self.cat_conv_final = nn.Conv1d(out_channels[-1], out_channels[-1], 1, 1)
        self.cat_act = nn.ReLU()
        self.final_conv = nn.ModuleList([])
        for step in range(len(out_channels) - 1):
            self.final_conv.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(out_channels[-1 - step], out_channels[-2 - step], 1, 1),
                        nn.ReLU(),
                        nn.Conv1d(out_channels[-2 - step], out_channels[-2 - step], 1, 1),
                        nn.BatchNorm1d(out_channels[-2 - step]),
                    ]
                )
            )

        self.latent_linear = nn.Linear(64, 21)
        self.final_linear1 = nn.Sequential(
            nn.Linear(out_channels[0], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        if pretrain:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            weight = torch.load(os.path.join(cur_dir, "../misc/densestack.pth"))
            self.load_state_dict(weight, strict=False)
            print("Load pre-trained weight: densestack.pth")

        self.act = nn.Sigmoid()

    def forward(self, x):
        pre_out = self.pre_layer(x)  # [1, 128, 128, 128]

        pre_out_reorg = self.reorg(pre_out)  # [1, 512, 64, 64]
        thrink = self.thrink(pre_out_reorg)

        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)
        #
        latent = self.mid_proj(stack2_mid)
        # import pdb; pdb.set_trace()
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        x = self.de_layer_conv(latent)  # [1, 512, 8, 8]
        x = x.view(x.shape[0], x.shape[1], -1)

        uv_reg = self.uv_linear(uv_reg)  # [B, 21, 128]?

        # x =  torch.cat([x, uv_feat], dim=2)

        x = self.cat_conv_final(x)
        x = self.cat_act(x)

        # import pdb; pdb.set_trace()
        for i, (conv, act, conv2, norm) in enumerate(self.final_conv):
            x = conv(x)
            x = act(x)
            x = conv2(x)
            x = norm(x)

        x = self.latent_linear(x).permute(0, 2, 1)
        x = x + uv_reg
        x = self.final_linear1(x)
        # x = x.permute(0, 2, 1)

        return x


class DenseStack_Backbone_prev(nn.Module):
    def __init__(self, input_channel=64, out_channel=24, latent_size=256, kpts_num=21, pretrain=False):
        super(DenseStack_Backbone_prev, self).__init__()
        self.pre_layer = nn.Sequential(
            conv_layer(3, input_channel // 2, 3, 2, 1), mobile_unit(input_channel // 2, input_channel)
        )
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)

        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(
            linear_layer(latent_size, 128, bn=False),
            linear_layer(128, 64, bn=False),
            linear_layer(64, 3, bn=False, relu=False),
        )

        self.reorg = conv_layer(input_channel, input_channel * 4, 3, 2, 1)  # Reorg()
        self.mid_proj = conv_layer(512, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)

        self.latent_size = latent_size
        out_channels = [128, 256, 512]
        uv_channels = 128

        self.de_layer_conv = conv_layer(self.latent_size, out_channels[-1], 1, bn=False, relu=False)
        self.uv_linear = nn.Linear(3, uv_channels)

        self.cat_conv_final = nn.Conv1d(out_channels[-1], out_channels[-1], 1, 1)
        self.cat_act = nn.ReLU()
        self.final_conv = nn.ModuleList([])
        for step in range(len(out_channels) - 1):
            self.final_conv.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(out_channels[-1 - step], out_channels[-2 - step], 1, 1),
                        nn.ReLU(),
                        nn.Conv1d(out_channels[-2 - step], out_channels[-2 - step], 1, 1),
                        nn.BatchNorm1d(out_channels[-2 - step]),
                    ]
                )
            )

        self.latent_linear = nn.Linear(64, 21)
        self.prev_linear = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, out_channels[0]))
        self.final_linear1 = nn.Sequential(
            nn.Linear(out_channels[0] + out_channels[0], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        if pretrain:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            weight = torch.load(os.path.join(cur_dir, "../misc/densestack.pth"))
            self.load_state_dict(weight, strict=False)
            print("Load pre-trained weight: densestack.pth")

        self.act = nn.Sigmoid()

    def forward(self, x, prev=None):
        pre_out = self.pre_layer(x)  # [1, 128, 128, 128]

        pre_out_reorg = self.reorg(pre_out)  # [1, 512, 64, 64]
        thrink = self.thrink(pre_out_reorg)

        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)
        #
        latent = self.mid_proj(stack2_mid)
        # import pdb; pdb.set_trace()
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        x = self.de_layer_conv(latent)  # [1, 512, 8, 8]
        x = x.view(x.shape[0], x.shape[1], -1)

        uv_reg = self.uv_linear(uv_reg)  # [B, 21, 128]?

        # x =  torch.cat([x, uv_feat], dim=2)

        x = self.cat_conv_final(x)
        x = self.cat_act(x)

        # import pdb; pdb.set_trace()
        for i, (conv, act, conv2, norm) in enumerate(self.final_conv):
            x = conv(x)
            x = act(x)
            x = conv2(x)
            x = norm(x)

        x = self.latent_linear(x).permute(0, 2, 1)

        # Prev part
        prev_x = self.prev_linear(prev)

        x = x + uv_reg

        x = torch.cat([x, prev_x], dim=-1)
        x = self.final_linear1(x)
        # x = x.permute(0, 2, 1)

        return x


class DenseStack_Backbone_like_prev(nn.Module):
    def __init__(self, input_channel=64, out_channel=24, latent_size=256, kpts_num=21, pretrain=False, in_chans=3):
        super(DenseStack_Backbone_like_prev, self).__init__()
        # in_chans: 3 for RGB (default, backward compatible), 4 for RGB-D early fusion.
        self.in_chans = in_chans
        self.pre_layer = nn.Sequential(
            conv_layer(in_chans, input_channel // 2, 3, 2, 1), mobile_unit(input_channel // 2, input_channel)
        )
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)

        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(
            linear_layer(latent_size, 128, bn=False),
            linear_layer(128, 64, bn=False),
            linear_layer(64, 3, bn=False, relu=False),
        )

        self.reorg = conv_layer(input_channel, input_channel * 4, 3, 2, 1)  # Reorg()
        self.mid_proj = conv_layer(512, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)

        self.latent_size = latent_size
        out_channels = [128, 256, 512]
        uv_channels = 128

        self.de_layer_conv = conv_layer(self.latent_size, out_channels[-1], 1, bn=False, relu=False)
        self.uv_linear = nn.Linear(3, uv_channels)

        self.cat_conv_final = nn.Conv1d(out_channels[-1], out_channels[-1], 1, 1)
        self.cat_act = nn.ReLU()
        self.final_conv = nn.ModuleList([])
        for step in range(len(out_channels) - 1):
            self.final_conv.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(out_channels[-1 - step], out_channels[-2 - step], 1, 1),
                        nn.ReLU(),
                        nn.Conv1d(out_channels[-2 - step], out_channels[-2 - step], 1, 1),
                        nn.BatchNorm1d(out_channels[-2 - step]),
                    ]
                )
            )

        self.latent_linear = nn.Linear(64, 21)
        self.prev_linear = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)),
                nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)),
            ]
        )
        self.final_linear1 = nn.Sequential(
            nn.Linear(out_channels[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        if pretrain:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            weight = torch.load(os.path.join(cur_dir, "../misc/densestack.pth"))
            self.load_state_dict(weight, strict=False)
            print("Load pre-trained weight: densestack.pth")

        self.act = nn.Sigmoid()

    def forward(self, x):
        pre_out = self.pre_layer(x)  # [1, 128, 128, 128]

        pre_out_reorg = self.reorg(pre_out)  # [1, 512, 64, 64]
        thrink = self.thrink(pre_out_reorg)

        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)
        #
        latent = self.mid_proj(stack2_mid)
        # import pdb; pdb.set_trace()
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))

        x = self.de_layer_conv(latent)  # [1, 512, 8, 8]
        x = x.view(x.shape[0], x.shape[1], -1)

        uv_reg = self.uv_linear(uv_reg)  # [B, 21, 128]?

        # x =  torch.cat([x, uv_feat], dim=2)

        x = self.cat_conv_final(x)
        x = self.cat_act(x)

        # import pdb; pdb.set_trace()
        for i, (conv, act, conv2, norm) in enumerate(self.final_conv):
            x = conv(x)
            x = act(x)
            x = conv2(x)
            x = norm(x)

        x = self.latent_linear(x).permute(0, 2, 1)
        x = x + uv_reg

        for layer in self.prev_linear:
            res_x = x
            x = layer(x) + res_x

        x = self.final_linear1(x)
        # x = x.permute(0, 2, 1)

        return x
