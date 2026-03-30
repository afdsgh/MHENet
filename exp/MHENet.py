import sys
import os
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.pvtv2 import pvt_v2_b2
from einops.einops import rearrange


def cbr(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def MixPooling(x):
    x_max = F.max_pool2d(x, kernel_size=2, stride=2)
    x_avg = F.avg_pool2d(x, kernel_size=2, stride=2)

    return x_max + x_avg


class CrossModalGlobalGuidance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convrgb = nn.Conv2d(channels, channels, 1, 1, 0)
        self.convdepth = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, rgb_feat, depth_feat):
        # Global max pooling: [B,C,1,1]
        rgb_global = F.adaptive_max_pool2d(rgb_feat, 1)
        depth_global = F.adaptive_max_pool2d(depth_feat, 1)

        # Cross-modal guidance
        rgb_guided = rgb_feat * self.convdepth(depth_global) + rgb_feat
        depth_guided = depth_feat * self.convrgb(rgb_global) + depth_feat

        return rgb_guided, depth_guided


class LearnableWeighted(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(LearnableWeighted, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, 1, 1)
        )

    def forward(self, rgb_feat, depth_feat):
        feat = torch.cat([rgb_feat, depth_feat], dim=1)

        raw_scores = self.net(feat)  # [B,2,H,W]
        sim_masks = torch.softmax(raw_scores, 1)

        rgb_mask = sim_masks[:, 0:1]
        depth_mask = sim_masks[:, 1:2]

        return rgb_mask, depth_mask


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        avg_result = self.avgpool(x)  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1)
        max_out = self.se(max_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        avg_out = self.se(avg_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,1)
        output = self.sigmoid(max_out + avg_out)  # 相加,然后通过sigmoid获得权重:(B,C,1,1)
        return output


class FusionRefine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionRefine, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        refined_feat = self.main_path(x)
        catt = self.ca(refined_feat)
        out = refined_feat * catt + residual
        return out


class ADFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ADFM, self).__init__()
        self.guidance = CrossModalGlobalGuidance(in_channels)
        self.weight_gen = LearnableWeighted(in_channels)
        self.fusion = FusionRefine(in_channels * 2, out_channels)

    def forward(self, rgb_feat, depth_feat, prev=None):
        # cross-modal guidance
        rgb_enh, depth_enh = self.guidance(rgb_feat, depth_feat)
        # dynamic weight
        rgb_mask, depth_mask = self.weight_gen(rgb_enh, depth_enh)
        # fusion
        fused = torch.cat((rgb_enh * rgb_mask, depth_enh * depth_mask), dim=1)
        out = self.fusion(fused)
        if prev is None:
            return out
        elif prev is not None:
            prev = F.interpolate(prev, scale_factor=2, mode='bilinear')
            return out + prev


class LearnableSobelConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 1):
        super(LearnableSobelConv, self).__init__()
        # 普通卷积核
        self.conv_kernel = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding,
                                     bias=False)

        P_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        P_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('P_h', P_h.repeat(out_channels, in_channels, 1, 1))
        self.register_buffer('P_v', P_v.repeat(out_channels, in_channels, 1, 1))

    def forward(self, x):
        kernel_h = self.conv_kernel.weight * self.P_h
        kernel_v = self.conv_kernel.weight * self.P_v

        grad_h = F.conv2d(x, kernel_h, stride=self.conv_kernel.stride, padding=self.conv_kernel.padding)
        grad_v = F.conv2d(x, kernel_v, stride=self.conv_kernel.stride, padding=self.conv_kernel.padding)

        grad = torch.sqrt(grad_h ** 2 + grad_v ** 2 + 1e-8)
        return grad


class GeometricBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeometricBlock, self).__init__()
        self.sobel_conv1 = LearnableSobelConv(in_channels, out_channels)
        self.sobel_conv2 = LearnableSobelConv(in_channels, out_channels)
        self.conv1 = cbr(in_channels, out_channels, 3, 1, 1)
        self.conv2 = cbr(in_channels, out_channels, 3, 1, 1)
        self.final_conv = cbr(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):

        g1 = self.sobel_conv1(x)
        x1 = self.conv1(g1 + x)
        g2 = self.sobel_conv2(x1)
        x2 = self.conv2(g2 + x1)
        return x2

class GSFM(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(GSFM, self).__init__()
        self.scale_factor = scale_factor
        self.semantic_block = SemanticBlock(in_channels, out_channels)
        self.geometric_block = GeometricBlock(in_channels, out_channels)
        self.convs = cbr(in_channels, out_channels, 3, 1, 1)
        self.convg = cbr(in_channels, out_channels, 3, 1, 1)
        self.fusion = cbr(in_channels, out_channels, 3, 1, 1)

    def forward(self, high_feat, low_feat):
        high_up = F.interpolate(high_feat, scale_factor=2, mode='bilinear')
        low_down = MixPooling(low_feat)
        geo_base = self.convg(low_feat + high_up)
        semantic_base = self.convs(high_feat + low_down)
        geo_feat = self.geometric_block(geo_base)
        semantic_feat = self.semantic_block(semantic_base)
        semantic_up = F.interpolate(semantic_feat, scale_factor=2, mode='bilinear')
        out = self.fusion(geo_feat + semantic_up)
        return out


class TextureBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextureBlock, self).__init__()
        self.conv1 = cbr(in_channels, out_channels, 1, 1, 0)
        self.conv2 = cbr(in_channels, out_channels, 3, 1, 1)
        self.conv3 = cbr(in_channels, out_channels, 5, 1, 2)
        self.conv = cbr(in_channels, out_channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        fused = self.conv(x1 + x2 + x3)
        x_pool = F.avg_pool2d(x, 3, stride=1, padding=1)
        texture = fused - x_pool
        ts = self.sigmoid(texture)
        out = ts * fused
        return out


class SemanticBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemanticBlock, self).__init__()
        self.mean_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = cbr(in_channels, out_channels, 3, 1, 1)
        self.conv2 = cbr(in_channels, out_channels, 3, 1, 1)
        self.conv3 = cbr(in_channels, out_channels, 3, 1, 1)
        self.final_conv = cbr(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        xm = self.mean_pool(x)
        x_down1 = MixPooling(x)
        x_down1 = self.conv1(x_down1)
        x_down2 = MixPooling(x_down1)
        x_down2 = self.conv2(x_down2)
        x_d2_up = F.interpolate(x_down2, size=x.size()[2:], mode='bilinear')
        x_d1_up = F.interpolate(x_down1, size=x.size()[2:], mode='bilinear')
        xd = self.conv3(x_d1_up + x_d2_up)
        out = self.final_conv(xd * xm + x)
        return out

class TSFM(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(TSFM, self).__init__()
        self.scale_factor = scale_factor
        self.texture_block = TextureBlock(in_channels, out_channels)
        self.semantic_block = SemanticBlock(in_channels, out_channels)
        self.convt = cbr(in_channels, out_channels, 3, 1, 1)
        self.convs = cbr(in_channels, out_channels, 3, 1, 1)
        self.fusion = cbr(in_channels, out_channels, 3, 1, 1)

    def forward(self, high_feat, low_feat):
        high_up = F.interpolate(high_feat, scale_factor=2, mode='bilinear')
        low_down = MixPooling(low_feat)
        text_base = self.convt(low_feat + high_up)
        semantic_base = self.convs(high_feat + low_down)
        text_feat = self.texture_block(text_base)
        semantic_feat = self.semantic_block(semantic_base)
        semantic_up = F.interpolate(semantic_feat, scale_factor=2, mode='bilinear')
        out = self.fusion(text_feat + semantic_up)
        return out

class MHENet(nn.Module):
    def __init__(self):
        super(MHENet, self).__init__()

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.pvt = pvt_v2_b2()
        self.pvtd = pvt_v2_b2()
        path = ''

        save_model = torch.load(path)
        model_dict = self.pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.pvt.load_state_dict(model_dict)
        self.pvtd.load_state_dict(model_dict)

        self.tsfm3 = TSFM(64, 64)
        self.tsfm2 = TSFM(64, 64)
        self.tsfm1 = TSFM(64, 64)
        self.gafm3 = GSFM(64, 64)
        self.gafm2 = GSFM(64, 64)
        self.gafm1 = GSFM(64, 64)

        self.adfm3 = ADFM(64, 64)
        self.adfm2 = ADFM(64, 64)
        self.adfm1 = ADFM(64, 64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.predtrans3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False),
        )
        self.predtrans2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False),
        )
        self.predtrans1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False),
        )
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False),
        )

        self.convr1 = nn.Conv2d(64, 64, 1, 1, 0)
        self.convr2 = nn.Conv2d(128, 64, 1, 1, 0)
        self.convr3 = nn.Conv2d(320, 64, 1, 1, 0)
        self.convr4 = nn.Conv2d(512, 64, 1, 1, 0)
        self.convd1 = nn.Conv2d(64, 64, 1, 1, 0)
        self.convd2 = nn.Conv2d(128, 64, 1, 1, 0)
        self.convd3 = nn.Conv2d(320, 64, 1, 1, 0)
        self.convd4 = nn.Conv2d(512, 64, 1, 1, 0)

    def forward(self, x, d):
        rgb_list = self.pvt(x)
        depth_list = self.pvtd(d)

        r1 = rgb_list[0]
        r2 = rgb_list[1]
        r3 = rgb_list[2]
        r4 = rgb_list[3]
        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]

        r1 = self.convr1(r1)
        r2 = self.convr2(r2)
        r3 = self.convr3(r3)
        r4 = self.convr4(r4)
        d1 = self.convd1(d1)
        d2 = self.convd2(d2)
        d3 = self.convd3(d3)
        d4 = self.convd4(d4)

        rf_3 = self.tsfm3(r4, r3)
        df_3 = self.gafm3(d4, d3)

        f_3 = self.adfm3(rf_3, df_3)

        rf_2 = self.tsfm2(rf_3, r2)
        df_2 = self.gafm2(df_3, d2)

        f_2 = self.adfm2(rf_2, df_2, f_3)

        rf_1 = self.tsfm1(rf_2, r1)
        df_1 = self.gafm1(df_2, d1)

        f_1 = self.adfm1(rf_1, df_1, f_2)

        y1 = F.interpolate(self.predtrans1(f_1), size=416, mode='bilinear')
        y2 = F.interpolate(self.predtrans2(rf_1), size=416, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_1), size=416, mode='bilinear')

        return y1, y2, y3


