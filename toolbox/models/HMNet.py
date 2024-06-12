import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
import copy
# from mmcv.cnn import build_norm_layer
# from mmengine.model import BaseModule
# from mmcv.cnn.bricks import DropPath
# from mmengine.model.weight_init import (constant_init, normal_init,
#                                         trunc_normal_init)
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from .seaformer import *
from .repghost import RepGhostModule
from .odconv import *



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        init_channels = in_features
        new_channels = hidden_features - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_features, init_channels, 1, 1, 1 // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) ,
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 3 // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Conv2d(hidden_features,out_features,1)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        x2 = self.drop(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.drop(self.outc(out))
        return out
class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            # build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            # build_norm_layer(norm_cfg, out_channels)[1],
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

def DMDC(rate=5,dim=64):

    b = nn.Sequential(
        nn.AdaptiveAvgPool2d((rate,rate)),
        nn.Conv2d(dim,dim//4,1),
        nn.Conv2d(dim//4,dim,kernel_size=rate,padding=0)
    )
    return b


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.att_path = DMDC(rate=5,dim=dim)

    def forward(self, x):
        u = x.clone()
        #attn = self.conv0(x)
        attn = x
        ap = self.att_path(attn)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)*ap

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
class SpatialAttention_sea(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = Sea_Attention(d_model, 16, 8, 2, activation=nn.ReLU6)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 att = 'sea',
                 ks = 5,
                 ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)  # build_norm_layer(norm_cfg, dim)[1]
        if att=='sea':
            self.attn = SpatialAttention_sea(dim)
        else:
            self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)  # build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = RepGhostModule(dim,dim)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.conv0 = nn.Conv2d(dim, dim, ks, padding=ks//2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.conv0(x))
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)  # build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class HMNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 ):
        super(HMNet, self).__init__()



        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        ks = [5,5,5,5]

        for i in range(num_stages):

            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])

            else:

                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j], att='msc' if i<=2 else 'sea',ks=ks[i])
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(
                    m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)




    def forward(self, x):
        B = x.shape[0]
        # B = len(x)
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs
class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out

class DAGM(nn.Module):
    def __init__(self,inc1,inc2,outc):
        super(DAGM, self).__init__()
        midc = inc2//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc1,midc,1),
            nn.BatchNorm2d(midc)
        )
        self.conv2 =ODConv2d(inc2,inc2,3,groups=inc2,padding=1)
        # self.conv2 = nn.Conv2d(inc2,inc2,3,groups=)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(inc2,midc,1),
            nn.BatchNorm2d(midc)
        )
        self.relu = nn.ReLU()
        self.midconv = nn.Sequential(
            nn.Conv2d(midc, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.outc = nn.Conv2d(inc2,outc,1)

    def forward(self,in1,in2):
        in1 = F.interpolate(in1,in2.size()[2:],mode='bilinear',align_corners=False)
        in2 = self.conv2(in2)
        mid = self.conv1(in1)+self.conv2_2(in2)
        att = self.midconv(self.relu(mid))
        out = in2 * att + in2
        return self.outc(out)
class ocsapp(nn.Module):
    def __init__(self,inc1):
        super().__init__()
        rate = [3,7,11]

        in_chans = inc1
        #self.inc = nn.Conv2d(inc1,inc1//2,1)
        self.aspp2 = nn.Conv2d(in_chans,in_chans,3,1,rate[0],rate[0],in_chans)
        self.aspp4 = nn.Conv2d(in_chans, in_chans, 3, 1, rate[1], rate[1], in_chans)
        self.aspp8 = nn.Conv2d(in_chans, in_chans, 3, 1, rate[2], rate[2], in_chans)
        self.outc = nn.Conv2d(in_chans * 3, in_chans, 1)
    def forward(self,x):

        #x = self.inc(in1)
        a1 = self.aspp2(x)
        a2 = self.aspp4(x)
        a3 = self.aspp8(x)
        a = torch.cat([a1, a2, a3], dim=1)
        a = self.outc(a)
        return a
class FPNHEAD(nn.Module):
    def __init__(self, channels=[64, 128, 320, 512], out_channels=256, num_classes=12):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels[-1], out_channels=out_channels)


        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)
        self.cls_seg = nn.Conv2d(out_channels,num_classes,1)
        self.aux_head = nn.Conv2d(out_channels, num_classes, 1)
        self.ag1 = DAGM(channels[-1],channels[-2],out_channels)
        self.ag2 = DAGM(channels[-1],channels[-3],out_channels)
        self.ag3 = DAGM(channels[-1], channels[-4], out_channels)
        #self.ocspp = ocsapp(out_channels)

    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.ag1(x1,input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.ag2(x1,input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.ag3(x1,input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))


        return self.cls_seg(x),self.aux_head(x1)
def CBR(i,o,k):
    return nn.Sequential(
        nn.Conv2d(in_channels=i,out_channels=o,kernel_size=k,padding=k//2,stride=1),
        nn.BatchNorm2d(o),
        nn.ReLU()
    )
class DN_auxhead(nn.Module):
    def __init__(self,channels=[64, 128, 320, 512],middim=128,outc=1):
        super().__init__()
        self.conv1 = CBR(channels[0],middim,1)
        self.conv2 = CBR(channels[1],middim,1)
        self.conv3 = CBR(channels[2],middim,1)
        self.aconv = CBR(middim*3,middim,1)
        self.a2conv = nn.Sequential(nn.Conv2d(middim,middim//4,3,1,1),nn.BatchNorm2d(middim//4))
        self.o1conv = nn.Sequential(nn.Conv2d(middim//4, middim // 32, 3, 1, 1), nn.BatchNorm2d(middim // 32))
        self.o2conv = CBR(middim//32,outc,3)
    def forward(self,inputs):
        x1 = self.conv1(inputs[-4])
        x2 = self.conv2(inputs[-3])
        x3 = self.conv3(inputs[-2])
        x1 = F.interpolate(x1,x3.size()[-2:])
        x2 = F.interpolate(x2, x3.size()[-2:])
        xa = torch.cat([x1,x2,x3],1)
        xa = self.aconv(xa)
        xa = self.a2conv(xa)
        xa = F.interpolate(xa,scale_factor=8)
        o =self.o1conv(xa)
        o = self.o2conv(o)
        return o
def HMNet_small():
    return HMNet(embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],drop_path_rate=0.1)
def HMNet_tiny():
    return HMNet(embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],drop_path_rate=0.1)
def HMNet_c():
    return HMNet(embed_dims=[32, 64, 160, 256],
        depths=[2, 2, 4, 2],drop_path_rate=0.1)
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

class HMNet_seg(nn.Module):
    def __init__(self, numclass=12):
        super().__init__()
        self.encoder = HMNet_tiny()
        self.decoder = FPNHEAD(channels=[32, 64, 160, 256],num_classes=numclass)
        self.aux = DN_auxhead(channels=[32, 64, 160, 256])
        self.t = True

    def convert_to_deploy(self):
        model_convert(self, do_copy=False)
        del self.aux
        self.t = False

    def forward(self, x):
        f = self.encoder(x)
        out,out2 = self.decoder(f)
        if self.t:
            pdn = self.aux(f)
        else:pdn = torch.zeros_like(out)
        out = F.interpolate(out,scale_factor=4,mode='bilinear',align_corners=False)
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
        pdn = F.interpolate(pdn, scale_factor=2, mode='bilinear', align_corners=False)
        return out,out2,pdn
if __name__ == "__main__":
    net = HMNet_seg()
    i = torch.randn(2,3,480,320)
    o = net(i)
    for a in o:
        print(a.shape)
