import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
from .mit import MixVisionTransformer as MIT
from .mit import TransformerEncoderLayer,OverlapPatchEmbed,nchw_to_nlc,nlc_to_nchw
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        # self.expand = nn.Conv2d(dim, 2*dim, kernel_size=3, stride=2,
        #                       padding=1)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        from einops import rearrange
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class MITD(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleList()
        embed_dims =32
        num_heads = [1, 2, 4, 8]
        sr_ratios = [8, 4, 2, 1]

        self.mlp1 = nn.Conv2d(256,256,1)
        self.pex = PatchExpand(dim=embed_dims*num_heads[3], input_resolution=[15, 10])
        self.pex2 = PatchExpand(dim=embed_dims*num_heads[2], input_resolution=[30, 20])
        self.pex3 = PatchExpand(dim=embed_dims*num_heads[1], input_resolution=[60, 40])
        self.pex4 = PatchExpand(dim=embed_dims*num_heads[0], input_resolution=[120, 80])

        mlp_ratio =4
        for i, num_layer in range(4):
            mlp = nn.Conv2d(256,256,1)
            embed_dims_i = embed_dims * num_heads[i]

            patch_embed = PatchExpand(
                in_chans=in_channels,
                embed_dim=embed_dims_i,
                patch_size=patch_sizes[i],
                stride=strides[i],

                norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,

                    qkv_bias=True,

                    sr_ratio=sr_ratios[i]) for idx in range(2)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))

def Block(dim,head,sr):
    layer = ModuleList([
        TransformerEncoderLayer(
            embed_dims=dim,
            num_heads=head,
            feedforward_channels=4 * dim,

            qkv_bias=True,

            sr_ratio=sr) for idx in range(2)
    ])
    return layer
class Transdecoder(nn.Module):
    def __init__(self,numclass=12):
        super(Transdecoder, self).__init__()
        embed_dims = 32
        num_heads = [1, 2, 4, 8]
        sr_ratios = [8, 4, 2, 1]
        # self.pe = OverlapPatchEmbed(in_chans=512, patch_size=3, stride=1, embed_dim=640)
        # self.pe2 = OverlapPatchEmbed(in_chans=320, patch_size=3, stride=1, embed_dim=320)
        # self.pe3 = OverlapPatchEmbed(in_chans=128, patch_size=3, stride=1, embed_dim=128)
        # self.pe4 = OverlapPatchEmbed(in_chans=64, patch_size=3, stride=1, embed_dim=64)
        self.pex = PatchExpand(dim=embed_dims*num_heads[3], input_resolution=[15, 10])
        self.pex2 = PatchExpand(dim=embed_dims*num_heads[2], input_resolution=[30, 20])
        self.pex3 = PatchExpand(dim=embed_dims*num_heads[1], input_resolution=[60, 40])
        self.pex4 = PatchExpand(dim=embed_dims*num_heads[0], input_resolution=[120, 80])
        self.block = Block(embed_dims*num_heads[3]//2,num_heads[3],sr_ratios[3])
        self.block2 = Block(embed_dims*num_heads[2]//2,num_heads[2],sr_ratios[2])
        self.block3 = Block(embed_dims*num_heads[1]//2,num_heads[1],sr_ratios[1])
        self.block4 = Block(embed_dims*num_heads[0]//2,num_heads[0],sr_ratios[0])
        self.linear = nn.Conv2d(embed_dims*num_heads[3], embed_dims*num_heads[3],1)
        self.linear2 = nn.Conv2d(embed_dims*num_heads[3]//2+embed_dims*num_heads[2], embed_dims*num_heads[2],1)
        self.linear3 = nn.Conv2d(embed_dims*num_heads[2]//2+embed_dims*num_heads[1], embed_dims*num_heads[1],1)
        self.linear4 = nn.Conv2d(embed_dims*num_heads[1]//2+embed_dims*num_heads[0], embed_dims*num_heads[0],1)
        self.cls_seg = nn.Conv2d(256*4, numclass, 1)
        self.omlp1 = nn.Conv2d(16,256,1)
        self.omlp2 = nn.Conv2d(64,256,1)
        self.omlp3 = nn.Conv2d(128,256,1)
        self.omlp4 = nn.Conv2d(256,256,1)


    def forward(self,f1,f2,f3,f4):
        #f4 = self.pe(f4)[0]

        f4 = self.linear(f4)
        o4 = f4
        f4 = nchw_to_nlc(f4)
        f4 = self.pex(f4)
        for block in self.block:
            f4 = block(f4, (30, 20))

        f4 = nlc_to_nchw(f4,(30,20))




        f3 = torch.cat([f3, f4], 1)
        f3 = self.linear2(f3)
        o3 = f3
        f3 = nchw_to_nlc(f3)
        f3 = self.pex2(f3)
        for block in self.block2:
            f3 = block(f3, (60, 40))

        f3 = nlc_to_nchw(f3, (60, 40))


        f2 = torch.cat([f2, f3], 1)
        f2 = self.linear3(f2)
        o2 = f2
        f2 = nchw_to_nlc(f2)
        f2 = self.pex3(f2)
        for block in self.block3:
            f2 = block(f2, (120, 80))

        f2 = nlc_to_nchw(f2, (120, 80))

        f1 = torch.cat([f1, f2], 1)
        f1 = self.linear4(f1)
        o1 = f1
        f1 = nchw_to_nlc(f1)
        f1 = self.pex4(f1)
        for block in self.block4:
            f1 = block(f1, (240, 160))

        f1 = nlc_to_nchw(f1, (240, 160))
        o0 = f1

        # B, N, C = f1.shape
        # f1 = f1.permute(0, 2, 1).view(B, C, 240, 160)
        o0 = self.omlp1(o0)
        o2 = F.interpolate(self.omlp2(o2),scale_factor=4,mode='bilinear',align_corners=False)
        o3 = F.interpolate(self.omlp3(o3),scale_factor=8,mode='bilinear',align_corners=False)
        o4 = F.interpolate(self.omlp4(o4),scale_factor=16,mode='bilinear',align_corners=False)
        out = torch.cat([o0,o2,o3,o4],1)



        out = F.interpolate(self.cls_seg(out), scale_factor=2, mode='bilinear')
        return out


class MiT_Unet(nn.Module):
    def __init__(self,ncls):
        super().__init__()
        self.encoder = MIT(embed_dims=32,num_layers=[2,2,2,2])
        self.decoder = Transdecoder(ncls)
    def forward(self,x):
        f1,f2,f3,f4 = self.encoder(x)
        out = self.decoder(f1,f2,f3,f4)
        return out

# a = torch.randn(2,3,480,320)
# att = MIT(embed_dims=32,num_layers=[2,2,2,2])
# f1,f2,f3,f4 = att(a)
# de = Transdecoder()
#
# o = de(f1,f2,f3,f4)
# print(o.shape)
#
# net = MiT_Unet(12)
# print(net(a).shape)