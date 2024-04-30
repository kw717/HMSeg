
import torch
import torch.nn as nn
import torch.nn.functional as F

from .u2net import *

class AG_block(nn.Module):
    def __init__(self,inc1,inc2,midc):
        super(AG_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc1,midc,1),
            nn.BatchNorm2d(midc)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inc2, midc, 1),
            nn.BatchNorm2d(midc)
        )
        self.relu = nn.ReLU()
        self.midconv = nn.Sequential(
            nn.Conv2d(midc, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self,in1,in2):
        mid = self.conv1(in1)+self.conv2(in2)
        att = self.midconv(self.relu(mid))
        out = in2 * att
        return out

class AP_block(nn.Module):
    def __init__(self,inc,outc,rate = [1,6,12,18]):
        super(AP_block, self).__init__()

        self.rc1 = REBNCONV(inc,inc,rate[0])
        self.rc2 = REBNCONV(inc, inc, rate[1])
        self.rc3 = REBNCONV(inc, inc, rate[2])
        self.rc4 = REBNCONV(inc, inc, rate[3])
        self.outc = REBNCONV(inc*4,outc,1)
        self.dp = nn.Dropout(0.1)

    def forward(self,x):
        r1 = self.rc1(x)
        r2 = self.rc2(x)
        r3 = self.rc3(x)
        r4 = self.rc4(x)
        r = torch.cat([r1,r2,r3,r4],dim=1)
        out = self.dp(self.outc(r))
        return out

class AA(nn.Module):
    def __init__(self,inc1,inc2,inc3,rate = [1,6,12,18]):
        super(AA, self).__init__()
        self.ap = AP_block(inc3,inc2,rate)
        self.ag = AG_block(inc1,inc2,inc2//2)

    def forward(self,in1,in2,in3):
        size = in2.size()[2:]
        in1 = F.interpolate(in1,size,mode='bilinear',align_corners=False)
        in3 = F.interpolate(in3, size, mode='bilinear', align_corners=False)
        ag = self.ag(in1,in2)
        ap = self.ap(in3)
        a = ap+ag
        out = torch.cat([in1,a],dim=1)
        return out
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class MaanuNet(nn.Module):

    def __init__(self,in_ch=3,out_ch=12):
        super(MaanuNet,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        # AA module
        self.AA4 = AA(512,512,256,rate=[1,2,4,8])
        self.AA3 = AA(256,256,128)
        self.AA2 = AA(128,128,64)
        self.AA1 = AA(64,64,in_ch)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Sequential(nn.Conv2d(6*out_ch,6*out_ch,3,1,1),nn.Conv2d(6*out_ch,out_ch,1))

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(self.AA4(hx5dup,hx4,hx3))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(self.AA3(hx4dup,hx3,hx2))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(self.AA2(hx3dup,hx2,hx1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(self.AA1(hx2dup,hx1,x))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return d0

