
import vgg
from lib.MCGFuse import MCG
from lib.cnn_trans_fuse import *
from lib.ConTriNet import BCFG
# from lib.SIA import *
from lib.BCFFM import BCFFM
from lib.swin_transformer import SwinTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn import Conv2d, Parameter, Softmax

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)
def convblock(in_, out_, ks, st=1, pad=0,dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class CASA(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(CASA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        x = x.mul(self.sigmoid(out))

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

# 全局信息感知（Global Information Aware Module, GIAM）
class GlobalInfo(nn.Module):  # ASPP

    def __init__(self):  # dim = 512*2  in_dim = 512  down_dim = 128
        super(GlobalInfo, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(128), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.PReLU()
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1), nn.BatchNorm2d(128),  nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * 128, 1024, kernel_size=1), nn.BatchNorm2d(1024), nn.PReLU()
        )

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')

        x_fuse = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

        return x_fuse
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#渐进式跨模态协同注意力融合模块（Progressive Cross-modal Collaborative Attention Fusion Module, PCCAFM）
class PCCAFM(nn.Module):
    def __init__(self, in_channel):
        super(PCCAFM, self).__init__()
        self.relu = nn.ReLU(True)
        self.rgb_branch1 = BasicConv2d(in_channel, (in_channel // 4), 3, padding=1, dilation=1)
        self.rgb_branch2 = BasicConv2d(in_channel, (in_channel // 4), 3, padding=3, dilation=3)
        self.rgb_branch3 = BasicConv2d(in_channel, (in_channel // 4), 3, padding=5, dilation=5)
        self.rgb_branch4 = BasicConv2d(in_channel, in_channel // 4, 3, padding=7, dilation=7)

        self.d_branch1 = BasicConv2d(in_channel, in_channel // 4, 3, padding=1, dilation=1)
        self.d_branch2 = BasicConv2d(in_channel, in_channel // 4, 3, padding=3, dilation=3)
        self.d_branch3 = BasicConv2d(in_channel, in_channel // 4, 3, padding=5, dilation=5)
        self.d_branch4 = BasicConv2d(in_channel, in_channel // 4, 3, padding=7, dilation=7)

        self.rgb_branch1_sa = SpatialAttention()
        self.rgb_branch2_sa = SpatialAttention()
        self.rgb_branch3_sa = SpatialAttention()
        self.rgb_branch4_sa = SpatialAttention()

        self.rgb_branch1_ca = ChannelAttention(int(in_channel / 4))
        self.rgb_branch2_ca = ChannelAttention(int(in_channel / 4))
        self.rgb_branch3_ca = ChannelAttention(int(in_channel / 4))
        self.rgb_branch4_ca = ChannelAttention(int(in_channel / 4))

        self.r_branch1_sa = SpatialAttention()
        self.r_branch2_sa = SpatialAttention()
        self.r_branch3_sa = SpatialAttention()
        self.r_branch4_sa = SpatialAttention()

        self.r_branch1_ca = ChannelAttention(int(in_channel / 4))
        self.r_branch2_ca = ChannelAttention(int(in_channel / 4))
        self.r_branch3_ca = ChannelAttention(int(in_channel / 4))
        self.r_branch4_ca = ChannelAttention(int(in_channel / 4))

        self.ca = ChannelAttention(in_channel)


    def forward(self, x_rgb, x_d, x_t):
        x1_rgb = self.rgb_branch1(x_rgb)
        x2_rgb = self.rgb_branch2(x_rgb)
        x3_rgb = self.rgb_branch3(x_rgb)
        x4_rgb = self.rgb_branch4(x_rgb)

        x1_d = self.d_branch1(x_d)
        x2_d = self.d_branch2(x_d)
        x3_d = self.d_branch3(x_d)
        x4_d = self.d_branch4(x_d)

        x1_t = self.d_branch1(x_t)
        x2_t = self.d_branch2(x_t)
        x3_t = self.d_branch3(x_t)
        x4_t = self.d_branch4(x_t)

        x1_rgb_ca = x1_rgb.mul(self.rgb_branch1_ca(x1_rgb))
        x1_d_sa = x1_d.mul(self.rgb_branch1_sa(x1_rgb_ca))
        x1_t_sa = x1_t.mul(self.rgb_branch1_sa(x1_rgb_ca))
        x1_d = x1_d + x1_d_sa
        x1_d_ca = x1_d.mul(self.r_branch1_ca(x1_d))
        x1_t = x1_t + x1_t_sa
        x1_t_ca = x1_t.mul(self.r_branch1_ca(x1_t))
        x1_rgb_sa = x1_rgb.mul(self.r_branch1_sa(x1_d_ca))#vd
        x1_rgb_sa = x1_rgb_sa.mul(self.r_branch1_sa(x1_t_ca))
        x2_rgb = x2_rgb + x1_rgb_sa

        x2_rgb_ca = x2_rgb.mul(self.rgb_branch2_ca(x2_rgb))
        x2_d_sa = x2_d.mul(self.rgb_branch2_sa(x2_rgb_ca))
        x2_t_sa = x2_t.mul(self.rgb_branch2_sa(x2_rgb_ca))
        x2_d = x2_d + x2_d_sa
        x2_d_ca = x2_d.mul(self.r_branch2_ca(x2_d))
        x2_t = x2_t + x2_t_sa
        x2_t_ca = x2_t.mul(self.r_branch2_ca(x2_t))
        x2_rgb_sa = x2_rgb.mul(self.r_branch2_sa(x2_d_ca))
        x2_rgb_sa = x2_rgb_sa.mul(self.r_branch2_sa(x2_t_ca))
        x3_rgb = x3_rgb + x2_rgb_sa

        x3_rgb_ca = x3_rgb.mul(self.rgb_branch3_ca(x3_rgb))
        x3_d_sa = x3_d.mul(self.rgb_branch3_sa(x3_rgb_ca))
        x3_t_sa = x3_t.mul(self.rgb_branch3_sa(x3_rgb_ca))
        x3_d = x3_d + x3_d_sa
        x3_d_ca = x3_d.mul(self.r_branch3_ca(x3_d))
        x3_t = x3_t + x3_t_sa
        x3_t_ca = x3_t.mul(self.r_branch3_ca(x3_t))
        x3_rgb_sa = x3_rgb.mul(self.r_branch3_sa(x3_d_ca))
        x3_rgb_sa = x3_rgb_sa.mul(self.r_branch3_sa(x3_t_ca))
        x4_rgb = x4_rgb + x3_rgb_sa

        x4_rgb_ca = x4_rgb.mul(self.rgb_branch4_ca(x4_rgb))
        x4_d_sa = x4_d.mul(self.rgb_branch4_sa(x4_rgb_ca))
        x4_t_sa = x4_t.mul(self.rgb_branch4_sa(x4_rgb_ca))
        x4_d = x4_d + x4_d_sa
        x4_d_ca = x4_d.mul(self.r_branch4_ca(x4_d))
        x4_t = x4_t + x4_t_sa
        x4_t_ca = x4_t.mul(self.r_branch4_ca(x4_t))
        x4_rgb_sa = x4_rgb.mul(self.r_branch4_sa(x4_d_ca))
        x4_rgb_sa = x4_rgb_sa.mul(self.r_branch4_sa(x4_t_ca))

        y = torch.cat((x1_rgb_sa, x2_rgb_sa, x3_rgb_sa, x4_rgb_sa), 1)
        y_ca = y.mul(self.ca(y))

        #z = y_ca + x_rgb + x_d + x_t
        z = y_ca + x_rgb
        # then try z = y_ca + x_rgb + x_d, choose the better performance

        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.glo = PCCAFM(1024)
        # self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(1024, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.mcg_vd4 = MCG(1024,1024,12,12)
        self.mcg_vd3 = MCG(512,512,24,24)
        self.mcg_vd2 = MCG(256,256,48,48)
        self.mcg_vd1 = MCG(128,128,96,96)

        self.mcg_vt4 = MCG(1024,1024,12,12)
        self.mcg_vt3 = MCG(512,512,24,24)
        self.mcg_vt2 = MCG(256,256,48,48)
        self.mcg_vt1 = MCG(128,128,96,96)

        self.bcffm4 = BCFFM(1024, 1024)
        self.bcffm3 = BCFFM(512, 1024)
        self.bcffm2 = BCFFM(256, 512)
        self.bcffm1 = BCFFM(128, 256)

        self.rgb_global = GlobalInfo()
        self.d_global = GlobalInfo()
        self.t_global = GlobalInfo()

        self.fu_0 = BCFG(128, 128)
        self.fu_1 = BCFG(256, 256)
        self.fu_2 = BCFG(512, 512)
        self.fu_3 = BCFG(1024, 1024)

        # self.ei4 = EI(1024)
        # self.ei3 = EI(512)
        # self.ei2 = EI(256)
        # self.ei1 = EI(128)

        self.res4 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        # 通道注意力模块
        self.ca4 = ChannelAttention(1024)
        self.ca3 = ChannelAttention(512)
        self.ca2 = ChannelAttention(256)
        self.ca1 = ChannelAttention(128)

        self.conv1024to512 = conv3x3(1024,512)
        self.conv512to256 = conv3x3(512,256)
        self.conv256to128 = conv3x3(256,128)
        self.conv128to64 = conv3x3(64,128)






    def forward(self, rgb, t, d):
        # print(rgb[0].shape)#128,96
        # print(rgb[1].shape)#256,48
        # print(rgb[2].shape)#512,24
        # print(rgb[3].shape)#1024,12

        # xsize = rgb[0].size()[2:]
        rgb_global = self.rgb_global(rgb[3])  # 4,512,22,22
        d_global = self.d_global(d[3])  # 4,512,22,22
        t_global = self.t_global(t[3])  # 4,512,22,22
        rgbdt_g = self.glo(rgb_global,d_global,t_global)

        # rgbdt_g = self.glo(rgb[3],t[3],d[3])
        # rgbdt_g = rgb[3]+t[3]+d[3]
        # 第4层 (1024,1024,12,12)
        # rgbd4 = rgb[3] + d[3]
        # rgbt4 = rgb[3] + t[3]
        rgbd4 = self.mcg_vd4(rgb[3], d[3])
        rgbt4 = self.mcg_vt4(rgb[3], t[3])
        ful_3 = self.fu_3(rgbd4, rgbt4)#512
        # ful_3 = rgbd4 + rgbt4
        # rgbdt4 = rgbdt_g + ful_3
        rgbdt4 = self.bcffm4(ful_3,rgbdt_g)
        res4_out = self.res4(rgbdt4) + rgbdt4
        att4_out = self.ca4(res4_out) * res4_out#1024,12,12
        s4 = self.S4(att4_out)

        # 第3层 (512,512,24,24)
        # rgbd3 = rgb[2] + d[2]
        # rgbt3 = rgb[2] + t[2]
        rgbd3 = self.mcg_vd3(rgb[2], d[2])
        rgbt3 = self.mcg_vt3(rgb[2], t[2])
        ful_2 = self.fu_2(rgbd3, rgbt3)#512,24,24
        # ful_2 = rgbd3 + rgbt3
        # att4_out = self.conv1024to512(att4_out)
        att4_out = F.interpolate(att4_out, [24,24], mode='bilinear', align_corners=True)
        # rgbdt3 = att4_out + ful_2
        rgbdt3 = self.bcffm3(ful_2,att4_out)
        res3_out = self.res3(rgbdt3) + rgbdt3
        att3_out = self.ca3(res3_out) * res3_out#512
        s3 = self.S3(att3_out)

        # 第2层 (256,256,48,48)
        # rgbd2 = rgb[1] + d[1]
        # rgbt2 = rgb[1] + t[1]
        rgbd2 = self.mcg_vd2(rgb[1], d[1])
        rgbt2 = self.mcg_vt2(rgb[1], t[1])
        ful_1 = self.fu_1(rgbd2, rgbt2)#256,48,48
        # ful_1 = rgbd2 + rgbt2
        # att3_out = self.conv512to256(att3_out)
        att3_out = F.interpolate(att3_out, [48,48], mode='bilinear', align_corners=True)
        # rgbdt2 = att3_out + ful_1
        rgbdt2 = self.bcffm2(ful_1,att3_out)
        res2_out = self.res2(rgbdt2) + rgbdt2
        att2_out = self.ca2(res2_out) * res2_out#256

        # att2_out = self.ei2(att2_out)

        s2 = self.S2(att2_out)

        # 第1层 (128,128,96,96)
        # rgbd1 = rgb[0] + d[0]
        # rgbt1 = rgb[0] + t[0]
        rgbd1 = self.mcg_vd1(rgb[0], d[0])
        rgbt1 = self.mcg_vt1(rgb[0], t[0])
        ful_0 = self.fu_0(rgbd1, rgbt1)
        # ful_0 = rgbd1 + rgbt1
        # att2_out = self.conv256to128(att2_out)
        att2_out = F.interpolate(att2_out, [96,96], mode='bilinear', align_corners=True)
        # rgbdt1 = att2_out + ful_0
        rgbdt1 = self.bcffm1(ful_0,att2_out)
        res1_out = self.res1(rgbdt1) + rgbdt1
        att1_out = self.ca1(res1_out) * res1_out#128

        # att1_out = self.ei1(att1_out)

        s1 = self.S1(att1_out)



        s4 = F.interpolate(s4, [384,384], mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, [384,384], mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, [384,384], mode='bilinear', align_corners=True)
        s1 = F.interpolate(s1, [384,384], mode='bilinear', align_corners=True)

        return s1, s2, s3, s4, rgbdt_g, rgbdt4, rgbdt3, rgbdt2, rgbdt1

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        # self.rgb_net = vgg.a_vgg16()
        # self.t_net = vgg.a_vgg16()
        # self.d_net = vgg.a_vgg16()
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])


        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb, t, d):
        # rgb_f = self.rgb_net(rgb)
        # t_f = self.t_net(t)
        # d_f = self.d_net(d)
        score_list_t, score_PE = self.swin1(t)
        score_list_rgb, score_PE = self.swin1(rgb)
        score_list_d, score_PE = self.swin1(d)

        rgb_f = [score_list_rgb[0], score_list_rgb[1], score_list_rgb[2], score_list_rgb[3]]
        t_f = [score_list_t[0], score_list_t[1], score_list_t[2], score_list_t[3]]
        d_f = [score_list_d[0], score_list_d[1], score_list_d[2], score_list_d[3]]

        # x1 = [score_list_rgb[0], score_list_t[0], score_list_d[0]]
        # x2 = [score_list_rgb[1], score_list_t[1], score_list_d[1]]
        # x3 = [score_list_rgb[2], score_list_t[2], score_list_d[2]]
        # x4 = [score_list_rgb[3], score_list_t[3], score_list_d[3]]

        score1, score2, score3, score4, rgbdt_g, rgbdt4, rgbdt3, rgbdt2, rgbdt1 = self.decoder(rgb_f, t_f, d_f)
        return score1, score2, score3, score4,self.sigmoid(score1), self.sigmoid(score2), self.sigmoid(score3), self.sigmoid(score4), rgbdt_g, rgbdt4, rgbdt3, rgbdt2, rgbdt1

    # def load_pretrained_model(self):
    #     st=torch.load("vgg16.pth")
    #     st2={}
    #     for key in st.keys():
    #         st2['base.'+key]=st[key]
    #     self.rgb_net.load_state_dict(st2)
    #     self.t_net.load_state_dict(st2)
    #     self.d_net.load_state_dict(st2)
    #     print('loading pretrained model success!')

    def load_pretrained_model(self):
        self.swin1.load_state_dict(torch.load('D:\Chinese\HWSI\swin_base_patch4_window12_384_22k.pth')['model'],strict=False)
        print('loading pretrained model success!')