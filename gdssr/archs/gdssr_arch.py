from operator import concat
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights, pixel_unshuffle
from basicsr.archs.rrdbnet_arch import RRDB
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np

class AffineModulate(nn.Module):

    def __init__(self, degradation_dim=128, num_feat=64):
        super(AffineModulate, self).__init__()
        #degradation_dim= 512 #256 #64
        self.fc = nn.Sequential(
            nn.Linear(degradation_dim, (degradation_dim + num_feat * 2) // 2),
            nn.ReLU(True),
            nn.Linear((degradation_dim + num_feat * 2) // 2, (degradation_dim + num_feat * 2) // 2),
            nn.ReLU(True),
            nn.Linear((degradation_dim + num_feat * 2) // 2, num_feat * 2),
        )
        default_init_weights([self.fc], 0.1)

    def forward(self, x, d):
        d = self.fc(d.view(d.size(0),-1)) #B,d->B,feat*2
        d = d.view(d.size(0), d.size(1), 1, 1) #[B,feat*2]->[B,feat*2,1,1]
        gamma, beta = torch.chunk(d, chunks=2, dim=1) #[B,feat,1,1]

        return (1 + gamma) * x + beta #SFT

@ARCH_REGISTRY.register()
class GDSSR_RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN. v2.1

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 scale=4,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 **kwargs):
        super(GDSSR_RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        # affine modulate list
        self.am_list = nn.ModuleList()
        num_degradation = kwargs['de_net']['num_degradation'] 
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch=num_grow_ch))
            self.am_list.append(AffineModulate(degradation_dim=kwargs['de_net']['num_feats'][-1]*num_degradation, num_feat=num_feat))

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.num_degradation = num_degradation
        self.num_block = num_block

        # degradation net
        kwargs['de_net']['num_in_ch'] = num_in_ch
        self.de_net = ARCH_REGISTRY.get(kwargs['de_net_type'])(**kwargs['de_net'])


    def forward(self, x, anchor=None, lqx=None, anchorx=None):
        b=x.shape[0]
        if lqx is None:
            lqx = x
        if anchor is not None:
            b,n,c,w,h = x.shape
            b,n,c,w_f,h_f = lqx.shape
            x = torch.cat([x,anchor], dim=1).contiguous()
            x_full = torch.cat([lqx,anchorx], dim=1).contiguous() 
            x = x.view(b*5,c,w,h)
            x_full = x_full.view(b*5,c,w_f,h_f)
            #print(x.shape,x_full.shape)
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_res = feat

        # predict the degradation embeddings and degrees
        degrees,degrees_feat = self.de_net(x_full)

        for i in range(self.num_block):
            feat = self.body[i](feat)
            feat = self.am_list[i](feat, degrees_feat)
        feat = self.conv_body(feat)
        feat = feat_res + feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        #print(degrees,concat_degrees)
        return out, degrees

@ARCH_REGISTRY.register()
class GDSSR_RRDBNet_decouple(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN. v2.1

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 scale=4,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 **kwargs):
        super(GDSSR_RRDBNet_decouple, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        # affine modulate list
        self.am_list = nn.ModuleList()
        num_degradation = kwargs['de_net']['num_degradation'] 
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch=num_grow_ch))
            self.am_list.append(AffineModulate(degradation_dim=kwargs['de_net']['num_feats'][-1]*num_degradation, num_feat=num_feat))
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.num_degradation = num_degradation
        self.num_block = num_block

        # degradation net
        kwargs['de_net']['num_in_ch'] = num_in_ch
        self.de_net = ARCH_REGISTRY.get(kwargs['de_net_type'])(**kwargs['de_net'])
        

    def forward(self, x, anchor=None, lqx=None, anchorx=None):
        b=x.shape[0]
        if lqx is None:
            lqx = x
        if anchor is not None:
            b,n,c,w,h = x.shape
            b,n,c,w_f,h_f = lqx.shape
            x = torch.cat([x,anchor], dim=1).contiguous()
            x_full = torch.cat([lqx,anchorx], dim=1).contiguous() 
            x = x.view(b*5,c,w,h)
            x_full = x_full.view(b*5,c,w_f,h_f)
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_res = feat

        # predict the degradation embeddings and degrees
        with torch.no_grad():
            degrees,degrees_feat = self.de_net(x_full)
        for i in range(self.num_block):
            feat = self.body[i](feat)
            feat = self.am_list[i](feat, degrees_feat)
        feat = self.conv_body(feat)
        feat = feat_res + feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out, degrees


@ARCH_REGISTRY.register()
class GDSSR_RRDBNet_test(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN. v2.1

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 scale=4,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 **kwargs):
        super(GDSSR_RRDBNet_test, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        # affine modulate list
        self.am_list = nn.ModuleList()
        num_degradation = kwargs['de_net']['num_degradation'] 
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch=num_grow_ch))
            self.am_list.append(AffineModulate(degradation_dim=kwargs['de_net']['num_feats'][-1]*num_degradation, num_feat=num_feat))

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.num_degradation = num_degradation
        self.num_block = num_block

        # degradation net
        kwargs['de_net']['num_in_ch'] = num_in_ch
        self.de_net = ARCH_REGISTRY.get(kwargs['de_net_type'])(**kwargs['de_net'])

    def forward(self, x, x_full=None, deg_feat=None):
        if x_full is None:
            x_full = x
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_res = feat

        # predict the degradation embeddings and degrees
        degrees, degrees_feat = self.de_net(x_full)
        if deg_feat is not None:
            degrees_feat = deg_feat


        for i in range(self.num_block):
            feat = self.body[i](feat)
            feat = self.am_list[i](feat, degrees_feat)

        feat = self.conv_body(feat)
        feat = feat_res + feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out, degrees