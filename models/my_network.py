import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import argparse
from torch.nn.utils import spectral_norm

import functools
# 相比于down128_sever作出的改动在于生成器的结构
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, fin,fout=None, dilation=1,norm_layer=None):
        super(ResnetBlock, self).__init__()
        if fout==None:
            fout=fin
        fmiddle = min(fin, fout)
        self.learned_shortcut = (fin != fout)
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            norm_layer(nn.Conv2d(in_channels=fin, out_channels=fmiddle, kernel_size=3, padding=0, dilation=dilation)),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            norm_layer((nn.Conv2d(in_channels=fmiddle, out_channels=fout, kernel_size=3, padding=0, dilation=1))))
        self.conv_s=nn.Sequential(
            nn.ReflectionPad2d(1),
            norm_layer(nn.Conv2d(in_channels=fin, out_channels=fout, kernel_size=3, padding=0, dilation=1)))

    def forward(self, x):
        out = self.shortcut(x) + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# 如果有多个鉴别器，那么鉴别器得到的结果是一个列表中装列表2，列表2 里面是鉴别器的结果，如果要求返回中间特征，就是多层特征图的列表
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, need_Feat,num_D,input_nc,output_nc,n_layers_D,norm_layer=None,ndf=64):
        super().__init__()
        self.need_Feat=need_Feat
        self.num_D=num_D
        for i in range(self.num_D):
            subnetD = NLayerDiscriminator(self.need_Feat,input_nc,output_nc,n_layers_D,norm_layer,ndf)
            self.add_module('discriminator_%d' % i, subnetD)


    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        for name, D in self.named_children():
            out = D(input)
            if not self.need_Feat:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


class PPADDiscriminator(nn.Module):
    def __init__(self,w,h,input_nc):
        super().__init__()
        subnetD = PPADNLayerDiscriminator(w,h,input_nc,kw=2)
        self.add_module('discriminator_%d' % 0, subnetD)
        subnetD = PPADNLayerDiscriminator(w,h,input_nc,kw=5)
        self.add_module('discriminator_%d' % 1, subnetD)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        for name, D in self.named_children():
            out = D(input)
            out = [out]
            result.append(out)
        return result


class PPADNLayerDiscriminator(nn.Module):
    def __init__(self,w,h,input_nc,kw=5):
        super().__init__()
        self.w=w
        self.h=h
        self.kw=kw
        n_layers_D = 4
        self.input_nc=input_nc
        ndf = 64
        norm_layer=get_nonspade_norm_layer("batch")
        self.n_layers_D=n_layers_D
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf

        sequence = [[nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2,padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1,n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

        if (self.w==512 and self.h==512 and kw==5) :
            self.linear = nn.Linear(512 * 32 * 32, 1)
        elif self.w==512 and self.h==512 and kw==2:
            self.linear=nn.Linear(512*33*33,1)
        elif self.w==256 and self.h==256 and kw==2:
            self.linear = nn.Linear(512 * 17 * 17, 1)
        elif self.w==256 and self.h==256 and kw==5:
            self.linear = nn.Linear(512 * 16 * 16, 1)
        elif self.w==512 and self.h==256 and kw==2:
            self.linear = nn.Linear(512 * 17 * 33, 1)
        elif self.w == 512 and self.h == 256 and kw == 5:
            self.linear = nn.Linear(512 * 16 * 32, 1)

        else:
            raise (f"你的尺寸为{self.w},{self.h},没有相应的尺寸，请检查linear")
    def forward(self, input):
        for name,submodel in self.named_children():
            if name=="linear":
                input = input.view(input.size(0), -1)
            input = submodel(input)
        return input


class NLayerDiscriminator(nn.Module):
    def __init__(self,need_Feat,input_nc,output_nc,n_layers_D,norm_layer=None,ndf=64,):
        super().__init__()
        self.need_Feat=need_Feat
        self.n_layers_D=n_layers_D
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1,n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, output_nc, kernel_size=kw, stride=1, padding=padw)]]
        # if opt.D_pool:
        #     sequence[-1]=sequence[-1]+[nn.MaxPool2d(11, 1, 5)]
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        if self.need_Feat:
            return results[1:]
        else:
            return results[-1]

class Pix2pixDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Pix2pixDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


# 只有 最后一个逐像素级别判别器
class Unet_Discriminator_1(nn.Module):
    def __init__(self, input_nc,output_nc,norm_layer):
        super().__init__()
        self.channels = [input_nc, 128, 128, 256, 256, 512, 512]
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(6):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], norm_layer, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2],norm_layer,  1))# 第一层上采样
        for i in range(1, 5):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i],norm_layer, 1))
        self.body_up.append(residual_block_D(2*self.channels[1], 64,norm_layer,  1))
        self.layer_up_last = nn.Conv2d(64, output_nc, 1, 1, 0)
    def forward(self, input):
        x = input
        #encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        #decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            catx=torch.cat((encoder_res[-i - 1], x), dim=1)
            x = self.body_up[i](catx)
        ans = self.layer_up_last(x)
        return ans

class residual_block_D(nn.Module):
    def __init__(self, fin, fout, norm_layer, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.up_or_down > 0:
            x = self.sampling(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.up_or_down < 0:
            x = self.sampling(x)
        x_s = x
        return x_s

class residual_block_G(nn.Module):
    def __init__(self, fin, fout, norm_layer, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.ReLU(), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.ReLU(), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.ReLU(), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.up_or_down > 0:
            x = self.sampling(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.up_or_down < 0:
            x = self.sampling(x)
        x_s = x
        return x_s

# norm组件，使用时先用此函数返回一个nrm加成器，norm加成器为一个函数，对一个层使用，可添加norm层
def get_nonspade_norm_layer( norm_type='spectralinstance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        elif hasattr(layer, 'out_c'):
            return getattr(layer, 'out_c')
        elif hasattr(layer, 'outc'):
            return getattr(layer, 'outc')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        subnorm_type=norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        # 使用了spectral 还可以叠加用其他norm

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc,norm_layer, num_downs, ngf=64,use_dropout=False,model="inpaint"):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.num_downs=num_downs
        self.downlist=[]
        self.uplist=[]
        self.model=model
        self.outdownlayer=nn.Sequential(
            nn.Conv2d(input_nc,ngf,4,2,1),
            nn.LeakyReLU(0.2,True)
        )
        self.downlist.append(self.outdownlayer)
        # construct unet structure
        def downlayer(inc,outc):
            layer=nn.Sequential(
                norm_layer(nn.Conv2d(inc,outc, kernel_size=4,stride=2, padding=1)),
                nn.LeakyReLU(0.2, True))
            return layer

        def uplayer(inc, outc,use_dropout):
            list=[
                norm_layer(nn.ConvTranspose2d(inc,outc,kernel_size=4, stride=2, padding=1)),
                nn.ReLU( True)]
            if use_dropout:
                list.append(nn.Dropout(0.5))
            return nn.Sequential(*list)

        inc=ngf
        outc=ngf*2
        #三层卷积到通道数为8*ngf，之后所有的下采样卷积的通道数都为8*ngf
        for i in range(3):
            self.add_module('down_%d' % (i+1), downlayer(inc,outc))
            self.downlist.append(self.get_submodule('down_%d' % (i+1)))
            inc=outc
            outc=outc*2

        for i in range(num_downs-5):
            self.add_module('down_%d' % (i+4), downlayer(8*ngf,8*ngf))
            self.downlist.append(self.get_submodule('down_%d' % (i+4)))

        self.mostinnerdownlayer=nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.mostinneruplayer= nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )

        # 上采样有拼接所以输入通道变成了ngf*16
        for i in range(num_downs-5):
            self.add_module('up_%d' % (i+1), uplayer(ngf*16,ngf*8,use_dropout))
            self.uplist.append(self.get_submodule('up_%d' % (i+1)))
        # 注意由于下采样过程要添加通道过来，所以实际上输出通道是输入通道的1/4
        inc=ngf*16
        outc=ngf*4
        for i in range(3):
            self.add_module('up_%d' % (i+num_downs-5+1), uplayer(inc,outc,use_dropout))
            self.uplist.append(self.get_submodule('up_%d' % (i+num_downs-5+1)))
            inc=inc//2
            outc=outc//2

        outuplayerlist=[nn.ConvTranspose2d(ngf*2,output_nc,4,2,1)]
        if model=="inpaint":
            outuplayerlist.append(nn.Tanh())
        else:
            outuplayerlist.append(nn.Sigmoid())
        self.outuplayer=nn.Sequential(*outuplayerlist)
        self.uplist.append(self.outuplayer)

    def forward(self, input):# alpha越大，隐私保护越强
        """Standard forward"""
        feature_list=[]
        for i in range(len(self.downlist)):
            input=self.downlist[i](input)
            feature_list.append(input)
        input=self.mostinnerdownlayer(input)
        input=self.mostinneruplayer(input)

        i=0
        for up in self.uplist:
            input=up(torch.cat((feature_list[-1-i],input),dim=1))
            i=i+1
        return input


def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 0, 0]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features,style_mean, style_std):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """

    content_mean, content_std = calc_mean_std(content_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

# adain的模块化，用于嵌入UNET中
class adain_module(nn.Module):
    def __init__(self):
        super(adain_module, self).__init__()
    def forward(self,x,style_mean,style_std):
        return adain(x,style_mean,style_std)

class alpha_adain_module(nn.Module):
    def __init__(self):
        super(alpha_adain_module, self).__init__()
    def forward(self,input,style_feature,alpha):
        style_mean,style_std=calc_mean_std(style_feature)
        # 如果alpha是浮点数就可以直接乘张量，否则需要让他的维度一致
        if torch.is_tensor(alpha):
            alpha=alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return input*alpha+adain(input,style_mean,style_std)*(1-alpha)

class orginal_adain_module(nn.Module):
    def __init__(self):
        super(orginal_adain_module, self).__init__()
    def forward(self,input,style_feature,is_returnmeanstd=False):
        style_mean,style_std=calc_mean_std(style_feature)
        if is_returnmeanstd:
            return adain(input,style_mean,style_std),style_mean,style_std
        return adain(input,style_mean,style_std)

# 学习风格均值风格标准差的模块
class LearnAdain(nn.Module):
    def __init__(self,adainrefertype,adainrefer_nc,current_nc,adainformin=False):
        super(LearnAdain, self).__init__()
        self.adainrefertype=adainrefertype
        self.adain=adain_module()
        self.current_nc=current_nc
        self.adainformin=adainformin
        if adainrefertype in ["mask","img","img_edge","edge"] :
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(adainrefer_nc, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Sequential(nn.Conv2d(128,self.current_nc , kernel_size=3, padding=1),nn.ReLU())
            self.mlp_beta =  nn.Sequential(nn.Conv2d(128, self.current_nc, kernel_size=3, padding=1),nn.ReLU())

        elif adainrefertype=="noise":
            self.mlp_shared = nn.Sequential(
                nn.Linear(128,256),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Sequential(nn.Linear(256,self.current_nc),nn.ReLU())
            self.mlp_beta =  nn.Sequential(nn.Linear(256,self.current_nc),nn.ReLU())

    def forward(self,input,adainrefer,alpha=0,is_returnmeanstd=False):
        mlp_shard = self.mlp_shared(adainrefer)
        gamma = self.mlp_gamma(mlp_shard)
        beta = self.mlp_beta(mlp_shard)
        if self.adainrefertype in ["mask","img","img_edge","edge"]:
            mask_style_mean = gamma.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(2)
            mask_style_std = beta.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(2)
        else:
            mask_style_mean = gamma.unsqueeze(2).unsqueeze(2)
            mask_style_std = beta.unsqueeze(2).unsqueeze(2)
        # 如果alpha是浮点数就可以直接乘张量，否则需要让他的维度一致
        if torch.is_tensor(alpha):
            alpha=alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if self.adainformin:
            input = self.adain(input, mask_style_mean, mask_style_std) * (1-alpha) + input * alpha
        else:
            input = self.adain(input, mask_style_mean, mask_style_std) * alpha + input * (1 - alpha)
        if is_returnmeanstd:
            return input,mask_style_mean,mask_style_std
        return input

def upsampleLayer(inplanes, outplanes, norm_layer,upsample='basic',outtest=False):
    # padding_type = 'zero'
    if upsample == 'basic':
        if not outtest:
            upconv = [norm_layer(nn.ConvTranspose2d(
                inplanes, outplanes, kernel_size=4, stride=2, padding=1))]
        else:
            upconv=[nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        if not outtest:
            upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      norm_layer(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=0)),
                      nn.ReLU(),
                      nn.ReflectionPad2d(1),
                      norm_layer(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))
                      ]
        else:
            upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=0),
                      nn.ReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)
                      ]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


# 上面Adain_UnetGenerator是自己学习方差和均值，这个由输入的style原图计算的房车和均值作为min的方差和均值。
# 所以Adain_UnetGenerator1需要两个输入了，一个是img，一个是edge_img
class Adain_UnetGenerator2(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc,norm_layer, num_downs, ngf=64,adainrefer_nc=1,adainrefertype="mask",
                 adain_start_layer=4,adain_num=1,adainformin=False,use_dropout=False,is_train=True,upsample="basic"):# adain_start_layer是倒着计算的，意味着上采样倒数第几层之前安排adain模块
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Adain_UnetGenerator2, self).__init__()
        self.is_train=is_train
        self.adain_start_layer=adain_start_layer
        self.adain_num=adain_num
        self.num_downs=num_downs
        self.downlist=[]
        self.uplist=[]
        self.outdownlayer=nn.Sequential(
            nn.Conv2d(input_nc,ngf,4,2,1),
            nn.LeakyReLU(0.2,True)
        )
        self.downlist.append(self.outdownlayer)
        # construct unet structure
        def downlayer(inc,outc):
            layer=nn.Sequential(
                norm_layer(nn.Conv2d(inc,outc, kernel_size=4,stride=2, padding=1)),
                nn.LeakyReLU(0.2, True))
            return layer

        def uplayer(inc, outc,use_dropout):
            list=upsampleLayer(inc,outc,norm_layer,upsample)+[nn.ReLU( True)]
            if use_dropout:
                list.append(nn.Dropout(0.5))
            return nn.Sequential(*list)

        inc=ngf
        outc=ngf*2
        #三层卷积到通道数为8*ngf，之后所有的下采样卷积的通道数都为8*ngf
        for i in range(3):
            self.add_module('down_%d' % (i+1), downlayer(inc,outc))
            self.downlist.append(self.get_submodule('down_%d' % (i+1)))
            inc=outc
            outc=outc*2

        for i in range(num_downs-5):
            self.add_module('down_%d' % (i+4), downlayer(8*ngf,8*ngf))
            self.downlist.append(self.get_submodule('down_%d' % (i+4)))

        self.mostinnerdownlayer=nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.mostinneruplayer= nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )

        # 上采样有拼接所以输入通道变成了ngf*16
        for i in range(num_downs-5):
            self.add_module('up_%d' % (i+1), uplayer(ngf*16,ngf*8,use_dropout))
            self.uplist.append(self.get_submodule('up_%d' % (i+1)))
        # 注意由于下采样过程要添加通道过来，所以实际上输出通道是输入通道的1/4

        inc=ngf*16
        outc=ngf*4
        for i in range(3):
            self.add_module('up_%d' % (i+num_downs-5+1), uplayer(inc,outc,use_dropout))
            self.uplist.append(self.get_submodule('up_%d' % (i+num_downs-5+1)))
            inc=inc//2
            outc=outc//2
        outuplayerlsit=upsampleLayer(ngf*2,output_nc,norm_layer,outtest=True)+[nn.Tanh()]
        self.outuplayer=nn.Sequential(*outuplayerlsit)
        self.uplist.append(self.outuplayer)

        for i in range(self.adain_num):
            setattr(self,"adain_%d" % (self.adain_num-i),orginal_adain_module())#倒着一个个装，但是序号是顺着的

        for i in range(self.adain_num):
            number_adain=adain_start_layer+i
            current_nc=min(ngf*(2**(number_adain-1)),512)
            setattr(self,"learnadain_%d" % (self.adain_num-i),LearnAdain(adainrefertype,adainrefer_nc,current_nc,adainformin))#倒着一个个装，但是序号是顺着的

    def forward(self, input,adainrefer,alpha=0,img=None):# alpha越大，隐私保护越强
        """Standard forward"""
        inputlist=[img,input]
        feature_list=[[],[]]
        meanstdlist=[[],[]]
        index_start_adain=self.num_downs-2-(self.adain_start_layer+self.adain_num-1)# 正数第一个层x后有adain模块，这个x层的在uplist中的索引
        for k in range(2):
            for i in range(len(self.downlist)):
                inputlist[k]=self.downlist[i](inputlist[k])
                feature_list[k].append(inputlist[k])
            inputlist[k]=self.mostinnerdownlayer(inputlist[k])
            inputlist[k]=self.mostinneruplayer(inputlist[k])
        # for i in range(len(feature_list[0])):
        #     print(feature_list[0][i].size())
        for i,up in enumerate(self.uplist):
            if i == index_start_adain:
                inputlist.append(inputlist[1])  # 0自我风格，1minfake，2maxfake
            if i-index_start_adain+1<=self.adain_num:# 在最后一层adain之前，input都需要向前计算
                for k in range(len(inputlist)):
                    inputlist[k]=torch.cat([feature_list[(k>=1)][-1 - i], inputlist[k]],dim=1)
                    inputlist[k] = up(inputlist[k])
                    if (i-index_start_adain+1<=self.adain_num and i>=index_start_adain ):
                        if k==0:#如果是img风格特征,就保存特征
                            stylefeature=inputlist[k]
                        else:#如果是input特征，就拿保存的style 均值方差来做adain
                            inputlist[2],maxmean,maxstd=getattr(self,"learnadain_%d"%(i-index_start_adain+1))\
                                (inputlist[2],adainrefer,alpha=1,is_returnmeanstd=True)
                            inputlist[1],minmean,minstd=getattr(self,"adain_%d"%(i-index_start_adain+1))(inputlist[1],\
                                                                 stylefeature,True)

                            meanstdlist[0].append((minmean,minstd))
                            meanstdlist[1].append((maxmean,maxstd))
            else:
                if not self.is_train:
                    if i==index_start_adain+self.adain_num:
                        if torch.is_tensor(alpha):
                            alpha = alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                        inputlist[1] = inputlist[1] * (1 - alpha) + inputlist[2] * alpha
                inputlist[1] = torch.cat([feature_list[1][-1 - i], inputlist[1]], dim=1)
                inputlist[1] = up(inputlist[1])
                if self.is_train:
                    inputlist[2] = torch.cat([feature_list[1][-1 - i], inputlist[2]], dim=1)
                    inputlist[2] = up(inputlist[2])
        if not self.is_train:
            return inputlist[1]
        else:
            return inputlist[1],inputlist[2],meanstdlist


class InpaintGenerator(nn.Module):
    def __init__(self, input_nc,output_nc,residual_blocks=8):
        super(InpaintGenerator, self).__init__()
        norm_layer=get_nonspade_norm_layer("instance")
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 256,norm_layer=norm_layer)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=7, padding=0),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x



class PPADGenerator(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64,num_downs=7,use_dropout=False,model="inpaint"):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(PPADGenerator, self).__init__()
        self.num_downs=num_downs
        self.downlist=[]
        self.uplist=[]
        self.model=model
        padw = 1
        norm_layer=get_nonspade_norm_layer("batch")
        self.num_downs=8
        self.outdownlayer=nn.Sequential(
            nn.Conv2d(input_nc,ngf,5,2,1),
            nn.LeakyReLU(0.2,True)
        )
        self.downlist.append(self.outdownlayer)
        # construct unet structure
        def downlayer(inc,outc):
            layer=nn.Sequential(
                norm_layer(nn.Conv2d(inc,outc, kernel_size=5,stride=2, padding=1)),
                nn.LeakyReLU(0.2, True))
            return layer

        def uplayer(inc, outc,use_dropout):
            list=[
                norm_layer(nn.ConvTranspose2d(inc,outc,kernel_size=5, stride=2, padding=1)),
                nn.ReLU( True)]
            if use_dropout:
                list.append(nn.Dropout(0.5))
            return nn.Sequential(*list)

        inc=ngf
        outc=ngf*2
        #三层卷积到通道数为8*ngf，之后所有的下采样卷积的通道数都为8*ngf
        for i in range(3):
            self.add_module('down_%d' % (i+1), downlayer(inc,outc))
            self.downlist.append(self.get_submodule('down_%d' % (i+1)))
            inc=outc
            outc=outc*2

        for i in range(num_downs-5):
            self.add_module('down_%d' % (i+4), downlayer(8*ngf,8*ngf))
            self.downlist.append(self.get_submodule('down_%d' % (i+4)))

        self.mostinnerdownlayer=nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.mostinneruplayer= nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=5, stride=2, padding=1),
            nn.ReLU(True),
        )

        # 上采样有拼接所以输入通道变成了ngf*16
        for i in range(num_downs-5):
            self.add_module('up_%d' % (i+1), uplayer(ngf*16,ngf*8,use_dropout))
            self.uplist.append(self.get_submodule('up_%d' % (i+1)))
        # 注意由于下采样过程要添加通道过来，所以实际上输出通道是输入通道的1/4
        inc=ngf*16
        outc=ngf*4
        for i in range(3):
            self.add_module('up_%d' % (i+num_downs-5+1), uplayer(inc,outc,use_dropout))
            self.uplist.append(self.get_submodule('up_%d' % (i+num_downs-5+1)))
            inc=inc//2
            outc=outc//2

        outuplayerlist=[nn.ConvTranspose2d(ngf*2,output_nc,5,2,1,output_padding=1)]
        if model=="inpaint":
            outuplayerlist.append(nn.Tanh())
        else:
            outuplayerlist.append(nn.Sigmoid())
        self.outuplayer=nn.Sequential(*outuplayerlist)
        self.uplist.append(self.outuplayer)

    def forward(self, input):# alpha越大，隐私保护越强
        """Standard forward"""
        feature_list=[]
        for i in range(len(self.downlist)):
            input=self.downlist[i](input)
            feature_list.append(input)
        input=self.mostinnerdownlayer(input)
        input=self.mostinneruplayer(input)

        i=0
        for up in self.uplist:
            input=up(torch.cat((feature_list[-1-i],input),dim=1))
            i=i+1
        return input
