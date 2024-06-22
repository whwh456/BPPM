import torch.nn as nn
import  torch
import torch.nn.functional as F
import numpy as np
import torchvision
# VGG architecter, used for the perceptual loss using a pretrained VGG network
from skimage.feature import canny
import util.my_visualizer as mv
import util.util


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False,pretrained_path=None,device=None):
        super().__init__()
        if pretrained_path==None:
            vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        else:
            vgg19_model = torchvision.models.vgg19(pretrained=False)
            state_dict = torch.load(pretrained_path, map_location=device)
            vgg19_model.load_state_dict(state_dict)
            vgg_pretrained_features = vgg19_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None,device=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        self.device=device
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input).to(self.device)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input).to(self.device)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).to(self.device)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

# target_is_real是想让他判别是否为真，而不是输入是否为真
    def __call__(self, input, target_is_real, for_discriminator=True,label=None,mask=None):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class SegGANLoss(nn.Module):
    def __init__(self,seggantype,device,segclassnum=21,weight=0.5):# 此处的segclassnum算上了虚假类，即是D_output_nc
        super(SegGANLoss, self).__init__()
        self.seggantype=seggantype
        self.segclassnum=segclassnum
        self.device=device
        self.weight=weight
        assert weight<=1.0 and weight>=0

    def loss(self,input,target_is_real,for_discriminator,label,mask=None):
        weight=torch.ones_like(label).to(self.device)
        if mask!=None and self.weight!=0.5:
            weight=(weight*self.weight*mask+weight*(1-mask)*(1-self.weight))*2
        N,C,H,W=input.size()
        label=F.interpolate(label,(H,W)).long().requires_grad_(False).squeeze()

        if self.seggantype=="seg":# 把生成图像进行像素级分类， 0——18为真实的街景图分类，最后一类代表虚假
            if target_is_real:
                labelidx=label
            else:
                fake_label_tensor = torch.LongTensor(label.size()).fill_(self.segclassnum-1).to(self.device)
                fake_label_tensor.requires_grad_(False)
                labelidx=fake_label_tensor

        elif self.seggantype=="mask":
            if target_is_real:
                labelidx=label
            else:
                labelidx=label+2
        # print("计算mask损失时输入input的形状")
        loss=F.cross_entropy(input, labelidx, ignore_index=19,reduction="none")
        # print("计算mask损失后loss的形状")
        loss=(loss*weight[:,0,:,:]).mean()
        return loss

    def forward(self,input, target_is_real,for_discriminator,label,mask):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real,label,mask)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, label,mask)

# 对于一个列表的预测值，列表最后一个使用SegGANLoss,除此以外前面的使用GANLoss
class MixGANLoss(nn.Module):
    def __init__(self,ganmode1,device,weight=0.5,ganmode2="mask"):
        super(MixGANLoss, self).__init__()
        ganmode1=ganmode1[4:]
        self.criteriengan=GANLoss(device=device,gan_mode=ganmode1)
        self.criterienseggan=SegGANLoss(device=device,weight=weight,seggantype=ganmode2)
    def forward(self,x,target_is_real,for_discriminator,label,mask):
        assert isinstance(x, list)==True
        loss1=self.criterienseggan(x[-1][0],target_is_real,for_discriminator,label,mask)
        loss2=self.criteriengan(x[0:len(x)-1],target_is_real,for_discriminator,label,mask)
        return loss1+loss2

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids,location):
        super(VGGLoss, self).__init__()
        if location == 1:
            self.vgg = VGG19(pretrained_path="/haowang_ms/chenlqn/mycode/pretrained/vgg19-dcbb9e9d.pth",device=gpu_ids)
        else:
            self.vgg = VGG19(device=gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.device=gpu_ids
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGGgramLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, gpu_ids,location):
        super(VGGgramLoss, self).__init__()
        if location == 1:
            self.vgg = VGG19(pretrained_path="/haowang_ms/chenlqn/mycode/pretrained/vgg19-dcbb9e9d.pth").to(gpu_ids)
        else:
            self.vgg = VGG19().to(gpu_ids)
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # Compute loss
        style_loss = 0.0
        for i in range(len(x_vgg)):
            style_loss += self.criterion(self.compute_gram(x_vgg[i]), self.compute_gram(y_vgg[i]))
            # style_loss += self.criterion(self.compute_gram(x_vgg[i]), self.compute_gram(y_vgg[i]))
            # style_loss += self.criterion(self.compute_gram(x_vgg[i]), self.compute_gram(y_vgg[i]))
            # style_loss += self.criterion(self.compute_gram(x_vgg[i]), self.compute_gram(y_vgg[i]))
        return style_loss

class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        # feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat =  torch.clamp(feat, min=-1, max=1)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids

class PatchSimLoss(nn.Module):
    def __init__(self,patch_nums=256, patch_size=None, norm=True):
        super(PatchSimLoss, self).__init__()
        self.patchsim=PatchSim(patch_nums=patch_nums, patch_size=patch_size, norm=norm)
        self.criterien=nn.L1Loss()
    def __call__(self,x,y):
        y=(y/y.max())*2-1
        if isinstance(x, list):
            loss = 0
            for pred_i in x:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                y = F.interpolate(y, size=(pred_i.size(2), pred_i.size(3)),
                                  mode='bilinear')
                if pred_i.size(1) != y.size(1):
                    y = y + 2
                    onehot_mask = F.one_hot(y, 4)
                    assert pred_i.size(1) == y.size(1)
                patch_sim_x, patch_ids = self.patchsim(pred_i)
                patch_sim_y ,_= self.patchsim(y, patch_ids)
                new_loss=self.criterien(patch_sim_x, patch_sim_y)
                loss += new_loss
            return loss / len(x)
        else:
            y = F.interpolate(y, size=(x.size(2), x.size(3)),
                              mode='bilinear')
            patch_sim_x,patch_ids=self.patchsim(x)
            patch_sim_y,_=self.patchsim(y,patch_ids)
            return self.criterien(patch_sim_x,patch_sim_y)



class DifcannyLoss(nn.Module):
    def __init__(self,sigma,high,device):
        super(DifcannyLoss, self).__init__()
        self.sigma=sigma
        self.high=high
        self.criterien=nn.L1Loss()
        self.device=device
    def __call__(self, x,y,mask):
        N,C,H,W=x.size()
        x_numpy=x.detach().cpu().numpy()
        loss=0
        for i in range(N):
            x_edge=torch.Tensor(canny(x_numpy[i][0],sigma=self.sigma,high_threshold=self.high)).to(self.device)
            loss=loss+self.criterien(x_edge*mask,y[i][0]*mask)
        return loss



class edgesumConv(nn.Module):
    def __init__(self,k=3,stride=2):
        super(edgesumConv, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=k, stride=stride,padding=0,bias=False)
        self.kernel=torch.ones([1,1,k,k],dtype=torch.float)
        self.conv_layer.weight = nn.Parameter(self.kernel, requires_grad=False)
    def forward(self,x):
        return self.conv_layer(x)


# 用于鉴别两个edge图像的相似程度
class Cal_Div_Loss(nn.Module):
    def __init__(self,x,device):# x为下采样层数
        super(Cal_Div_Loss,self).__init__()
        self.device=device
        self.layer_num=x
        self.edgesumconv=edgesumConv().to(device)
    def forward(self,x,y,alpha,is_return_tensor=False):
        # 特征有layernum+1个特征，因为除了layernum个卷积层，还有一个原始输入
        # 档次有layernum+2，从全正，逐渐到1正，再到0正，所以有layernum+2个档次
        # 最终的sumconvloss_tensor的通道维度为layernum+2，layernum个卷积层+原始输入+一个全图损失，其形状为（N,layernum+2）
        k_layer_list=torch.div(alpha, (1/(self.layer_num+2))).long()
        sumconvloss_tensor=torch.zeros((alpha.size(0),self.layer_num+2),requires_grad=False).to(self.device)

        xmaplist = [x]  # 注意第一维度是层次，第二维度是N
        ymaplist = [y]
        fuhao = -1
        for i in range(self.layer_num):
            xmaplist.append(self.edgesumconv(xmaplist[-1]))
            ymaplist.append(self.edgesumconv(ymaplist[-1]))

        for k,k_layer in enumerate(k_layer_list):#这是遍历的N的维度
            mse=nn.L1Loss().to(self.device)
            # for i,j in zip(poolxlist,poolylist):
            #     poolloss+=mse(i,j)
            for i in range(self.layer_num+1):#注意0正的时候，不需要作任何变化，所以范围为self.layer_num+1而不是layer_num+2
                if i>=k_layer:
                    fuhao=1
                sumconvloss_tensor[k][i]=mse(xmaplist[i][k],ymaplist[i][k])*fuhao
            sumconvloss=mse(torch.sum(x[k]),torch.sum(y[k]))*(self.layer_num+1) # 整体数量
            sumconvloss_tensor[k][-1]=sumconvloss
        if is_return_tensor:
            return sumconvloss_tensor
        return sumconvloss_tensor.mean()

class ms_Loss(nn.Module):
    def __init__(self):
        super(ms_Loss, self).__init__()
        self.MSE=torch.nn.MSELoss()
    def forward(self,meanstdlist):
        loss=0
        for i in range(len(meanstdlist)):
            loss=self.MSE(meanstdlist[0][i][0],meanstdlist[1][i][0])+loss
            loss=self.MSE(meanstdlist[0][i][1],meanstdlist[1][i][1])+loss
        loss=loss/(len(meanstdlist)*2)
        return loss

class PPAD_loss(nn.Module):
    def __init__(self):
        super(PPAD_loss, self).__init__()
    def forward(self,img1,img2):
        from util.evaluate import ssim
        ssimloss=(1-ssim(img1,img2))*200
        l1loss=(1/(nn.L1Loss()(img1,img2)))*10
        return ssimloss+l1loss