
import math
from skimage.metrics import structural_similarity as skissim
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import os
from util.my_visualizer import readlist,writelist
from data.cityscape_dataset import CityscapeDataset, replace_first_occurrence
from data.test_dataset import TestDataset

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr1(img1, img2,outmean=True):
    n=img1.size()[0]
    psnrsum=torch.zeros(n)
    for i in range(n):
        img1i=img1[i]
        img2i=img2[i]
        mse = torch.mean((img1i / 1.0 - img2i / 1.0) ** 2)
        # 计算PSNR
        if mse<1e-10:
            psnri=100.0
        else:
            psnr=20 * torch.log10(1.0 / torch.sqrt(mse))
        psnrsum[i]=psnr
        # 将计算结果转换为 Python float（如果需要）
    if outmean==True:
        out=psnrsum.mean()
    else:
        out=psnrsum
    return out


def ski_ssim(img1,img2):
    if(len(img1.size())==4):
        img1_np=img1.cpu().numpy()
        img2_np=img2.cpu().numpy()
        ssim_value=0.0
        for i in range(img1.size(0)):
            ssim_value=ssim_value+skissim(img1_np[i],img2_np[i],channel_axis=0)
        return ssim_value/(img1.size(0))
    elif(len(img1.size())==3 and img1.size(0)==3):
        ssim=skissim(img1.cpu().numpy(), img2.cpu().numpy(), channel_axis=0)
        return ssim

def decreator(func):
    def wrapper(*args,**kwargs):
        args[0].size()


def psnr2(img1, img2):#第二种法：归一化
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    PIXEL_MAX = 1
    psnr2 = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr2


def L1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    meanL1= torch.mean(abs(img1 / 1.0 - img2 / 1.0))
    # compute psnr
    return meanL1

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# data_list是真实seg的路径列表，pred_folder
def cal_acc(data_list, pred_folder, classes,logger,k):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for i, target_path in enumerate(data_list):
        image_name = target_path.split('/')[-1].split('.')[0]

        if k=="cityscape":#真實seg的文件名和保存的predseg不一致，保存的predseg是和真实img原图一致的
            image_name=image_name.replace("gtFine_labelTrainIds","leftImg8bit")
        predpath=os.path.join(pred_folder, image_name+'.png')
        pred = cv2.imread(predpath, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        print(pred)
        print(predpath)
        target=cv2.resize(target,dsize=(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    fbuildingAcc=(sum(intersection_meter.sum)-intersection_meter.sum[2])/(sum(target_meter.sum)-target_meter.sum[2]+ 1e-10)
    fbuildingiou=(sum(intersection_meter.sum)-intersection_meter.sum[2])/(sum(union_meter.sum)-union_meter.sum[2]+ 1e-10)
    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc,fbuildingAcc))
    print('Eval result: buildingAcc/buildingiou/fbuildingAcc/fbuildingiou {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(accuracy_class[2],iou_class[2],
                                                                                                  fbuildingAcc,fbuildingiou))
    for i in range(classes):
        print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], i))

def cal_acciou(dir):

    valimglist=readlist("F:/mycode/cityscape/vaimages.txt")
    graylist=[]
    for i in valimglist:
        gray=replace_first_occurrence(i, "leftImg8bit", "gtFine_labelTrainIds")
        graylist.append(gray)# gray就是label
    mylist=os.listdir("")
    cal_acc(mylist, graylist, 19)

# 效用部分真实填充，img1为真实图像
def ski_pri_ssim(img1,img2,mask):
    img2=img2*(mask)+img1*(1-mask)
    if(len(img1.size())==4):
        img1_np=img1.cpu().numpy()
        img2_np=img2.cpu().numpy()
        ssim_value=0.0
        for i in range(img1.size(0)):
            ssim_value=ssim_value+skissim(img1_np[i],img2_np[i],channel_axis=0)
        return ssim_value/(img1.size(0))
    elif(len(img1.size())==3 and img1.size(0)==3):
        ssim=skissim(img1.cpu().numpy(), img2.cpu().numpy(), channel_axis=0)
        return ssim

# 效用部分噪声填充，img1为真实图像
def ski_pri_ssim1(img1,img2,mask):
    noisea=torch.rand(size=img1.size())
    noiseb=torch.rand(size=img1.size())
    img2=img2*(mask)+noisea*(1-mask)
    img1=img1*(mask)+noiseb*(1-mask)
    if(len(img1.size())==4):
        img1_np=img1.cpu().numpy()
        img2_np=img2.cpu().numpy()
        ssim_value=0.0
        for i in range(img1.size(0)):
            ssim_value=ssim_value+skissim(img1_np[i],img2_np[i],channel_axis=0)
        return ssim_value/(img1.size(0))
    elif(len(img1.size())==3 and img1.size(0)==3):
        ssim=skissim(img1.cpu().numpy(), img2.cpu().numpy(), channel_axis=0)
        return ssim

# 隐私部分噪声填充，img1为真实图像
def ski_uti_ssim(img1,img2,mask):
    noisea=torch.rand(size=img1.size())
    noiseb=torch.rand(size=img1.size())
    img1=img1*(1-mask)+noisea*mask
    img2=img2*(1-mask)+noiseb*mask
    if(len(img1.size())==4):
        img1_np=img1.cpu().numpy()
        img2_np=img2.cpu().numpy()
        ssim_value=0.0
        for i in range(img1.size(0)):
            ssim_value=ssim_value+skissim(img1_np[i],img2_np[i],channel_axis=0)
        return ssim_value/(img1.size(0))
    elif(len(img1.size())==3 and img1.size(0)==3):
        ssim=skissim(img1.cpu().numpy(), img2.cpu().numpy(), channel_axis=0)
        return ssim

# 隐私部分效用填充，img1为真实图像
def ski_uti_ssim1(img1, img2, mask):
    img2 = img2 * (1 - mask) + img1 * mask
    if (len(img1.size()) == 4):
        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()
        ssim_value = 0.0
        for i in range(img1.size(0)):
            ssim_value = ssim_value + skissim(img1_np[i], img2_np[i], channel_axis=0)
        return ssim_value / (img1.size(0))
    elif (len(img1.size()) == 3 and img1.size(0) == 3):
        ssim = skissim(img1.cpu().numpy(), img2.cpu().numpy(), channel_axis=0)
        return ssim


def pri_psnr(img1,img2,mask):
    if img1.max()>1:
        img1=img1/255.0
        img2=img2/255.0
    cal_img1=img1*(mask)
    cal_img2=img2*(mask)
    if isinstance(img1, torch.Tensor):
        cal_img1=cal_img1.numpy()
        cal_img2=cal_img2.numpy()
    mse = np.mean((cal_img1 - cal_img2) ** 2)
    if mse < 1e-10:
        return 100
    PIXEL_MAX = 1
    psnr2 = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr2


def uti_psnr(img1,img2,mask):
    if img1.max()<=1:
        img1=img1/255.0
        img2=img2/255.0
    cal_img1=img1*(1-mask)
    cal_img2=img2*(1-mask)
    if isinstance(img1, torch.Tensor):
        cal_img1=cal_img1.numpy()
        cal_img2=cal_img2.numpy()
    mse = np.mean((cal_img1 - cal_img2) ** 2)
    if mse < 1e-10:
        return 100
    PIXEL_MAX = 1
    psnr2 = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr2