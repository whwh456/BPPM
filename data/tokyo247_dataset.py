import os
import torch
from torchvision import transforms
import numpy as np
from skimage.feature import canny
from data.base_dataset import BaseDataset
from PIL import Image
import  argparse

# def gettokyoroot(opt,location):
#     if location == 0:
#         opt.dataroot="/home/liujian/chenlqn/mycode/Tokyo247"
#     elif location == 1:
#         opt.dataroot="/haowang_ms/chenlqn/mycode/Tokyo247"
#     elif location == 2:
#         opt.dataroot= "E:/Tokyo247"
#     elif location==3:
#         opt.dataroot = "/root/autodl-tmp/Tokyo247"
#     else:
#         raise NotImplementedError('location [%s] is valid is not found')

def readlist(dir):
    with open(dir,'r',encoding='utf-8') as f:  #使用with方法
        a=f.readlines()
        losslist=[]
        for i in a:
            losslist.append(i.strip())
        return losslist


def saveseg2mask():
    opt=Tokyo247Dataset.modify_commandline_options()
    opt.location=2
    dataset=Tokyo247Dataset(opt)
    for i in dataset:
        i["mask"]

# Tokyo247Dataset用来test生成Tokyo图像
class Tokyo247Dataset(BaseDataset):
    # 新添加的参数
    @staticmethod
    # 不传入parser，单单想要得到一个用于创建数据集的opt
    def modify_commandline_options(parser=None):
        if parser ==None:
            parser=argparse.ArgumentParser()
        parser.add_argument('--dataroot', type=str, default=None, help='scale images to this size')
        parser.add_argument('--h', type=int, default=256, help='scale images to this size')
        parser.add_argument('--w', type=int, default=256, help='then crop to this size')
        parser.add_argument('--resizeh', type=int, default=300, help='scale images to this size')
        parser.add_argument('--resizew', type=int, default=700, help='then crop to this size')
        parser.add_argument('--flip_rate', type=float, default=0, help='翻转概率')
        parser.add_argument('--is_canny', type=bool, default=256, help='scale images to this size')
        parser.add_argument('--simple_edge', type=bool, default=True, help='翻转概率')
        parser.add_argument('--standard_high', type=float, default=0.8, help='')
        parser.add_argument('--standard_sigma', type=float, default=1.5, help='')
        parser.add_argument('--range_sigma', type=float, default=5.0, help='')
        parser.add_argument('--range_high', type=float, default=0.2, help='')
        parser.add_argument('--datasettype', type=str, default="only_train", help='翻转概率')
        parser.add_argument('--resource_type', type=str, default="seg", help='真正的源图像是什么,mask,seg,mask_littleedge,seg_littleedge,img_edge')
        parser.add_argument('--target_type', type=str, default="simple_edge", help='真正的要生成的目标图像是什么,simple_edge,mask_edge,full_edge,img')
        parser.add_argument('--return_edge', type=bool,default=True,  help="")
        parser.add_argument('--return_img',  type=bool,default=True,   help='')
        parser.add_argument('--return_mask',  type=bool,default=True,  help='')
        parser.add_argument('--return_seg',  type=bool,default=False,  help='')
        parser.add_argument('--location', type=int, default=2, help='')
        parser.add_argument('--num_worker', type=int, default=8, help='')
        parser.add_argument('--shuffle', type=bool, default=False, help='')
        parser.add_argument('--dataset_mode', type=str, default='tokyo247', help='aligned,single')
        parser.add_argument('--batch_size', type=int, default=1, help='aligned,single')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser

    def __init__(self, opt):
        super(Tokyo247Dataset, self).__init__(opt)
        # gettokyoroot(opt,opt.location)
        self.w=opt.w
        self.h=opt.h
        self.resizeh=opt.resizeh
        self.resizew=opt.resizew
        self.flip_rate=opt.flip_rate
        self.opt=opt

        if self.opt.datasettype=="try":
            self.image_files = readlist(os.path.join(self.opt.dataroot, "querypaths.txt"))[0:30]
        elif self.opt.datasettype=="images":
            self.image_files=readlist(os.path.join(self.opt.dataroot, "imagespaths.txt"))
        elif self.opt.datasettype=="query":
            self.image_files = readlist(os.path.join(self.opt.dataroot, "querypaths.txt"))
        else:
            raise NotImplementedError('datasettype is valid')

        # self.maskmodel=maskmodel

    def __getitem__(self, idx):
        # 如果resizeh和resizew都为None，则不缩放，如果其中一个不为None，则按最短边等比例缩放
        # 如果h,w都不为None,则进行裁剪
        # 加载图像
        imgfile=self.image_files[idx]
        img = Image.open(imgfile)
        if self.opt.datasettype=="images":
            segfile=imgfile.replace("Tokyo247","Tokyo247/seg")
        elif self.opt.datasettype=="query" and self.opt.location==2:
            imgname=os.path.basename(imgfile)
            segfile=os.path.join("E:/Tokyo247/seg/query",imgname)
        seg=Image.open(segfile)


        if self.flip_rate > 0:
            if torch.rand(1) < self.flip_rate:
                img = transforms.functional.hflip(img)
                seg = transforms.functional.hflip(seg)
        transformslist=[]
        masktransformslist=[]

        if self.resizeh != None and self.resizew != None:
            transformslist.append(transforms.Resize((self.resizeh, self.resizew)))  # 缩放图像到目标尺寸
            masktransformslist.append(transforms.Resize((self.resizeh, self.resizew), interpolation=transforms.InterpolationMode.NEAREST))
        elif self.resizeh == None and self.resizew != None:
            transformslist.append(transforms.Resize(self.resizew))  # 缩放图像到目标尺寸
            masktransformslist.append(transforms.Resize((self.resizew), interpolation=transforms.InterpolationMode.NEAREST))
        elif self.resizeh != None and self.resizew == None:
            transformslist.append(transforms.Resize(self.resizeh))  # 缩放图像到目标尺寸
            masktransformslist.append(transforms.Resize((self.resizeh), interpolation=transforms.InterpolationMode.NEAREST))
        if self.h!=None and self.w!=None:
            transformslist.append(transforms.RandomCrop((self.h, self.w)))
            masktransformslist.append(transforms.RandomCrop((self.h, self.w)))
        cropedimg=transforms.Compose(transformslist)(img)
        transformslist.append(transforms.ToTensor())  # 将图片(Image)转成Tensor，归一化至[0, 0]
        transformslist.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        masktransformslist.append(transforms.ToTensor())
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label_transform = transforms.Compose(masktransformslist)
        img_transform=transforms.Compose(transformslist)
        img1 = img_transform(img)
        seg = label_transform(seg)*255
        if self.opt.return_seg or self.opt.resource_type == "seg":
            label=seg.clone()
        else:
            label=seg
        label[label!=2]=0
        label[label==2]=1

        gray_img = cropedimg.convert('L')
        img_array = np.array(gray_img)
        edge = torch.Tensor(
            canny(img_array, sigma=self.opt.standard_sigma, use_quantiles=True, high_threshold=self.opt.standard_high)) \
            .unsqueeze(0)
        dict={}
        dict["path"]=imgfile
        dict["alpha"] = 0
        if self.opt.return_img:
            dict["img"] = img1
        if self.opt.return_edge:
            dict["edge"] = edge
        if self.opt.return_mask:
            dict["mask"] = label
        if self.opt.return_seg:
            dict["seg"] = seg
        if self.opt.resource_type == "mask":
            dict["A"] = label
        elif self.opt.resource_type == "img":
            dict["A"] = img1
        elif self.opt.resource_type == "seg":
            dict["A"] = seg
        elif self.opt.resource_type == "onehot_mask_littleedge" or self.opt.resource_type == "littleedge" or \
                self.opt.resource_type == "mask_littleedge":
            random_alpha = torch.rand(1)
            random_sigma = random_alpha.item() * self.opt.range_sigma + self.opt.standard_sigma
            random_high = random_alpha.item() * self.opt.range_high + self.opt.standard_high
            input_edge = torch.Tensor(
                canny(img_array, sigma=random_sigma, use_quantiles=True, high_threshold=random_high)).unsqueeze(0)
            if self.opt.resource_type == "mask_littleedge":
                mask_littleedge = input_edge * label
            if self.opt.resource_type == "onehot_mask_littleedge":
                mask_littleedge = label + input_edge * label
                mask_littleedge = torch.nn.functional.one_hot(mask_littleedge.squeeze(0).long(), 3).permute(2, 0,
                                                                                                            1).float()
            if self.opt.resource_type == "littleedge":
                mask_littleedge = input_edge
            dict["A"] = mask_littleedge
            dict["alpha"] = 1 - (mask_littleedge.sum() / edge.sum()).cpu().item()

        elif self.opt.resource_type == "img_littleedge":
            random_alpha = torch.rand(1)
            random_sigma = random_alpha.item() * self.opt.range_sigma + self.standard_sigma
            random_high = random_alpha.item() * self.opt.range_high + self.opt.standard_high
            input_edge = torch.Tensor(
                canny(img_array, sigma=random_sigma, use_quantiles=True, high_threshold=random_high)) * label
            img_littleedge = img1 * (1 - label) + label * input_edge
            dict["A"] = img_littleedge
        elif self.opt.resource_type == "img_edge":
            dict["A"] = img1 * (1 - label) + label * edge
        elif self.opt.resource_type == "img_hole":
            dict["A"] = img1 * (1 - label)
        else:
            raise NotImplementedError('{} not implemented'.format(self.opt.resource_type))
        if self.opt.target_type == "full_edge":
            dict["B"] = edge
        elif self.opt.target_type == "mask_edge":
            dict["B"] = edge * label
        elif self.opt.target_type == "img":
            dict["B"] = img1
        else:
            raise NotImplementedError('{} not implemented'.format(self.opt.target_type))
        return dict

    def __len__(self):
        return len(self.image_files)

    def tensorimg2MaskLittlelEdge(self,tensorimg,mask,alpha):
        img_array=tensorimg.cpu().numpy()
        random_sigma = (alpha * self.opt.range_sigma + self.opt.standard_sigma).cpu().numpy()
        random_high = (alpha* (1 - self.opt.standard_high) + self.opt.standard_high).cpu().numpy()
        N,C,H,W=tensorimg.size().to(device)

        original_edges=torch.zeros((N,1,H,W)).to(device)
        from skimage.feature import canny
        for i in range(N):
            input_edge = torch.Tensor(
            canny(img_array[i][0], sigma=random_sigma[i], use_quantiles=True, high_threshold=random_high[i])) * mask[i]
            original_edges[i] = input_edge.unsqueeze(0)
        return original_edges

    def tensorimg2MaskLittlelEdge1(self,tensorimg,mask,sigma,high):
        img_array=tensorimg.cpu().numpy()
        random_sigma = (np.ones((self.opt.batch_size,))*sigma)
        random_high = (np.ones((self.opt.batch_size,))*high)
        N,C,H,W=tensorimg.size()

        original_edges=torch.zeros((N,1,H,W)).to(tensorimg.device)
        from skimage.feature import canny
        for i in range(N):
            input_edge = torch.Tensor(
            canny(img_array[i][0], sigma=random_sigma[i], use_quantiles=True, high_threshold=random_high[i]))
            input_edge=input_edge.to(device)*mask[i]
            original_edges[i] = input_edge.unsqueeze(0)
        return original_edges.to(device)
