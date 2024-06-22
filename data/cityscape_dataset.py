import os
import torch
from torchvision import transforms
import numpy as np
from skimage.feature import canny
from data.base_dataset import BaseDataset
from PIL import Image
import  argparse
import util
from util.util import seg2onehot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def readlist(dir):
    with open(dir,'r',encoding='utf-8') as f:  #使用with方法
        a=f.readlines()
        losslist=[]
        for i in a:
            losslist.append(i.strip())
        return losslist

def replace_first_occurrence(input_string, search_str, replace_str):
    # 使用 find() 方法找到第一个匹配的位置
    first_occurrence = input_string.find(search_str)

    if first_occurrence != -1:
        # 使用切片将字符串分成三部分：匹配前、匹配、匹配后
        before_match = input_string[:first_occurrence]
        match = input_string[first_occurrence:first_occurrence + len(search_str)]
        after_match = input_string[first_occurrence + len(search_str):]

        # 将匹配的部分替换为新的字符串
        modified_string = before_match + replace_str + after_match
        return modified_string
    else:
        # 如果没有找到匹配，返回原始字符串
        return input_string



# def getcityscaperoot(opt,location):
#     if location == 0:
#         opt.dataroot="/home/liujian/chenlqn/mycode/cityscape"
#     elif location == 1:
#         opt.dataroot="/haowang_ms/chenlqn/mycode/cityscape"
#     elif location == 2:
#         opt.dataroot= "F:\mycode\cityscape"
#     elif location==3:
#         opt.dataroot = "/root/autodl-tmp/cityscape"
#     else:
#         raise NotImplementedError('location [%s] is valid is not found')

# datasettype确定数据集的类型，only_val,only_train,try,train，cityscaperoot表示cityscape的目录位置，resizeh是要缩放的图像尺寸的高度
# resizew是要缩放图像的高度，flip_rate为水平翻转的概率
class CityscapeDataset(BaseDataset):
    # 新添加的参数
    @staticmethod
    # 不传入parser，单单想要得到一个用于创建数据集的opt
    def modify_commandline_options(parser=None,location=None):
        if parser ==None:
            parser=argparse.ArgumentParser()
        parser.add_argument('--location', type=int, default=2, help='')
        parser.add_argument('--num_worker', type=int, default=8, help='')
        parser.add_argument('--shuffle', type=bool, default=False, help='')
        parser.add_argument('--dataset_mode', type=str, default='cityscape', help='aligned,single')
        parser.add_argument('--batch_size', type=int, default=1, help='aligned,single')
        parser.add_argument('--dataroot', type=str, default=None, help='scale images to this size')
        parser.add_argument('--h', type=int, default=256, help='scale images to this size')
        parser.add_argument('--w', type=int, default=256, help='then crop to this size')
        parser.add_argument('--resizeh', type=int, default=300, help='scale images to this size')
        parser.add_argument('--resizew', type=int, default=700, help='then crop to this size')
        parser.add_argument('--flip_rate', type=float, default=0.5, help='翻转概率')
        parser.add_argument('--is_canny', type=bool, default=256, help='scale images to this size')
        parser.add_argument('--simple_edge', type=bool, default=True, help='翻转概率')
        parser.add_argument('--standard_high', type=float, default=0.8, help='')
        parser.add_argument('--standard_sigma', type=float, default=1.5, help='')
        parser.add_argument('--range_sigma', type=float, default=6.0, help='')
        parser.add_argument('--range_high', type=float, default=0.2, help='')
        parser.add_argument('--datasettype', type=str, default="only_train", help='翻转概率')
        parser.add_argument('--resource_type', type=str, default="seg", help='真正的源图像是什么,mask,seg,mask_littleedge,seg_littleedge,img_edge')
        parser.add_argument('--target_type', type=str, default="simple_edge", help='真正的要生成的目标图像是什么,simple_edge,mask_edge,full_edge,img')
        parser.add_argument('--return_edge', action="store_true",  help="")
        parser.add_argument('--return_img',  action="store_true",  help='')
        parser.add_argument('--return_seg',  action="store_true", help='')
        parser.add_argument('--return_mask',  action="store_true", help='')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        return parser

    def __init__(self, opt):
        self.standard_sigma=opt.standard_sigma
        self.standard_high=opt.standard_high
        self.w=opt.w
        self.h=opt.h
        self.datasettype=opt.datasettype
        self.resizeh=opt.resizeh
        self.resizew=opt.resizew
        self.flip_rate=opt.flip_rate
        self.test_list=[]
        self.counter=0
        self.sanmplesum=0
        self.is_canny=opt.is_canny
        self.high=opt.standard_high
        self.opt=opt
        self.cityscaperoot=opt.dataroot
        list0 = readlist(os.path.join(self.opt.dataroot, "testImages.txt"))#读取测试集的图像
        list1 = readlist(os.path.join(self.opt.dataroot, "trainImages.txt"))#读取训练集的图像
        list2 = readlist(os.path.join(self.opt.dataroot, "valImages.txt"))#读取验证集的图像
        if self.opt.datasettype=="train":
            self.image_files= list1 + list2
        elif self.opt.datasettype=="try":
            self.image_files=os.listdir(os.path.join(self.cityscaperoot, "leftImg8bit/train/darmstadt"))
        elif self.opt.datasettype=="test":
            self.image_files=list0
        elif self.opt.datasettype=="only_train":
            self.image_files = list1
        elif self.opt.datasettype=="only_val":
            self.image_files = list2
        else:
            raise NotImplementedError('datasettype is valid')

    def __getitem__(self, idx):
        gtFine_name = "gtFine_labelTrainIds"
        if self.opt.location == 3:
            gtFine_name = "gtFine_labelIds"
        zhongString=""
        if self.datasettype=="try":
            zhongString="leftImg8bit/train/darmstadt"
        image_filename = os.path.join(self.cityscaperoot,zhongString, self.image_files[idx])
        temp=replace_first_occurrence(image_filename,"leftImg8bit","gtFine_mask")
        label_filename=replace_first_occurrence(temp, "leftImg8bit", "gtFine_labelTrainIds")

        temp=replace_first_occurrence(image_filename,"leftImg8bit", "gtFine")
        seg_filename = replace_first_occurrence(temp, "leftImg8bit", gtFine_name)


        # 加载图像
        img = Image.open(image_filename)
        label=Image.open(label_filename)
        seg=Image.open(seg_filename)
        img_transform = transforms.Compose([
            transforms.Resize((self.resizeh,self.resizew)),  # 缩放图像到目标尺寸
        ])
        label_transform = transforms.Compose([
            transforms.Resize((self.resizeh,self.resizew),interpolation=transforms.InterpolationMode.NEAREST),  # 缩放标签图像到目标尺寸
        ])
        img = img_transform(img)
        label = label_transform(label)
        seg = label_transform(seg)
        if self.resizeh>self.h and self.resizew>self.w:
            basici=0
            #  我们希望图像中裁剪到更多的建筑物，所以想让裁剪垂直方向上靠上，如果以0.6的概率g小于4，则舍弃掉图像下面的一部分，不选择裁剪到他们
            g = (torch.randint(1, 6, size=(1,)).item())
            basici = 0
            if g < 4:
                basici = int((self.resizeh - self.h) * (g / 3))
            t=0.5
            while(1):
                i = torch.randint(0,self.resizeh-self.h-basici+1,size=(1,)).item()
                j = torch.randint(0,self.resizew-self.w+1,size=(1,)).item()
                label_tensor=transforms.ToTensor()(label)
                label_sum=label_tensor[label_tensor==1].sum().item()

                try_label = transforms.functional.crop(label, i, j, self.h, self.w)
                try_label_tensor=transforms.ToTensor()(try_label)
                rate=try_label_tensor[try_label_tensor == 1].sum().item() / (label_sum + 1)
                if  rate>t:
                    img = transforms.functional.crop(img, i, j, self.h, self.w)
                    label=transforms.functional.crop(label, i, j, self.h, self.w)

                    seg=transforms.functional.crop(seg, i, j, self.h, self.w)
                    break
                self.sanmplesum+=1
                t=t-0.01
            self.test_list.append(rate)
            self.counter+=1

        if self.flip_rate>0:
            if torch.rand(1)<self.flip_rate:
                img=transforms.functional.hflip(img)
                label=transforms.functional.hflip(label)
                seg = transforms.functional.hflip(seg)

        img_transform1 = transforms.Compose([
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 0]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])  # 标准化至[-0, 0]，规定均值和标准差
        ])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label_transform1 = transforms.Compose([
            transforms.ToTensor(),
        ])

        img1=img_transform1(img)
        label=label_transform1(label)
        seg = label_transform1(seg)*255
        seg[seg == 255.0] = 19.0

        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        edge = torch.Tensor(canny(img_array, sigma=self.opt.standard_sigma, use_quantiles=True, high_threshold=self.opt.standard_high))\
            .unsqueeze(0)
        dict = {}
        dict['seg_path']= seg_filename
        dict['img_path']= image_filename
        dict["mask_path"]=label_filename
        dict["alpha"]=0
        dict["path"]=image_filename
        if self.opt.return_img:
            dict["img"]=img1
        if self.opt.return_edge:
            dict["edge"]=edge
        if self.opt.return_mask:
            dict["mask"]=label
        if self.opt.return_seg:
            dict["seg"]=seg
        if self.opt.resource_type=="mask":
            dict["A"]=label
        elif self.opt.resource_type == "img":
            dict["A"] = img1
        elif self.opt.resource_type=="seg":
            dict["A"]=seg
        elif self.opt.resource_type=="onehot_mask_littleedge" or self.opt.resource_type=="littleedge" or \
                self.opt.resource_type=="mask_littleedge":
            random_alpha= torch.rand(1)
            random_sigma = random_alpha.item()*self.opt.range_sigma + self.standard_sigma
            random_high = random_alpha.item() * self.opt.range_high + self.opt.standard_high
            input_edge = torch.Tensor(
                canny(img_array, sigma=random_sigma, use_quantiles=True, high_threshold=random_high)).unsqueeze(0)
            if self.opt.resource_type == "mask_littleedge":
                mask_littleedge =  input_edge*label
            if self.opt.resource_type=="onehot_mask_littleedge":
                mask_littleedge = label + input_edge*label
                mask_littleedge=torch.nn.functional.one_hot(mask_littleedge.squeeze(0).long(),3).permute(2,0,1).float()
            if self.opt.resource_type == "littleedge":
                mask_littleedge=input_edge
            dict["A"] = mask_littleedge
            dict["alpha"]=1-(mask_littleedge.sum()/edge.sum()).cpu().item()

        elif self.opt.resource_type=="seg_littleedge":
            random_alpha= torch.rand(1)
            random_sigma = random_alpha.item()*self.opt.range_sigma + self.standard_sigma
            random_high = random_alpha.item() * self.opt.range_high + self.opt.standard_high
            input_edge = torch.Tensor(
                canny(img_array, sigma=random_sigma, use_quantiles=True, high_threshold=random_high)) * label
            seg = seg + input_edge * 17
            dict["A"]=seg
        elif self.opt.resource_type=="img_littleedge":
            random_alpha= torch.rand(1)
            random_sigma = random_alpha.item()*self.opt.range_sigma + self.standard_sigma
            random_high = random_alpha.item() * self.opt.range_high + self.opt.standard_high
            input_edge = torch.Tensor(
                canny(img_array, sigma=random_sigma, use_quantiles=True, high_threshold=random_high)) * label
            img_littleedge = img1*(1-label)+label*input_edge
            dict["A"]=img_littleedge
        elif self.opt.resource_type=="img_edge":
            dict["A"]=img1*(1-label)+label*edge
        elif self.opt.resource_type == "img_hole":
            dict["A"] = img1 * (1 - label)
        else:
            raise NotImplementedError('{} not implemented'.format(self.opt.resource_type))
        if self.opt.target_type=="full_edge":
            dict["B"]=edge
        elif self.opt.target_type == "mask_edge":
            dict["B"] = edge*label
        elif self.opt.target_type == "simple_edge":# 建筑物全edge，其他边缘edge
            seg_edge = torch.Tensor(canny(np.array(seg[0])))
            edge_melt = seg_edge * (1 - label) + label*edge
            dict["B"]=edge_melt
        elif self.opt.target_type == "img":
            dict["B"]=img1
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

        original_edges=torch.zeros((N,1,H,W)).to(device)
        from skimage.feature import canny
        for i in range(N):
            input_edge = torch.Tensor(
            canny(img_array[i][0], sigma=random_sigma[i], use_quantiles=True, high_threshold=random_high[i]))
            input_edge=input_edge.to(device)*mask[i]
            original_edges[i] = input_edge.unsqueeze(0)
        return original_edges.to(device)
