import os
import torch
from torchvision import transforms
from data.base_dataset import BaseDataset
from PIL import Image
import  argparse


# test_dataset数据集返回的只有图像，没有mask
class TestDataset(BaseDataset):
    # 新添加的参数
    @staticmethod
    # 不传入parser，单单想要得到一个用于创建数据集的opt
    def modify_commandline_options(parser=None,location=None):
        if parser ==None:
            parser=argparse.ArgumentParser()
        parser.add_argument('--dataroot', type=str, default=None, help='scale images to this size')
        parser.add_argument('--h', type=int, default=None, help='scale images to this size')
        parser.add_argument('--w', type=int, default=256, help='then crop to this size')
        parser.add_argument('--resizeh', type=int, default=None, help='scale images to this size')
        parser.add_argument('--resizew', type=int, default=None, help='then crop to this size')
        parser.add_argument('--flip_rate', type=float, default=0, help='翻转概率')
        parser.add_argument('--h', type=int, default=256, help='scale images to this size')
        parser.add_argument('--w', type=int, default=256, help='then crop to this size')
        if parser==None:
            parser.add_argument('--dataset_mode', type=str, default='test', help='aligned,single')
        return parser

    def __init__(self, opt):
        self.w=opt.w
        self.h=opt.h
        self.resizeh=opt.resizeh
        self.resizew=opt.resizew
        self.flip_rate=opt.flip_rate
        self.opt=opt
        self.cityscaperoot=opt.dataroot
        self.img_files=os.listdir(opt.dataroot)
        # self.maskmodel=maskmodel
        t = 0
        for i in range(len(self.img_files)):
            path=os.path.join(self.dataroot, self.img_files[t])
            if os.path.isdir(path):
                self.img_files.pop(t)   # 删除某一个元素后，列表后面的数会移动到前面来，所以刚删除后不用移动指针到下一位，继续访问就可以，于是要t=t-1使得下一个位置还是原位置
                t=t-1
            else:
                self.img_files[t]=path
            t=t+1
    def __getitem__(self, idx):
        # 如果resizeh和resizew都为None，则不缩放，如果其中一个不为None，则按最短边等比例缩放
        # 如果h,w都不为None,则进行裁剪
        # 加载图像
        img = Image.open(self.img_files[idx])
        list=[]
        if self.resizeh != None and self.resizew != None:
            list.append(transforms.Resize((self.resizeh, self.resizew)))  # 缩放图像到目标尺寸
        elif self.resizeh == None and self.resizew != None:
            list.append(transforms.Resize(self.resizew))  # 缩放图像到目标尺寸
        elif self.resizeh != None and self.resizew == None:
            list.append(transforms.Resize(self.resizeh))  # 缩放图像到目标尺寸
        if self.h!=None and self.w!=None:
            list.append(transforms.RandomCrop((self.h, self.w)))
        img_transform = transforms.Compose(list
        )
        img = img_transform(img)

        if self.flip_rate>0:
            if torch.rand(1)<self.flip_rate:
                img=transforms.functional.hflip(img)
        img_transform1 = transforms.Compose([
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 0]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])  # 标准化至[-0, 0]，规定均值和标准差
        ])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        img1=img_transform1(img)
        # label=self.maskmodel(img1)
        dict={}
        dict["img"]=img1
        return dict
    def __len__(self):
        return len(self.img_files)


    def tensorimg2MaskLittlelEdge(self,tensorimg,mask,alpha):
        img_array=tensorimg.cpu().numpy()
        random_sigma = (alpha * self.opt.range_sigma + self.opt.standard_sigma).cpu().numpy()
        random_high = (alpha* (1 - self.opt.standard_high) + self.opt.standard_high).cpu().numpy()
        N,C,H,W=tensorimg.size()

        original_edges=torch.zeros((N,1,H,W))
        from skimage.feature import canny
        for i in range(N):
            input_edge = torch.Tensor(
            canny(img_array[i][0], sigma=random_sigma[i], use_quantiles=True, high_threshold=random_high[i])) * mask[i]
            original_edges[i] = input_edge.unsqueeze(0)
        return original_edges


