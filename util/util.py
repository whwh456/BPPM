from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle
import torchvision
from options.train_options import upgradeopt


# 把一张tensor图像转换为numpy
def tensor2im(input_image, imtype=np.uint8):
    """"Convert a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def colorize(prediction,color_path=None):
    gray = np.uint8(prediction)
    colors = np.loadtxt("/haowang_ms/chenlqn/mycode/BicycleGAN-master/util/cityscapes_colors.txt").astype('uint8')
    # gray: numpy array of the label and 0*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    # color是一个PIL图像对象
    color.putpalette(colors)
    if color_path!=None:
        color.save(color_path)
    return color.convert("RGB")

#把张量预测图（无通道维度，N,H,W）转换成np上色，再转换为张量
def tensorlabeltocolor(tensor):
    #单个张量标签转换
    def temp(tensor):
        if len(tensor.size())==3:
            tensor=tensor[0]
        numpyx=tensor.detach().cpu().numpy()
        color=colorize(numpyx)
        tensor_color=torchvision.transforms.functional.to_tensor(color)
        return tensor_color

    tensor_length=len(tensor.size())
    if tensor_length==4:
        N, C, H, W = tensor.size()
        color_batch=torch.zeros(N,3,H,W)
        for i in range(N):
            tensor_colori=temp(tensor[i])
            color_batch[i]=tensor_colori

    if tensor_length==3:
        N,H,W=tensor.size()
        color_batch=torch.zeros(N,3,H,W)
        for i in range(N):
            tensor_colori=temp(tensor[i])
            color_batch[i]=tensor_colori
    return color_batch



#把19通道张量转化为上色后的张量
def t19totcolor(tensor):
    pred=outputtopred(tensor)
    colortensor=tensorlabeltocolor(pred)
    return colortensor

# 把模型生成的通道维n_class的图，转化为真实的预测图，其像素值为预测值,预测图无通道维度
def outputtopred(output):
    N, _, h, w = output.size()
    pred = output.argmax(axis=1)
    # 得到最终的预测图
    return pred

def seg2onehot(seg,device,nc,convert255to19=True):
    b,c,h,w=seg.size()
    seg=seg.long()
    input_label = torch.FloatTensor(b, nc, h, w).zero_().to(device)
    if convert255to19:
        seg[seg==255]=19
    segonehot = input_label.scatter_(1, seg, 1.0)
    return segonehot

def getinput(input,xi):

    if len(input.size())==4:
        N,C,H,W=input.size()
        input=input.unsqueeze(1)
        input=input.repeat(1,xi,1,1,1)
        input=input.view(N*xi,C,H,W)
    elif len(input.size())==2:
        N,L=input.size()
        input=input.unsqueeze(1)
        input=input.repeat(1,xi,1)
        input=input.view(N*xi,L)
    return input

# is_continue指是否是连续的点，否则是离散的点
def create_z(n,xi,scope,is_continue=True):# 注意这个scope能够取到右端点,这个z是针对于xi来创造的
    z = torch.linspace(0, scope, xi).repeat(n)
    if not is_continue:
        z=  z.long().float()
    return z

# 每个系列以insertimg开头，接着跟xi个output,output的长度是insertimg列表里的张量的长度的xi倍
def adjustoutshow(xi,output,*insertimg):
    insertimg=list(insertimg)
    if len(output.size())==4 and output.size()[1]==1:
        output = output.repeat(1, 3, 1, 1)
    for i in range(len(insertimg)):
        if len(insertimg[i].size())==4 and insertimg[i].size()[1]==1:
            insertimg[i]=insertimg[i].repeat(1,3,1,1)
    N,C,H,W=output.size()
    length=len(insertimg)
    s_xi=len(insertimg[0])
    assert s_xi*xi==len(output)
    zonglength=len(output)+length*s_xi
    zongsum=torch.zeros(zonglength,C,H,W) #总拼接的容器
    for ii in range (s_xi):
        for iii in range(length):
            zongsum[ii*(xi+length)+iii]=insertimg[iii][ii]# 装入insertimg
        zongsum[ii*(xi+length)+iii+1:ii*(xi+length)+iii+1+xi]=output[ii*xi:ii*xi+xi]
        # zongsum[ii*(xi+length)+length:ii*(xi+length)+length+xi]=output[ii*xi:(ii+1)*xi]#装入output
    return zongsum

def save_optobject(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
        print(f"保存opt对象至{file_path}")

def load_optobject(file_path,batch_size,is_upgradeopt=True,location=2,epoch=60):
    with open(file_path, 'rb') as file:
        loaded_obj = pickle.load(file)
    loaded_obj.location=location
    loaded_obj.continue_train=True
    loaded_obj.name=file_path[file_path.find("checkpoints")+len("checkpoints/"):len(file_path)-4]# 把路径文件夹的名字赋予name
    loaded_obj.epoch=epoch
    loaded_obj.batch_size=batch_size
    loaded_obj.phase="test"
    loaded_obj.isTrain=False
    if is_upgradeopt:
        loaded_obj=upgradeopt(loaded_obj)
    return loaded_obj
