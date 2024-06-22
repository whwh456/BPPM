"""General-purpose training script for image-to-image translation.
This script works for various models (with option '--model': e.g., bicycle_gan, pix2pix, test) and
different datasets (with option '--dataset_mode': e.g., aligned, single).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
Example:
    Train a BiCycleGAN model:
        python train.py --dataroot ./datasets/facades --name facades_bicyclegan --model bicycle_gan --direction BtoA
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/train_options.py for more training options.
"""
import util.util
from options.train_options import TrainOptions
from data import create_dataset,get_option_setter
from models import create_model
import util.evaluate
import copy
import torch
import util.my_visualizer as mv
from util.my_visualizer import to_img,colshow,imshow
import torch.nn as nn
from models.losses import Cal_Div_Loss
from data import tokyo247_dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os

def loadmodel(pathList, batch_size,location=2, epoch=60):
    list = []
    for path in pathList:
        opt = util.util.load_optobject(path, batch_size=batch_size,location=location, epoch=epoch)
        print(opt.input_nc)
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        list.append(model)
    return list

def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 显示整个过程，edge，pix两个模型都用到
def edgepix_xiout(edgemodellist,pixmodellist,xi=4):
    alpha=util.util.create_z(datasetopt.batch_size,xi,0.5)
    for k,i in enumerate(dataset):
        xi_real_img=util.util.getinput(i["img"],xi)
        xi_mask = util.util.getinput(i["mask"], xi)

        for emi,edgemodel in enumerate(edgemodellist):
            for pmi,pixmodel in enumerate(pixmodellist):
                xi_inputs=dataset.dataset.tensorimg2MaskLittlelEdge(xi_real_img,xi_mask,alpha)
                onehot_xi_inputs=edgemodel.maskLittlelEdge2onehot(xi_inputs,xi_mask)
                edgeout=edgemodel.generate(onehot_xi_inputs)
                pixinput=edgeout*xi_mask+(1-xi_mask)*xi_real_img
                pixout=pixmodel.generate(pixinput,xi_real_img,alpha)
                skipone_pixout=pixmodel.generate(xi_inputs*xi_mask+(1-xi_mask)*xi_real_img,xi_real_img,alpha)


def edgepixsave(edgemodel,pixmodel,alpha,sigma,high):
    alpha=torch.ones((datasetopt.batch_size,))*alpha
    for k, i in enumerate(dataset):
        mask=i["mask"]
        img=i["img"]

        inputs = dataset.dataset.tensorimg2MaskLittlelEdge1(img,mask, sigma,high)
        onehot_inputs = edgemodel.maskLittlelEdge2onehot(inputs, i["mask"])
        edgeout = edgemodel.generate(onehot_inputs)
        pixinput = edgeout * mask + (1 - mask) * img
        i["A"]=pixinput
        pixmodel.set_input(i)
        pixout = pixmodel.selfgenerate(alpha)
        savepath=i["path"][0].replace("originalquery",
                          f"{os.path.basename(edgemodel.opt.name)}#{os.path.basename(pixmodel.opt.name)}_sty{alpha.item()}sig{sigma}high{high}")

        check_makedirs(os.path.dirname(savepath))
        imshow(to_img(pixout[0]),savepath)
        print(f"{k}/{len(dataset)}:已经生成保存到{savepath}")


# 显示比较多个组合模型，alpha固定并不成系列展示
def conmpareEdgepix(modelTuplelList):
    alpha=torch.zeros((datasetopt.batch_size,1))+0.5  #这个alpha是用来控制输入edge的疏密程度的
    for k,i in enumerate(dataset):
        real_A=i["A"]
        mask=i["mask"]
        inputs, original_edge = model.tensorimg2MaskLittlelEdge(real_A, mask, alpha, True)
        for edgemodel,pimodel in modelTuplelList:
            edgeout=edgemodel(real_A)
            pixout=pimodel(edgeout,alpha)
            pass

def testdataset(xi=4):
    for k,i in enumerate(dataset):
        print(i["mask"])
        imshow(util.util.tensorlabeltocolor(i["mask"])[0])


if __name__ == '__main__':
# 决定dataroot的值的操作，在创建数据集的参数时进行，所以在之后更改location的值不会改变dataroot

    datasetopt = tokyo247_dataset.Tokyo247Dataset.modify_commandline_options().parse_args() # get training options
    datasetopt.resource_type = "img"
    datasetopt.target_type = "img"
    datasetopt.datasettype = "query"
    datasetopt.return_mask = True
    datasetopt.return_img=True
    datasetopt.location = 2
    datasetopt.return_edge = True
    datasetopt.resizew = 512
    datasetopt.w = 512
    datasetopt.h = 512
    datasetopt.resizeh = 512
    datasetopt.batch_size = 1
    datasetopt.range_sigma=4
    print(datasetopt.dataset_mode)
    dataset = create_dataset(datasetopt)  # create a dataset given opt.dataset_mode and other options

    pathlist=["F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a6\opt"]
    pixlist=["F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_l1\opt"]
    pixmodellilst = loadmodel(pixlist, batch_size=datasetopt.batch_size, epoch=80)
    modellist=loadmodel(pathlist,batch_size=datasetopt.batch_size,epoch=60)

#"F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_a1\opt"
#"F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a0\opt"
# "F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a0\opt",
#     compareedge_xiout(*modellist)
#     edgepix_xiout(modellist,pixmodellilst)
#     testdataset()
    edgepixsave(*modellist,*pixmodellilst,alpha=0,sigma=3.0,high=0.85)
