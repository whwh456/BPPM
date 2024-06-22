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
from util.my_visualizer import to_img,colshow
import torch.nn as nn
from models.losses import Cal_Div_Loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadmodel(pathList, batch_size,location=2, epoch=60):
    list = []
    for path in pathList:
        opt = util.util.load_optobject(path, batch_size=batch_size,location=location, epoch=epoch)
        print(opt.input_nc)
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        list.append(model)
    return list

# edge控制强度是直接依靠制造的输入的边缘多少来决定的
def compareedge_xiout(*args,xi=4):
    cal_divloss=Cal_Div_Loss(5,device)
    alpha=util.util.create_z(datasetopt.batch_size,xi,1)
    print(alpha)
    for k,i in enumerate(dataset):
        xi_real_A=util.util.getinput(i["A"],xi)
        xi_mask=util.util.getinput(i["mask"],xi)
        xi_real_B=util.util.getinput(i["B"],xi)
        list=[]
        for i,model in enumerate(args):
            xi_inputs = dataset.dataset.tensorimg2MaskLittlelEdge(xi_real_A, xi_mask, alpha)  # xi_inputs=masklitteledge
            if model.opt.input_nc == 3:
                xi_inputs = model.maskLittlelEdge2onehot(xi_inputs, xi_mask)
            fake_B=model.generate(xi_inputs)
            if i==0:
                list.append(xi_inputs)
            list.append(fake_B*xi_mask)
            divtensor=cal_divloss(fake_B,xi_real_B,alpha,True)
            for tempi  in range(fake_B.size(0)):
                print((fake_B[tempi].sum()-xi_real_B[tempi].sum())*5)
            print(divtensor)
        mv.colshow(1,1,*list,num=4,batch_x=1)


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
                mv.colshow(1,1,onehot_xi_inputs,to_img(pixinput),to_img(pixout),
                           to_img(skipone_pixout),num=4,batch_x=1)
        if k>20:
            break

def compareedgeout(*args):
    for k,i in enumerate(dataset):
        real_A=i["A"]
        real_B=i["B"]
        list=[]
        for model in args:
            model.set_input(i)
            model.forward()
            list.append(to_img(model.fake_B))
        mv.colshow(1,1,to_img(real_A),to_img(real_B),*list,num=4)
        if k>20:
            break

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
        alpha=util.util.create_z(datasetopt.batch_size,xi,1)
        print(alpha)
        xi_real_A=util.util.getinput(i["A"],xi)
        xi_mask=util.util.getinput(i["mask"],xi)
        original_edge = dataset.dataset.tensorimg2MaskLittlelEdge(xi_real_A, xi_mask, alpha)
        mv.colshow(2,4,original_edge,num=2)
        divLoss=Cal_Div_Loss(5,device)


if __name__ == '__main__':
# 决定dataroot的值的操作，在创建数据集的参数时进行，所以在之后更改location的值不会改变dataroot
    trainoption = TrainOptions()
    datasetopt = trainoption.parse("inpaint", "cityscape")  # get training options
    datasetopt.resource_type = "mask_littleedge"
    datasetopt.target_type = "mask_edge"
    datasetopt.datasettype = "only_val"
    datasetopt.return_mask = True
    datasetopt.return_seg = True
    datasetopt.return_img=True
    datasetopt.epoch = 60
    datasetopt.location = 2
    datasetopt.isTrain = False
    datasetopt.return_edge = True
    datasetopt.resizew = 512
    datasetopt.w = 512
    datasetopt.h = 256
    datasetopt.resizeh = 256
    datasetopt.serial_batches = True
    datasetopt.batch_size = 1
    datasetopt.range_sigma=4
    dataset = create_dataset(datasetopt)  # create a dataset given opt.dataset_mode and other options

    pathlist=["F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a5\opt","F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a6\opt"
         ,"F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a7\opt"]
    pixlist=["F:\mycode\BicycleGAN-master\checkpoints\multi_sim_l2\opt"]
    # pixmodellilst = loadmodel(pixlist, batch_size=datasetopt.batch_size, epoch=60)
    modellist=loadmodel(pathlist,batch_size=datasetopt.batch_size,epoch=60)

#"F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_a1\opt"
#"F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a0\opt"
# "F:\mycode\BicycleGAN-master\checkpoints\edge\edge_div_a0\opt",
#     compareedge_xiout(*modellist)
#     edgepix_xiout(modellist,pixmodellilst)
    # calgram(model,10)
    # testdataset()
