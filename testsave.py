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
import util.evaluate as eva
import copy
import torch
from util.my_visualizer import to_img,colshow,imshow
from data import tokyo247_dataset,cityscape_dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
from torchvision import transforms
from PIL import Image
import logging
import os





def loadmodel(pathList, batch_size,location=2, epoch=60,no_loadD=False):
    list = []
    for path in pathList:
        opt = util.util.load_optobject(path, batch_size=batch_size,location=location, epoch=epoch)
        print(opt.input_nc)
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt,no_loadD=no_loadD)  # regular setup: load and print networks; create schedulers
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


def edgepixsave(pixmodel,alpha,sigma,high,edgemodel=None,istokyo=True,message="",savedir=None):
    alpha=torch.ones((datasetopt.batch_size,))*alpha
    prissim_avg=eva.AverageMeter()
    utissim_avg=eva.AverageMeter()
    for k, i in enumerate(dataset):
        mask=i["mask"]
        img=i["img"]
        if edgemodel!=None:
            inputs = dataset.dataset.tensorimg2MaskLittlelEdge1(img,mask, sigma,high)
            onehot_inputs = edgemodel.maskLittlelEdge2onehot(inputs, i["mask"])
            edgeout = edgemodel.generate(onehot_inputs)
            pixinput = edgeout * mask + (1 - mask) * img
        else:
            pixinput=i["A"]
        i["A"]=pixinput
        pixmodel.set_input(i)
        pixout = pixmodel.selfgenerate(alpha)

        if edgemodel==None:
            edgemodelname ="None"
        else:
            edgemodelname=edgemodel.opt.name
        if istokyo:
            savedir = os.path.join(savedir,
                                    f"{os.path.basename(edgemodelname)}#{os.path.basename(pixmodel.opt.name)}",
                                    f"{os.path.basename(edgemodelname)}#{os.path.basename(pixmodel.opt.name)}_sty{alpha.item()}sig{sigma}high{high}{message}")

        else :
            savedir=os.path.join(savedir,f"{os.path.basename(edgemodelname)}#{os.path.basename(pixmodel.opt.name)}",
                              f"{os.path.basename(edgemodelname)}#{os.path.basename(pixmodel.opt.name)}_sty{alpha.item()}sig{sigma}high{high}{message}")

        imgname=os.path.basename(i["path"][0])
        savepath=os.path.join(savedir,imgname)
        check_makedirs(savedir)
        imshow(to_img(pixout[0]),savepath)
        print(f"{k}/{len(dataset)}:已经生成保存到{savepath}")
        prissim=eva.ski_pri_ssim(img,pixout,mask)
        utissim=eva.ski_uti_ssim(img,pixout,mask)
        prissim_avg.update(prissim,pixout.size(0))
        utissim_avg.update(utissim,pixout.size(0))


    # 配置日志记录器
    log_file_dir = os.path.join(savedir,"log")
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    log_file_path=os.path.join(log_file_dir,"log.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file_path),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    print(f"prissim:{prissim_avg.avg}")
    print(f"utissim:{utissim_avg.avg}")


def cal_ssim(dirlist):
    for dir in dirlist:
        prissim_avg=eva.AverageMeter()
        utissim_avg=eva.AverageMeter()

        prissim1_avg=eva.AverageMeter()
        utissim1_avg=eva.AverageMeter()
        for k, i in enumerate(dataset.dataset):
            mask=i["mask"]
            img = i["img"]
            imgname=os.path.basename(i["path"])
            savepath=os.path.join(dir,imgname)
            pixout = Image.open(os.path.join(dir,imgname))
            pixout=pixout.convert("RGB")
            transform = transforms.ToTensor()
            pixout = transform(pixout)


            check_makedirs(os.path.dirname(savepath))
            prissim=eva.ski_pri_ssim(to_img(img),pixout,mask)
            utissim=eva.ski_uti_ssim(to_img(img),pixout,mask)
            prissim_avg.update(prissim,pixout.size(0))
            utissim_avg.update(utissim,pixout.size(0))

            prissim1=eva.ski_pri_ssim1(to_img(img),pixout,mask)
            utissim1=eva.ski_uti_ssim1(to_img(img),pixout,mask)
            prissim1_avg.update(prissim1,pixout.size(0))
            utissim1_avg.update(utissim1,pixout.size(0))
        # 配置日志记录器
        print(dir)
        logger = logging.getLogger(__name__)
        print("_____________________________________________________________________________________________")
        print(f"prissim:{prissim_avg.avg:.4f}")
        print(f"utissim:{utissim_avg.avg:.4f}")
        print(f"prissim1:{prissim1_avg.avg:.4f}")
        print(f"utissim1:{utissim1_avg.avg:.4f}")

def cal_psnr(dirlist):
    for dir in dirlist:
        pripsnr_avg=eva.AverageMeter()
        utipsnr_avg=eva.AverageMeter()
        for k, i in enumerate(dataset.dataset):
            mask=i["mask"]
            img = i["img"]
            imgname=os.path.basename(i["path"])
            savepath=os.path.join(dir,imgname)
            pixout = Image.open(os.path.join(dir,imgname))
            pixout=pixout.convert("RGB")
            transform = transforms.ToTensor()
            pixout = transform(pixout)
            check_makedirs(os.path.dirname(savepath))
            pripsnr=eva.pri_psnr(to_img(img),pixout,mask)
            utipsnr=eva.uti_psnr(to_img(img),pixout,mask)
            pripsnr_avg.update(pripsnr,pixout.size(0))
            utipsnr_avg.update(utipsnr,pixout.size(0))

        # 配置日志记录器
        print(dir)
        print("_____________________________________________________________________________________________")
        print(f"pripsnr:{pripsnr_avg.avg:.4f}")
        print(f"utipsnr:{utipsnr_avg.avg:.4f}")


def edgepixshow(edgemodel,pixmodel,alpha,sigma,high):
    alpha=torch.ones((datasetopt.batch_size,))*alpha
    for k, i in enumerate(dataset):
        mask=i["mask"]
        img=i["img"]

        inputs = dataset.dataset.tensorimg2MaskLittlelEdge1(img,mask, sigma,high)
        onehot_inputs = edgemodel.maskLittlelEdge2onehot(inputs, i["mask"])
        edgeout = edgemodel.generate(onehot_inputs)
        pixinput = edgeout * mask + (1 - mask) * img
        pixinput2=inputs*mask+(1-mask)*img
        i["A"]=pixinput
        pixmodel.set_input(i)
        pixout = pixmodel.selfgenerate(alpha)
        i["A"]=pixinput2
        pixmodel.set_input(i)
        pixout2 = pixmodel.selfgenerate(alpha)
        colshow(1,1,to_img(img),to_img(pixinput),to_img(pixout),to_img(pixout2),num=4)



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
    # datasetopt = cityscape_dataset.CityscapeDataset.modify_commandline_options().parse_args()
    datasetopt.resource_type = "img_edge"
    datasetopt.target_type = "img"
    datasetopt.datasettype = "only_val"
    datasetopt.return_mask = True
    datasetopt.return_img=True
    datasetopt.location = 2
    datasetopt.return_edge = True
    datasetopt.shuffle=False
    datasetopt.flip_rate=0
    if datasetopt.dataset_mode=="cityscape":
        datasetopt.resizew = 512
        datasetopt.w = 512
        datasetopt.h = 256
        datasetopt.resizeh = 256
    elif datasetopt.dataset_mode=="tokyo247":
        datasetopt.resizew = 512
        datasetopt.w = 512
        datasetopt.h = 512
        datasetopt.resizeh = 512
        datasetopt.datasettype="query"
    datasetopt.batch_size = 1
    datasetopt.range_sigma=4
    print(datasetopt.dataset_mode)
    dataset = create_dataset(datasetopt)  # create a dataset given opt.dataset_mode and other options
    testsavedir=""
    pathlist=["the path of trained edgemodel's opt"]
    pixlist=["the path of trained pixmodel's opt"
    ]
    pixmodellilst = loadmodel(pixlist, batch_size=datasetopt.batch_size, epoch=10)
    modellist=loadmodel(pathlist,batch_size=datasetopt.batch_size,epoch=60,no_loadD=True)

    edgepixsave(*pixmodellilst, edgemodel=modellist[0], alpha=0.2, sigma=2, high=0.85,
                istokyo=(datasetopt.dataset_mode == "tokyo247"), message="",savedir=testsavedir)

    dirlist=[
        "the directory of the tested images",
    ]
    cal_psnr(dirlist)
    cal_ssim(dirlist)
