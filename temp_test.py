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
import torch.nn.functional as F
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadmodel(pathList, batch_size,location=2, epoch=60):
    list = []
    for path in pathList:
        opt = util.util.load_optobject(path, batch_size=batch_size,location=location, epoch=epoch)
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        list.append(model)
    return list


def showDout(model, dataset):
    for i in dataset:
        model.set_input(i)
        model.forward()
        D_fake = model.netD(model.fake_data_forD)
        D_real = model.netD(model.real_data_forD)
        mask = i["mask"]
        for k in range(opt.num_D):
            D_cfake = (torch.clamp(D_fake[k][0], -1, 1) + 1) / 2
            D_creal = (torch.clamp(D_real[k][0], -1, 1) + 1) / 2
            print(D_creal.size())
            mask = F.interpolate(mask, size=(D_creal.size(2), D_creal.size(3)),
                                 mode='bilinear')
            mv.colshow(1, 1, mask, D_creal, D_cfake, num=4)


def showDonehot(modellist):
    for i in dataset:
        for argi,model in enumerate(modellist):
            print(f"第{argi}个模型：")
            model.set_input(i)
            fake_B=model.selfgenerate()
            # D_fake = model.netD(model.fake_data_forD)
            # D_real = model.netD(model.real_data_forD)
            mask = i["mask"]
            # for k in range(model.opt.num_D):
            #     D_cfake = D_fake[k][0]
            #     D_creal = D_real[k][0]
            #     print(D_creal.size())
            #     mask = F.interpolate(mask, size=(D_creal.size(2), D_creal.size(3)),
            #                          mode='bilinear')
            #     # D_cfake = util.util.t19totcolor(D_cfake)
            #     # D_creal = util.util.t19totcolor(D_creal)
            #     mv.colshow(1, 1, mask, D_creal, D_cfake, num=4)
            #     mv.colshow(1, 1, to_img(model.real_B),to_img(model.fake_B), num=4)

            if model.opt.use_D2:
                print("D2")
                D_fake2 = model.netD2(fake_B)
                D_real2 = model.netD2(model.real_B)
                D_cfake2 = util.util.t19totcolor(D_fake2)
                D_creal2 = util.util.t19totcolor(D_real2)
                mv.colshow(1,1,to_img(model.real_B),to_img(fake_B),mask,D_creal2,D_cfake2,num=2)

def showsegD2(modellist):
    for i in dataset:
        for argi,model in enumerate(modellist):
            print(f"第{argi}个模型：")
            print(i["img"].size())
            seg=model.netD2(i["img"])
            # for k in range(model.opt.num_D):
            #     D_cfake = D_fake[k][0]
            #     D_creal = D_real[k][0]
            #     print(D_creal.size())
            #     mask = F.interpolate(mask, size=(D_creal.size(2), D_creal.size(3)),
            #                          mode='bilinear')
            #     # D_cfake = util.util.t19totcolor(D_cfake)
            #     # D_creal = util.util.t19totcolor(D_creal)
            #     mv.colshow(1, 1, mask, D_creal, D_cfake, num=4)
            #     mv.colshow(1, 1, to_img(model.real_B),to_img(model.fake_B), num=4)

            D_cfake2 = util.util.t19totcolor(seg)

            mv.colshow(1,1,to_img(i["img"]),D_cfake2,num=1)


def comparepix_xiout(*args, xi=6):
    numpy.set_printoptions(suppress=True)
    alpha = util.util.create_z(datasetopt.batch_size, xi, 1)
    print(f"有多少个模型：{len(args)}")
    print(alpha)
    ssimlist = numpy.zeros((len(args),xi))# 用来记录同一系列相邻隐私强度fake之间的ssim差值
    for k, i in enumerate(dataset):
        list = []
        for argi,model in enumerate(args):
            model.set_input(i)
            xi_real_A = util.util.getinput(model.real_A, xi)
            xi_adainrefer=None
            if hasattr(model, 'adainrefer'):
                xi_adainrefer = util.util.getinput(model.adainrefer, xi)
            if model.img!=None:
                xi_img=util.util.getinput(model.img,xi)
            fake_B = model.generate(xi_real_A, xi_adainrefer, alpha,xi_img)
            # show_fake_B = util.util.adjustoutshow(xi, to_img(fake_B), to_img(model.real_A), to_img(model.real_B),i["mask"],i["edge"])
            show_fake_B = util.util.adjustoutshow(xi, to_img(fake_B), to_img(model.real_A), to_img(model.real_B))
            list.append(show_fake_B)
            xi_real_B = util.util.getinput(model.real_B, xi)
            lastssim=0
            print(f"模型{argi}")
            for ii in range(xi):
                ssim = util.evaluate.ski_ssim(xi_real_B[ii], fake_B[ii])
                ssimlist[argi][ii]=round((ssim-lastssim)+ssimlist[argi][ii],5)
                # ssimlist[argi][ii] = ssim + ssimlist[argi][ii]
                print(ssim)
                lastssim=ssim

        # mv.colshow(1, 1, *list, num=5, batch_x=2)
        print(k)
        if k >200:
            print(ssimlist / (200))
            break




def comparepixout(*args):
    alpha = torch.zeros((datasetopt.batch_size, 1)) + 0.5  # 这个alpha是用来控制输入edge的疏密程度的
    ssimlist = numpy.zeros((len(args)))
    for k, i in enumerate(dataset):
        real_A = i["A"]
        mask = i["mask"]
        real_B = i["B"]
        list = []
        for argi,model in enumerate(args):
            model.set_input(i)
            fake_B=model.selfgenerate()
            list.append(to_img(fake_B))
            ssim = util.evaluate.ski_ssim(fake_B, model.real_B)
            print(f"模型{argi}")
            print(ssim)
            ssimlist[argi] = ssim + ssimlist[argi]
        mv.colshow(1, 1, to_img(real_A), to_img(real_B), *list, num=4)
        if k > 20:
            break

# nb为采样多少个批次
def cal_ssim(nb, *args, forentire=True):
    ssimlist = [0 for i in range(len(args))]
    if not forentire:
        pri_ssimlist = [0 for i in range(len(args))]
        uti_ssimlist = [0 for i in range(len(args))]
    for k, i in enumerate(dataset):
        if k == nb:
            break
        real_A = i["A"]
        real_B = i["B"]
        mask = i["mask"]
        for ii, model in enumerate(args):
            model.set_input(i)
            if hasattr(model, 'adainrefer'):
                fake_B = model.generate(real_A,model.adainrefer)
            else:
                fake_B = model.generate(real_A)
            ssim_value = util.evaluate.ski_ssim(fake_B, real_B)
            ssimlist[ii] = ssimlist[ii] + ssim_value
            if not forentire:
                pri_ssim_value = util.evaluate.ski_ssim(fake_B * mask, real_B * mask)
                pri_ssimlist[ii] = pri_ssimlist[ii] + pri_ssim_value
                uti_ssim_value = util.evaluate.ski_ssim(fake_B * (1 - mask), real_B * (1 - mask))
                uti_ssimlist[ii] = uti_ssimlist[ii] + uti_ssim_value
    for i in range(len(ssimlist)):
        ssimlist[i] = ssimlist[i] / nb
        print(f"entireimg_ssim{i}:{ssimlist[i]:.4f}")
        if not forentire:
            pri_ssimlist[i] = pri_ssimlist[i] / nb
            print(f"privacypart_ssim{i}:{pri_ssimlist[i]:.4f}")
            uti_ssimlist[i] = uti_ssimlist[i] / nb
            print(f"utilitypart_ssim{i}:{uti_ssimlist[i]:.4f}")


def calgram(model, xi=8):
    from models.losses import VGGgramLoss
    alpha = util.util.create_z(datasetopt.batch_size, xi, 1)
    print(alpha)
    VGGgramloss = VGGgramLoss(device, 2)
    for k, i in enumerate(dataset):
        model.set_input(i)
        xi_real_A = util.util.getinput(model.real_A, xi)
        xi_adainrefer = None
        if hasattr(model, 'adainrefer'):
            xi_adainrefer = util.util.getinput(model.adainrefer, xi)
        fake_B = model.generate(xi_real_A, xi_adainrefer, alpha=alpha)
        gramlist = []
        for i in range(xi):
            gramlist.append(format(VGGgramloss(xi_real_A[i].unsqueeze(0), fake_B[i].unsqueeze(0)).item(), ".10f"))
        print(gramlist)
        colshow(1,1,to_img(fake_B),batch_x=4,num=4)




if __name__ == '__main__':
# 决定dataroot的值的操作，在创建数据集的参数时进行，所以在之后更改location的值不会改变dataroot
    trainoption=TrainOptions()
    datasetopt = trainoption.parse("inpaint","cityscape")   # get training options
    # datasetopt.dataroot="E:\pittsburgh250k\queries_real\\000"
    datasetopt.shuffle=False
    datasetopt.return_img=True
    datasetopt.return_mask = True
    datasetopt.resource_type = "img_edge"
    datasetopt.target_type = "img"
    datasetopt.datasettype = "only_val"
    datasetopt.return_mask=True
    datasetopt.return_seg=True
    datasetopt.epoch=60
    datasetopt.location=2
    datasetopt.isTrain=False
    datasetopt.return_edge=True
    datasetopt.resizew=512
    datasetopt.resizeh = 256
    datasetopt.w=512
    datasetopt.h=256
    datasetopt.serial_batches=True
    datasetopt.batch_size=1
    dataset = create_dataset(datasetopt)  # create a dataset given opt.dataset_mode and other options

    pathlist=["F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_l1_1\opt",
              "F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_l2\opt" ,
              "F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_l3\opt" ]
#                "F:\mycode\BicycleGAN-master\checkpoints\multisim_1\opt"
#"F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_a1\opt",
             # "F:\mycode\BicycleGAN-master\checkpoints\\adain\\adain_imgu1_a3\opt",

#"F:\mycode\BicycleGAN-master\checkpoints\multisim_1\opt"
#"F:\mycode\BicycleGAN-master\checkpoints\\adain_noise_a_1\opt",
#"F:\mycode\BicycleGAN-master\checkpoints\\adain_mask_a_2\opt"
    modellist=loadmodel(pathlist,batch_size=datasetopt.batch_size,epoch=90)
    # calgram(*modellist)
    comparepix_xiout(*modellist,xi=6)
    # comparepixout(*modellist)
    # showsegD2(modellist)
    # showDonehot(modellist)
    # cal_ssim(100,*modellist,forentire=False)
    # calgram(model,10)
