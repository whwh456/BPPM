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
import time
from util.my_visualizer import to_img,save_all_loss
import util.my_visualizer
from options.train_options import TrainOptions
from data import create_dataset,get_option_setter
from models import create_model
from util.util import update_EMA
import copy
import torch
import os

if __name__ == '__main__':
# 决定dataroot的值的操作，在创建数据集的参数时进行，所以在之后更改location的值不会改变dataroot
    trainoption=TrainOptions()
    opt = trainoption.parse("inpaint","cityscape")   # get training options

    opt.message="测试canny损失，不用条件gan，opt.lambda_difcanny=0.2.当lambda为10的时候只能重复输入的边缘。为1时不重复边缘，但是依然是边缘"
    opt.conditional_D=False
    opt.lambda_difcanny=0.02
    opt.D_output_nc=4
    # opt.lambda_D_sim = 0
    opt.return_mask=True
    opt.return_seg=True
    opt.lambda_GAN=1.0
    opt.lambda_vgg=10.0
    # opt.lambda_vgggram=0.0
    opt.lambda_L1=10.0
    opt.nz=0
    opt.location=1
    opt.niter = 60
    opt.model = "inpaint"
    opt.name = "cannyloss_1"
    opt.G_norm="batch"
    opt.D_norm="spectralinstance"
    # opt.no_EMA=False
    opt.netD = "patchgan_multi"
    opt.netG = "unet_256"
    opt.gan_mode="hinge"
    opt.input_nc = 3
    opt.output_nc = 3
    opt.save_epoch_freq=10
    opt.resource_type = "img_edge"
    opt.target_type = "img"
    opt.datasettype = "only_train"
    opt.print_freq=200

    trainoption.print_options(opt)


    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    print('The number of training images = %d' % dataset_size)
    visualizer=util.my_visualizer.Visualizer(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq< opt.batch_size:
                t_data = iter_start_time - iter_data_time #t_data是每个iteration数据加载的时间
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if data["A"].size(0)!=opt.batch_size:      # if this batch of input data is enough for training.
                print('skip this batch')
                continue
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq < opt.batch_size:    # print training losses and save logging information to the disk
                model.add_losstolossdict()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter,model.lossdict, t_comp, t_data)
            iter_data_time = time.time()

        if not opt.no_EMA:
            run_stats=(i==len(dataset.dataloader)-1 or i==len(dataset.dataloader)-2)
            update_EMA(model, dataset.dataloader, opt,run_stats)

        with torch.no_grad():
            visualizer.showexpample(model.lossdict, opt.display_epoch_pagenum, 1, to_img(model.real_B.cpu()),
                                    to_img(model.real_A.cpu()), to_img(model.fake_B).cpu(),
                                    num=6)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_all_loss(model.lossdict,dir,opt.o)
