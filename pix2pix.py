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
import copy
import torch
import os
import sys

if __name__ == '__main__':
# 决定dataroot的值的操作，在创建数据集的参数时进行，所以在之后更改location的值不会改变dataroot
    trainoption=TrainOptions()
    opt = trainoption.parse("inpaint","tokyo247",is_print=False)   # get training options

    opt.message=""
    opt.conditional_D=True
    opt.return_mask = True
    opt.use_dropout = True

    opt.lambda_L1=10
    opt.location=1
    opt.niter = 200
    opt.niter_decay=0
    opt.name = "tokyo_pix2pix"
    opt.G_norm="batch"
    opt.D_norm="batch"
    opt.gan_mode = "original"

    opt.use_D2 = False
    opt.lambda_G2=2.0
    opt.D2_output_nc = 4
    opt.D2_norm="spectral"

    opt.netD = "patchgan_multi"
    opt.netD2 = "net"
    opt.netG = "unet"
    opt.upsample="basic"
    # opt.addspade_nc=20
    opt.model = "inpaint"
    opt.G_lr = 0.0001
    opt.D_lr = 0.0004
    opt.input_nc = 3
    opt.output_nc = 3
    opt.resource_type = "img_edge"
    # opt.range_sigma=0.0
    # opt.range_high=0.0
    opt.w=512
    opt.h=512
    opt.resizew=512
    opt.resizeh=512
    opt.target_type = "img"
    opt.datasettype = "images"
    opt.print_freq=1000
    opt.save_epoch_freq=15
    opt.display_epoch_freq = 3000
    opt.building_weight=0.8
    opt.batch_size=4
    trainoption.print_options(opt)

    opt.num_worker=12
    # opt.continue_train=True
    # opt.epoch=30
    # opt.epoch_count=31
    # opt.o="1"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.

    print('The number of training images = %d' % dataset_size)
    visualizer = util.my_visualizer.Visualizer(opt)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    cross_cpu_time = 0.0
    cross_gpu_time = 0.0
    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            t_data = iter_start_time - iter_data_time  # t_data是每个iteration数据加载的时间
            cross_cpu_time += t_data

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if data["A"].size(0) != opt.batch_size:  # if this batch of input data is enough for training.
                print('skip this batch')
                continue
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            model.add_losstolossdict()

            b_comp_t = time.time() - iter_start_time
            t_comp = (b_comp_t) / opt.batch_size
            cross_gpu_time += b_comp_t

            if total_iters % opt.print_freq < opt.batch_size:  # print training losses and save logging information to the disk
                model.add_losstolossdict()
                visualizer.print_current_losses(epoch, epoch_iter, model.lossdict, t_comp, t_data, cross_gpu_time,
                                                cross_cpu_time)
                cross_gpu_time = 0.0
                cross_cpu_time = 0.0
            iter_data_time = time.time()

            with torch.no_grad():
                visualizer.showexpample((epoch-1)*dataset_size+(i*opt.batch_size), opt.display_epoch_pagenum, 1, to_img(model.real_B.cpu()),
                                      to_img(model.real_A.cpu()), to_img(model.fake_B).cpu(),
                                      num=4)

        with torch.no_grad():
            visualizer.savelossgraph(epoch,model.lossdict)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

