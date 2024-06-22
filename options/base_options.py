import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False
    def initialize(self, parser):
        """This class defines options used during both training and test time.

        It also implements several helper functions such as parsing, printing, and saving the options.
        It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
        """
        # parser.add_argument('--dataroot',type=str,default=None, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        parser.add_argument('--gpu_ids', type=str, default='0', help='')
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default=None, help='chooses which model to use. bicycle,, ...')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=0, type=int, help='# sthreads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--message', type=str,default="", help='if specified, do not flip the images for data argumentation')
        # model parameters
        parser.add_argument('--G_num_downs', type=int, default=8, help='生成器下采样的层数')
        parser.add_argument('--D_output_nc', type=int, default=1, help='')
        parser.add_argument('--addspade_nc', type=int,default=20, help='表示onehot的语义标签图，20类是因为还有一类是无关类')

        parser.add_argument('--num_D', type=int, default=2, help='number of Discrminators')
        parser.add_argument('--netD', type=str, default='patchgan_multi', help='selects model to use for netD')
        parser.add_argument('--netG', type=str, default='adainunetgenerator', help='selects model to use for netG')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--G_norm', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--D_norm', type=str, default='spectral', help='instance normalization or batch normalization')
        parser.add_argument('--E_norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--mask_weight', type=float,default=1.0, help='鉴别器建筑物部分加权')
        parser.add_argument('--use_D', type=bool,default=True, help='鉴别器建筑物部分加权')
        parser.add_argument('--addspadetype', type=str,default="segonehot", help='addspade的类型')

        # 注意D2的ganmode只能是seg,或者mask
        parser.add_argument('--netD2', type=str, default='unetdiscriminator_1', help='selects model to use for netD')
        parser.add_argument('--gan_mode2', type=str,default="mask", help='鉴别器建筑物部分加权')
        parser.add_argument('--use_D2', type=bool,default=False, help='是否使用第二个鉴别器')
        parser.add_argument('--D2_output_nc', type=int,default=None, help='D2的输入通道数')
        parser.add_argument('--D2_norm', type=int,default=None, help='D2的输入通道数')
        parser.add_argument('--num_D2', type=int,default=None, help='D2的输入通道数')
        parser.add_argument('--conditional_D2', type=bool,default=False, help='if use conditional GAN for D2')

        # extra parameters
        parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        parser.add_argument('--conditional_D', type=bool,default=True, help='if use conditional GAN for D')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--isTrain', type=bool,default=True, help='if specified, print more debugging information')


        # special tasks
        self.initialized = True
        return parser

    def gather_options(self,model_name=None,dataset_name=None):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are difined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        if model_name==None:
            model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        # 返回一个模型设置参数的静态方法
        parser = model_option_setter(parser)
        # 应用静态方法
        opt, _ = parser.parse_known_args()  # parse again with new defaults
        # 用opt获取所有参数，经过应用模型静态方法paser已经整合了模型相关的参数

        # modify dataset-related parser options
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt,is_save=True):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        validateopt(opt)
        message = ''
        message += '----------------- Options ---------------\n'
        message+=opt.message+"\n"
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if is_save:
            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
            util.save_optobject(opt,expr_dir+"/opt")


    def parse(self,model_name=None,dataset_name=None,is_print=False):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(model_name,dataset_name)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if is_print:
            self.print_options(opt)

        # set gpu ids
        if len(opt.gpu_ids)>1:
            str_ids = opt.gpu_ids.split(',')
            opt.gpu_ids = []
            for str_id in str_ids:
                print(str_id)
                id = int(str_id)
                if id >= 0:
                    opt.gpu_ids.append(id)
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

def validateopt(opt):
    if opt.gan_mode=="seg":
        if not opt.D_output_nc>10:
            opt.D_output_nc=20
            print("the parameter D_output_nc!=20,have set as 20")
            raise
    if "adain" in opt.netG:
        if not opt.lambda_vgggram>0.0:
            print("adain必须要使用vgggram损失")
            raise
        if opt.adainrefertype=="img" and ((not opt.adainrefer_nc==3) or (not opt.return_img ==True)):
            opt.adainrefer_nc = 3
            opt.return_img=True
            print("adain类型为img时候，其adainrefer应为3,当前不为3，已经自动更改")
        if (opt.netG=="adainunetgenerator" and(not opt.adainrefertype=="img" or (not opt.return_img ==True)))\
                or opt.netG=="adainunetgenerator2":
            opt.return_img=True