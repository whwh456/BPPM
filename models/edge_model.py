import torch
import util.util
from .base_model import BaseModel
from . import networks
from . import my_network
from . import losses
import copy

# 注意当前的代码支持用multidiscrimnator。不支持单独用nlayerdiscriminator，并且还要利用D的中间特征计算F的情形，这种情况可以设置num_D=1,依然使用
# multidiscrminator
class EdgeModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        # parser.add_argument('--F_type', type=str, default="normal", help='正常感知损失，使用D的中间特征')
        parser.add_argument('--lambda_ADAIN', type=float, default=10.0, help='F感知损失的超参数')
        parser.add_argument('--lambda_F', type=float, default=0.0, help='F感知损失的超参数')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='F感知损失的超参数')
        parser.add_argument('--lambda_vgggram', type=float, default=0.0, help='F感知损失的超参数')
        parser.add_argument('--lambda_D_sim', type=float, default=0.0, help='F感知损失的超参数')
        parser.add_argument('--simtype', type=str, default="mask", help='sim的类型，可选有mask,real')
        parser.add_argument('--lambda_difcanny', type=float, default=0.0, help='canny（fakeB）-edge')
        parser.add_argument('--patch_nums', type=float, default=256, help='F感知损失的超参数')
        parser.add_argument('--patch_size', type=float, default=16, help='F感知损失的超参数')
        parser.add_argument('--lambda_D2_sim', type=float, default=0.0, help='F感知损失的超参数')
        parser.add_argument('--simtype2', type=str, default="real", help='sim的类型，可选有mask,real')
        parser.add_argument('--patch_nums2', type=float, default=256, help='F感知损失的超参数')
        parser.add_argument('--patch_size2', type=float, default=20, help='F感知损失的超参数')
        parser.add_argument('--adaintraintype', type=str, default="0.5melt", help="0.5melt:融合训练，twofake:"
                                                                                  "生成两种隐私保护强度的fake，都参与反向传播")
        parser.add_argument('--adainrefertype', type=str,default="mask", help='addrefer的类型')
        parser.add_argument('--adainformin', type=bool,default=False, help='如果为TRUE，则完全经过adain模块才为隐私强度最小的图像')
        parser.add_argument('--start_adain_layer', type=int, default=4, help='上采样从0从后往前数第一个层后有adain的层序号')
        parser.add_argument('--adainrefer_nc', type=int, default=1, help='adainrefer的通道数')
        parser.add_argument('--adain_layer_num', type=int, default=1, help='有adain层的数量')

        parser.add_argument('--lambda_divloss', type=float, default=0.0, help='F感知损失的超参数')
        parser.add_argument('--div_layer_num', type=int, default=5, help='F感知损失的超参数')
        parser.add_argument('--samediv', type=bool, default=False, help='是否把div的各层都使得相似，alpha全为0')
        parser.set_defaults(input_nc=3)
        parser.set_defaults(output_nc=3)
        parser.add_argument('--random_type', type=str, default="uni", help='uni,or gauss')
        return parser

    def __init__(self, opt):
        # if opt.isTrain:
        #     assert opt.batch_size % 2 == 0  # load two images at one time.
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.opt=opt
        self.lossdict = {'G_GAN':[],'D_GAN':[]}
        if opt.lambda_F>0.0:
            self.lossdict["F"]=[]
        if opt.lambda_L1>0.0:
            self.lossdict["L1"]=[]
        if opt.lambda_vgg>0.0:
            self.lossdict["VGG"]=[]
        if opt.lambda_vgggram>0.0:
            self.lossdict["VGG_GRAM"]=[]
        if opt.lambda_D_sim>0.0:
            self.lossdict["D_sim"]=[]
        if opt.lambda_difcanny>0.0:
            self.lossdict["Difcanny"]=[]
            self.opt.return_mask=True
        if "adain" in opt.netG:
            self.lossdict["ADAIN"]=[]
        if opt.lambda_divloss > 0.0:
            self.lossdict["Div"] = []
        if opt.use_D2:
            self.lossdict["G_GAN2"]=[]
            self.lossdict["D2"] = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'real_B', 'fake_B']
        print(f"loss:{self.lossdict.keys()}")
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        # use_D = opt.isTrain and opt.lambda_GAN > 0.0
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,addspade_nc=opt.addspade_nc,
                                      norm=opt.G_norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample,w=opt.w,h=opt.h,model=opt.model)

        D_input_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        D2_input_nc = opt.input_nc + opt.output_nc if opt.conditional_D2 else opt.output_nc
        print(f"net_G:{self.netG.__class__.__name__}")
        if opt.use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_input_nc, opt.ndf,output_nc=opt.D_output_nc, netD=opt.netD, norm=opt.D_norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_D, gpu_ids=self.gpu_ids,need_Feat=opt.lambda_F>0)
            print(f"net_D:{self.netD.__class__.__name__}")
            if opt.use_D2:
                self.model_names += ['D2']
                self.netD2 = networks.define_D(D2_input_nc, opt.ndf,output_nc=opt.D2_output_nc, netD=opt.netD2, norm=opt.D2_norm, nl=opt.nl,
                                              init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_D2, gpu_ids=self.gpu_ids,need_Feat=opt.lambda_F>0)
                print(f"net_D2:{self.netD2.__class__.__name__}")
        with torch.no_grad():
            if not opt.no_EMA:
                self.model_names+=["EMA"]
                self.netEMA = copy.deepcopy(self.netG)

        if opt.isTrain:
            if opt.netD=="unetdiscriminator" or "melt" in opt.gan_mode:
                self.criterionGAN=losses.MixGANLoss(opt.gan_mode,self.device,weight=opt.building_weight)
            elif not (opt.gan_mode=="mask" or opt.gan_mode=="seg") :
                self.criterionGAN = losses.GANLoss(gan_mode=opt.gan_mode,device=self.device).to(self.device)
            else:
                self.criterionGAN=losses.SegGANLoss(seggantype=opt.gan_mode,device=self.device,weight=opt.building_weight)
            if opt.use_D2:
                if opt.gan_mode2 == "mask" or opt.gan_mode2 == "seg":
                    self.criterionGAN2=losses.SegGANLoss(seggantype=opt.gan_mode2,device=self.device,weight=opt.building_weight)
                else:
                    self.criterionGAN2 = losses.GANLoss(gan_mode=opt.gan_mode, device=self.device).to(self.device)
            self.criterionF = torch.nn.L1Loss()
            self.criterionL1=torch.nn.L1Loss()
            self.criterionVGG=losses.VGGLoss(self.device,opt.location).to(self.device)
            if self.opt.lambda_vgggram>0.0:
                self.criterionVGGgram=losses.VGGgramLoss(self.device,opt.location)
            if self.opt.lambda_D_sim>0.0:
                self.criterionDsim=losses.PatchSimLoss(self.opt.patch_nums,self.opt.patch_size).to(self.device)
            if self.opt.lambda_D2_sim>0.0 and opt.use_D2:
                self.criterionD2sim=losses.PatchSimLoss(self.opt.patch_nums2,self.opt.patch_size2).to(self.device)
            if self.opt.lambda_difcanny>0.0:
                self.criteriondDifcanny=losses.DifcannyLoss(sigma=2.0,high=0.8,device=self.device)
            if opt.lambda_divloss > 0.0:
                self.DivLoss = losses.Cal_Div_Loss(self.opt.div_layer_num, device=self.device)
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.G_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if opt.use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.D_lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
                if opt.use_D2:
                    self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.D_lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D2)
        print(self.model_names)

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        self.istwofake="adain" in self.opt.netG and self.opt.adaintraintype=="twofake"
        self.real_A = input['A' ].to(self.device)
        self.real_B = input['B' ].to(self.device)

        self.calloss_realB=self.real_B
        if self.istwofake:
            self.calloss_realB = self.firstrepeat2(self.real_B)
        if self.opt.lambda_divloss>0.0:
            if self.opt.samediv==False:
                self.alpha=input["alpha"]
            else:
                self.alpha=torch.zeros(size=(self.opt.batch_size,))

        self.mask=None
        self.label=None
        self.fortrainfake_label=None
        self.fortrainfake_mask=None
        if self.opt.return_mask or self.opt.lambda_difcanny>0.0 or self.opt.building_weight!=0.5 or \
            ("adain" in self.opt.netG and self.opt.adainrefertype == "mask"):
            self.mask=input["mask"].to(self.device)
            self.fortrainfake_mask=self.mask
            if self.istwofake:
                self.fortrainfake_mask = self.firstrepeat2(self.mask)
        if self.opt.gan_mode=="seg" or (self.opt.gan_mode2=="seg" and self.opt.use_D2) :
            self.label=input["seg"].to(self.device)
            self.fortrainfake_label=self.label
            if self.istwofake:
                self.fortrainfake_label = self.firstrepeat2(self.label)
        if self.opt.gan_mode=="mask" or (self.opt.gan_mode2=="mask" and self.opt.use_D2):
            self.label=input["mask"].to(self.device)
            self.fortrainfake_label=self.label
            if self.istwofake:
                self.fortrainfake_label = self.firstrepeat2(self.label)
        if "withspade" in self.opt.netG:
            if self.opt.addspadetype=="segonehot":
                self.seg = input["seg"].to(self.device)
                self.segonehot = util.util.seg2onehot(self.seg, self.device, self.opt.addspade_nc)
                self.addspade=self.segonehot
            elif self.opt.addspadetype=="mask":
                self.addspade=input["mask"].to(self.device)

        if self.opt.netG == "spade":
            self.seg = input["seg"].to(self.device)
            self.segonehot = util.util.seg2onehot(self.seg, self.device, self.opt.addspade_nc)
            self.real_A = self.segonehot

        if "adain" in self.opt.netG:
            if self.opt.adainrefertype=="mask":
                self.adainrefer=self.mask
            elif self.opt.adainrefertype=="noise":
                self.adainrefer=torch.randn((self.opt.batch_size,128)).to(self.device)
            elif self.opt.adainrefertype=="img":
                self.adainrefer=input["img"].to(self.device)

    def generate(self,input,input2=None,alpha=0):
        with torch.no_grad():
            if "withspade" in self.opt.netG:
                fake_B = self.netG(input, input2)
            elif "spade" == self.opt.netG:
                fake_B = self.netG(input)
            elif "adain" in self.opt.netG:
                fake_B = self.netG(input, input2, alpha)
            else:
                fake_B = self.netG(input)
        return fake_B

    def forward(self):
        # get real images
        # generate fake_B_random
        
        if "withspade" in self.opt.netG:
            self.fake_B = self.netG(self.real_A,self.addspade)
            self.calossfake=self.fake_B
        elif "spade" ==self.opt.netG:
            self.fake_B = self.netG(self.segonehot)
            self.calossfake=self.fake_B
        elif "adain" in self.opt.netG and self.opt.adaintraintype=="0.5melt":
            self.fake_B=self.netG(self.real_A,self.adainrefer,0.5)  # 设置alpha为0.5就在训练中不生成两种fake，而是融合adain进行训练
            self.calossfake=self.fake_B
        elif "adain" in self.opt.netG and self.opt.adaintraintype == "twofake":
            self.fake_B=self.netG(self.real_A,self.adainrefer,0)
            self.maxalpha_fake=self.netG(self.real_A,self.adainrefer,1)
            self.calossfake=torch.cat((self.fake_B,self.maxalpha_fake),dim=0)
            #在训练时由于此时fake_B的大小不再是批量大小，任何参与和fake进行损失计算的辅助张量都需要调整

            self.fortrainfake_mask=self.mask.repeat(2,1,1,1)
            self.fortrainfake_real=self.real_A.repeat(2,1,1,1)
        else:
            self.fake_B = self.netG(self.real_A)
            self.calossfake=self.fake_B
            
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_forD = torch.cat([self.real_A, self.fake_B], 1)
            self.real_data_forD = torch.cat([self.real_A, self.real_B], 1)
            if self.istwofake:
                maxfake_data_forD=torch.cat([self.real_A, self.maxalpha_fake], 1)
                self.fake_data_forD=torch.cat([self.fake_data_forD,maxfake_data_forD],0)
        else:
            self.fake_data_forD =  self.fake_B
            self.real_data_forD = self.real_A
            if self.istwofake:
                self.fake_data_forD = torch.cat([self.fake_data_forD, self.maxalpha_fake], 0)

        if self.opt.use_D2:
            if self.opt.conditional_D2:   # tedious conditoinal data
                self.real_data_forD2 = self.real_data_forD
                self.fake_data_forD2 =self.fake_data_forD
            else:
                self.fake_data_forD2 =  self.fake_B
                self.real_data_forD2 = self.real_A
                if self.istwofake:
                    self.fake_data_forD2 = torch.cat([self.fake_data_forD2, self.maxalpha_fake], 0)

    def discriminate(self,fake_forD,real_forD):
        fake_and_real = torch.cat([real_forD,fake_forD], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def discriminateD2(self,fake_forD,real_forD):
        fake_and_real = torch.cat([real_forD, fake_forD], dim=0)
        discriminator_out = self.netD2(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        k = 2
        if self.istwofake:
            k = 3
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                real.append([tensor[:tensor.size(0) // k] for tensor in p])
                fake.append([tensor[tensor.size(0) // k:] for tensor in p])
        else:
            real = pred[:pred.size(0) // k]
            fake = pred[pred.size(0) // k:]
        return fake, real

    def backward_D(self):
        # Fake, stop backprop to the generator by detaching fake_B
        # 注意如果adain训练模式为twofake，则此处的pred_fake的第0维是fake和maxfake的拼接
        pred_fake,pred_real=self.discriminate(self.fake_data_forD.detach(),self.real_data_forD)
        self.D_fake_loss=self.criterionGAN(pred_fake,False,True,self.label,self.fortrainfake_mask)
        self.D_real_loss=self.criterionGAN(pred_real,True,True,self.label,self.mask)

        if "adain" in self.opt.netG and self.opt.adainrefertype == "twofake":
            self.D_GAN_loss = (self.D_fake_loss + self.D_real_loss) / 3 * self.opt.lambda_GAN
        else:
            self.D_GAN_loss=(self.D_fake_loss+self.D_real_loss)/2*self.opt.lambda_GAN

        if self.opt.lambda_D_sim>0.0:
            if self.opt.simtype=="mask":
                self.D_sim_loss=self.criterionDsim(pred_fake,self.fortrainfake_mask)*self.opt.lambda_D_sim
            elif self.opt.simtype=="real":
                fortrainfake_predreal=pred_real
                if self.istwofake:
                    fortrainfake_predreal=self.fortrainfake_predreal(pred_real)
                self.D_sim_loss=self.criterionDsim(pred_fake,fortrainfake_predreal)*self.opt.lambda_D_sim
        else:
            self.D_sim_loss=0.0
        self.D_loss=self.D_sim_loss+self.D_GAN_loss
        self.D_loss.backward()


    def backward_D2(self):
        pred_fake2, pred_real2 = self.discriminateD2(self.fake_data_forD2.detach(), self.real_data_forD2)
        self.D2_fake_loss = self.criterionGAN2(pred_fake2, False, True,self.fortrainfake_label,self.fortrainfake_mask)
        self.D2_real_loss = self.criterionGAN2(pred_real2, True, True,self.label,self.mask)
        self.D2_GAN_loss = (self.D2_fake_loss + self.D2_real_loss) / 2 * self.opt.lambda_D2
        if self.opt.lambda_D2_sim>0.0:
            if self.opt.simtype=="mask":
                self.D2_sim_loss=self.criterionDsim(pred_fake2,self.fortrainfake_mask)*self.opt.lambda_D2_sim
            elif self.opt.simtype=="real":
                fortrainfake_predreal=pred_real2
                if self.istwofake:
                    fortrainfake_predreal = self.fortrainfake_predreal(pred_real2)
                self.D2_sim_loss = self.criterionDsim(pred_fake2, fortrainfake_predreal) * self.opt.lambda_D2_sim
        else:
            self.D2_sim_loss=0.0
        self.D2_loss = self.D2_sim_loss+self.D2_GAN_loss
        self.D2_loss.backward()

    # 输fake_data_forD
    def backward_G(self):
        netD=self.netD
        # 当使用adain的时候把原始生成的建筑物图像（alpha为0）作为隐私强度最低的生成效果，而当alpha设置为1时候，中间特征自动根据adainrefer来学习
        # adain 的mean和std，所以此时的隐私强度认为是最大，为了做到这一点就要求，隐私强度低的时候的风格与原始图像接近，这项损失是gramloss，隐私图
        # 像高的时候的风格与隐私设置低的时候生成的风格差距保持大，这项损失设置为adain_loss
        if "adain" in self.opt.netG:
            if not self.istwofake:
                self.maxalpha_fake=self.netG(self.real_A,self.adainrefer,alpha=1)
            self.ADAIN_loss=-self.criterionVGGgram(self.fake_B,self.maxalpha_fake)*self.opt.lambda_ADAIN
            if self.opt.adaintraintype=="twofake":
                pass
            # 使用这个就是说要生成两种fake，一种隐私保护强度低，一种隐私保护强度高
            # self.fake_B=torch.cat((self.fake_B,self.maxalpha_fake),dim=1)
        else:
            self.ADAIN_loss=0.0
        # gan损失
        if self.opt.lambda_GAN > 0.0:
            pred_fake,_=self.discriminate(self.fake_data_forD,self.real_data_forD)
            self.G_GAN_loss = self.criterionGAN(pred_fake, True,False,self.fortrainfake_label,self.fortrainfake_mask)*self.opt.lambda_GAN
        else:
            self.G_GAN_loss=0.0

        if self.opt.use_D2:
            pred_fake2,_=self.discriminateD2(self.fake_data_forD2,self.real_data_forD2)
            self.G_GAN2_loss = self.criterionGAN2(pred_fake2, True,False,self.fortrainfake_label,self.fortrainfake_mask)*self.opt.lambda_G2
        else:
            self.G_GAN2_loss=0.0

        # 针对鉴别器的中间特征的感知损失
        if self.opt.lambda_F>0.0:
            pred_real = netD(self.real_data_forD)
            if self.istwofake:
                pred_real=self.fortrainfake_predreal(pred_real)
            num_D = len(pred_fake)

            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionF(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_F / num_D
            self.GAN_Feat_loss=GAN_Feat_loss
        else:
            self.GAN_Feat_loss=0.0


        if self.opt.lambda_L1>0.0:
            self.L1_loss=self.criterionL1(self.calossfake,self.calloss_realB)
        else:
            self.L1_loss=0.0

        # 针对vgg的感知损失
        if self.opt.lambda_vgg>0.0:

            self.VGG_loss=self.criterionVGG(self.calloss_realB,self.calossfake)  * self.opt.lambda_vgg
        else:
            self.VGG_loss =0.0

        if self.opt.lambda_vgggram>0.0:
            self.VGG_GRAM_loss=self.criterionVGGgram(self.calloss_realB,self.calossfake)* self.opt.lambda_vgggram
        else:
            self.VGG_GRAM_loss=0.0

        if self.opt.lambda_difcanny:
            self.Difcanny_loss=self.criteriondDifcanny(self.calossfake,self.calloss_realB,self.fortrainfake_mask)*self.opt.lambda_difcanny
        else:
            self.Difcanny_loss=0.0

        if self.opt.lambda_divloss:

            self.Div_loss=self.DivLoss(self.fake_B,self.real_B,self.alpha)*self.opt.lambda_divloss
        else:
            self.Div_loss=0.0

        loss_G=self.G_GAN_loss+self.GAN_Feat_loss+self.VGG_loss+self.L1_loss+self.VGG_GRAM_loss+self.Difcanny_loss+self.ADAIN_loss
        loss_G.backward()

#retain_graph=True
    def update_D(self):
        self.set_requires_grad([self.netD], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        if self.opt.use_D2:
            self.optimizer_D2.zero_grad()
            self.set_requires_grad([self.netD2], True)
            self.backward_D2()
            self.optimizer_D2.step()

    def update_G(self):
        # update G and E
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()

    def add_losstolossdict(self):
        for i in self.lossdict:
            self.lossdict[i].append(getattr(self,i+"_loss").cpu().item())

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()
        # print("一轮结束")
    # 把predreal变为和拼接了predfake对应的形式，注意predfake可能是list
    def fortrainfake_predreal(self,pred_real):
        if isinstance(pred_real, list):
            for pred_i in pred_real:
                if isinstance(pred_i, list):
                    for pred_ii in pred_i:
                        pred_ii=self.firstrepeat2(pred_ii)
                else:
                    pred_i=pred_i.repeat(2, *([1] * (len(pred_i.size()) - 1)))
        else:
            pred_real=pred_real.repeat(2, *([1] * (len(pred_real.size()) - 1)))
        return pred_real

    def firstrepeat2(self,tensor):
        return tensor.repeat(2, *([1] * (len(tensor.size()) - 1)))


    # 根据masklittleedge生成onehot形式，要求masklittleedge的形状为(N,1,H,W)

    def maskLittlelEdge2onehot(self,masklitteledge,mask):
        N,_,H,W=masklitteledge.size()
        onehot_masklittleedges=torch.zeros((N,3,H,W))
        for i in range(N):
            mask_littleedge = mask[i] + masklitteledge.to(mask.device)
            mask_littleedge = torch.nn.functional.one_hot(mask_littleedge[i].squeeze(0).long(), 3).permute(2, 0,
                                                                                               1).float()

            onehot_masklittleedges[i]=mask_littleedge
        return onehot_masklittleedges
