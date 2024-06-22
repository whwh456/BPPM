from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--display_epoch_freq', type=int, default=1, help='the frequency of saving images')
        parser.add_argument('--display_epoch_pagenum', type=int, default=1, help='how many pages of images will be saved per epoch')

        parser.add_argument('--test_n', type=int, default=16, help='测试时生成多少张图片')
        parser.add_argument('--o', type=str, default="0", help='每次训练的序号')
        parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='the type of GAN objective. [vanilla | lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        # training parameters
        parser.add_argument('--niter', type=int, default=None, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero') #多少个epoch学习率降低一次
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--G_lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: linear | step | plateau | cosine')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--no_EMA', type=bool, default=True, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
        parser.add_argument('--building_weight', type=float, default=0.5, help='decay in exponential moving averages')
        # lambda parameters

        parser.add_argument('--lambda_D2', type=float, default=1.0, help='运用于第二个鉴别器的对抗损失参数')
        parser.add_argument('--lambda_G2', type=float, default=1.0, help='运用于第二个鉴别器的对抗时，生成相对于此项的损失参数')
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
        parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
        self.isTrain = True
        return parser

def upgradeopt(opt):
    trainoption = TrainOptions()
    upopt = trainoption.parse(opt.model,opt.dataset_mode)  # get training options
    for arg in vars(upopt):
        optattr=getattr(opt, arg, "noattr")
        if optattr!="noattr":
            setattr(upopt, arg, optattr)
    return upopt
