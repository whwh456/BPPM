import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from . import util

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0,1)
    return x

def imshow(img,filename=None):
    fig=plt.figure()
    a=len(img.size())
    c=img.size()[0]
    npimg = img.detach().cpu().numpy()
    plt.axis('off')
    array = np.transpose(npimg,(1,2,0))
    if filename != None:
        matplotlib.image.imsave(filename,array)
    else:
        if a==2 or c==1:
            plt.imshow(array,cmap='gray')
        else:
            plt.imshow(array)
        plt.show()
    plt.close('all')


def writelist(dir,losslist):#把loss列表写入文件
    with open(dir,'w',encoding='utf-8') as f:  #使用with方法
        for i in losslist:
            f.write(str(i))
            f.write('\r')

def readlist(dir):
    with open(dir,'r',encoding='utf-8') as f:  #使用with方法
        a=f.readlines()
        losslist=[]
        for i in a:
            losslist.append(float(i))
        return losslist

def save_crossloss(crossloss_dir,now_loss):#保存断续训练所有的损失列表文本
    if(os.path.exists(crossloss_dir)):#crossloss文件是否存在，存在则读出来，连上本次的loss，再存进去
        with open(crossloss_dir,'r',encoding='utf-8') as f:
            crosslosslist=readlist(crossloss_dir)
            for i in now_loss:
                crosslosslist.append(i)
        writelist(crossloss_dir,crosslosslist)
    else:
        writelist(crossloss_dir,now_loss)

def saveloss(loss,complete_dir):#显示并保存图像
    plt.plot(loss)
    plt.savefig(complete_dir)
    plt.close('all')

def save_losstogether(dictloss,filename):#给出多个损失列表的字典，保存共同的折线图到filename
    for name,list in dictloss.items():
        plt.plot(list,label=name)
    plt.legend()
    plt.savefig(filename)
    plt.close('all')

def save_all_loss(dictloss,filename):
    if (os.path.isdir(filename) == False):
        os.makedirs(filename, exist_ok=True)  # 创建输出文件夹
    crossdict={}
    save_losstogether(dictloss,os.path.join(filename,'this_allloss.png'))
    for name,list in dictloss.items():
        dir=filename + '/' + name+'.png'
        saveloss(list,dir)#保存本次的loss图
        crossloss_png_dir = filename + '/' + name+'_crossloss.png'  #总损失的图表名
        crossloss_txt_dir = filename + '/' +name+ '_crosslosslist.txt'#总损失的txt文件名
        save_crossloss(crossloss_txt_dir,list)
        newlist=readlist(crossloss_txt_dir)
        crossdict[name]=newlist
    save_losstogether(crossdict,os.path.join(filename,'allloss.png'))#保存本次的联合损失图像

#把多个批次图像，对应一行一行地显示，num是每行显示的图像数量，x是行数（成对应的行的数量，不是实际行数），y是生成的大图像数量，args是,batch_x是每个arg占多少行
#输入图像的多个批次
def colshow(x,y,*args,filename=None,num=8,batch_x=1):
    args=list(args)
    for i in range (len(args)):
        if len(args[i].size())==4 and args[i].size()[1]==1:
            args[i]=args[i].repeat(1,3,1,1)
    imgnum=args[0].size()[0]
    lenarg=len(args)
    a = 0
    if imgnum >= x * y * num:
        for g in range(y):
            rel = args[0][a:a + num*batch_x]
            for g1 in args[1:]:
                rel=torch.cat((rel,g1[a:a+num*batch_x]),axis=0)
            a=a+num*batch_x
            for i in range(x-1):
                for g2 in args:
                    rel = torch.cat((rel,g2[a:a + num*batch_x]),axis=0)
                a = a + num*batch_x
            # rel=fake_images
            rel = torchvision.utils.make_grid(rel,num,normalize=False)
            imshow(rel,filename)
    else:
        print(f"当前批量为{imgnum},而你想生成{x}行，每行{num}列个图像")

# 给出一个四维张量图像，把他们逐一保存至文件
def savewholeimg(img,save_dir,name):
    for i in range(img.size()[0]):
        imshow(img[i],save_dir+"/"+name+f"{i}"+".jpg")


def imgcat(list,x,t,*args,num=8):#拼接显示给出列表中的序号的图像
    y=len(list)
    new_args=[]
    for arg in args:
        rel = arg[list[0]].unsqueeze(0)
        for i in range(y-1):
            rel=torch.cat((rel,arg[list[i+1]].unsqueeze(0)))
        new_args.append(rel)
    z=(y//8)
    if z==0:
        z=1
    colshow(x,t,*new_args)
# 0.5 定义函数，实现可视化模型结果：获取一部分测试数据，显示由模型生成的模拟数据。


def writeeva(dir,list1,list2,pcslist,ranklist):#把loss列表写入文件
    with open(dir,'w',encoding='utf-8') as f:  #使用with方法
        for i in range(len(list1)):
            f.write(f"{list1[i]:.2f}|")
            f.write(f"{list2[i]:.2f}|")
            f.write(f"{pcslist[i]:.2f}|")
            f.write(f"{ranklist[i]:d}   ")
            if((i+1)%8==0):
                f.write('\n')

class Visualizer():
    def __init__(self,opt):
        self.opt=opt
        self.o=opt.o
        self.display_epoch_freq=opt.display_epoch_freq
        self.print_preq=opt.print_freq
        self.save_dir=opt.checkpoints_dir
        self.batchsize=opt.batch_size
        self.save_epoch_freq=opt.save_epoch_freq
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    # 保存图像
    def showexpample(self,xuhao,ye,hang,*args,num=8,batch_x=1):
        if self.opt.batch_size<num:
            num=self.opt.batch_size
        if (xuhao% self.display_epoch_freq)<self.batchsize:
            for i in range(ye):
                img_dir=os.path.join(self.save_dir,self.opt.name,"img")
                if (os.path.isdir(img_dir)== False):
                    os.makedirs(img_dir, exist_ok=True)  # 创建输出文件夹
                filename="{}img{}_{}.png".format(self.o,xuhao,i+1)
                path=os.path.join(img_dir,filename)
                colshow(1,hang,*args,filename=path,num=num,batch_x=batch_x)
                print(f"保存测试图像至{path}")

# 只保存图
    def savelossgraph(self,epoch,dictloss):
        if epoch%self.save_epoch_freq==0:
            for name, list in dictloss.items():
                loss_dir=os.path.join(self.save_dir,self.opt.name,"loss")
                if (os.path.isdir(loss_dir) == False):
                    os.makedirs(loss_dir, exist_ok=True)  # 创建输出文件夹
                path = os.path.join(loss_dir,self.o+f"_latest" + name + '.png')
                saveloss(list, path)  # 保存本次的loss图
            print(f"保存loss图至{path}")
            gross_dir = os.path.join(self.save_dir, self.opt.name, "grossloss")
            if (os.path.isdir(gross_dir) == False):
                os.makedirs(gross_dir, exist_ok=True)  # 创建输出文件夹
            save_all_loss(dictloss, gross_dir)

    # losses是损失列表字典
    def print_current_losses(self, epoch,iters, losses, t_comp, t_data,cross_t_comp=None,cross_t_data=None):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point
            t_data (float) -- data loading time per data point
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v[-1])
        print(message)  # print the message
        if not cross_t_data==None:
            print("cross_t_comp:%.3f, cross_t_data:%.3f" % (cross_t_comp,cross_t_data))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

