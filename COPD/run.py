import os
import opts
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from model import VGG
from dataloader import copd_dataloader
from utils import data_utils
from utils import run_utils
from  train import *
from validate import *




def run(opt):
    model=VGG.vgg16_bn()
    model.features=torch.nn.DataParallel(model.features)
    model.cuda()

    data_df,_=data_utils.preprocese_data(opt)
    train_df,test_df=data_utils.split_data(data_df)

    train_loader=copd_dataloader.CopdDataloader(opt,train_df).get_train_loader()
    val_loader=copd_dataloader.CopdDataloader(opt,test_df).get_test_loader()

    # print len(test_loader)
    # for i,b in enumerate(test_loader):
    #     image=b['image']
    #     label=b['label']
    #     print image.shape
    #     print label.shape
    # exit()


    best_result=0

    if opt.model_path:
        if os.path.isfile(opt.model_path):
            print "=>loading checkpoint '{}'".format(opt.model_path)
            checkpoint=torch.load(opt.model_path)
            opt.start_epoch=checkpoint['epoch']
            # best_prec1=checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=>loaded checkpoint(epoch{})"
                    .format(checkpoint['epoch']))
            losslist=checkpoint['losslist']
            # best_result=checkpoint['best_result']
        else:
            print "=>no checkpoint found at '{}'".format(opt.model_path)

    cudnn.benchmark=True 

    criterion=nn.BCELoss().cuda()

    if opt.half:
        model.half()
        criterion.half()
    learning_rate=opt.learning_rate
    # optimizer=torch.optim.SGD(model.parameters(),learning_rate,momentum=opt.momentum,weight_decay=opt.weight_decay)
    optimizer=torch.optim.Adam(model.parameters(),learning_rate,weight_decay=opt.weight_decay)

    if opt.evaluate:
         validate(opt,val_loader,model,criterion)
         return

    #loss
    losslist=[]


    for epoch in range(opt.start_epoch,opt.epochs):
        run_utils.adjust_learning_rate(optimizer,epoch,learning_rate)

        train(opt,train_loader,model,criterion,optimizer,epoch,losslist)
        result=validate(opt,val_loader,model,criterion)
        print ('the binary accurracy is:{0}'.format(result))  
        is_best=result>best_result
        best_result=max(result,best_result)
        run_utils.save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'losslist':losslist,
            'best_result':best_result,
        },filename=os.path.join(opt.save_dir,'checkpoint_{}.tar'.format(epoch)))


opt=opts.parse_opt()
run(opt)
