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
import train




def run(opt):
    model=VGG.vgg16()
    model.features=torch.nn.DataParallel(model.features)
    model.cuda()


    data_df,_=data_utils.preprocese_data(opt)
    train_df,test_df=data_utils.split_data(data_df)

    train_loader=copd_dataloader.CopdDataloader(opt,train_df).get_train_loader()
    test_loader=copd_dataloader.CopdDataloader(opt,test_df).get_test_loader()


    # print len(test_loader)
    # for i,b in enumerate(test_loader):
    #     image=b['image']
    #     label=b['label']
    #     print image.shape
    #     print label.shape
    # exit()

    if opt.model_path:
        if os.path.isfile(opt.model_path):
            print "=>loading checkpoint '{}'".format(opt.model_path)
            checkpoint=torch.load(opt.model_path)
            opt.start_epoch=checkpoint['epoch']
            # best_prec1=checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=>loaded checkpoint(epoch{})"
                    .format(checkpoint['epoch']))
        else:
            print "=>no checkpoint found at '{}'".format(opt.model_path)

    cudnn.benchmark=True

    criterion=nn.BCELoss().cuda()

    if opt.half:
        model.half()
        criterion.half()
    learning_rate=opt.learning_rate
    optimizer=torch.optim.SGD(model.parameters(),learning_rate,momentum=opt.momentum,weight_decay=opt.weight_decay)

    # if opt.evaluate:
    #     validate(val_loader,model,criterion)
    #     return

    for epoch in range(opt.start_epoch,opt.epochs):
        run_utils.adjust_learning_rate(optimizer,epoch,learning_rate)

        train.train(opt,train_loader,model,criterion,optimizer,epoch)
        # prec1=validate(val_loader,model,criterion)
        # is_best=prec1>best_prec1
        # best_prec1=max(prec1,best_prec1)
        save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dcit()
            # 'best_prec1':best_prec1,
        },filename=os.path.join(opt.save_dir,'checkpoint_{}.tar'.format(epoch)))


opt=opts.parse_opt()
run(opt)
