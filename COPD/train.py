import torch
import time
from utils.run_utils import *

def train(opt,train_loader,model,criterion,optimizer,epoch):

    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses=AverageMeter()
    top1=AverageMeter()

    model.train()
    end=time.time()

    for i,sample in enumerate(train_loader):

        input=sample['image']
        target=sample['label']
        data_time.update(time.time()-end)
        target=target.cuda(async=True)
        input_var=torch.autograd.Variable(input).cuda()
        target_var=torch.autograd.Variable(target)
        # print target

        # exit()
        if opt.half:
            input_var=input_var.half()

        output=model(input_var)
        loss=criterion(output,target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.data[0],input.size(0))
        # top1.update(prec1[0])

        if i % opt.print_freq==0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  .format(epoch,i,len(train_loader),batch_time=batch_time,data_time=data_time,
                      loss=losses))
