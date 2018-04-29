import torch
import time
from utils.run_utils import *
import torch.nn as nn

def train(opt,train_loader,model,criterion,optimizer,epoch,losslist):

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

        if opt.half:
            input_var=input_var.half()

        output=model(input_var)

        loss=criterion(output,target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output=output.float()
        result=accuracy(output.data,target)
         

        losses.update(loss.data[0],input.size(0))
        losslist.append(losses.val)

        top1.update(result,input.size(0))
        batch_time.update(time.time()-end)
        end=time.time()

        if i % opt.print_freq==0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  'Accuracy{top1.val:.3f}({top1.avg:.3f})'
                  .format(epoch,i,len(train_loader),batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,top1=top1))

