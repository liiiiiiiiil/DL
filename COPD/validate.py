import torch
import time
from utils.run_utils import *
# import utils.run_utils

def validate(opt,val_loader,model,criterion):

    batch_time=AverageMeter()
    losses=AverageMeter()
    top1=AverageMeter()
    model.eval()

    end=time.time()

    for i,sample in enumerate(val_loader):
        input=sample['image']
        target=sample['label']
        target=target.cuda(async=True)
        input_var=torch.autograd.Variable(input,volatile=True).cuda()
        target_var=torch.autograd.Variable(target,volatile=True)

        if opt.half:
            input_var=input_var.half()

        output=model(input_var)
        loss=criterion(output,target_var)
        losses.update(loss.data[0],input.size(0))
        
        output=output.float()
        result=accuracy(output.data,target)        
        top1.update(result,input.size(0))


        batch_time.update(time.time()-end)
        end=time.time()
        
        if i %opt.print_freq==0:
            print('Test:[{0}/{1}]\t'
                  'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy:{top1.val:.3f}({top1.avg:.3f})'.format(i,len(val_loader),
                  batch_time=batch_time,loss=losses,top1=top1))

    return result 

