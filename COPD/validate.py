import torch
import time
from utils.run_utils import *

def validate(opt,val_loader,model,criterion):

    batch_time=AverageMeter()
    losses=AverageMeter()
    model.eval()

    end=time.time()

    for i,sample in enumerate(val_loader):
        target=target.cuda(async=True)
        input_var=torch.autograd.Variable(input,volatile=True).cuda()
        target_var=torch.autograd.Variable(target,volatile=True)

        if opt.half:
            input_var=input_var.half()

        output=model(input_var)
        loss=criterion(output,target_var)

        output=output.float()
        loss=loss.float()
        
