import torch
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_checkpoint(state,filename='checkpoint.pth.tar'):
    """save the checkpoint in file path"""
    torch.save(state,filename)

def adjust_learning_rate(optimizer,epoch,learning_rate):
    """"""
    learning_rate=learning_rate*(0.5**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['learning_rate']=learning_rate

def accuracy(output,target):
    
    batch_size=output.size(0)
    pred=output.round()

    accuracy=torch.mean(torch.eq(target,pred).type(torch.FloatTensor))
    return accuracy

def top_accuracy(output, target, topk=(4,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)#pred:tensor,batch_size*k
    
    target_args=_get_arg_of_target(target)#label_args:[tensor,tensor...],batch_size
    top1_acy=_topk_accuracy(target_args,pred,1)
    top2_acy=_topk_accuracy(target_args,pred,2)
    top3_acy=_topk_accuracy(target_args,pred,3)
    top4_acy=_topk_accuracy(target_args,pred,4)
    result={}
    result['top1_acy']=top1_acy
    result['top2_acy']=top2_acy
    result['top3_acy']=top3_acy
    result['top4_acy']=top4_acy
    return result


def _get_arg_of_target(target):

    batch_size=target.size(0)
    target_args=[]
    labelnum=torch.sum(target,dim=1)
    for i in range(batch_size):
        # print int(labelnum[i])
        if labelnum[i]==0:
            target_args.append(torch.Tensor([1,1,1,1,1,1,1,1,1,1,1,1,1]))
            continue
        _,pred=torch.topk(target[i],int(labelnum[i]))
        target_args.append(pred)

    return target_args

def _topk_accuracy(target_args,output,k):
    
    batch_size=output.size(0)
    result=[]
    x=0
    output=output[:,:k]
    for i in range(batch_size):
       t_set=_tensor2set(target_args[i]) 
       # print t_set
       o_set=_tensor2set(output[i])
       # print o_set
       if len(t_set)<=len(o_set):
           if t_set&o_set==t_set:
               x=1
           else:
               x=0
       else:
           if o_set&t_set==o_set:
               x=1
           else:
               x=0
       # print x
       # print "loop"
       result.append(x)

    # print result
    result=(np.sum(result)+0.0)/len(result)
    return result

def _tensor2set(tensor):
    # batch_size=tensor.size(0)
    lenth=tensor.size(0)
    result=set()
    for i in range(lenth):
        result.add(tensor[i])
    
    return result

    

