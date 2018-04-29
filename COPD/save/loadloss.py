import torch
import matplotlib.pyplot as plt 
import numpy as np

checkpoint=torch.load('./checkpoint_9.tar')
losslist=checkpoint['losslist']
# print losslist.shape
lossarray=np.array(losslist)
np.save('./lossarray.npy',lossarray)
# print lossarray.shape
