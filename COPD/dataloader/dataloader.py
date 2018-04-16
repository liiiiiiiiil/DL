import os
import sys
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils

sys.path.insert(0,"../")
from utils.data_utils import *

class CopdDataset(Dataset):

    def __init__(self,root_dir,data_df,transform=None):
        self.data_df=data_df
        self.root_dir=root_dir
        self.transform=transform 

    def __len__(self):
        return len(self.data_df)

    def __getitem_(self,idx):
        img_name=os.path.join(self.root_dir,self.data_df.iloc[idx]['Image Index'])
        image=io.imread(img_name)
        label=self.data_df.iloc[idx]['disease_vec'].as_matrix()
        sample={'image':image,'label':label}

        if self.transform:
            sample=self.transform(sample)

        return sample

class CopdDataloader():

    def __init__(self,opt,data_df,transform=None):
        self.batch_size=opt.batch_size
        self.shuffle=opt.shuffle
        self.num_workers=opt.num_workers
        self.root_dir=opt.root_dir
        self.data_df=data_df
        self.transform=transform
    
    def get_loader(self):
        dataset=CopdDataset(self.root_dir,self.data_df,self.transform)
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)
        



