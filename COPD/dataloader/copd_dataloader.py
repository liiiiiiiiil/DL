import os
import sys
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize,Resize,RandomCrop,RandomHorizontalFlip,RandomRotation,ToTensor,ToPILImage
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from PIL import Image
sys.path.insert(0,"../")
# from torchvision.nn.trans

# def _process_sample(sample):
#     image=sample['image']
#     label=sample['label']
#     if len(image.shape)==2:
#         image=image[:,:,np.newaxis]
#     label=label.astype(float)
#     return image,label
class PreTransX(object):

    def __call__(self,image):


        if len(image.shape)==3:
            image=image[:,:,0]

        arrays=[image for _ in range(3)]
        image=np.stack(arrays,axis=2)
        # print image.shape
       
        return image

class AfterTransX(object):

    def __call__(self,image):

        image=np.array(image)
        # print image
        image=image[:,:,0]
        image=image[:,:,np.newaxis]
        return image


class RescaleX(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size=output_size

    def __call__(self,sample):
        image,label=sample['image'],sample['label']

        h,w=image.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w=self.output_size*h/w,self.output_size
            else:
                new_h,new_w=self.output_size,self.output_size*w/h
        else:
            new_h,new_w=self.output_size

        new_h,new_w=int(new_h),int(new_w)
        img=transform.resize(image,(new_h,new_w))
        return {'image':img,'label':label}


class ToTensorX(object):

    def __call__(self,sample):
        image,label=sample['image'],sample['label']

        return {'image':torch.from_numpy(image),
                'label':torch.from_numpy(label)}

class RandomCropX(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size=(output_size,output_size)

        else:
            assert len(output_size)==2
            self.output_size=output_size

    def __call__(self,sample):
        image,label=sample['image'],sample['label']
        h,w=image.shape[:2]
        new_h,new_w=self.output_size

        top=np.random.randint(0,h-new_h)
        left=np.random.randint(0,w-new_w)
        image=image[top:top+new_h,left:left+new_w]
        return {'image':image,'label':label}

class RotateX(object):

    def __init__(self,max_degree):
        self.max_degree=max_degree

    def __call__(self,sample):

        image,label=sample['image'],sample['label']
        degree=np.random.randint(-self.max_degree,self.max_degree)
        image=transform.rotate(image,degree)
        return {'image':image,'label':label}

class NormalizeX(object):

    def __call__(self,sample):

        image,label=sample['image'],sample['label']
        image=image/np.std(image)
        return {'image':image,'label':label}

class RemoveCenterX(object):

    def __call__(self,sample):

        image,label=sample['image'],sample['label']
        image=image[:,:,0]
        mean=image.mean(axis=1)
        image=image-mean
        image=image[:,:,np.newaxis]
        return {'image':image,'label':label}

class TransposeX(object):

    def __call__(self,sample):

        image,label=sample['image'],sample['label']
        # image=image[:,:,0]
        # image=image.transpose([2,0,1])
        image=np.transpose(image,(2,0,1))
        return {'image':image,'label':label}

class CopdDataset(Dataset):

    def __init__(self,root_dir,data_df,transform=None):
        self.data_df=data_df
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self,idx):
        img_name=os.path.join(self.root_dir,self.data_df.iloc[idx]['Image Index'])
        image=io.imread(img_name)
        label=self.data_df.iloc[idx]['disease_vec']
        label=label.astype(float)
        label=torch.from_numpy(label)

        if self.transform:
            image=self.transform(image)


        sample={'image':image,'label':label}
        return sample



class CopdDataloader():

    def __init__(self,opt,data_df):
        self.batch_size=opt.batch_size
        self.shuffle=opt.shuffle
        self.num_workers=opt.num_workers
        self.root_dir=opt.root_dir
        self.data_df=data_df
        self.test_batch_size=opt.test_batch_size
        self.transform=transforms.Compose([
            PreTransX(),
            ToPILImage(),
            Resize(opt.rescale_size),
            RandomRotation(opt.max_rotate_degree),
            RandomHorizontalFlip(),
            RandomCrop(opt.cnn_image_size),
            # AfterTransX(),
            ToTensor(),
            Normalize([0],[2])
            ])

    def get_train_loader(self):
        self.dataset=CopdDataset(self.root_dir,self.data_df,self.transform)
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)

    def get_test_loader(self):
        self.dataset=CopdDataset(self.root_dir,self.data_df,self.transform)
        return DataLoader(self.dataset,batch_size=self.test_batch_size,shuffle=self.shuffle,num_workers=self.num_workers)



