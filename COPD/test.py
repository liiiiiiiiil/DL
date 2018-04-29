from utils.data_utils import *
import matplotlib.pyplot as plt
import opts
from sklearn.model_selection import train_test_split
from dataloader.copd_dataloader import CopdDataloader,CopdDataset
from PIL import Image
import torch
import numpy as np
opt=opts.parse_opt()


data,all_label=preprocese_data(opt)
# print a.iloc[22]

# train_data,test_data=train_test_split(data,test_size=0.25,stratify=data['Finding Labels'].map(lambda x:x[:4]))
# print [(c_label,int(data[c_label].sum())) for c_label in all_label]

train_data,test_data=split_data(data)
vecs=test_data['disease_vec'].values
for i in range(800):
   vec=vecs[i] 
   if np.sum(vec)==0:
       print i

exit()
dataloader=CopdDataloader(opt,test_data)


# dataset=CopdDataset(root_dir=opt.root_dir,data_df=data)
# for i in range(len(dataset)):
    # sample=dataset[i]
    # if sample['image'].shape==(1024,1024,4):
        # print sample['image']
        # print sample['label']
l=dataloader.get_test_loader()
for a,b in enumerate(l):
    
    image_batch=b['image']
    label_batch=b['label']
    # print label_batch.shape
    result=label_batch.sum(dim=1)
    print result


    exit()
    if a==5:
        break
    print label_batch[0]
    # print image_batch.shape

    # print image_batch.dtype


# image_batch=image_batch[:,:,:,0]
# image=image_batch[,:,:]

np.save('./lihao.npy',image_batch)





# exit()
# label_counts = data['Finding Labels'].value_counts()[0:15]
# fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
# ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
# ax1.set_xticks(np.arange(len(label_counts))+0.5)
# _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
# plt.show()
