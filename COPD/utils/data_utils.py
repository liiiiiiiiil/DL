import pandas as pd
import numpy as np
from itertools import chain
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def _sample(opt,data):
    sample_weights=data['Finding Labels'].map(lambda x:len(x.split('|'))if len(x)>0 else 0).values+4e-2
    sample_weights/=sample_weights.sum()
    data=data.sample(opt.num_sample,weights=sample_weights)
    return data



def read_data(opt):
    return pd.read_csv(opt.input_csv_path)



def preprocese_data(opt):
    max_classes=opt.max_classes
    data=read_data(opt)


    data['Finding Labels']=data['Finding Labels'].map(lambda x:x.replace('No Finding',''))
    all_labels=np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels=[x for x in all_labels if len(x)>0]
    for c_label in all_labels:
        if len(c_label)>1:
            data[c_label]=data['Finding Labels'].map(lambda finding:1.0 if c_label in finding else 0)

    all_labels=[c_label for c_label in all_labels if data[c_label].sum()>opt.min_cases]

    data=_sample(opt,data)
    data['disease_vec']=data.apply(lambda x:[x[all_labels].values],1).map(lambda x:x[0])
    return data,all_labels

def split_data(data):
    #####
    train_df,valid_df=train_test_split(data,test_size=0.25,stratify=data['Finding Labels'].map(lambda x:x[:4]))
    return train_df,valid_df


