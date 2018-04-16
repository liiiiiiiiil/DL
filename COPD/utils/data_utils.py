import pandas as pd
import numpy as np
from itertools import chain
import os
# import sys
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
# sys.path.insert(0,"../COPD/")


def _sample(opt,data):
    sample_weights=data['Finding Labels'].map(lambda x:len(x.split('|'))if len(x)>0 else 0).values+4e-2
    sample_weights/=sample_weights.sum()
    return data.sample(opt.num_sample)


def read_data(opt):
    return pd.read_csv(opt.input_csv_path)



def preprocese_labels(opt):
    max_classes=opt.max_classes
    data=read_data(opt)
    label_counts=data['Finding Labels'].value_counts()[:max_classes]
    data['Finding Labels']=data['Finding Labels'].map(lambda x:x.replace('No Finding',''))
    all_labels=np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels=[x for x in all_labels if len(x)>0]
    #delete empty labels
    for c_label in all_labels:
        if len(c_label)>1:
            data[c_label]=data['Finding Labels'].map(lambda finding:1.0 if c_label in finding else 0)

    all_labels=[c_label for c_label in all_labels if data[c_label].sum()>opt.min_cases]




    return all_labels,data




