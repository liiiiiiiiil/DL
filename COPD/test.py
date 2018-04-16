from utils.data_utils import *
import opts

opt=opts.parse_opt()

all_label,data=preprocese_labels(opt)
# print a.iloc[22]
print [(c_label,int(data[c_label].sum())) for c_label in all_label]
