from utils.data_utils import *
import matplotlib.pyplot as plt
import opts
from sklearn.model_selection import train_test_split


opt=opts.parse_opt()

data,all_label=preprocese_data(opt)
# print a.iloc[22]
# print [(c_label,int(data[c_label].sum())) for c_label in all_label]
train_df, valid_df = train_test_split(data, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = data['Finding Labels'].map(lambda x: x[:4]))
data=data.iloc[:10]
data=add_image_path(opt,data)
print data.iloc[0]


# exit()
# label_counts = data['Finding Labels'].value_counts()[0:15]
# fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
# ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
# ax1.set_xticks(np.arange(len(label_counts))+0.5)
# _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
# plt.show()
