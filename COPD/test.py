from utils.data_utils import *
import matplotlib.pyplot as plt
import opts
from sklearn.model_selection import train_test_split
from dataloader.copd_dataloader import CopdDataloader,CopdDataset


opt=opts.parse_opt()


data,all_label=preprocese_data(opt)
# print a.iloc[22]



# print [(c_label,int(data[c_label].sum())) for c_label in all_label]
dataloader=CopdDataloader(opt,data)

# dataset=CopdDataset(root_dir=opt.root_dir,data_df=data)
# for i in range(len(dataset)):
    # sample=dataset[i]
    # if sample['image'].shape==(1024,1024,4):
        # print sample['image']
        # print sample['label']
l=dataloader.get_loader()

for a,b in enumerate(l):
    print (a,b['image'].size())






# exit()
# label_counts = data['Finding Labels'].value_counts()[0:15]
# fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
# ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
# ax1.set_xticks(np.arange(len(label_counts))+0.5)
# _ = ax1.set_xticklabels(label_counts.index, rotation = 90)
# plt.show()
