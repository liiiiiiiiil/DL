import argparse

def parse_opt():
    parser=argparse.ArgumentParser()

    parser.add_argument('--root_dir',type=str,default='/home/lihao/data/DL/copd_data/images')
    parser.add_argument('--input_csv_path',type=str,default='~/data/DL/copd_data/Data_Entry_2017.csv')
    parser.add_argument('--max_classes',type=int,default=15)
    parser.add_argument('--min_cases',type=int,default=1000)
    parser.add_argument('--num_sample',type=int,default=40000)
    parser.add_argument('--test_size_rate',type=float,default=0.25)
    parser.add_argument('--batch_size',type=int,default=200)
    parser.add_argument('--test_batch_size',type=int,default=1000)
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--max_rotate_degree',type=int,default=10)
    parser.add_argument('--rescale_size',type=int,default=280)
    parser.add_argument('--cnn_image_size',type=int,default=256)
    parser.add_argument('--model_path',type=str,default=None)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--half',type=bool,default=False)
    parser.add_argument('--learning_rate',type=float,default=0.005)
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--save_dir',type=str,default='save/')
    # parser.add_



    args=parser.parse_args()
    return args
