import argparse

def parse_opt():
    parser=argparse.ArgumentParser()

    parser.add_argument('--root_dir',type=str,default='/home/lihao/data/DL/copd_data/images')
    parser.add_argument('--input_csv_path',type=str,default='~/data/DL/copd_data/Data_Entry_2017.csv')
    parser.add_argument('--max_classes',type=int,default=15)
    parser.add_argument('--min_cases',type=int,default=1000)
    parser.add_argument('--num_sample',type=int,default=500)
    parser.add_argument('--test_size_rate',type=float,default=0.25)
    parser.add_argument('--batch_size',type=int,default=33)
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--num_workers',type=int,default=0)



    args=parser.parse_args()
    return args
