import argparse

def parse_opt():
    parser=argparse.ArgumentParser()

    parser.add_argument('--input_csv_path',type=str,default='data/Data_Entry_2017.csv')
    parser.add_argument('--max_classes',type=int,default=15)
    parser.add_argument('--min_cases',type=int,default=1000)


    args=parser.parse_args()
    return args
