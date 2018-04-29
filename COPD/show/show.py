import pandas as pd
import numpy as np
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,'../COPD')

from utils.data_utils import *
import opts






def show_raw_csv_data(opts):
    data=read_data(opts)
    print "the raw csv data:"
    print data.head()




