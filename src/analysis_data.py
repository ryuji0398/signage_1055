"""
data analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from mk_data import mk_data

DATA_PATH = '../../data/'

def main():
    # # CSVファイルを読み込んでDataFrameに格納
    # train_file = 'train.csv'
    # test_file = 'test.csv'
    # train_df = pd.read_csv(os.path.join(DATA_PATH, train_file))
    # test_df = pd.read_csv(os.path.join(DATA_PATH, test_file))

    # print(train_df.head())
    # breakpoint()

    # plt.hist(train_df["price"], bins=10)
    # plt.xlabel("Year")
    # plt.ylabel("Frequency")
    # plt.title("Age Distribution")
    # plt.show()
    mk_data()

if __name__=='__main__':
    main()