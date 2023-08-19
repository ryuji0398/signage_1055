"""
データ読み込み、データ修正

X_train.keys():
['region', 'year', 'manufacturer', 'condition', 'cylinders',
       'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size',
       'type', 'paint_color', 'state']
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_PATH = '../../data/'

def mk_data(
        use_col = ['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']
        ):
    # データの読み込み
    # CSVファイルを読み込んでDataFrameに格納
    train_file = 'train.csv'
    test_file = 'test.csv'
    train_df = pd.read_csv(os.path.join(DATA_PATH, train_file))
    test_df = pd.read_csv(os.path.join(DATA_PATH, test_file))

    train_df = repair_data(train_df)
    # test_df = repair_data(test_df)
    
    # 特徴量とターゲットに分割
    X = train_df.drop(['id', 'price'], axis=1)
    y = train_df['price']
    X = data_drop(X, use_col)

    # # 特徴量とターゲットに分割
    # X_train = train_df.drop(['id', 'price'], axis=1)
    # y_train = train_df['price']
    # X_train = data_drop(X_train, use_col)

    # # 特徴量とターゲットに分割
    # X_test = test_df.drop(['id', 'price'], axis=1)
    # y_test = test_df['price']
    # X_test = data_drop(X_test, use_col)

    # データの前処理と分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # return X_train, X_test, y_train, y_test
    return X, X_test, y, y_test

def mk_data_eval(
        use_col = ['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']
        ):
    # データの読み込み
    # CSVファイルを読み込んでDataFrameに格納
    test_file = 'test.csv'
    test_df = pd.read_csv(os.path.join(DATA_PATH, test_file))
    test_df = repair_data(test_df)

    # 特徴量とターゲットに分割
    X_test = test_df.drop(['id'], axis=1)
    X_id = test_df['id']
    X_test = data_drop(X_test, use_col)

    return X_test, X_id

def mk_data_tmp(use_col):
    # データの読み込み
    # CSVファイルを読み込んでDataFrameに格納
    train_file = 'train.csv'
    test_file = 'test.csv'
    train_df = pd.read_csv(os.path.join(DATA_PATH, train_file))
    test_df = pd.read_csv(os.path.join(DATA_PATH, test_file))

    # 文字列のデータに対して、lightGBMで使える形に修正
    from sklearn import preprocessing
    for column in train_df.columns:
        target_column = train_df[column]
        if target_column.dtype == object:
            le = preprocessing.LabelEncoder()
            le.fit(target_column)
            label_encoded_column = le.transform(target_column)
            train_df[column] = pd.Series(label_encoded_column).astype('category')
    

    # 特徴量とターゲットに分割
    X = train_df.drop(['id', 'price'], axis=1)
    y = train_df['price']
    X = data_drop(X, use_col)

    # データの前処理と分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def data_drop(X, use_col):
    drop_col = [*set(X.keys()) -set(use_col),]
    X_ = X.drop(drop_col, axis=1)

    return X_

def repair_data(df):
    # 文字列のデータに対して、lightGBMで使える形に修正
    from sklearn import preprocessing
    for column in df.columns:
        target_column = df[column]
        if target_column.dtype == object:
            le = preprocessing.LabelEncoder()
            le.fit(target_column)
            label_encoded_column = le.transform(target_column)
            df[column] = pd.Series(label_encoded_column).astype('category')
    return df

if __name__=='__main__':
    X_train, X_test, y_train, y_test = mk_data()
