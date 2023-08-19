"""
main コード

"""

import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import mk_data
from predicting import eval_model

def main():
    # use_col = ['region', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'state']
    use_col = ['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']
    X_train, X_test, y_train, y_test = mk_data(use_col)

    # LightGBM用のデータセットを作成
    train_data = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=train_data)

    # ハイパーパラメータの設定
    # params = {
    #     'objective': 'regression',  # 回帰問題の場合
    #     'metric': 'rmse',  # 平均二乗誤差（Root Mean Squared Error）を使用
    # }
    params = {
        'objective': 'mean_absolute_percentage_error', 
        'metric': 'mape',
    }

    # モデルの学習
    num_rounds = 200  # 学習のラウンド数（エポック数）
    model = lgb.train(params, train_data, num_rounds, valid_sets=lgb_eval,)


    # テストデータを使って予測を行う
    y_pred = model.predict(X_test)

    # # 平均二乗誤差（RMSE）を計算して評価
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print("RMSE:", rmse)

    # MAPEを計算して評価
    mape = np.sqrt(mean_absolute_percentage_error(y_test, y_pred))
    print("MAPE:", mape)

    eval_model(model)

def main_1():
    # use_col = ['region', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'state']
    use_col = ['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']
    X_train, X_test, y_train, y_test = mk_data(use_col)

    # LightGBM用のデータセットを作成
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        weight=compute_sample_weight(class_weight='balanced', y=y_train).astype('float32'))

    # ハイパーパラメータの設定
    params = {
        'objective': 'mean_absolute_percentage_error', 
        'metric': 'mape',
    }

    # モデルの学習
    num_rounds = 100  # 学習のラウンド数（エポック数）
    model = lgb.train(params, train_data, num_rounds)

    # # テストデータを使って予測を行う
    # y_pred = model.predict(X_test)
    # # MAPEを計算して評価
    # mape = np.sqrt(mean_absolute_percentage_error(y_test, y_pred))
    # print("MAPE:", mape)

    eval_model(model)



# # Datasetへ変換時に引数weightにcompute_sample_weightの結果を渡す。
# trn_data = lgb.Dataset(X_train, label=y_train.Target, weight=compute_sample_weight(class_weight='balanced', y=y_train.Target).astype('float32'))
# # 検証データには全て１になっているデータを渡す。
# val_data = lgb.Dataset(X_test, label=y_test.Target, weight=np.ones(len(X_test)).astype('float32'))

# clf = lgb.train(
#     model_params, trn_data, **fit_params,
#     valid_sets=[trn_data, val_data],
#     fobj=fobj, feval=feval

if __name__=='__main__':
    main()
    # main_1()