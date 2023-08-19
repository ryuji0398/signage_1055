"""
評価用、提出ファイル作成
"""
import os
import csv
import datetime

from preprocessing import mk_data_eval

OUTPUT_PATH = '../../output/'

def eval_model(model):
    X_test, X_id = mk_data_eval()

    y_pred = model.predict(X_test)

    write_csv(y_pred, X_id, fol='lightgbm')
    # breakpoint()
    # print(y_pred)

def write_csv(y_pred, X_id, fol='test'):
    now = datetime.datetime.now()
    filename = 'sample_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
    path = os.path.join(OUTPUT_PATH, fol)
    with open(os.path.join(path, filename), mode='w') as f:
        writer = csv.writer(f)
        for id, price in zip(X_id, y_pred):
            writer.writerow([id, price])