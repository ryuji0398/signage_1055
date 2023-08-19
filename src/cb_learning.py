from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor, CBLocPoissonRegressor
from cyclic_boosting import flags

from preprocessing import mk_data, mk_data_eval
from predicting import write_csv

def main():
    X_train, X_test_, y_train, y_test_ = mk_data()
    CB_est = pipeline_CBPoissonRegressor()
    # CB_est = CBLocPoissonRegressor()
    CB_est.fit(X_train, y_train)
    # breakpoint()
    # yhat = CB_est.predict(X_test)

    X_test, X_id = mk_data_eval()
    y_pred = CB_est.predict(X_test)

    write_csv(y_pred=y_pred, X_id=X_id, fol='cb')

def main_1():
    X_train, X_test_, y_train, y_test_ = mk_data()

    fp = {}
    fp['year'] = flags.IS_CONTINUOUS
    fp['manufacturer'] = flags.IS_UNORDERED
    fp['condition'] = flags.IS_UNORDERED
    fp['cylinders'] = flags.IS_UNORDERED
    fp['odometer'] = flags.IS_CONTINUOUS
    fp['title_status'] = flags.IS_UNORDERED
    fp['transmission'] = flags.IS_UNORDERED
    fp['drive'] = flags.IS_UNORDERED
    fp['size'] = flags.IS_UNORDERED
    fp['type'] = flags.IS_UNORDERED
    fp['fuel'] = flags.IS_UNORDERED
    # fp['feature1'] = flags.IS_UNORDERED
    # fp['feature1'] = flags.IS_CONTINUOUS #| flags.HAS_MISSING | flags.MISSING_NOT_LEARNED IS_LINEAR
    # ['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']

    CB_est = pipeline_CBPoissonRegressor(feature_properties=fp)

    CB_est.fit(X_train, y_train)

    X_test, X_id = mk_data_eval()
    y_pred = CB_est.predict(X_test)

    write_csv(y_pred=y_pred, X_id=X_id, fol='cb')

if __name__=='__main__':
    # main()
    main_1()