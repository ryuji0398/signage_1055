"""
numpyro or pyro
MCMC or 変分ベイズ
確立モデルを推定する
"""

from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, Trace_ELBO, MCMC, NUTS

import numpy as np
import itertools
from preprocessing import mk_data, mk_data_eval
from predicting import write_csv

def model(
        X,
        y=None,
):
    # X_col = X.keys()
    # col_num = len(X_col)

    # samples = dist.MultivariateNormal(loc=jnp.zeros(2), covariance_matrix=jnp.eye(2)).sample(random.PRNGKey(0), (1000,))
    # plt.scatter(samples[:, 0], samples[:, 1])
    # 切片
    intercept = numpyro.sample("intercept", dist.Normal(0., 100.))
    # 重み
    coef = numpyro.sample("coef", dist.Normal(0.0, 100).expand([X.shape[1]]))
    
    # muを計算
    mu = numpyro.deterministic("mu", jnp.dot(X, coef) + intercept)
    
    # ノイズ
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    # 正規分布からのサンプリング.yは観測値なので、obs=yを追加
    with numpyro.plate("N", len(X)):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

if __name__=='__main__':

    use_col = ['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']
    X_train, X_test, y_train, y_test = mk_data(use_col)

    # 乱数の固定に必要
    rng_key= random.PRNGKey(0)

    # NUTSでMCMCを実行する
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4, thinning=1)
    # mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1, thinning=1)
    mcmc.run(
        rng_key=rng_key,
        X=X_train[['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']].values,
        y=y_train.values,
        # X=df[["A", "Score"]].values,
        # y=df["Y"].values,
    )
    mcmc.print_summary()
    # breakpoint()
    # get param (num_samples=2000)
    mcmc_samples = mcmc.get_samples()

    # print("mcmc_samples:", mcmc_samples)
    # X_range = jnp.linspace(0, 50, 50)
    X_pred, X_id = mk_data_eval()

    predictive = Predictive(model, mcmc_samples)
    predict_samples = predictive(
        random.PRNGKey(0), 
        X=X_pred[['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size', 'type']].values, 
        y=None)
    # print(predict_samples)

    y_pred = predict_samples['obs']
    # print(y_pred)
    # breakpoint()
    y_pred = np.mean(y_pred, axis=0)
    write_csv(y_pred=y_pred, X_id=X_id, fol='svi')