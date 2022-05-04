from benchopt.datasets.simulated import make_correlated_data
from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm.solvers import cd_solver_path
import numpy as np
import time

from celer import celer_path

X, y, _ = make_correlated_data(n_samples=500, n_features=2000)

datafit = Quadratic()
penalty = L1(1)


n_iter = 10
lambda_max = abs(X.T.dot(y)).max()
lambda_min_ratio = 1e-2

lambdas = np.logspace(
    np.log(lambda_max),
    np.log(lambda_max * lambda_min_ratio),
    num=100,
    base=np.exp(1),
)

cd_solver_path(X, y, datafit, penalty, alphas=lambdas[-2:] / len(y))
t0 = time.time()
res_ = cd_solver_path(
    X,
    y,
    datafit,
    penalty,
    alphas=lambdas / len(y),
    tol=1e-6,
    max_iter=n_iter,
    max_epochs=100_000,
    verbose=1
)
t_skglm = time.time() - t0

t0 = time.time()
res_celer = celer_path(
    X,
    y,
    pb="lasso",
    alphas=lambdas / len(y),
    prune=1,
    tol=1e-10,
    max_iter=1_000,
    max_epochs=100_000,
    verbose=1,
)
t_celer = time.time() - t0
