import numpy as np


def func(w, extra=None):
    return np.sum(w * w)


def grad(w, extra=None):
    return 2 * w


def grad_mse(w, xy):
    x, y = xy
    rows, cols = x.shape

    o = np.sum(x*w, axis=1)
    diff = y-o
    diff = np.tile(diff.reshape((rows, 1)), (1, cols))
    grad = -np.sum(diff*x, axis=0)
    return grad


def mse(w, xy):
    x, y = xy
    o = np.sum(x*w, axis=1)
    mse = np.sum((y-o)**2)
    return mse/2


def error(w,xy):
    x, y = xy
    o = np.sum(x * w, axis=1)
    
    ind_0, ind_1 = np.where(o <= 0.5), np.where(o > 0.5)
    o[ind_1] = 1
    o[ind_0] = 0
    return np.sum((o - y) * (o - y)) / y.size * 100

