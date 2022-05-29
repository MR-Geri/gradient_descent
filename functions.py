import numpy as np


def func(w, extra=None):
    return np.sum(w * w)


def grad(w, extra=None):
    return 2 * w


def hesse(w, extra=None):
    return np.array(((2, 0), (0, 2))).transpose()


def grad_mse(w, xy):
    x, y = xy
    rows, cols = x.shape

    o = np.sum(x*w, axis=1)
    diff = y - o
    diff = np.tile(diff.reshape((rows, 1)), (1, cols))
    grad = -np.sum(diff*x, axis=0)
    return grad


def mse(w, xy):
    x, y = xy
    o = np.sum(x*w, axis=1)
    mse = np.sum((y-o)**2)
    return mse/2


def hesse_mse(w, xy):
    x, y = xy
    o = np.sum(x*w, axis=1)
    ab = 2 * np.sum(x) / len(x)
    return np.array((
        (2, ab),
        (ab, 2 * np.sum(x**2) / len(x))
    )).transpose()

