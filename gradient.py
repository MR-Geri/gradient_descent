import numpy as np


def gradient_descent(max_iterations, threshold, w,
                     obj_func, grad_func, 
                     learning_rate, momentum,
                     params=None):
    if params is None:
        params = []

    w_history = w
    f_history = obj_func(w, params)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1e10

    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate * grad_func(w, params) + momentum * delta_w
        w += delta_w

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, params)))

        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


if __name__ == '__main__':
    pass

