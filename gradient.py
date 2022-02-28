import numpy as np


def gradient_descent(max_iterations, threshold, w_init,
                     obj_func, grad_func, extra_param=None,
                     learning_rate=0.05, momentum=0.8):
    if extra_param is None:
        extra_param = []

    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10

    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate*grad_func(w, extra_param) + momentum*delta_w
        w += delta_w

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


if __name__ == '__main__':
    pass

