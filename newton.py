import numpy as np


def is_pos_def(hesse):
    return np.all(np.linalg.eigvals(hesse) > 0)


def newton_func(
        max_iterations, threshold, w,
        obj_func, grad_func, hesse_func,
        learning_rate, params=None
    ):
    if params is None:
        params = tuple()

    w_history = w.copy()
    f_history = obj_func(w, params)
    i, diff = 0, 1e10

    while i < max_iterations and diff > threshold:
        hesse = hesse_func(w, params)
        if is_pos_def(hesse):
            hesse_inverse = np.linalg.inv(hesse)
            w -= learning_rate * np.dot(hesse_inverse, grad_func(w, params))
        else:
            w -= learning_rate * grad_func(w, params)

        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, params)))

        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history
