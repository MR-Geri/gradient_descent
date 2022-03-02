import numpy as np
import matplotlib.pyplot as plt

from gradient import gradient_descent
from visual import *
from functions import func, grad


def visualize_learning(w_history):
    function_plot(pts, f_vals)

    plt.plot(w_history[:, 0], w_history[:, 1], marker='o', c='magenta')

    annotate_pt('минимум', w_history[-1], (-1, 7), 'green')
    for i, w in enumerate(w_history[:-1]):
        plt.annotate(
            "",
            xy=w, xycoords='data', xytext=w_history[i+1, :], textcoords='data',
            arrowprops={
                "arrowstyle": '<-',
                "connectionstyle": 'angle3'
            })


def solve_fw():
    rand = np.random.RandomState(19)
    w = rand.uniform(-10, 10, 2)
    fig, _ = plt.subplots(nrows=4, ncols=4, figsize=(54, 54))
    learning_rates = [0.05, 0.3, 0.5, 0.9]
    momentum = [0, 0.2, 0.7]
    ind = 1

    for alpha in momentum:
        for col, rate in enumerate(learning_rates):
            plt.subplot(3, 4, ind)
            w_history, _ = gradient_descent(
                max_iterations=10, 
                threshold=-1, 
                w=w.copy(), 
                obj_func=func, 
                grad_func=grad, 
                learning_rate=rate, 
                momentum=alpha
            )

            visualize_learning(w_history)
            ind += 1
            plt.text(-9, 12, f'Скорость = {rate}', fontsize=13)
            if col == 1:
                plt.text(10, 15, f'Импульс = {alpha}', fontsize=20)

    fig.subplots_adjust(hspace=.5, wspace=.3)
    plt.show()


if __name__ == '__main__':
    pts, f_vals = visualize_fw()
    solve_fw()
