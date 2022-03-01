import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

from gradient import gradient_descent
from visual import *


def f(w, extra=[]):
    return np.sum(w * w)


def grad(w, extra=[]):
    return 2 * w


def visualize_learning(w_history):
    # Make the function plot
    function_plot(pts, f_vals)

    # Plot the history
    plt.plot(w_history[:, 0], w_history[:, 1], marker='o', c='magenta')

    # Annotate the point found at last iteration
    annotate_pt('minimum found',
                (w_history[-1, 0], w_history[-1, 1]),
                (-1, 7), 'green')
    iter = w_history.shape[0]
    for w, i in zip(w_history, range(iter-1)):
        # Annotate with arrows to show history
        plt.annotate("",
                     xy=w, xycoords='data',
                     xytext=w_history[i+1, :], textcoords='data',
                     arrowprops=dict(arrowstyle='<-',
                                     connectionstyle='angle3'))


def solve_fw():
    rand = np.random.RandomState(19)
    w_init = rand.uniform(-10, 10, 2)
    fig, _ = plt.subplots(nrows=4, ncols=4, figsize=(18 * 3, 12 * 3))
    learning_rates = [0.05, 0.2, 0.5, 0.8]
    momentum = [0, 0.5, 0.9]
    ind = 1

    # Iteration through all possible parameter combinations
    for alpha in momentum:
        for eta, col in zip(learning_rates, [0, 1, 2, 3]):
            plt.subplot(3, 4, ind)
            w_history, _ = gradient_descent(
                5, -1, w_init.copy(), f, grad, [], eta, alpha)

            visualize_learning(w_history)
            ind += 1
            plt.text(-9, 12, 'Learning Rate = ' + str(eta), fontsize=13)
            if col == 1:
                plt.text(10, 15, 'momentum = ' + str(alpha), fontsize=20)

    fig.subplots_adjust(hspace=0.5, wspace=.3)
    plt.show()


if __name__ == '__main__':
    pts, f_vals = visualize_fw()
    solve_fw()
