import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

from gradient import gradient_descent
from visual import *
from functions import func, grad, mse, grad_mse, error


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
    learning_rates = [.05, .3, .5, .9]
    momentum = [0, .2, .7]
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


def main(flag_show_data: bool = False):
    digits, target = dt.load_digits(n_class=2, return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        digits, target, test_size=.2, random_state=10
    )

    x_train = np.hstack((np.ones((len(y_train), 1)), x_train))
    x_test = np.hstack((np.ones((len(y_test), 1)), x_test))

    rand = np.random.RandomState(19)
    w = rand.uniform(-1, 1, x_train.shape[1]) * 1e-6

    if flag_show_data:
        fig, ax = plt.subplots(
            nrows=1, ncols=10, figsize=(12, 4), 
            subplot_kw={"xticks": [], "yticks": [] }
        )
        for i in np.arange(10):
            ax[i].imshow(digits[i, :].reshape(8, 8))
        plt.show()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 4))
    for ind, alpha in enumerate((0, .5, .9)):
        w_history, mse_history = gradient_descent(
            max_iterations=100,
            threshold=1e-2,
            w=w,
            obj_func=mse,
            grad_func=grad_mse,
            learning_rate=1e-6,
            momentum=alpha,
            params=(x_train, y_train)
        )

        plt.subplot(131 + ind)
        plt.plot(np.arange(mse_history.size), mse_history, color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Square Error')

        plt.title(f'Momentum = {alpha}\nIter={mse_history.size - 1}')

        train_error = error(w_history[-1], (x_train, y_train))
        test_error = error(w_history[-1], (x_test, y_test))

        print(f"Train Error Rate: {train_error}")
        print(f"Test Error Rate: {test_error}\n")
    plt.show()


if __name__ == '__main__':
    pts, f_vals = visualize_fw()
    solve_fw()
#    main(True)

