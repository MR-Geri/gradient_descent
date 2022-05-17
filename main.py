import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

from gradient import gradient_descent
from visual import *
from functions import func, grad, mse, grad_mse, hesse, hesse_mse
from newton import newton_func


def draw_one_graph(w_history, pts, f_vals):
    draw_plot(pts, f_vals)

    plt.plot(w_history[:, 0], w_history[:, 1], marker='o', c='magenta')

    draw_arrowprops('минимум', w_history[-1], (-1, 7), 'green')
    for i, w in enumerate(w_history[:-1]):
        plt.annotate(
            "",
            xy=w, xycoords='data', xytext=w_history[i+1, :], textcoords='data',
            arrowprops={
                "arrowstyle": '<-',
                "connectionstyle": 'angle3'
            })


def solve_fw_newton():
    rand = np.random.RandomState(19)
    w = rand.uniform(-10, 10, 2)
    fig, _ = plt.subplots(nrows=4, ncols=4, figsize=(54, 54))
    learning_rates = [.05, .3, .7, .9]
    ind = 1
    pts, f_vals = init_graph()

    for rate in learning_rates:
        plt.subplot(2, 4, ind)
        w_history, _ = gradient_descent(
            max_iterations=10, 
            threshold=1e-2, 
            w=w.copy(), 
            obj_func=func, 
            grad_func=grad, 
            learning_rate=rate, 
            momentum=.5
        )
        draw_one_graph(w_history, pts, f_vals)
        plt.subplot(2, 4, ind+1)
        w_history, _ = newton_func(
            max_iterations=10,
            threshold=1e-2,
            w=w.copy(),
            obj_func=func, 
            grad_func=grad,
            hesse_func=hesse,
            learning_rate=rate 
        )
        draw_one_graph(w_history, pts, f_vals)
        ind += 2
        plt.text(-39, 12, f'Градиент', fontsize=13)
        plt.text(-3, 12, f'Ньютон', fontsize=13)
        plt.text(-25, 15, f'Скорость = {rate}', fontsize=13)

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
            ax[i].imshow(digits[i].reshape(8, 8))
        plt.show()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 4))
    for ind, alpha in enumerate((0, .7, .9)):
        w_history, y_history = gradient_descent(
            max_iterations=100,
            threshold=1e-4,
            w=w.copy(),
            obj_func=mse,
            grad_func=grad_mse,
            learning_rate=1e-6,
            momentum=alpha,
            params=(x_train, y_train)
        )
        plt.subplot(131 + ind)
        plt.plot(np.arange(y_history.size), y_history, color='green', 
                 label='Градиент')

        w_history, y_history = newton_func(
            max_iterations=100,
            threshold=1e-4,
            w=w.copy(),
            obj_func=mse, 
            grad_func=grad_mse,
            hesse_func=hesse_mse,
            learning_rate=1e-6,
            params=(x_train, y_train)
        )
        plt.plot(np.arange(y_history.size), y_history, color='red', 
                 label='Ньютон')
        plt.legend()
        if ind == 1:
            plt.xlabel('Итерация')
        if ind == 0:
            plt.ylabel('Среднеквадратичная ошибка')

        plt.title(f'Импульс = {alpha}\n')
    plt.show()


if __name__ == '__main__':
#    solve_fw_newton()
    main(True)

