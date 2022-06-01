import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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


def show_main_graph(ax, w, momentum, learning_rate, params):
    ax.clear()
    _, y_history = gradient_descent(
        max_iterations=100,
        threshold=1e-4,
        w=w.copy(),
        obj_func=mse,
        grad_func=grad_mse,
        learning_rate=learning_rate,
        momentum=momentum,
        params=params
    )
    ax.plot(np.arange(y_history.size), y_history, color='green', 
            label='Градиент')

    _, y_history = newton_func(
        max_iterations=100,
        threshold=1e-4,
        w=w.copy(),
        obj_func=mse, 
        grad_func=grad_mse,
        hesse_func=hesse_mse,
        learning_rate=learning_rate,
        params=params
    )
    ax.plot(np.arange(y_history.size), y_history, color='red', 
             label='Ньютон')
    ax.legend()



def main(flag_show_data: bool = False):
    def update_slider(tmp_value) -> None:
        for i in range(3):
            ax[i].clear()

        show_main_graph(ax[0], w.copy(), 0.7, s1.val, (x_train, y_train))
        show_main_graph(ax[1], w.copy(), 0.7, s2.val, (x_train, y_train))
        show_main_graph(ax[2], w.copy(), 0.7, s3.val, (x_train, y_train))

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

#    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 4))
    fig, ax = plt.subplot_mosaic([
        [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]
    ])

    for ind, momentum in enumerate((0, .7, .9)):
        show_main_graph(ax[ind], w.copy(), momentum, 1e-6, (x_train, y_train))
        if ind == 1:
            ax[ind].set_xlabel('Итерация')
        if ind == 0:
            ax[ind].set_ylabel('Среднеквадратичная ошибка')

        ax[ind].set_title(f'Импульс = {momentum}\n')

    s1 = Slider(ax[3], '[1]', 1e-9, 1e-6, 1e-6)
    s2 = Slider(ax[4], '[2]', 1e-9, 1e-6, 1e-6)
    s3 = Slider(ax[5], '[3]', 1e-9, 1e-6, 1e-6)
    s1.on_changed(update_slider)
    s2.on_changed(update_slider)
    s3.on_changed(update_slider)
    plt.show()


if __name__ == '__main__':
    solve_fw_newton()
#    main(False)

