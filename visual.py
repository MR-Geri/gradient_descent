import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from itertools import chain

from graph import Paraboloid, MSE
from dataclasses import dataclass 


@dataclass
class Data:
    x: np.ndarray
    y: np.ndarray

    def get(self):
        return self.x, self.y


class ViewMSE:
    def __init__(self):
        self.graph = MSE(tuple())
        digits, target = dt.load_digits(n_class=2, return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(
            digits, target, test_size=.2, random_state=10
        )
        x_train = self.alignment(x_train)
        x_test = self.alignment(x_test)
        self.train = Data(x_train, y_train)
        self.test = Data(x_test, y_test)

    def show_numbers(self, data):
        _, ax = plt.subplots(
            nrows=1, ncols=10, figsize=(12, 4),
            subplot_kw={"xticks": [], "yticks": []}
        )
        for i in np.arange(10):
            ax[i].imshow(data[i].reshape(8, 6))

    @staticmethod
    def alignment(data: np.ndarray):
        del_index = list(range(0, 64, 8)) + list(range(7, 64, 8))
        data = np.array([np.delete(i, del_index) for i in data])
        return data

    def show(self):
        cords_copy = self.graph.get_random_cords(
            -1, 1, self.train.x.shape[1]
        ) * 1e-6

#        self.show_numbers(self.train.x)

        _, ax = plt.subplot_mosaic([
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]
        ])

        self.graph.params = self.train.get()

        for ind_grad, rate_grad in enumerate(range(-7, -3)):
            for ind_newton, rate_newton in enumerate(range(-4, 0)):
                ax[ind_grad * 4 + ind_newton].clear()
                cords_grad, f_grad = self.graph.gradient_descent(
                    learning_rate=10 ** rate_grad,
                    momentum=0.7,
                    max_iterations=100,
                    threshold=0.01,
                    cords_copy=cords_copy.copy()
                )
                self.graph.draw_graph_on_ax(
                    ax[ind_grad * 4 + ind_newton],
                    (cords_grad, f_grad),
                    'green',
                    'Градиент'
                )
                cords_newton, f_newton = self.graph.newton(
                    learning_rate=10 ** rate_newton,
                    max_iterations=100,
                    threshold=0.01,
                    cords_copy=cords_copy.copy()
                )
                self.graph.draw_graph_on_ax(
                    ax[ind_grad * 4 + ind_newton],
                    (cords_newton, f_newton),
                    'red',
                    'Ньютон'
                )
                ax[ind_grad * 4 + ind_newton].set_title(
                    f'Град = {rate_grad} Ньют = {rate_newton}')
                ax[ind_grad * 4 + ind_newton].legend()
                ax[ind_grad * 4 + ind_newton].set_xticklabels([])
                print(
                    f'grad[{len(f_grad)}]='
                    f'{self.graph.function(cords_grad[-1], self.test.get())}'
                )
                print(
                    f'newton[{len(f_newton)}]='
                    f'{self.graph.function(cords_newton[-1], self.test.get())}'
                )

        plt.show()


class ViewParaboloid:
    def __init__(self):
        self.graph = Paraboloid()

    def show(self):
        cords = self.graph.get_random_cords(-10, 10, 2)
        fig, _ = plt.subplots(nrows=4, ncols=4, figsize=(54, 54))
        learning_rates = [.05, .3, .7, .9]
        ind = 1

        for rate in learning_rates:
            plt.subplot(2, 4, ind)
            self.graph.draw_graph_on_board(self.graph.gradient_descent(
                learning_rate=rate,
                momentum=.5,
                max_iterations=100,
                threshold=1e-2,
                cords_copy=cords.copy()
            )[0])
            plt.subplot(2, 4, ind+1)
            self.graph.draw_graph_on_board(self.graph.newton(
                learning_rate=rate,
                max_iterations=100,
                threshold=1e-2,
                cords_copy=cords.copy()
            )[0])
            ind += 2
            plt.text(-39, 12, f'Градиент', fontsize=13)
            plt.text(-3, 12, f'Ньютон', fontsize=13)
            plt.text(-25, 15, f'Скорость = {rate}', fontsize=13)

        fig.subplots_adjust(hspace=.5, wspace=.3)
        plt.show()

    def show_grad(self):
        cords = self.graph.get_random_cords(-10, 10, 2)
        fig, _ = plt.subplots(nrows=4, ncols=3, figsize=(54, 54))
        ind = 1

        for rate in (.05, .3, .7, .9):
            for momentum in (0, .2, .7):
                plt.subplot(4, 3, ind)
                self.graph.draw_graph_on_board(self.graph.gradient_descent(
                    learning_rate=rate,
                    momentum=momentum,
                    max_iterations=10,
                    threshold=0.1,
                    cords_copy=cords.copy()
                )[0])
                if rate == .05:
                    plt.text(-8, 20, f'Импульс = {momentum}', fontsize=15)
                plt.text(-8, 12, f'Скорость = {rate}', fontsize=13)
                ind += 1

        fig.subplots_adjust(hspace=.5, wspace=.3)
        plt.show()

