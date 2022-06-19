import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
from itertools import chain

from graph import Paraboloid, MSE, init_graph
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
        for i in np.arange(-10, 0):
            ax[i].imshow(data[i].reshape(8, 8))

    @staticmethod
    def alignment(data: np.ndarray):
#        first = np.sum(data[:,::8], axis=1)
#        last = np.sum(data[:,7::8], axis=1)
#        print('first', first == 0)
#        print('last', last == 0)
        del_index = list(range(0, 64, 8)) + list(range(7, 64, 8))
        data = np.array([np.delete(i, del_index) for i in data])
        return data

    def show(self):
        cords_copy = self.graph.get_random_cords(
            -1, 1, self.train.x.shape[1]
        ) * 1e-6

#        self.show_numbers(self.train.x)

        _, ax = plt.subplot_mosaic([
            [0, 1, 2]
        ])
#
        self.graph.params = self.train.get()

#        cords_newton, errors_newton = self.graph.newton(
#            learning_rate=1e-6,
#            max_iterations=100,
#            threshold=0.005,
#            cords_copy=cords_copy.copy()
#        )
#        print(f'len_newton={len(errors_newton)} {errors_newton}')
#        cords_grad, errors_grad = self.graph.gradient_descent(
#            learning_rate=1e-6,
#            max_iterations=100,
#            momentum=0.7,
#            threshold=0.005,
#            cords_copy=cords_copy.copy()
#        )
#        print(f'len_grad={len(errors_grad)} {errors_grad}')
#        self.graph.params = (x_test, y_test)
#        print(f'{self.graph.function(cords_grad[-1])}')

        for ind, rate in enumerate((1e-1, 1e-2, 1e-3)):
            ax[ind].clear()
            cords_grad, f_grad = self.graph.gradient_descent(
                learning_rate=1e-6,
                momentum=0.7,
                max_iterations=100,
                threshold=0.005,
                cords_copy=cords_copy.copy()
            )
            self.graph.draw_graph_on_ax(
                ax[ind],
                (cords_grad, f_grad),
                'green',
                'Градиент'
            )
            cords_newton, f_newton = self.graph.newton(
                learning_rate=rate,
                max_iterations=100,
                threshold=0.005,
                cords_copy=cords_copy.copy()
            )
            self.graph.draw_graph_on_ax(
                ax[ind],
                (cords_newton, f_newton),
                'red',
                'Ньютон'
            )
            ax[ind].set_title(f'Скорость = {rate}')
            ax[ind].legend()
            print(
                f'grad='
                f'{self.graph.function(cords_grad[-1], self.test.get())}'
            )
            print(
                f'newton='
                f'{self.graph.function(cords_newton[-1], self.test.get())}'
            )

        plt.show()


class ViewParaboloid:
    def __init__(self):
        self.graph = Paraboloid(init_graph())

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

