import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split

from graph import Paraboloid, MSE


class ViewMSE:
    def __init__(self):
        self.graph = MSE(init_graph())
        self.sliders = []

    def show_numbers(self, data):
        print(f'len_show={len(data)}\nshow={data}')
        _, ax = plt.subplots(
            nrows=1, ncols=10, figsize=(12, 4), 
            subplot_kw={"xticks": [], "yticks": [] }
        )
        for i in np.arange(10):
            ax[i].imshow(data[i].reshape(8, 8))

    def show(self):
        digits, target = dt.load_digits(n_class=2, return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(
            digits, target, test_size=.2, random_state=10
        )

        cords_copy = self.graph.get_random_cords(
            -1, 1, x_train.shape[1]
        ) * 1e-6

#        self.show_numbers(x_train)

#        x_train = np.hstack((np.ones((len(y_train), 1)), x_train))
#        x_test = np.hstack((np.ones((len(y_test), 1)), x_test))

#        _, ax = plt.subplot_mosaic([
#            [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]
#        ])
#
        self.graph.params = (x_train, y_train)

        _, errors_newton = self.graph.newton(
            learning_rate=1e-6,
            max_iterations=100,
            threshold=0.005,
            cords_copy=cords_copy.copy()
        )
        print(f'len_newton={len(errors_newton)} last={errors_newton[-1]}')
        _, errors_grad = self.graph.gradient_descent(
            learning_rate=1e-6,
            max_iterations=100,
            momentum=0.7,
            threshold=0.005,
            cords_copy=cords_copy.copy()
        )
        print(f'len_grad={len(errors_grad)} last={errors_grad[-1]}')

#        for ind, rate in enumerate((1e-6, 1e-7, 1e-8)):
#            ax[ind].clear()
#            self.graph.draw_graph_on_ax(
#                ax[ind],
#                self.graph.gradient_descent(
#                    learning_rate=rate,
#                    momentum=0.9,
#                    max_iterations=100,
#                    threshold=1e-2,
#                    cords_copy=cords_copy.copy()
#                ),
#                'green',
#                'Градиент'
#            )
#            self.graph.draw_graph_on_ax(
#                ax[ind],
#                self.graph.newton(
#                    learning_rate=rate,
#                    max_iterations=100,
#                    threshold=1e-2,
#                    cords_copy=cords_copy.copy()
#                ),
#                'red',
#                'Ньютон'
#            )
#            ax[ind].legend()

#        for ind, momentum in enumerate((0, .7, .9)):
#            show_main_graph(ax[ind], w.copy(), momentum, 1e-6, (x_train, y_train))
#            if ind == 1:
#                ax[ind].set_xlabel('Итерация')
#            if ind == 0:
#                ax[ind].set_ylabel('Среднеквадратичная ошибка')
#
#            ax[ind].set_title(f'Импульс = {momentum}\n')
#
#        s1 = Slider(ax[3], '[1]', 1e-9, 1e-6, 1e-6)
#        s2 = Slider(ax[4], '[2]', 1e-9, 1e-6, 1e-6)
#        s3 = Slider(ax[5], '[3]', 1e-9, 1e-6, 1e-6)
#        s1.on_changed(update_slider)
#        s2.on_changed(update_slider)
#        s3.on_changed(update_slider)
        plt.show()

class ViewParabaloid:
    def __init__(self):
        self.graph = Paraboloid(init_graph())
        self.sliders = []

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


def init_graph():
    x = np.linspace(-10.0, 10.0, 100)
    y = np.linspace(-10.0, 10.0, 100)
    w1, w2 = np.meshgrid(x, y)
    pts = np.vstack((w1.flatten(), w2.flatten()))
    pts = pts.transpose()

    f_vals = np.sum(pts * pts, axis=1)
    return pts, f_vals


def draw_arrowprops(text, xy, xytext, color) -> None:
    plt.plot(xy=xy, marker='P', markersize=10, c=color)
    plt.annotate(text, xy=xy, xytext=xytext,
                 arrowprops={
                     "arrowstyle": "->",
                     "color": color,
                     "connectionstyle": 'arc3'
                 })


def draw_plot(pts, f_val) -> None:
    f_plot = plt.scatter(
        pts[:, 0], pts[:, 1],
        c=f_val, vmin=min(f_val), vmax=max(f_val), cmap='RdBu_r')
    plt.colorbar(f_plot)
    draw_arrowprops('глобальный минимум', (0, 0), (-5, -10), 'yellow')


if __name__ == '__main__':
#    pts, f_vals = init_graph()
#    draw_plot(pts, f_vals)
#    plt.show()
    ViewMSE().show()

