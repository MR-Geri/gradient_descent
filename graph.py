import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, board: tuple[np.ndarray, np.ndarray], params=None) -> None:
        self.board = board
        self.params = params if params is not None else tuple()

        self.random_seed = np.random.RandomState(19)

    def get_random_cords(self,
                         left: float | int,
                         right: float | int,
                         quantity: int
                         ) -> np.ndarray:
        return self.random_seed.uniform(left, right, quantity)

    def draw_board(self) -> None:
        pts, f_vals = self.board
        f_plot = plt.scatter(
            pts[:, 0], pts[:, 1],
            c=f_vals, vmin=min(f_vals), vmax=max(f_vals), cmap='RdBu_r')
        plt.colorbar(f_plot)
        self.draw_arrow(
            min(zip(*self.board), key=lambda x: x[1])[0],
            'yellow', (-5, -10), 'глобальный минимум'
        )

    def draw_graph_on_board(self, points) -> None:
        self.draw_board()
        plt.plot(points[:, 0], points[:, 1], marker='o', c='magenta')
        self.draw_arrow(points[-1], 'green', (-1, 7), 'минимум')

        for i, w in enumerate(points[:-1]):
            plt.annotate(
                "",
                xy=w, xycoords='data', xytext=points[i+1, :], textcoords='data',
                arrowprops={
                    "arrowstyle": '<-',
                    "connectionstyle": 'angle3'
                })

    def draw_graph_on_ax(self, ax: Axes, points, color: str, label: str) -> None:
        _, y = points
        ax.plot(np.arange(y.size), y, color=color, label=label)

    def draw_arrow(self,
                   cord_point: tuple[float | int, float | int], color: str,
                   cord_text: tuple[float | int, float | int], text: str
                   ) -> None:
        plt.plot(xy=cord_point, marker='P', markersize=10, c=color)
        plt.annotate(text, xy=cord_point, xytext=cord_text,
                     arrowprops={
                         "arrowstyle": "->",
                         "color": color,
                         "connectionstyle": 'arc3'
                     })

    def function(self, cords: np.ndarray):
        raise NotImplementedError("Обязательно к переопределению.")

    def function_derivative(self, cords: np.ndarray):
        raise NotImplementedError("Обязательно к переопределению.")

    def function_hesse(self, cords: np.ndarray):
        raise NotImplementedError("Обязательно к переопределению.")

    @staticmethod
    def calculate_diff(pred, pred_pred):
        return np.absolute(pred - pred_pred)

    def gradient_descent(self,
                         learning_rate: float,
                         momentum: float,
                         max_iterations: int,
                         threshold: float,
                         cords_copy: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        w_history = cords_copy.copy()
        f_history = self.function(cords_copy.copy())
        delta_w = np.zeros(cords_copy.shape)
        i, diff = 0, 1e10

        while i < max_iterations and diff > threshold:
            delta_w = -learning_rate * self.function_derivative(
                cords_copy
            ) + momentum * delta_w
            cords_copy += delta_w

            w_history = np.vstack((w_history, cords_copy))
            f_history = np.vstack((f_history, self.function(cords_copy)))

            i += 1
            diff = self.calculate_diff(f_history[-1], f_history[-2])

        return w_history, f_history

    def is_pos_def(self, matrix_hesse):
        return np.all(np.linalg.eigvals(matrix_hesse) > 0)

    def newton(self,
               learning_rate: float,
               max_iterations: int,
               threshold: float,
               cords_copy: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:

        w_history = cords_copy.copy()
        f_history = self.function(cords_copy)
        i, diff = 0, 1e10

        while i < max_iterations and diff > threshold:
            hesse = self.function_hesse(cords_copy)
            grad = self.function_derivative(cords_copy)
            if self.is_pos_def(hesse):
                hesse_inverse = np.linalg.inv(hesse)
                cords_copy -= learning_rate * np.dot(hesse_inverse, grad)
            else:
                cords_copy -= learning_rate * grad

            w_history = np.vstack((w_history, cords_copy))
            f_history = np.vstack((f_history, self.function(cords_copy)))

            i += 1
            diff = self.calculate_diff(f_history[-1], f_history[-2])

        return w_history, f_history


class Paraboloid(Graph):
    def __init__(self, board: tuple[np.ndarray, np.ndarray], params=None) -> None:
        super().__init__(board, params)

    def function(self, cords: np.ndarray):
        return np.sum(cords * cords)

    def function_derivative(self, cords: np.ndarray):
        return 2 * cords

    def function_hesse(self, cords: np.ndarray):
        return np.array(((2, 0), (0, 2)))


class MSE(Graph):
    def __init__(self, board: tuple[np.ndarray, np.ndarray], params=None) -> None:
        super().__init__(board, params)

    @staticmethod
    def calculate_diff(pred, pred_pred):
        return pred

    def function(self, cords: np.ndarray, temp_params = None):
        params = self.params if temp_params is None else temp_params
        o = np.sum(params[0] * cords, axis=1)

#        ind_1 = np.where(o > 0.5)
#        ind_0 = np.where(o <= 0.5)
#        o[ind_1] = 1
#        o[ind_0] = 0

        mse = np.sum((params[1] - o) ** 2)
        return mse / params[1].size

    def function_derivative(self, cords: np.ndarray):
        rows, cols = self.params[0].shape

        o = np.sum(self.params[0] * cords, axis=1)
        zn = self.params[1] - o

        diff = np.tile(zn.reshape((rows, 1)), (1, cols))
        grad = -2 * np.sum(diff * self.params[0], axis=0)
        return grad

    def function_hesse(self, cords: np.ndarray):
        hesse = 2 * np.sum([np.outer(i, i) for i in self.params[0]], axis=0)

#        hesse = np.zeros((64, 64))
#        for matrix in self.params[0]:
#            tmp = np.array([[2 * i * j for j in matrix] for i in matrix])
#            hesse += np.array(tmp)

        return hesse
    
    def analize_hesse(self):
        for i in self.params[0]:
            matrix = 2 * np.outer(i, i)
            hesse = list(np.linalg.eigvals(matrix) > 0)
            value_false = len(list(filter(lambda x: not x, hesse)))
            print(value_false)


def init_graph():
    x = np.linspace(-10.0, 10.0, 100)
    y = np.linspace(-10.0, 10.0, 100)
    w1, w2 = np.meshgrid(x, y)
    pts = np.vstack((w1.flatten(), w2.flatten()))
    pts = pts.transpose()

    f_vals = np.sum(pts * pts, axis=1)
    return pts, f_vals

