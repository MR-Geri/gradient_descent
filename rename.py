import numpy as np
import matplotlib.pyplot as plt

from visual import init_graph


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
        self.draw_arrow((0, 0), 'yellow', (-5, -10), 'глобальный минимум')

    def draw_graph(self, points) -> None:
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

    def function_grad(self, cords: np.ndarray):
        raise NotImplementedError("Обязательно к переопределению.")

    def function_hesse(self, cords: np.ndarray):
        raise NotImplementedError("Обязательно к переопределению.")

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
            delta_w = -learning_rate * self.function_grad(
                cords_copy
            ) + momentum * delta_w
            cords_copy += delta_w

            w_history = np.vstack((w_history, cords_copy))
            f_history = np.vstack((f_history, self.function(cords_copy)))

            i += 1
            diff = np.absolute(f_history[-1] - f_history[-2])

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
            grad = self.function_grad(cords_copy)
            if self.is_pos_def(hesse):
                hesse_inverse = np.linalg.inv(hesse)
                cords_copy -= learning_rate * np.dot(hesse_inverse, grad)
            else:
                cords_copy -= learning_rate * grad

            w_history = np.vstack((w_history, cords_copy))
            f_history = np.vstack((f_history, self.function(cords_copy)))

            i += 1
            diff = np.absolute(f_history[-1] - f_history[-2])

        return w_history, f_history

    def visualize(self):
        cords = self.get_random_cords(-10, 10, 2)
        fig, _ = plt.subplots(nrows=4, ncols=4, figsize=(54, 54))
        learning_rates = [.05, .3, .7, .9]
        ind = 1

        for rate in learning_rates:
            plt.subplot(2, 4, ind)
            self.draw_graph(self.gradient_descent(
                learning_rate=rate,
                momentum=.5,
                max_iterations=100,
                threshold=1e-2,
                cords_copy=cords.copy()
            ))
            plt.subplot(2, 4, ind+1)
            self.draw_graph(self.newton(
                learning_rate=rate,
                max_iterations=100,
                threshold=1e-2,
                cords_copy=cords.copy()
            ))
            ind += 2
            plt.text(-39, 12, f'Градиент', fontsize=13)
            plt.text(-3, 12, f'Ньютон', fontsize=13)
            plt.text(-25, 15, f'Скорость = {rate}', fontsize=13)

        fig.subplots_adjust(hspace=.5, wspace=.3)

if __name__ == "__main__":
    tmp = Graph(init_graph())
    tmp.visualize()
    plt.show()

