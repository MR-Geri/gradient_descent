import numpy as np
import matplotlib.pyplot as plt


def visualize_fw():
    x = np.linspace(-10.0, 10.0, 100)
    y = np.linspace(-10.0, 10.0, 100)
    w1, w2 = np.meshgrid(x, y)
    pts = np.vstack((w1.flatten(), w2.flatten()))

    pts = pts.transpose()

    f_vals = np.sum(pts * pts, axis=1)
    return pts, f_vals


def annotate_pt(text, xy, xytext, color):
    plt.plot(xy=xy, marker='P', markersize=10, c=color)
    plt.annotate(text, xy=xy, xytext=xytext,
                 arrowprops={
                     "arrowstyle": "->",
                     "color": color,
                     "connectionstyle": 'arc3'
                 })


def function_plot(pts, f_val):
    f_plot = plt.scatter(
        pts[:, 0], pts[:, 1],
        c=f_val, vmin=min(f_val), vmax=max(f_val), cmap='RdBu_r')
    plt.colorbar(f_plot)
    annotate_pt('глобальный минимум', (0, 0), (-5, -10), 'yellow')


if __name__ == '__main__':
    pts, f_vals = visualize_fw()
    function_plot(pts, f_vals)
    plt.show()

