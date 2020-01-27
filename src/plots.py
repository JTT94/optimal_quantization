
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# plot settings
#rc('text', usetex=True)

def scatter_plot(x, y, col = 'r',
                 title='', xlab='x', ylab= 'y',
                 file_path=None, fig_size = (6 ,5),
                 base_x=None, base_y=None, base_col = 'b'):
    fig = plt.figure(figsize=fig_size)

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    plt.scatter(x, y, c=col)

    if (base_x is not None) and (base_y is not None):
        plt.plot(base_x, base_y, base_col)

    plt.show()

    if file_path is not None:
        plt.savefig(file_path)


def arrow_plot(init_ys, final_ys,
               col1 = 'b', col2='g',
               title='',
               xlab='x',
               ylab='y',
               file_path=None,
               fig_size=(6 ,5),
               head_width=0.01,
               arrow_alpha=0.1):

    fig = plt.figure(figsize=fig_size)
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    for i in range(np.shape(init_ys)[0]):
        dx, dy = final_ys[i] - init_ys[i]
        plt.arrow(init_ys[i, 0], init_ys[i, 1], dx, dy, head_width=head_width, alpha=arrow_alpha)
        plt.plot(init_ys[i, 0], init_ys[i, 1], col1 + "x")
        plt.plot(final_ys[i, 0], final_ys[i, 1], col2 + "o")
    plt.show()

    if file_path is not None:
        plt.savefig(file_path)
