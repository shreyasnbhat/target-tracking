import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
from mpl_toolkits.mplot3d import axes3d, Axes3D


def smooth_plot_2d(x, y, plot_path):
    t = range(len(x))
    ipl_t = np.linspace(0.0, len(x) - 1, 100)

    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)

    x_list = list(x_tup)
    x_list[1] = x + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    y_list[1] = y + [0.0, 0.0, 0.0, 0.0]

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)

    print(x_i)
    print(y_i)

    # plt.plot(x, y, '-o')
    plt.plot(x_i, y_i, 'r')
    plt.xlim([min(x) - 0.3, max(x) + 0.3])
    plt.ylim([min(y) - 0.3, max(y) + 0.3])
    plt.xlabel("X Coordinate")
    plt.ylabel("Depth")
    plt.savefig(plot_path)


def smooth_plot_3d(x, y, z, plot_path):
    t = range(len(x))

    ipl_t = np.linspace(0.0, len(x) - 1, 100)

    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)
    z_tup = si.splrep(t, z, k=3)

    x_list = list(x_tup)
    x_list[1] = x + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    y_list[1] = y + [0.0, 0.0, 0.0, 0.0]

    z_list = list(z_tup)
    z_list[1] = z + [0.0, 0.0, 0.0, 0.0]

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)
    z_i = si.splev(ipl_t, z_list)

    fig = plt.figure()
    ax  = Axes3D(fig)

    ax.set_xlim3d(0, 100)
    ax.set_ylim3d(0, 100)
    ax.set_zlim3d(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(x_i, y_i, z_i, color='b')
    plt.savefig(plot_path)
