import numpy as np
import scipy.interpolate as si
import plotly.graph_objs as go
import plotly

SMOOTH_FACTOR = 0.05

def exponential_process(x, smooth_factor):
    if len(x) > 5:
        x_res = list()
        x_res.append(x[0])
        for i in range(1, len((x))):
            xn = x[i] * smooth_factor + (1 - smooth_factor) * x_res[-1]
            x_res.append(xn)
        return x_res
    else:
        return x


def smooth_plot_2d(x, y, plot_path):
    t = range(len(x))
    ipl_t = np.linspace(0.0, len(x) - 1, 1000)

    x = exponential_process(x, SMOOTH_FACTOR)
    y = exponential_process(y, SMOOTH_FACTOR)

    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)

    x_list = list(x_tup)
    x_list[1] = x + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    y_list[1] = y + [0.0, 0.0, 0.0, 0.0]

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)

    trace2 = go.Scatter(
        x=x_i,
        y=y_i
    )

    data = [trace2]
    fig = go.Figure(data=data)
    plotly.offline.plot(fig, filename=plot_path, auto_open=True)


def smooth_plot_3d(x, y, z, plot_path):
    t = range(len(x))

    x = exponential_process(x, SMOOTH_FACTOR)
    y = exponential_process(y, SMOOTH_FACTOR)
    z = exponential_process(z, SMOOTH_FACTOR)

    ipl_t = np.linspace(0.0, len(x) - 1, 1000)

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

    trace1 = go.Scatter3d(
        x=x_i,
        y=y_i,
        z=z_i,
        mode='markers',
        marker=dict(
            size=4,
            color=z,
            colorscale='Viridis',
        ),
        line=dict(
            color='#1f77b4',
            width=2
        )
    )


    data = [trace1]
    layout = go.Layout(
        title='Object Track',
        showlegend=True,
        width=1600,
        height=1000,
        scene=dict(
            xaxis=dict(
                title='X Coordinate',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                ),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title='Y Coordinate',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                ),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                title='Depth',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                ),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=plot_path, auto_open=True)
