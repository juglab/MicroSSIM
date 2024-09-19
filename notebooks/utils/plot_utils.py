import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def add_text(ax, text, img_shape, place="TOP_LEFT", edgecolor="black"):
    """
    Adding text on image
    """
    assert place in ["TOP_LEFT", "BOTTOM_RIGHT"]
    if place == "TOP_LEFT":
        ax.text(
            img_shape[1] * 15 / 500,
            img_shape[0] * 55 / 500,
            text,
            bbox=dict(facecolor="white", edgecolor=edgecolor, alpha=0.9),
        )
    elif place == "BOTTOM_RIGHT":
        s0 = img_shape[1]
        s1 = img_shape[0]
        ax.text(
            s0 - s0 * 150 / 500,
            s1 - s1 * 35 / 500,
            text,
            bbox=dict(facecolor="white", edgecolor=edgecolor, alpha=0.9),
        )


def clean_ax(ax):
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def add_subplot_axes(
    ax, rect: list[float], facecolor: str = "w", min_labelsize: int = 5
):
    """
    Add an axes inside an axes. This can be used to create an inset plot.
    Adapted from https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    Args:
        ax: matplotblib.axes
        rect: Array with 4 elements describing where to position the new axes inside the current axes ax.
            eg: [0.1,0.1,0.4,0.2]
        facecolor: what should be the background color of the new axes
        min_labelsize: what should be the minimum labelsize in the new axes
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    # transAxes: co-ordinate system of the axes: 0,0 is bottomleft and 1,1 is top right.
    # With below command, we want to get to a position which would be the position of new plot in the axes coordinate
    # system
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    # with below command, we now have a position of the new plot in the figure coordinate system. we need this because
    # we can create a new axes in the figure coordinate system. so we want to get everything in that system.
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    # subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=max(min_labelsize, x_labelsize))
    subax.yaxis.set_tick_params(labelsize=max(min_labelsize, y_labelsize))
    return subax


def add_pixel_kde(
    ax,
    rect: list[float],
    data_list: list[np.ndarray],
    min_labelsize: int,
    plot_xmax_value: int = None,
    plot_xmin_value: int = None,
    plot_kwargs_list=None,
    color_list=None,
    label_list=None,
    color_xtick="white",
):
    """
    Adds KDE (density plot) of data1(eg: target) and data2(ex: predicted) image pixel values as an inset
    """
    if plot_kwargs_list is None:
        plot_kwargs_list = [{} for _ in range(len(data_list))]

    inset_ax = add_subplot_axes(ax, rect, facecolor="None", min_labelsize=min_labelsize)

    inset_ax.tick_params(axis="x", colors=color_xtick)
    # xmin, xmax = inset_ax.get_xlim()

    if plot_xmax_value is not None:
        xmax_data = plot_xmax_value
    else:
        xmax_data = [int(datak.max()) for datak in data_list]
        if len(xmax_data) > 1:
            xmax_data = max(*xmax_data) + 1
        else:
            xmax_data = xmax_data[0] + 1

    xmin_data = 0
    if plot_xmin_value is not None:
        xmin_data = plot_xmin_value
    else:
        xmin_data = [int(datak.min()) for datak in data_list]
        if len(xmin_data) > 1:
            xmin_data = min(*xmin_data) - 1
        else:
            xmin_data = xmin_data[0] - 1

    for datak, colork, labelk, plot_kwargsk in zip(
        data_list, color_list, label_list, plot_kwargs_list
    ):
        sns.kdeplot(
            data=datak.reshape(
                -1,
            ),
            ax=inset_ax,
            color=colork,
            label=labelk,
            clip=(xmin_data, None),
            **plot_kwargsk,
        )

    inset_ax.set_aspect("auto")
    inset_ax.set_xlim([xmin_data, xmax_data])  # xmin=0,xmax= xmax_data
    inset_ax.set_xbound(lower=xmin_data, upper=xmax_data)

    xticks = inset_ax.get_xticks()
    inset_ax.set_xticks([xticks[0], xticks[-1]])
    clean_for_xaxis_plot(inset_ax)
    return inset_ax


def clean_for_xaxis_plot(inset_ax):
    """
    For an xaxis plot, the y axis values don't matter. Neither the axes borders save the bottom one.
    """
    # Removing y-axes ticks and text
    inset_ax.set_yticklabels([])
    inset_ax.tick_params(left=False, right=False)
    inset_ax.set_ylabel("")

    # removing the axes border lines.
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.spines["left"].set_visible(False)


def boundary_color(ax_cur, color):
    for spine in ax_cur.spines.values():
        spine.set_edgecolor(color)


def get_zoomin_hw(img, zoomin_size=80, seed=42):
    rng = np.random.default_rng(seed)

    hw = None
    hw_std = None
    for _ in range(200):
        # h = np.random.randint(0, img.shape[0]-zoomin_size)
        # w = np.random.randint(0, img.shape[1]-zoomin_size)
        h = rng.integers(0, img.shape[0] - zoomin_size)
        w = rng.integers(0, img.shape[1] - zoomin_size)
        if hw is None:
            hw = [h, w]
            hw_std = img[h : h + 80, w : w + 80].std()
        else:
            cur_std = img[h : h + 80, w : w + 80].std()
            if cur_std > hw_std:
                hw = [h, w]
                hw_std = cur_std
    return hw
