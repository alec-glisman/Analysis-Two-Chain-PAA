"""
Script contains the desired styling parameters for plots.
"""
# Standard library
import gc

# Third-party packages
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_style():
    """
    Set the style of the plots.
    """

    # Pyplot parameters
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["agg.path.chunksize"] = 10000
    plt.style.use(["seaborn-colorblind"])

    # Matplotlib parameters
    # mpl.use("TKAgg")

    mpl.rcParams.update(
        {
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "axes.linewidth": 1.5,
            "axes.unicode_minus": False,
            "figure.autolayout": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            "font.size": 14,
            "legend.columnspacing": 1,
            "legend.fontsize": 14,
            "legend.handlelength": 1.25,
            "legend.labelspacing": 0.25,
            "legend.loc": "best",
            "legend.title_fontsize": 16,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "k",
            "lines.linewidth": 2,
            "lines.markersize": 10,
            "mathtext.fontset": "cm",
            "savefig.dpi": 1200,
            "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amsbsy,"
            + r"amssymb,bm,amsthm,mathrsfs,fixmath,gensymb}",
            "text.usetex": True,
            "xtick.labelsize": 16,
            "xtick.major.size": 5,
            "xtick.major.width": 1.2,
            "xtick.minor.size": 3,
            "xtick.minor.width": 0.9,
            "xtick.minor.visible": True,
            "ytick.labelsize": 16,
            "ytick.major.size": 5,
            "ytick.major.width": 1.2,
            "ytick.minor.size": 3,
            "ytick.minor.width": 0.9,
            "ytick.minor.visible": True,
        }
    )


def close_fig(fig: plt.figure, variables: list) -> None:
    """
    Delete the figure and variables from memory.

    :param fig: The figure to close
    :type fig: plt.figure
    :param variables: The variables to delete
    :type variables: list
    """

    fig.clear()
    plt.figure(fig.number)
    plt.clf()
    plt.close()

    del fig
    for var in variables:
        del var

    gc.collect()
