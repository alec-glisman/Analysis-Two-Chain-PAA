"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-09
Description: Functions to load, parse, modify, and plot data from the Plumed output
files generated during the MD simulations.
"""

# Standard library
import os
from pathlib import Path
import re
import warnings

# Third party packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plumed
from scipy import integrate
from tqdm import tqdm

# Internal dependencies
from figures.style import set_style
from stats.block_error import BlockError
from utils.logs import setup_logging


def plumed_df(
    working_dir: str,
    data_dir: str,
    tag: str,
    kbt: float,
    t_ps_remove: float = 100,
    refresh: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load Plumed data from a directory. Data is automatically saved to a
    parquet file for faster loading in the future. If the parquet file
    exists, it is loaded instead re-parsing the data.

    Parameters
    ----------
    working_dir : str
        Path to the desired working directory for data output
    data_dir : str
        Path to the directory containing the plumed data files
    tag : str
        Unique tag for the data set
    kbt : float
        Boltzmann constant times the temperature of the system
        in units of kJ/mol
    refresh : bool, optional
        If True, the data is reloaded from the parquet file
        instead of the original data files, by default False
    verbose : bool, optional
        If True, the logging level is set to DEBUG, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe containing the Plumed data
    """
    log = setup_logging(verbose=verbose, log_file="plumed_df.log")

    # create plumed main directory and change to it
    dir_plumed = Path(f"{working_dir}/plumed")
    dir_plumed.mkdir(parents=True, exist_ok=True)

    # create plumed data sub-directory and change to it
    dir_plumed_data = Path(f"{dir_plumed}/data")
    dir_plumed_data.mkdir(parents=True, exist_ok=True)
    os.chdir(f"{dir_plumed_data}")
    log.debug(f"Moved to directory: {dir_plumed_data}")

    # load previously saved data if it exists
    df_filename = f"df_plumed_{tag}.parquet"
    if not refresh and Path(df_filename).exists():
        log.debug(f"Loading data from {df_filename}")
        df = pd.read_parquet(df_filename)
        os.chdir(f"{working_dir}")
        return df
    log.debug(f"Data not found. Loading from original files.")

    # add base replica number to HREMD simulation data
    if "hremd" in tag:
        path_append = ".0"
        log.debug(f"Appending {path_append} to file paths")
    else:
        path_append = ""

    # find plumed bias and collective variable files
    d_data = Path(data_dir)
    colvar = d_data / f"COLVAR{path_append}.dat"
    bias_walls = d_data / f"BIAS_WALLS{path_append}.dat"
    bias_metad = d_data / f"BIAS_METAD{path_append}.dat"
    files = [colvar, bias_walls, bias_metad]

    if not files[0].exists():
        raise FileNotFoundError(f"COLVAR must exist. Error with {files[0]}")
    if not files[1].exists():
        files[1] = None
        log.info("BIAS_WALLS not found. Skipping.")
    if not files[2].exists():
        log.info("BIAS_METAD not found. Skipping.")
        files[2] = None

    # load plumed files
    dfs = []
    for path in files:
        if path is None:
            continue

        try:
            df = plumed.read_as_pandas(str(path), enable_constants="columns")
            print(f"Loaded {path}")

        except pd.errors.ParserError:
            # first line of file contains column names
            with open(str(path), encoding="utf8") as f:
                header = f.readline()
            header = header.split()[2:]  # remove "#!" FIELDS
            n_cols = len(header)

            log.warning(
                f"WARNING: Error reading {path}. Attempting manual load."
                + f"\nColumns: {header}"
            )

            # create dataframe from file
            df = pd.read_csv(
                str(path),
                names=header,
                comment="#",
                delim_whitespace=True,
                skipinitialspace=True,
                usecols=list(range(n_cols)),
            )

        # clean data
        df.loc["time"] = df["time"].astype(float)

        # convert inf to nan
        df = df.replace([np.inf, -np.inf], np.nan)
        # drop nan rows
        df = df.dropna()
        # drop duplicate rows
        df = df.drop_duplicates(subset="time", keep="last")
        # drop first nanosecond of data
        df = df[df["time"] >= t_ps_remove]

        # data check
        if len(df) == 0:
            raise ValueError(f"Empty dataframe for {path}")

        dfs.append(df)

    df = dfs[0]
    df["tag"] = tag
    for i in range(1, len(dfs)):
        df = pd.merge(df, dfs[i], on="time")

    # convert bias columns to np.float64
    df.astype({"lwall.bias": np.float64, "uwall.bias": np.float64})
    try:
        df.astype({"metad.rbias": np.float64})
    except KeyError:
        pass

    # add metad.rbias column if it doesn't exist
    if "metad.rbias" not in df.columns:
        log.warning("WARNING: metad.rbias column not found. Setting to zero.")
        df["metad.rbias"] = 0

    # sum bias columns
    try:
        df["bias"] = df[["lwall.bias", "uwall.bias", "metad.rbias"]].sum(axis=1)
    except KeyError as exc:
        print(f"WARNING: Bias columns not found, setting equal to zero. {exc}")
        df["bias"] = 0.0

    # boltzmann weight of bias
    df["bias_nondim"] = (df["bias"] - df["bias"].max()) / (kbt)
    df["weight"] = np.exp(df["bias_nondim"], dtype=np.float64)

    # normalize weights to maximum of 1
    max_weight = np.nanmax(df["weight"])
    norm_weight = np.divide(df["weight"], max_weight, dtype=np.float64)
    df["weight"] = norm_weight

    # account for periodic boundary conditions
    if "d12" in df.columns:
        box_length = float(re.findall(r"([0-9]*[.])?[0-9]+(?=nm_box)", tag)[0])
        df["d12"] = np.where(
            df["d12"] > 0.5 * box_length, box_length - df["d12"], df["d12"]
        )

    # save compressed dataframe
    df.to_parquet(df_filename)
    log.debug(f"Saved data to {df_filename}")
    log.debug(f"Final simulation time: {df['time'].max()/1e3:.1f} ns")
    log.debug(f"Number of data points: {len(df)}")

    # return to original directory and return dataframe
    os.chdir(f"{working_dir}")
    return df


def b2_integrand(r: np.ndarray, pmf: np.ndarray, pmf_err: np.ndarray):
    integrand = (np.exp(-pmf) - 1.0) * np.square(r)
    integrand_err = np.abs(pmf_err) * np.abs(integrand)
    return integrand, integrand_err


def integrate_simpson(
    r: np.ndarray,
    integrand: np.ndarray,
    integrand_err: np.ndarray,
    prefactor: float = -2 * np.pi,
):
    integrand_var = integrand_err**2
    dr = np.diff(r)

    integral = integrate.simps(integrand, x=r)
    integral_err = np.sqrt(
        0.25
        * (
            integrand_var[0] * (dr[0] ** 2)
            + integrand_var[-1] * (dr[-1] ** 2)
            + np.sum(integrand_var[1:-1] * (dr[:-1] + dr[1:]) ** 2)
        )
    )
    integral *= prefactor
    integral_err *= np.abs(prefactor)

    return integral, integral_err


def prepare_data(
    df: pd.DataFrame,
    cv: str,
    block_size: int = None,
    pmf: bool = True,
    kde: bool = False,
    x_min_data: float = 0.0,
    bandwidth: float = 0.05,
    n_pts: int = 200,
    x_min_kde=None,
    x_max_kde=None,
    volume_correction_plumed: bool = False,
    volume_correction_gr: bool = False,
    integrand: bool = False,
    integral: bool = False,
) -> tuple[np.array, np.array, np.array]:
    """
    Prepare time series data for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing data to plot.
    cv : str
        Name of collective variable to plot.
    block_size : int, optional
        Size of blocks to average over, by default None
    pmf : bool, optional
        Plot potential of mean force free energy, by default True
    kde : bool, optional
        Plot kernel density estimate, by default False
    x_min_data : float, optional
        Minimum value of cv for analysis, by default 0.0
    bandwidth : float, optional
        Bandwidth of kernel density estimate, by default 0.05
    n_pts : int, optional
        Number of points to plot, by default 200
    x_min_kde : float, optional
        Minimum value of cv for kde, by default None
    x_max_kde : float, optional
        Maximum value of cv for kde, by default None
    volume_correction_plumed : bool, optional
        Plot volume correction from plumed, by default False
    volume_correction_gr : bool, optional
        Plot volume correction from gr, by default False
    integrand : bool, optional
        Plot integrand, by default False
    integral : bool, optional
        Plot integral, by default False

    Returns
    -------
    np.array
        x values
    np.array
        y values
    np.array
        y error values (standard error)

    """
    if integral and integrand:
        raise ValueError("Cannot plot integral and integrand")
    if volume_correction_plumed and volume_correction_gr:
        raise ValueError("Cannot plot volume correction from plumed and gr")
    if len(df[cv]) == 0:
        raise ValueError(f"Dataframe for {cv} is empty")

    # parse box length from tag
    try:
        tag = df["tag"].iloc[0]
    except KeyError:
        tag = df["Tag"].iloc[0]
    box_length = float(re.findall(r"([0-9]*[.])?[0-9]+(?=nm_box)", tag)[0])
    half_box_length = box_length / 2.0

    # drop NaN values
    dfc = df.replace([np.inf, -np.inf], np.nan)
    dfc = dfc.dropna(subset=[cv])
    if len(dfc[cv]) == 0:
        raise ValueError(f"Dropped NaN values for {cv} and all data is gone")

    # drop data outside of reasonable range
    dfc = dfc[dfc[cv] >= x_min_data]
    if cv == "d12":
        dfc = dfc[dfc[cv] <= 0.95 * half_box_length]
        xmin, xmax = 0.0, half_box_length
    else:
        xmin, xmax = np.nanmin(dfc[cv]), np.nanmax(dfc[cv])
    if x_min_kde is not None:
        xmin = x_min_kde
    if x_max_kde is not None:
        xmax = x_max_kde

    # verify that there is data in the range
    if len(dfc[(dfc[cv] >= xmin) & (dfc[cv] <= xmax)]) == 0:
        raise ValueError(f"No data in range {xmin} to {xmax} for {cv}")

    # gather data
    data = dfc[cv].values
    weights = dfc["weight"].values

    # Determine block size as 10 ns of data or 20 blocks, whichever is smaller
    if block_size is None:
        max_block_size = int(np.floor(len(dfc[cv]) // 20))
        n_frames = 10
        dt_frame_ns = (
            (dfc["time"].iloc[n_frames] - dfc["time"].iloc[0]) / n_frames / 1000.0
        )
        block_size = int(np.ceil(10.0 / dt_frame_ns))
        block_size = min(block_size, max_block_size)

    # transform data
    block_sizes = np.array([block_size])
    block_error = BlockError(block_sizes, data, weights=weights)
    if kde and pmf:
        edges, fes, _, fes_error = block_error.x_kde_fes(
            n_pts=n_pts, bandwidth=bandwidth, x_min=xmin, x_max=xmax
        )
        x, y, yerr = edges[1:], fes[-1, 1:], fes_error[-1, 1:]
    elif kde and not pmf:
        edges, pdf, _, pdf_error = block_error.x_kde(
            n_pts=n_pts, bandwidth=bandwidth, x_min=xmin, x_max=xmax
        )
        x, y, yerr = edges[1:], pdf[-1, 1:], pdf_error[-1, 1:]
    else:
        # Freedman-Diaconis rule for bin width
        if bandwidth is None:
            q75, q25 = dfc[cv].quantile([0.75, 0.25])
            bin_width = 2 * (q75 - q25) * len(dfc) ** (-1.0 / 3.0)
            bins = int(np.ceil((dfc[cv].max() - dfc[cv].min()) / bin_width))

        # if bandwidth is 1, use np.arange to bin data
        elif bandwidth == 1:
            bins = np.arange(0, np.nanmax(df[cv]) + 1, 1)
            print(bins)

        else:
            bin_width = bandwidth
            bins = int(np.ceil((dfc[cv].max() - dfc[cv].min()) / bin_width))

        if pmf:
            centers, fes, _, fes_error, edges = block_error.x_histo_fes(
                bins=bins, return_edges=True
            )
            x, y, yerr = centers, fes[-1], fes_error[-1]
        else:
            centers, pdf, _, pdf_error, edges = block_error.x_histo(
                bins=bins, return_edges=True
            )
            x, y, yerr = centers, pdf[-1], pdf_error[-1]

    if cv == "d12":
        # finite volume correction
        if volume_correction_plumed:
            y += 2 * np.log(x)
        if volume_correction_gr:
            shell_volumes = 4.0 / 3.0 * np.pi * np.diff(np.power(edges, 3))
            y += np.log(shell_volumes)

        # subtract average pmf in plateau region
        plateau_mean = np.mean(y[(x >= 4.8) & (x <= 5.2)])
        if not np.isnan(plateau_mean):
            y -= plateau_mean
        else:
            warnings.warn("Plateau region is empty, not subtracting mean")

        # calculate boltzmann weighted integrand
        if integrand or integral:
            y, yerr = b2_integrand(x, y, yerr)

        # integrate using simpson's rule
        if integral:
            y, yerr = integrate_simpson(x, y, yerr, prefactor=-2.0 * np.pi)

    return x, y, yerr


def plt_dyn(df: pd.DataFrame, cv: str) -> tuple[plt.figure, plt.Axes]:
    """
    Plot dynamics of a collective variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing data to plot.
    cv : str
        Name of collective variable to plot.

    Returns
    -------
    fig : plt.figure
        Figure object.
    ax : plt.Axes
        Axes object.
    """
    # remove periods from tag
    tag = df["tag"].iloc[0]
    tag = re.sub(r"\.", "", tag)

    x = df["time"].to_numpy() / 1e3
    y = df[cv].to_numpy()

    # generate FES figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(visible=True, which="both", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.plot(x, y, "o", markersize=1.25, alpha=0.5)
    # ax.set_title(f"{tag}", y=1.05)
    ax.set_xlabel(r"$t$ [ns]")
    if cv == "d12":
        ax.set_ylabel(r"$r$ [nm]")
    elif cv == "rg1":
        ax.set_ylabel(r"$R_{g1}$ [nm]")
    elif cv == "rg2":
        ax.set_ylabel(r"$R_{g2}$ [nm]")
    else:
        raise ValueError("Unknown cv")

    # Save figure and return objects
    fig.tight_layout()
    fig.savefig(f"{tag}_{cv}_dyn.png", dpi=600)
    return fig, ax


def plt_fes_conv(
    df: pd.DataFrame, cv: str, time_slices: np.array, cv_slices: np.array
) -> tuple[list[plt.figure], list[plt.Axes]]:
    """
    Plot dynamics of a collective variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing data to plot.
    cv : str
        Name of collective variable to plot.
    time_slices : np.array
        Time slices to plot.
    cv_slices : np.array
        CV slices to plot.

    Returns
    -------
    figs : list[plt.figure]
        Figure objects.
    axs : list[plt.Axes]
        Axes objects.
    """
    figs = []
    axs = []

    # remove periods from tag
    tag = df["tag"].iloc[0]
    tag = re.sub(r"\.", "", tag)

    # KDE information
    volcorr_plumed = False
    xmin_data = df[cv].min()
    xmin = df[cv].min()
    xmax = df[cv].max()
    if cv == "d12":
        volcorr_plumed = True
        xmin_data = 0.2
        xmin = 0.2
        xmax = 5.5

    # ANCHOR: generate FES mean convergence figure
    viridis = plt.get_cmap("viridis")
    cmap = viridis(np.linspace(0, 256, len(time_slices), dtype=int))
    linestyle = ["-", "--"] * len(time_slices // 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(visible=True, which="both", linestyle="-", linewidth=0.6, alpha=0.6)

    # plot PMF estimate for each sunset of df where time < time_slice
    for time_slice, color, line in tqdm(
        zip(time_slices, cmap, linestyle),
        dynamic_ncols=True,
        total=len(time_slices),
        colour="green",
        desc="FES mean convergence",
    ):
        df_slice = df[df["time"] <= time_slice]
        x, pmf, pmf_err = prepare_data(
            df_slice,
            cv,
            block_size=4000,
            pmf=True,
            kde=True,
            volume_correction_plumed=volcorr_plumed,
            x_min_data=xmin_data,
            n_pts=500,
            x_min_kde=xmin,
            x_max_kde=xmax,
        )
        ax.plot(x, pmf, label=f"{time_slice / 1e3:.0f}", color=color, linestyle=line)

    ax.fill_between(
        x,
        pmf - 1.96 * pmf_err,
        pmf + 1.96 * pmf_err,
        alpha=0.5,
        color=cmap[-1],
        label=r"$95\%$ CI",
    )

    # set plot text elements
    # ax.set_title(tag[:25], y=1.05)
    ax.set_ylabel(r"$\Delta F$ [$k_\mathrm{B} \, T$]")
    if cv == "d12":
        ax.set_xlabel(r"Chain Distance [nm]")
    elif cv == "rg1":
        ax.set_xlabel(r"$R_{g1}$ [nm]")
    elif cv == "rg2":
        ax.set_xlabel(r"$R_{g2}$ [nm]")
    else:
        raise ValueError("Unknown cv")
    ax.legend(title=r"$t_f$ [ns]", loc="best", ncol=2)

    # set plot limits
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, min(ymax, 20.0))

    # Save figure and return objects
    fig.tight_layout()
    fig.savefig(f"{tag}_{cv}_fes_mean_conv.png", dpi=600)
    figs.append(fig)
    axs.append(ax)

    # ANCHOR: generate FES standard error convergence figure
    cmap = viridis(np.linspace(0, 256, len(cv_slices), dtype=int))
    linestyle = ["-", "--"] * len(cv_slices // 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(visible=True, which="both", linestyle="-", linewidth=0.6, alpha=0.6)

    # run block error analysis for data
    block_sizes = np.linspace(10, len(df) // 30, 80, dtype=int)
    block_error = BlockError(block_sizes, df[cv].values, weights=df["weight"].values)
    edges, _, _, fes_error = block_error.x_kde_fes(
        n_pts=100, bandwidth=0.05, x_min=xmin, x_max=xmax
    )

    # locate indices of edges nearest to each cv_slice
    cv_slice_indices = []
    for cv_slice in cv_slices:
        cv_slice_indices.append(np.argmin(np.abs(edges - cv_slice)))

    # plot fes_error vs block_size for each cv_slice
    for cv_slice_index, color, line in zip(cv_slice_indices, cmap, linestyle):
        ax.plot(
            block_sizes,
            fes_error[:, cv_slice_index],
            label=f"{edges[cv_slice_index]:.2f}",
            color=color,
            linestyle=line,
        )

    # set plot text elements
    # ax.set_title(tag[:25], y=1.05)
    ax.set_xlabel("Block Size")
    if cv == "d12":
        ax.set_ylabel(r"${SE}_{\Delta F}$(Chain Distance) [$k_\mathrm{B} \, T$]")
        ax.legend(title=r"Chain Distance [nm]", loc="best", ncol=2)
    elif cv == "rg1":
        ax.set_ylabel(r"${SE}_{\Delta F}{R_{g1}}$ [$k_\mathrm{B} \, T$]")
        ax.legend(title=r"$R_{g1}$ [nm]", loc="best", ncol=2)
    elif cv == "rg2":
        ax.set_ylabel(r"${SE}_{\Delta F}{R_{g2}}$ [$k_\mathrm{B} \, T$]")
        ax.legend(title=r"$R_{g2}$ [nm]", loc="best", ncol=2)
    else:
        raise ValueError("Unknown cv")

    # Save figure and return objects
    fig.tight_layout()
    fig.savefig(f"{tag}_{cv}_fes_std_err_conv.png", dpi=600)
    figs.append(fig)
    axs.append(ax)

    return figs, axs


def plt_b2_conv(df: pd.DataFrame) -> tuple[plt.figure, plt.Axes]:
    """
    Generate B2 convergence plot. B2 is the two-chain second osmotic
    virial coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the data to be plotted

    Returns
    -------
    fig : plt.figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    # remove periods from tag
    tag = df["tag"].iloc[0]
    tag = re.sub(r"\.", "", tag)

    # calculate data
    t_subset_arr = np.linspace(10000, df["time"].max(), 100)
    b2_arr = np.zeros_like(t_subset_arr)
    b2_err_arr = np.zeros_like(t_subset_arr)
    for i in tqdm(
        range(len(t_subset_arr)),
        desc="Calculating B2",
        colour="green",
        dynamic_ncols=True,
    ):
        df_slice = df[df["time"] <= t_subset_arr[i]]
        block_size = min(4000, len(df_slice) // 20)
        _, b2, b2_err = prepare_data(
            df_slice,
            "d12",
            block_size=block_size,
            pmf=True,
            kde=True,
            bandwidth=0.05,
            volume_correction_plumed=True,
            integral=True,
            x_min_data=0.2,
            n_pts=1000,
            x_min_kde=0.1,
            x_max_kde=5.0,
        )
        b2_arr[i] = b2
        b2_err_arr[i] = b2_err

    # generate figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(visible=True, which="both", linestyle="-", linewidth=0.6, alpha=0.6)

    ax.plot(t_subset_arr / 1e3, b2_arr)
    ax.fill_between(
        t_subset_arr / 1e3,
        b2_arr - 1.96 * b2_err_arr,
        b2_arr + 1.96 * b2_err_arr,
        alpha=0.3,
    )

    ax.set_ylabel(r"$B_2$ [nm$^3$]")
    ax.set_xlabel(r"Time [ns]")
    # ax.set_title(tag[:25], y=1.05)

    fig.tight_layout()
    fig.savefig(f"{tag}_b2_conv.png", dpi=600)
    return fig, ax


def plumed_plots(data_dir: str, tag: str, cv: str) -> None:
    """
    Create Plumed plots for a given data set.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the plumed data files
    tag : str
        Unique tag for the data set
    cv : str
        Name of the collective variable
    """
    cwd = os.getcwd()

    log = setup_logging(verbose=False, log_file="plumed_plots.log")
    dir_plumed = Path(f"{data_dir}/plumed")

    # create plumed plots sub-directory and change to it
    dir_plumed_figures = Path(f"{dir_plumed}/figures")
    dir_plumed_figures.mkdir(parents=True, exist_ok=True)
    os.chdir(f"{dir_plumed_figures}")

    # load plumed data
    dir_plumed_data = Path(f"{dir_plumed}/data")
    df = pd.read_parquet(f"{dir_plumed_data}/df_plumed_{tag}.parquet")
    log.debug(f"Loaded data from {dir_plumed_data}/df_plumed_{tag}.parquet")

    # convert dask to pandas for plotting
    try:
        df = df.compute()
    except AttributeError:
        pass

    # get time slices to estimate FES
    time = df["time"].values
    time_min = np.nanmin(time)
    time_max = np.nanmax(time)
    time_range = time_max - time_min
    time_percentiles = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    time_slices = time_min + time_percentiles * time_range
    log.debug(f"Time slices: {time_slices}")

    # get slices of data for FES plots
    cv_data = df[cv].values
    cv_min = np.nanmin(cv_data)
    cv_max = np.nanmax(cv_data)
    cv_range = cv_max - cv_min
    cv_percentiles = np.array(
        [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )
    cv_slices = cv_min + cv_percentiles * cv_range
    log.debug(f"CV slices: {cv_slices}")

    set_style()

    # Plot CV vs time
    fig, _ = plt_dyn(df, cv)
    plt.close(fig)

    # Plot FES mean and error estimates vs time
    figs, _ = plt_fes_conv(df, cv, time_slices, cv_slices)
    for fig in figs:
        plt.close(fig)

    # Plot B2 vs time if CV is d12
    if cv == "d12":
        fig, _ = plt_b2_conv(df)
        plt.close(fig)

    os.chdir(cwd)
