"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-14
Description: Intermolecular RDF analysis with Dask parallelization

This module provides a class for calculating the intermolecular pair
distribution function (g(r)) with Dask parallelization. The class
inherits from :class:`ParallelAnalysisBase`. The `run` method calculates
desired quantities. The `save` method saves the results to a parquet file.
The `figures` method plots the results.

The `_single_frame` is similar to the method of the same name in
the default :class:`MDAnalysis.analysis.rdf.InterRDF` class. I added
functionality for weighted frames, center of mass distributions, and
excluding frames that do not meet other input boolean criteria.
"""

# Standard library
from pathlib import Path
import sys
import warnings

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# MDAnalysis inheritance
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import capped_distance

# Internal dependencies
from .base import ParallelAnalysisBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class InterRDF(ParallelAnalysisBase):
    """
    Intermolecular pair distribution function (g(r)). This class inherits
    from :class:`ParallelAnalysisBase`. The `run` method of this class
    calculates the intermolecular pair distribution function (g(r)).

    Note that the `run` method of this class allows for additional
    input parameters beyond the default in
    :class:`MDAnalysis.analysis.base.AnalysisBase`. These additional
    parameters are used for Dask parallelization.

    Additional methods are provided to plot the results as well as
    save the results to a parquet file.
    """

    def __init__(
        self,
        ag1: AtomGroup,
        ag2: AtomGroup,
        label: str = None,
        df_weights: pd.DataFrame = None,
        nbins: int = 75,
        domain: tuple[float] = (0.0, 15.0),
        norm: str = "rdf",
        center_of_mass: bool = False,
        verbose: bool = False,
        exclusion_block=None,
        **kwargs,
    ):
        """
        Initialize the RDF analysis object.

        Parameters
        ----------
        ag1 : AtomGroup
            First atom group.
        ag2 : AtomGroup
            Second atom group.
        label : str, optional
            Label for the RDF. Default is None.
        df_weights : DataFrame, optional
            Dataframe containing weights for each frame from biasing
            potential. If None, no weights are applied. Default is None.
        nbins : int, optional
            Number of bins for the RDF. Default is 75.
        domain : tuple[float], optional
            Domain of the RDF in units of Angstroms. Default is (0.0, 15.0).
        norm : str, optional
            Normalization of the RDF. Options are "rdf", "density", and
            "none". Default is "rdf".
        center_of_mass : bool, optional
            If True, the center of mass of each atom group is used for the
            calculation. If False, the coordinates of each atom are used.
            Default is False.
        exclusion_block : list, optional
            List of atom indices to exclude from the calculation. Default is
            None.
        verbose : bool, optional
            If True, print additional information. Default is False.
        **kwargs
            Additional keyword arguments for :class:`ParallelAnalysisBase`.
        """
        super().__init__(
            ag1.universe.trajectory, (ag1, ag2), label=label, verbose=verbose, **kwargs
        )
        self._logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        # print how many atoms are in each selection
        self._logger.debug(f"Number of atoms in ag1: {ag1.n_atoms}")
        self._logger.debug(f"Number of atoms in ag2: {ag2.n_atoms}")

        # class parameters
        self.center_of_mass: bool = center_of_mass
        self._logger.debug(f"center_of_mass: {self.center_of_mass}")

        # dataframe containing weight of each frame from biasing potential
        if df_weights is not None:
            self._weighted: bool = True
            self._df_weights = df_weights[["time", "weight"]].copy()
        else:
            self._weighted: bool = False
            self._df_weights: pd.DataFrame = None
        self._logger.debug(f"weighted: {self._weighted}")

        # output data
        self._dir_out: Path = Path(f"./mdanalysis_rdf")
        self._df_filename = f"rdf_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df = None
        self._columns = ["frame", "time", "volume"]
        for i in range(nbins):
            self._columns.append(f"bin_{i}")

        # rdf settings
        self._rdf_settings: dict = {"bins": nbins, "range": domain}
        self._exclusion_block = exclusion_block
        self._norm = str(norm).lower()
        if self._norm not in ["rdf", "density", "none"]:
            raise ValueError(
                f"'{self._norm}' is an invalid norm. " "Use 'rdf', 'density' or 'none'."
            )

        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self._rdf_settings)
        count = count.astype(np.float64)
        count *= 0.0
        self.results.count = count
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])
        self.results.rdf = None

        # Normalization method
        if self._norm == "rdf":
            # Cumulative volume for rdf normalization
            self._volume_cum = 0

        # Set the max range to filter the search radius
        self._maxrange = self._rdf_settings["range"][1]

        self._logger.info(f"Initialized RDF analysis for {self._tag}.")

    def _single_frame(self, idx_frame: int) -> np.array:
        """
        Analyze a single frame in the trajectory.

        Parameters
        ----------
        idx_frame: int
            Index of the frame to analyze.

        Returns
        -------
        np.array
            The weight, the volume of the box, and the histogram of the
            pair distances at the current frame.
        """
        # get the current frame
        ts = self._universe.trajectory[idx_frame]
        g1, g2 = self._atomgroups

        # prepare output array
        result = np.empty(3 + self._rdf_settings["bins"], dtype=np.float64)
        result.fill(np.nan)

        # save frame index, time
        result[0] = ts.frame
        result[1] = g1.universe.trajectory.time
        result[2] = ts.volume

        # Get pair separation distances of atom pairs within range
        pairs, dist = capped_distance(
            g1.positions,
            g2.positions,
            self._rdf_settings["range"][1],
            box=ts.dimensions,
        )

        # Exclude atom pairs with the same atoms or atoms from the
        # same residue
        if self._exclusion_block is not None:
            dist = dist[
                np.where(
                    pairs[:, 0] // self._exclusion_block[0]
                    != pairs[:, 1] // self._exclusion_block[1]
                )[0]
            ]

        # calculate histogram
        result[3:] = np.histogram(dist, **self._rdf_settings)[0]

        return result

    def _conclude(self) -> None:
        """
        Finalize the results of the calculation. This is called once
        after the analysis is finished inside the run() method. The method
        concatenates the results from the single frame analysis and
        normalizes the histograms.
        """
        self._logger.info(f"Finalizing RDF analysis for {self._tag}.")

        # Sum up the results
        self._volume_cum = np.sum(self._results[:, 2])

        if self._weighted:
            # merge the rdf results with the weights
            self._df = pd.DataFrame(self._results, columns=self._columns)
            self._df = pd.merge(self._df, self._df_weights, how="inner", on="time")

            # calculate the total weight and weighted histogram
            self._total_weight = np.sum(self._df["weight"].values)
            self.results.count = np.sum(
                self._df.iloc[:, 3:-1].values * self._df["weight"].values[:, None],
                axis=0,
            )

        else:
            self._total_weight = len(self._results[:, 0])
            self.results.count = np.sum(self._results[:, 3:], axis=0)

        norm = self.n_frames
        if self._norm in ["rdf", "density"]:
            # Volume in each radial shell
            vols = np.power(self.results.edges, 3)
            norm *= 4.0 / 3.0 * np.pi * np.diff(vols)

        if self._norm == "rdf":
            # Number of each selection
            na = self._atomgroups[0].n_atoms
            nb = self._atomgroups[1].n_atoms
            n = na * nb

            # If we had exclusions, take these into account
            if self._exclusion_block:
                xa, xb = self._exclusion_block
                n_blocks = na / xa
                n -= xb * xb * n_blocks

            # Average number density
            box_vol = self._volume_cum / self.n_frames
            self._logger.debug(f"Average box volume: {box_vol:.3f} A^3")
            norm *= n / box_vol

        # Normalize by the sum of weights
        self._logger.debug(f"Number of frames: {self.n_frames}")
        if self._weighted is not False:
            self._logger.debug(f"Total weight: {self._total_weight:.3f}")
            norm *= self._total_weight / self.n_frames

        # Normalize the histogram and save the results
        self.results.rdf = self.results.count / norm
        self._df = pd.DataFrame(
            {
                "bins": self.results.bins,
                "rdf": self.results.rdf,
                "count": self.results.count,
            }
        )

        self._logger.info(f"Finished RDF analysis for {self._tag}.")

    @property
    def cdf(self):
        """Calculate the cumulative distribution functions (CDF).
        Note that this is the actual count within a given radius, i.e.,
        :math:`N(r)`.
        Returns
        -------
              cdf : numpy array
                      a numpy array with the same structure as :attr:`rdf`
        .. versionadded:: 0.3.0
        """
        cdf = np.cumsum(self.results.count) / self.n_frames

        return cdf

    @staticmethod
    def _reduce(res, result_single_frame):
        """'add' action for an accumulator"""
        if isinstance(res, list) and len(res) == 0:
            # Convert res from an empty list to a numpy array
            # which has the same shape as the single frame result
            res = result_single_frame
        else:
            # Add two numpy arrays
            res += result_single_frame
        return res

    def figures(
        self, title: str = None, ext: str = "png"
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot the radial distribution function and the potential of mean
        force. The figures are saved to the `figures` directory in the
        `dir_out` directory. This is a wrapper for the `plt_rdf` and
        `plt_pmf` methods.

        This method should only be called after the analysis has been
        run.

        Parameters
        ----------
        title : str, optional
            The title of the plots.
        ext : str, optional
            The file extension of the saved figures. Default is "png".

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figures and axes of the plots.
        """
        self._logger.info(f"Plotting RDF analysis for {self._tag}.")

        figs = []
        axs = []

        fig, ax = self.plt_rdf(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        fig, ax = self.plt_pmf(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        self._logger.info(f"Finished plotting RDF analysis for {self._tag}.")
        return figs, axs

    def plt_rdf(self, title: str = None, ext: str = "png"):
        """
        Plot the radial distribution function. The figure is saved to the
        `figures` directory in the `dir_out` directory.

        This method should only be called after the analysis has been
        run.

        Parameters
        ----------
        title : str, optional
            The title of the plot.
        ext : str, optional
            The file extension of the saved figure. Default is "png".

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figure and axes of the plot.
        """
        self._logger.info(f"Plotting RDF for {self._tag}.")

        d = self._dir_out / "figures"
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(visible=True, which="both", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.8)

        # gather data
        x, y = self.results.bins / 10.0, self.results.rdf
        ax.plot(x, y)
        ax.set_xlabel("r [nm]")
        ax.set_ylabel("$g{(r)}$")
        if title is not None:
            ax.set_title(title, y=1.05)
        else:
            ax.set_title(self.label_system, y=1.05)

        self._logger.debug(f"Saving figure to {d}/plt_rdf_{self._tag}.{ext}.")
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{d}/plt_rdf_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        self._logger.info(f"Finished plotting RDF for {self._tag}.")
        return fig, ax

    def plt_pmf(self, title: str = None, ext: str = "png"):
        """
        Plot the potential of mean force. The figure is saved to the
        `figures` directory in the `dir_out` directory.

        This method should only be called after the analysis has been
        run.

        Parameters
        ----------
        title : str, optional
            The title of the plot.
        ext : str, optional
            The file extension of the saved figure. Default is "png".

        Returns
        -------
        tuple[plt.figure, plt.axes]
            The figure and axes of the plot.
        """
        self._logger.info(f"Plotting PMF for {self._tag}.")

        d = self._dir_out / "figures"
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(visible=True, which="both", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.8)

        # gather data
        x, y = self.results.bins / 10.0, self.results.rdf
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fes = -np.log(y)
        warnings.simplefilter("default", category=RuntimeWarning)
        # shift to zero at cutoff
        if not np.isnan(fes[-1]):
            fes -= fes[-1]
        else:
            fes -= np.nanmin(fes)
        ax.plot(x, fes)
        ax.set_xlabel("r [nm]")
        ax.set_ylabel(r"$\Delta F_\mathrm{PMF}$ [$k_B T$]")
        if title is not None:
            ax.set_title(title, y=1.05)
        else:
            ax.set_title(self.label_system, y=1.05)

        self._logger.debug(f"Saving figure to {d}/plt_rdf_pmf_{self._tag}.{ext}.")
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{d}/plt_rdf_pmf_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        self._logger.info(f"Finished plotting PMF for {self._tag}.")
        return fig, ax
