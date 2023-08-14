"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-15
Description: Radius of gyration and end-to-end distance analysis

This module provides a class for calculating the radius of gyration
and end to end distance with Dask parallelization. The class
inherits from :class:`ParallelAnalysisBase`. The `run` method calculates
desired quantities. The `save` method saves the results to a parquet file.
The `figures` method plots the results.
"""

# Standard library
from pathlib import Path
import sys
import warnings

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st

# MDAnalysis inheritance
from MDAnalysis.core.groups import AtomGroup

# Internal dependencies
from .base import ParallelAnalysisBase
from stats.block_error import BlockError

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class PolymerLengths(ParallelAnalysisBase):
    """
    Polymer radius of gyration (Rg) and end-to-end distance (Ree) analysis.

    This class inherits from :class:`ParallelAnalysisBase`. The `run`
    method of this class calculates the collective variables.

    Note that the `run` method of this class allows for additional
    input parameters beyond the default in
    :class:`MDAnalysis.analysis.base.AnalysisBase`. These additional
    parameters are used for Dask parallelization.

    Additional methods are provided to plot the results as well as
    save the results to a parquet file.
    """

    def __init__(
        self, atomgroup: AtomGroup, label: str = None, verbose: bool = False, **kwargs
    ):
        """
        Initialize the polymer length analysis object.

        Parameters
        ----------
        atomgroup : AtomGroup
            AtomGroup for polymer chain.
        label : str, optional
            text label for system. default is none.
        verbose : bool, optional
            If True, print additional information. Default is False.
        **kwargs
            Additional keyword arguments for :class:`ParallelAnalysisBase`.
        """
        super().__init__(
            atomgroup.universe.trajectory,
            (atomgroup,),
            label=label,
            verbose=verbose,
            **kwargs,
        )
        self._logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        # output data
        self._dir_out: Path = Path(f"./mdanalysis_polymer_lengths")
        self._df_filename = f"pl_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        # set output data structures
        self._columns: list[str] = [
            "frame",
            "time",
            "rg",
            "rg_xx",
            "rg_xy",
            "rg_xz",
            "rg_yx",
            "rg_yy",
            "rg_yz",
            "rg_zx",
            "rg_zy",
            "rg_zz",
            "rg_lambda_x",
            "rg_lambda_y",
            "rg_lambda_z",
            "principle_rg_x",
            "principle_rg_y",
            "principle_rg_z",
            "ree",
            "ree_x",
            "ree_y",
            "ree_z",
        ]

        self._logger.info(f"Initialized polymer length analysis for {self._tag}.")

    def _single_frame(self, idx_frame: int) -> np.ndarray:
        """
        Analyze a single frame in the trajectory.

        Parameters
        ----------
        idx_frame : int
            The index of the current frame.

        Returns
        -------
        np.array
            The weight, the volume of the box, and the histogram of the
            pair distances at the current frame.
        """
        # update frame and atomgroup
        ts = self._universe.trajectory[idx_frame]
        ag = self._atomgroups[0]

        results = np.empty(len(self._columns), dtype=np.float64)
        results.fill(np.nan)

        # save frame index, time
        results[0] = ts.frame
        results[1] = ag.universe.trajectory.time

        # ANCHOR: Calculate radius of gyration
        # get distance from center of mass
        try:
            ri = ag.positions - ag.center_of_mass(unwrap=True)

            # apply periodic boundary conditions
            xdim, ydim, zdim = ag.universe.dimensions[:3]
            ri[:, 0] -= xdim * np.rint(ri[:, 0] / xdim)
            ri[:, 1] -= ydim * np.rint(ri[:, 1] / ydim)
            ri[:, 2] -= zdim * np.rint(ri[:, 2] / zdim)

        except ValueError:
            ri = ag.positions - ag.center_of_mass()

        # calculate mass weighted distance from center of mass
        ri_mass = ag.masses[:, None] * ri
        total_mass = np.sum(ag.masses)

        # calculate mass weighted radius of gyration tensor
        rg_sq_tensor = np.einsum("ij,ik->jk", ri_mass, ri, optimize=True)
        rg_sq_tensor /= total_mass

        # square root to get radius of gyration
        results[2] = np.sqrt(np.trace(rg_sq_tensor))

        # output radius of gyration tensor elements
        results[3:12] = rg_sq_tensor.flatten()

        # get eigenvalues of the tensor
        rg_sq_lambda, rg_sq_vector = np.linalg.eig(rg_sq_tensor)
        results[12:15] = rg_sq_lambda
        
        # output eigenvector of largest eigenvalue
        results[15:18] = rg_sq_vector[:, np.argmax(rg_sq_lambda)]

        # ANCHOR: Calculate end-to-end distance
        try:
            ree = ag.residues[0].atoms.center_of_mass(unwrap=True) - ag.residues[
                -1
            ].atoms.center_of_mass(unwrap=True)

            # apply periodic boundary conditions
            ree[0] -= xdim * np.rint(ree[0] / xdim)
            ree[1] -= ydim * np.rint(ree[1] / ydim)
            ree[2] -= zdim * np.rint(ree[2] / zdim)

        except ValueError:
            ree = (
                ag.residues[0].atoms.center_of_mass()
                - ag.residues[-1].atoms.center_of_mass()
            )

        results[18] = np.linalg.norm(ree)
        results[19:] = ree

        return results

    def figures(
        self, title: str = None, ext: str = "png"
    ) -> tuple[plt.figure, plt.axes]:
        """
        Plot the radial distribution function and the potential of mean
        force. The figures are saved to the `figures` directory in the
        `dir_out` directory. This is a wrapper for all plotting methods.

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
        self._logger.info(f"Plotting polymer length analysis for {self._tag}.")

        figs = []
        axs = []

        fig, ax = self.plt_rg_dyn(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        fig, ax = self.plt_rg_fes(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        fig, ax = self.plt_rg_kde(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        fig, ax = self.plt_ree_kde(title=title, ext=ext)
        figs.append(fig)
        axs.append(ax)

        self._logger.info(f"Finished plotting polymer length analysis for {self._tag}.")
        return figs, axs

    def plt_rg_dyn(self, title: str = None, ext: str = "png"):
        self._logger.info(f"Plotting Rg dynamics for {self._tag}")
        d = self._dir_out / f"figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(
            self._df["time"] / 1e3, self._df["rg"] / 10.0, "o", markersize=1, alpha=0.5
        )

        ax.set_xlabel(r"Time [ns]")
        ax.set_ylabel(r"$R_g$ [nm]")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{d}/plt_rg_dyn_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        self._logger.debug("Saved figure")

        return fig, ax

    def plt_rg_fes(self, title: str = None, ext: str = "png"):
        self._logger.info(f"Plotting Rg free energy surface for {self._tag}")
        d = self._dir_out / f"figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # gather data
        data = self._df["rg"].values / 10.0  # nm
        if "weight" in self._df.columns:
            weights = self._df["weight"].values
        else:
            weights = np.ones_like(self._df["rg"].values)
            warnings.warn("No weights found. Using uniform weights.")

        block_sizes = np.linspace(50, len(data) // 30 + 2, 10, dtype=int)
        block_error = BlockError(block_sizes, data, weights=weights)
        centers, fes, _, fes_error, _ = block_error.x_histo_fes(
            bins=80, return_edges=True
        )
        x, y, yerr = centers, fes[-1], fes_error[-1]

        # set minimal value to 0
        y -= np.nanmin(y)

        ax.plot(x, y, label="FES")
        ax.fill_between(
            x, y - 1.96 * yerr, y + 1.96 * yerr, alpha=0.5, label=r"95\% CI"
        )

        ax.set_xlabel(r"$R_g$ [nm]")
        ax.set_ylabel(r"$\Delta F$ [k$_\mathrm{B} T$]")
        ax.set_ylim(-0.1, 10.1)
        ax.legend(loc="best")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{d}/plt_rg_fes_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        self._logger.debug("Saved figure")

        return fig, ax

    def plt_rg_kde(self, title: str = None, ext: str = "png"):
        self._logger.info(f"Plotting Rg distribution for {self._tag}")
        d = self._dir_out / "figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        self._logger.debug("Created figure")

        try:
            # gather data
            data = self._df["rg"].values / 10.0  # nm
            if "weight" in self._df.columns:
                weights = self._df["weight"].values
            else:
                weights = np.ones_like(self._df["rg"].values)
                warnings.warn("No weights found. Using uniform weights.")

            kde_obj = st.gaussian_kde(data, weights=weights)
            pdf_range = np.linspace(0.99 * max(data), 1.01 * min(data), 1000)
            pdf_estimate = kde_obj.evaluate(pdf_range)
            ax.plot(pdf_range, pdf_estimate)

        except np.linalg.LinAlgError:
            warnings.warn("KDE failed, skipping")

        ax.set_xlabel(r"$R_g$ [nm]")
        ax.legend(labels=[r"$R_g$ [nm]"], loc="best")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{d}/plt_rg_kde_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        self._logger.debug("Saved figure")

        return fig, ax

    def plt_ree_kde(self, title: str = None, ext: str = "png"):
        self._logger.info(f"Plotting Ree distribution for {self._tag}")
        d = self._dir_out / f"figures"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        try:
            # gather data
            data = self._df["ree"].values / 10.0  # nm
            if "weight" in self._df.columns:
                weights = self._df["weight"].values
            else:
                weights = np.ones_like(self._df["rg"].values)
                warnings.warn("No weights found. Using uniform weights.")

            kde_obj = st.gaussian_kde(data, weights=weights)
            pdf_range = np.linspace(0.99 * max(data), 1.01 * min(data), 1000)
            pdf_estimate = kde_obj.evaluate(pdf_range)
            ax.plot(pdf_range, pdf_estimate)

        except np.linalg.LinAlgError:
            warnings.warn("KDE failed, skipping")

        ax.set_xlabel(r"$R_{ee}$ [nm]")
        ax.legend(labels=[r"$R_{ee}$ [nm]"], loc="best")

        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{d}/plt_ree_kde_{self._tag}.{ext}", dpi=300, bbox_inches="tight")
        self._logger.debug("Saved figure")

        return fig, ax
