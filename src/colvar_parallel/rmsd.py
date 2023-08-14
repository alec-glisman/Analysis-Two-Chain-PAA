# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

"""
Calculating Root-Mean-Square Deviations (RMSD) --- :mod:`colvar_parallel.rms`
=====================================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rms.RMSD`.

See Also
--------
:mod:`MDAnalysis.analysis.rms`


.. autoclass:: RMSD
    :members:
    :inherited-members:

"""
# Standard library
from __future__ import absolute_import
from pathlib import Path
import sys

# External dependencies
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd

# Internal dependencies
from .base import ParallelAnalysisBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class RMSD(ParallelAnalysisBase):
    r"""Parallel RMSD analysis.

    Optimally superimpose the coordinates in the
    :class:`~MDAnalysis.core.groups.AtomGroup` `mobile` onto `ref` for
    each frame in the trajectory of `mobile`, and calculate the time
    series of the RMSD. The single frame calculation is performed with
    :func:`MDAnalysis.analysis.rms.rmsd` (with ``superposition=True``
    by default).

    Attributes
    ----------
    rmsd : array
          Contains the time series of the RMSD as a `Tx3`
          :class:`numpy.ndarray` array with content ``[[frame, time
          (ps), RMSD (Å)], [...], ...]``, where `T` is the number of
          time steps selected in the :meth:`run` method.

    Parameters
    ----------
    mobile : AtomGroup
         atoms that are optimally superimposed on `ref` before
         the RMSD is calculated for all atoms. The coordinates
         of `mobile` change with each frame in the trajectory.
    ref : AtomGroup
         fixed reference coordinates
    superposition : bool, optional
         ``True`` perform a RMSD-superposition, ``False`` only
         calculates the RMSD. The default is ``True``.

    See Also
    --------
    MDAnalysis.analysis.rms.RMSD

    Notes
    -----
    If you use trajectory data from simulations performed under *periodic
    boundary conditions* then you **must make your molecules whole** before
    performing RMSD calculations so that the centers of mass of the mobile and
    reference structure are properly superimposed.

    Run the analysis with :meth:`RMSD.run`, which stores the results
    in the array :attr:`RMSD.rmsd`.

    The root mean square deviation :math:`\rho(t)` of a group of :math:`N`
    atoms relative to a reference structure as a function of time is
    calculated as:

    .. math::

        \rho(t) = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(\mathbf{x}_i(t)
                        - \mathbf{x}_i^{\text{ref}}\right)^2}

    The selected coordinates from `atomgroup` are optimally superimposed
    (translation and rotation) on the `reference` coordinates at each time step
    as to minimize the RMSD.

    At the moment, this class has far fewer features than its serial
    counterpart, :class:`MDAnalysis.analysis.rms.RMSD`.

    Examples
    --------
    In this example we will globally fit a protein to a reference
    structure. The example is a DIMS trajectory of adenylate kinase, which
    samples a large closed-to-open transition.

    The trajectory is included in the MDAnalysis test data files. The data in
    :attr:`RMSD.rmsd` is plotted with :func:`matplotlib.pyplot.plot`::

        import MDAnalysis
        from MDAnalysis.tests.datafiles import PSF, DCD, CRD
        mobile = MDAnalysis.Universe(PSF,DCD).atoms
        # reference closed AdK (1AKE) (with the default ref_frame=0)
        ref = MDAnalysis.Universe(PSF,DCD).atoms

        from colvar_parallel.rms import RMSD

        R = RMSD(mobile, ref)
        R.run()

        import matplotlib.pyplot as plt
        rmsd = R.rmsd.T[2]  # transpose makes it easier for plotting
        time = rmsd[0]
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.plot(time, rmsd,  label="all")
        ax.legend(loc="best")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel(r"RMSD ($\\AA$)")
        fig.savefig("rmsd_all_CORE_LID_NMP_ref1AKE.pdf")

    """

    def __init__(
        self, mobile, ref, label=None, superposition=True, verbose=False, **kwargs
    ):
        """Set up the analysis.

        Parameters
        ----------
        mobile : AtomGroup
            atoms that are optimally superimposed on `ref` before
            the RMSD is calculated for all atoms. The coordinates
            of `mobile` change with each frame in the trajectory.
        ref : AtomGroup
            fixed reference coordinates
        label : str, optional
            Text label for system.
        superposition : bool, optional
            ``True`` perform a RMSD-superposition, ``False`` only
            calculates the RMSD. The default is ``True``.
        verbose : bool, optional
            ``True``: print progress bar to the screen; ``False``: no
            progress bar. The default is ``False``.
        """
        universe = mobile.universe

        super().__init__(universe.trajectory, (mobile,), **kwargs)
        self.logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        self._ref_pos = ref.positions.copy()
        self._superposition = superposition

        # output data
        self._dir_out: Path = Path(f"./mdanalysis_rmsd")
        self._df_filename = f"rmsd_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        # set output data structures
        self.columns: list[str] = ["frame", "time", "rmsd"]

        self._logger.info(f"Initialized RMSD analysis for {self._tag}.")

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
            The frame index, simulation time, and RMSD from the reference
            structure.
        """
        # update MDA objects
        ts = self._universe.trajectory[idx_frame]

        results = np.empty(3, dtype=np.float64)
        results.fill(np.nan)

        # get the current frame and time
        results[0] = ts.frame
        results[1] = ts.time

        # get the current rmsd
        results[2] = rms.rmsd(
            self._atomgroups[0].positions,
            self._ref_pos,
            superposition=self._superposition,
        )

        # return the results
        return results

    def _conclude(self):
        """Finalize the results array and set the `rmsd` attribute."""
        # call parent method
        super()._conclude()

        # set the output data
        self.results.rmsd = self._results[:, 2]
