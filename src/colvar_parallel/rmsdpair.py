# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#

"""
Calculating All Pairwise Root-Mean-Square Deviations (RMSD) --- :mod:`colvar_parallel.rmsdpair`
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
    rmsd : matrix
          Contains the time series of the RMSD as a `TxT`
          :class:`numpy.ndarray` array, where `T` is the number of
          frames in the trajectory.
    times : array
          Contains the times of each frame in the trajectory as a
          :class:`numpy.ndarray` array. The shape of this array is
          `(T,)`.
    frames : array
            Contains the frame indices of each frame in the trajectory
            as a :class:`numpy.ndarray` array. The shape of this array
            is `(T,)`.

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
        self._dir_out: Path = Path(f"./mdanalysis_rmsdpair")
        self._df_filename = f"rmsd_matrix_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        # set output data structures
        self.columns: list[str] = ["frame", "time"]

        self._logger.info(f"Initialized RMSD analysis for {self._tag}.")

    def _prepare(self):
        """Prepare the analysis for execution."""
        self._logger.info(f"Preparing RMSD analysis for {self._tag}.")

        # add frames to columns list
        for ts in self._trajectory[self.start : self.stop : self.step]:
            self.columns.append(f"frame_{ts.frame}")

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
            The frame index, simulation time, and pairwise RMSD of structure
            at current frame to all other frames in the trajectory.
        """
        # update MDA objects
        ts = self._universe.trajectory[idx_frame]

        results = np.empty(2 + self.n_frames, dtype=np.float64)
        results.fill(0.0)
        atoms = self._atomgroups[0]

        # get the current frame and time
        results[0] = ts.frame
        results[1] = ts.time

        # current positions
        iframe = ts.frame
        i_ref = atoms.positions

        # diagonal entries need not be calculated due to metric(x,x) == 0 in
        # theory, _ts not updated properly.
        for j, _ in enumerate(self._trajectory[iframe : self.stop : self.step]):
            # j frame position
            j_ref = atoms.positions

            dist = rms.rmsd(i_ref, j_ref, superposition=self.superposition)

            results.dist_matrix[j + self._frame_index] = dist

        # return the results
        return results

    def _conclude(self):
        """
        Called after the run() method to finish everything up.

        This method is called by the run() method and should not be called
        directly by the user.
        """

        # calculate the rmsd matrix
        self.results.dist_matrix = self.results[:, 2:]
        # verify dist_matrix is upper triangular
        assert np.allclose(self.results.dist_matrix, np.triu(self.results.dist_matrix))
        # add transpose of dist_matrix to itself to get full matrix
        self.results.dist_matrix += self.results.dist_matrix.T
        # output the results
        self.results[:, 2:] = self.results.dist_matrix

        # call base class method
        super()._conclude()
