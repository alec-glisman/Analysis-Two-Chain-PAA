"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-16
Description: Native contacts calculation.

This module calculates the coordination number for each reference atom
with respect to a set of coordinating atoms with Dask parallelization.
The class inherits from :class:`ParallelAnalysisBase`. The `run` method
calculates desired quantities. The `save` method saves the results to a
parquet file. The `figures` method plots the results.

The `_single_frame` is similar to the method of the same name in
the default :class:`MDAnalysis.analysis.contacts.Contacts` class. I added
functionality for weighted frames, center of mass distributions, and
excluding frames that do not meet other input boolean criteria.
"""

# Standard library
import functools
from pathlib import Path
import sys
import warnings

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# MDAnalysis inheritance
from MDAnalysis.core.groups import AtomGroup, UpdatingAtomGroup
from MDAnalysis.core.universe import Universe
from MDAnalysis.analysis.contacts import hard_cut_q, radius_cut_q, soft_cut_q
from MDAnalysis.lib.distances import distance_array

# Internal dependencies
from .base import ParallelAnalysisBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402

# pylint: disable=invalid-name


def coordination_rational(
    r: float,
    r0: float,
    d0: float = 0.0,
    n: int = 6,
    m: int = 0,
) -> float:
    """
    Calculate the coordination number for a given distance r, with
    characteristic distance r0, using a rational switching function.
    The function is defined as:
    .. math::
        C(r) = \\frac{1 - \frac{r - d_0}{r_0})^n}{1 - \frac{r - d_0}{r_0})^m}

    Parameters
    ----------
    r : float
        The distance at which to calculate the coordination number
    r0 : float
        The characteristic distance.
    d0 : float, optional
        Offset for the distance. Default is 0.0
    n : int, optional
        The power to which the distance is raised in the numerator.
        Default is 6.
    m : int, optional
        The power to which the distance is raised in the denominator.
        Default is 0, which is changed to 2 * n.

    Returns
    -------
    float
        The coordination number at distance r
    """
    m = 2 * n if m == 0 else m
    r_dimless = (r - d0) / r0
    return (1.0 - np.power(r_dimless, n)) / (1.0 - np.power(r_dimless, m))


class Contacts(ParallelAnalysisBase):
    """Calculate contacts based observables.
    The standard methods used in this class calculate the number of native
    contacts *Q* from a trajectory.
    .. rubric:: Contact API
    By defining your own method it is possible to calculate other observables
    that only depend on the distances and a possible reference distance. The
    **Contact API** prescribes that this method must be a function with call
    signature ``func(r, r0, **kwargs)`` and must be provided in the keyword
    argument `method`.

    Attributes
    ----------
    results.timeseries : numpy.ndarray
        2D array containing *Q* for all refgroup pairs and analyzed frames
    timeseries : numpy.ndarray
        Alias to the :attr:`results.timeseries` attribute.
        .. deprecated:: 2.0.0
           Will be removed in MDAnalysis 3.0.0. Please use
           :attr:`results.timeseries` instead.
    .. versionchanged:: 1.0.0
       ``save()`` method has been removed. Use ``np.savetxt()`` on
       :attr:`Contacts.results.timeseries` instead.
    .. versionchanged:: 1.0.0
        added ``pbc`` attribute to calculate distances using PBC.
    .. versionchanged:: 2.0.0
       :attr:`timeseries` results are now stored in a
       :class:`MDAnalysis.analysis.base.Results` instance.
    .. versionchanged:: 2.2.0
       :class:`Contacts` accepts both AtomGroup and string for `select`
    """

    def __init__(
        self,
        u: Universe,
        select: tuple[AtomGroup, AtomGroup],
        method: str = "hard_cut",
        radius: float = 4.5,
        label: str = None,
        verbose: bool = False,
        kwargs: dict = None,
        **basekwargs,
    ):
        """
        Parameters
        ----------
        u : Universe
            trajectory
        select : tuple(AtomGroup, AtomGroup) | tuple(string, string)
            two contacting groups that change over time
        method : string | callable (optional)
            Can either be one of ``["hard_cut" , "soft_cut", "radius_cut", "rational", "6_12"]``
            or a callable with call signature ``func(r, r0, **kwargs)`` (the "Contacts API").
        radius : float, optional (4.5 Angstroms)
            radius within which contacts exist in refgroup
        label : str, optional
            text label for system. default is none.
        verbose : bool (optional)
            Show detailed progress of the calculation if set to ``True``; the
            default is ``False``.
        kwargs : dict, optional
            dictionary of additional kwargs passed to `method`. Check
            respective functions for reasonable values.

        Notes
        -----
        .. versionchanged:: 1.0.0
           Changed `selection` keyword to `select`
        """
        super().__init__(
            u.trajectory, select, label=label, verbose=verbose, **basekwargs
        )
        self._logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        self._fraction_kwargs = kwargs if kwargs is not None else {}

        if method == "hard_cut":
            self._fraction_contacts = hard_cut_q
        elif method == "soft_cut":
            self._fraction_contacts = soft_cut_q
        elif method == "radius_cut":
            self._fraction_contacts = functools.partial(radius_cut_q, radius=radius)
        elif method == "rational":
            self._fraction_contacts = functools.partial(coordination_rational)
        elif method == "6_12":
            self._fraction_contacts = functools.partial(
                coordination_rational, n=6, m=12
            )
        else:
            if not callable(method):
                raise ValueError("method has to be callable")
            self._fraction_contacts = method

        self._select = select
        # ag1 is the coordinating group that can change over time
        # ag2 is the reference group that stays the same
        self._ag1 = self._get_atomgroup(u, select[0], updating=True)
        self._ag2 = self._get_atomgroup(u, select[1], updating=False)
        self._radius = radius

        # output data
        self._dir_out: Path = Path(f"./mdanalysis_contacts")
        self._df_filename = f"contact_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df = None
        self._columns = ["frame", "time"]
        for i in range(len(self._ag2)):
            self._columns.append(f"ag2_{i}")
        for i in range(len(self._ag2)):
            self._columns.append(f"bridge_ag2_{i}")

    @staticmethod
    def _get_atomgroup(u, sel, updating=False):
        select_error_message = (
            "selection must be either string or a "
            "static AtomGroup. Updating AtomGroups "
            "are not supported."
        )
        if isinstance(sel, str):
            return u.select_atoms(sel)
        elif isinstance(sel, AtomGroup):
            if isinstance(sel, UpdatingAtomGroup) and not updating:
                raise TypeError(select_error_message)
            else:
                return sel
        else:
            raise TypeError(select_error_message)

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
        # update MDA objects
        ts = self._universe.trajectory[idx_frame]
        ag1, ag2 = self._atomgroups

        n1, n2 = ag1.n_atoms, ag2.n_atoms
        func = np.vectorize(self._fraction_contacts)
        cutoff = self._radius

        # prepare results array
        results = np.empty(2 + 2 * n2, dtype=np.float64)
        results.fill(np.nan)

        # save simulation frame and time
        results[0] = ts.frame
        results[1] = ag1.universe.trajectory.time

        # compute distance array for a frame
        # Note: rows are group 1 (coordinating) atoms, columns are group 2 (reference) atoms
        dists = distance_array(
            ag1.positions,
            ag2.positions,
            box=ts.dimensions,
        )
        # apply cutoff and convert to fractional contacts
        d = func(dists, cutoff, **self._fraction_kwargs)

        # calculate coordination number of reference group (sum over rows)
        start = 2
        stop = start + n2
        results[start:stop] = np.sum(d, axis=0).flatten()

        # calculate if reference group is coordinated by first half of group 1
        # and second half of group 1, which makes it a bridge
        # REVIEW: this is highly system specific analysis and should not be
        #         part of the main functionality
        coord_by_first_half = np.sum(d[: n2 // 2, :], axis=0) > 0.33
        coord_by_second_half = np.sum(d[n2 // 2 :, :], axis=0) > 0.33
        coord_by_both_halves = np.logical_and(coord_by_first_half, coord_by_second_half)

        start = stop
        stop = start + n2
        results[start:stop] = coord_by_both_halves.astype(int).flatten()

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
        self._logger.info(f"Plotting contact number analysis for {self._tag}.")

        figs = []
        axs = []

        warnings.warn("This method has not been implemented yet.")

        self._logger.info(
            f"Finished plotting contact numbers analysis for {self._tag}."
        )
        return figs, axs
