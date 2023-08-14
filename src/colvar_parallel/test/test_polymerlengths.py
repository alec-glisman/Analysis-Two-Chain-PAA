# pylint: disable=missing-docstring
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import absolute_import
from pathlib import Path
import sys

from dask.distributed import Client, LocalCluster
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysisTests.datafiles import PSF, DCD
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import pytest

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from colvar_parallel.polymerlengths import PolymerLengths  # noqa: E402

# pylint: disable=invalid-name, redefined-outer-name, unused-argument, protected-access


@pytest.fixture(scope="module")
def universe() -> mda.Universe:
    return mda.Universe(PSF, DCD)


@pytest.fixture(scope="module")
def scheduler() -> Client:
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    return client


def radgyr(atomgroup: mda.AtomGroup, masses: np.ndarray, total_mass: float = None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates - center_of_mass) ** 2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:, [1, 2]], axis=1)  # sum over y and z
    sq_y = np.sum(ri_sq[:, [0, 2]], axis=1)  # sum over x and z
    sq_z = np.sum(ri_sq[:, [0, 1]], axis=1)  # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses * sq_rs, axis=1) / total_mass

    # square root and return
    return np.sqrt(rog_sq)


class RadiusOfGyration2(AnalysisBase):  # subclass AnalysisBase
    def __init__(self, atomgroup, verbose=True):
        """
        Set up the initial analysis parameters.
        """
        # must first run AnalysisBase.__init__ and pass the trajectory
        trajectory = atomgroup.universe.trajectory
        super(RadiusOfGyration2, self).__init__(trajectory, verbose=verbose)
        # set atomgroup as a property for access in other methods
        self.atomgroup = atomgroup
        # we can calculate masses now because they do not depend
        # on the trajectory frame.
        self.masses = self.atomgroup.masses
        self.total_mass = np.sum(self.masses)

    def _prepare(self):
        """
        Create array of zeroes as a placeholder for results.
        This is run before we begin looping over the trajectory.
        """
        # This must go here, instead of __init__, because
        # it depends on the number of frames specified in run().
        self.results = np.zeros((self.n_frames, 6))
        # We put in 6 columns: 1 for the frame index,
        # 1 for the time, 4 for the radii of gyration

    def _single_frame(self):
        """
        This function is called for every frame that we choose
        in run().
        """
        # call our earlier function
        rogs = radgyr(self.atomgroup, self.masses, total_mass=self.total_mass)
        # save it into self.results
        self.results[self._frame_index, 2:] = rogs
        # the current timestep of the trajectory is self._ts
        self.results[self._frame_index, 0] = self._ts.frame
        # the actual trajectory is at self._trajectory
        self.results[self._frame_index, 1] = self._trajectory.time

    def _conclude(self):
        """
        Finish up by calculating an average and transforming our
        results into a DataFrame.
        """
        # by now self.result is fully populated
        self.average = np.mean(self.results[:, 2:], axis=0)
        columns = [
            "Frame",
            "Time (ps)",
            "Radius of Gyration",
            "Radius of Gyration (x-axis)",
            "Radius of Gyration (y-axis)",
            "Radius of Gyration (z-axis)",
        ]
        self.df = pd.DataFrame(self.results, columns=columns)


class TestPolymerLengths(object):
    sel_poly = "protein or resname ACI LAI RAI ACN LAN RAN"
    label_poly = "polymer"
    label_system = "test"

    def _run_PolymerLengths(
        self, universe, start=None, step=None, stop=None, n_blocks=1, **kwargs
    ):
        polymer = universe.select_atoms(self.sel_poly)
        return PolymerLengths(polymer, self.label_system, **kwargs).run(
            start=start, stop=stop, step=step, n_blocks=n_blocks
        )

    def test_uneven_blocks(self, universe: mda.Universe, scheduler: Client):
        """Issue #140"""
        CA1 = self._run_PolymerLengths(universe, n_blocks=3)
        assert len(CA1._results) == universe.trajectory.n_frames

    def test_end_zero(self, universe: mda.Universe, scheduler: Client):
        """test_end_zero: TestContactAnalysis1: stop frame 0 is not ignored"""
        pl = self._run_PolymerLengths(universe, stop=0)
        assert len(pl._results) == 0

    def test_slicing(self, universe: mda.Universe, scheduler: Client):
        start, stop, step = 10, 30, 5
        CA1 = self._run_PolymerLengths(universe, start=start, stop=stop, step=step)
        frames = np.arange(universe.trajectory.n_frames)[start:stop:step]
        assert len(CA1._results) == len(frames)

    def test_mda_implementation(self, universe: mda.Universe, scheduler: Client):
        """test_mda_implementation: TestContactAnalysis1: compare with MDAnalysis"""
        CA1 = self._run_PolymerLengths(universe)
        CA2 = RadiusOfGyration2(universe.select_atoms(self.sel_poly))
        CA2.run()

        # compare frame numbers
        assert_almost_equal(CA1._results[:, 0], CA2.results[:, 0])
        # compare times in units of picoseconds
        assert_almost_equal(CA1._results[:, 1], CA2.results[:, 1])
        # compare Rg in units of Angstrom
        assert_almost_equal(CA1._results[:, 2], CA2.results[:, 2])
