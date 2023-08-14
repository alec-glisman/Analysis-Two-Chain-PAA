# pylint: disable=missing-docstring
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import absolute_import, division, print_function
from pathlib import Path
import sys

from dask.distributed import Client, LocalCluster
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysisTests.datafiles import (
    PSF,
    DCD,
    contacts_villin_folded,
    contacts_villin_unfolded,
)
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_almost_equal,
)
import pytest

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from colvar_parallel.contacts import Contacts  # noqa: E402

# pylint: disable=invalid-name, redefined-outer-name, unused-argument, protected-access


@pytest.fixture(scope="module")
def universe() -> mda.Universe:
    return mda.Universe(PSF, DCD)


@pytest.fixture(scope="module")
def scheduler() -> Client:
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    return client


def soft_cut(ref, u, selA, selB, radius=4.5, beta=5.0, lambda_constant=1.8):
    """
    Reference implementation for testing
    """
    # reference groups A and B from selection strings
    refA, refB = ref.select_atoms(selA), ref.select_atoms(selB)

    # 2D float array, reference distances (r0)
    dref = distance_array(refA.positions, refB.positions)

    # 2D bool array, select reference distances that are less than the cutoff
    # radius
    mask = dref < radius

    # group A and B in a trajectory
    grA, grB = u.select_atoms(selA), u.select_atoms(selB)
    results = []

    for ts in u.trajectory:
        d = distance_array(grA.positions, grB.positions)
        r, r0 = d[mask], dref[mask]
        x = 1 / (1 + np.exp(beta * (r - lambda_constant * r0)))

        # average/normalize and append to results
        results.append((ts.time, x.sum() / mask.sum()))

    return np.asarray(results)


class TestContacts(object):
    sel_basic = "(resname ARG LYS) and (name NH* NZ)"
    sel_acidic = "(resname ASP GLU) and (name OE* OD*)"

    def _run_Contacts(
        self, universe, start=None, step=None, stop=None, n_blocks=1, **kwargs
    ):
        acidic = universe.select_atoms(self.sel_acidic)
        basic = universe.select_atoms(self.sel_basic)
        return Contacts(universe, (acidic, basic), radius=6.0, **kwargs).run(
            start=start, stop=stop, step=step, n_blocks=n_blocks
        )

    def test_startframe(self, universe: mda.Universe, scheduler: Client):
        """test_startframe: TestContactAnalysis1: start frame set to 0 (resolution of
        Issue #624)

        """
        CA1 = self._run_Contacts(universe)
        assert len(CA1._results) == universe.trajectory.n_frames

    def test_uneven_blocks(self, universe: mda.Universe, scheduler: Client):
        """Issue #140"""
        CA1 = self._run_Contacts(universe, n_blocks=3)
        assert len(CA1._results) == universe.trajectory.n_frames

    def test_end_zero(self, universe: mda.Universe, scheduler: Client):
        """test_end_zero: TestContactAnalysis1: stop frame 0 is not ignored"""
        cn = self._run_Contacts(universe, stop=0)
        assert len(cn._results) == 0

    def test_slicing(self, universe: mda.Universe, scheduler: Client):
        start, stop, step = 10, 30, 5
        CA1 = self._run_Contacts(universe, start=start, stop=stop, step=step)
        frames = np.arange(universe.trajectory.n_frames)[start:stop:step]
        assert len(CA1._results) == len(frames)

    @staticmethod
    def _is_any_closer(r, r0, dist=2.5):  # pylint: disable=unused-argument
        return np.any(r < dist)

    @staticmethod
    def _weird_own_method(r, r0):  # pylint: disable=unused-argument
        return "aaa"

    def test_own_method_no_array_cast(self, universe: mda.Universe, scheduler: Client):
        with pytest.raises(Exception):
            self._run_Contacts(universe, method=self._weird_own_method, stop=2)

    def test_non_callable_method(self, universe: mda.Universe, scheduler: Client):
        with pytest.raises(Exception):
            self._run_Contacts(universe, method=2, stop=2)
