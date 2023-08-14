# pylint: disable=missing-docstring
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import absolute_import
from pathlib import Path
import sys

import dask
from dask.distributed import Client, LocalCluster
import MDAnalysis as mda
from MDAnalysisTests.datafiles import DCD, PSF
import numpy as np
import pytest

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from colvar_parallel.base import ParallelAnalysisBase  # noqa: E402

# pylint: disable=invalid-name, redefined-outer-name, protected-access, unused-argument


class NoneAnalysis(ParallelAnalysisBase):
    def __init__(self, atomgroup):
        universe = atomgroup.universe
        super().__init__(universe.trajectory, (atomgroup,))

    def _prepare(self):
        pass

    def _conclude(self):
        self.res = np.array(self._results, dtype=np.float64)

    def _single_frame(self, idx):  # pylint: disable=unused-argument
        ts = self._universe.trajectory[idx]
        return ts.frame


@pytest.fixture(scope="module")
def scheduler() -> Client:
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    return client


@pytest.fixture(scope="module")
def analysis() -> NoneAnalysis:
    u = mda.Universe(PSF, DCD)
    ana = NoneAnalysis(u.atoms)
    return ana


@pytest.mark.parametrize("n_jobs", (1, 2))
def test_all_frames(analysis: NoneAnalysis, scheduler: Client, n_jobs: int):
    analysis.run(n_jobs=n_jobs)
    u = mda.Universe(analysis._top, analysis._traj)
    assert len(analysis.res) == u.trajectory.n_frames


@pytest.mark.parametrize("n_jobs", (1, 2, 3, 4))
def test_sub_frames(analysis: NoneAnalysis, scheduler: Client, n_jobs: int):
    analysis.run(start=10, stop=50, step=10, n_jobs=n_jobs)
    np.testing.assert_almost_equal(analysis.res, [10, 20, 30, 40])


@pytest.mark.parametrize("n_jobs", (1, 2, 3))
def test_no_frames(analysis: NoneAnalysis, scheduler: Client, n_jobs: int):
    u = mda.Universe(analysis._top, analysis._traj)
    n_frames = u.trajectory.n_frames
    with pytest.warns(UserWarning):
        analysis.run(start=n_frames, stop=n_frames + 1, n_jobs=n_jobs)
    assert len(analysis.res) == 0
    np.testing.assert_equal(analysis.res, [])


def test_scheduler(analysis: NoneAnalysis, scheduler: Client):
    analysis.run()

    u = mda.Universe(analysis._top, analysis._traj)
    _ = u.trajectory.n_frames
    with pytest.warns(UserWarning):
        analysis.run(stop=2, n_blocks=4, n_jobs=2)


@pytest.mark.parametrize("n_blocks", np.arange(1, 11))
def test_nblocks(analysis: NoneAnalysis, scheduler: Client, n_blocks: int):
    analysis.run(n_blocks=n_blocks)
    assert len(analysis._results) == analysis.n_frames


def test_guess_nblocks(analysis: NoneAnalysis, scheduler: Client):
    with dask.config.set(scheduler="processes"):
        analysis.run(n_jobs=-1)
    assert len(analysis._results) == analysis.n_frames


def test_reduce(scheduler: Client):
    res = []
    u = mda.Universe(PSF, DCD)
    ana = NoneAnalysis(u.atoms)
    res = ana._reduce(res, [1])
    res = ana._reduce(res, [1])
    # Should see res become a list with 2 elements.
    assert res == [[1], [1]]
