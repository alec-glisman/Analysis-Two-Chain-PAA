# pylint: disable=missing-docstring
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

from __future__ import absolute_import, division, print_function
from pathlib import Path
import sys

from dask.distributed import Client, LocalCluster
import MDAnalysis
from MDAnalysisTests.datafiles import PSF, DCD
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from colvar_parallel.rmsd import RMSD  # noqa: E402

# pylint: disable=invalid-name, protected-access, unused-argument


class TestRMSD(object):
    @pytest.fixture()
    def universe(self) -> MDAnalysis.Universe:
        return MDAnalysis.Universe(PSF, DCD)

    @pytest.fixture(scope="module")
    def scheduler(self) -> Client:
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        client = Client(cluster)
        return client

    @pytest.fixture()
    def correct_values(self) -> list[list[float]]:
        return [[0, 1.0, 0], [49, 50.0, 4.68953]]

    @pytest.fixture()
    def correct_values_frame_5(self) -> list[list[float]]:
        return [[5, 6.0, 0.91544906]]

    def test_rmsd(self, universe: MDAnalysis.Universe, scheduler: Client):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(n_jobs=2)
        assert_array_equal(RMSD1._results.shape, (universe.trajectory.n_frames, 3))

    def test_rmsd_step(
        self,
        universe: MDAnalysis.Universe,
        correct_values: list[list[float]],
        scheduler: Client,
    ):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(step=49)
        assert_almost_equal(
            RMSD1._results,
            correct_values,
            4,
            err_msg="error: rmsd profile should match " + "test values",
        )

    def test_rmsd_single_frame(
        self,
        universe: MDAnalysis.Universe,
        correct_values_frame_5: list[list[float]],
        scheduler: Client,
    ):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(start=5, stop=6)
        assert_almost_equal(
            RMSD1._results,
            correct_values_frame_5,
            4,
            err_msg="error: rmsd profile should match " + "test values",
        )

    @pytest.mark.parametrize("n_blocks", [1, 2, 3])
    def test_rmsd_different_blocks(
        self, universe: MDAnalysis.Universe, n_blocks: int, scheduler: Client
    ):
        ca = universe.select_atoms("name CA")
        universe.trajectory.rewind()
        RMSD1 = RMSD(ca, ca).run(n_blocks=n_blocks)
        assert_array_equal(RMSD1._results.shape, [universe.trajectory.n_frames, 3])
