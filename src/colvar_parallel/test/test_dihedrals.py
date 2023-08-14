# pylint: disable=missing-docstring
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest

from dask.distributed import Client, LocalCluster
import MDAnalysis as mda
from MDAnalysisTests.datafiles import GRO, XTC, DihedralArray, DihedralsArray

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from colvar_parallel.dihedrals import Dihedral  # noqa: E402


# pylint: disable=invalid-name, protected-access, redefined-outer-name, unused-argument


@pytest.fixture(scope="module")
def scheduler():
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    return client


class TestDihedral(object):
    @pytest.fixture()
    def atomgroup(self):
        u = mda.Universe(GRO, XTC)
        ag = u.select_atoms("(resid 4 and name N CA C) or (resid 5 and name N)")
        return ag

    def test_dihedral(self, atomgroup, scheduler):
        dihedral = Dihedral(atomgroup).run()
        test_dihedral = np.load(DihedralArray)

        assert_almost_equal(
            dihedral._results[:, 4],
            test_dihedral.flatten(),
            5,
            err_msg="error: dihedral angles should " "match test values",
        )

    def test_dihedral_single_frame(self, atomgroup, scheduler):
        dihedral = Dihedral(atomgroup).run(start=5, stop=6)
        test_dihedral = [np.load(DihedralArray)[5]]

        assert_almost_equal(
            dihedral._results[0, 4],
            test_dihedral,
            5,
            err_msg="error: dihedral angles should " "match test vales",
        )
