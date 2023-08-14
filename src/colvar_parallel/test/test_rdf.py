# pylint: disable=missing-docstring
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
from __future__ import absolute_import
from pathlib import Path
import sys

from dask.distributed import Client, LocalCluster
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

# Local internal dependencies
from colvar_parallel.rdf import InterRDF  # noqa: E402

# pylint: disable=invalid-name, redefined-outer-name, protected-access, unused-argument


@pytest.fixture(scope="module")
def scheduler() -> Client:
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    return client


@pytest.fixture(scope="module")
def u() -> mda.Universe:
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


@pytest.fixture(scope="module")
def sels(u) -> tuple[mda.AtomGroup, mda.AtomGroup]:
    s1 = u.select_atoms("name OD1 and resname ASP")
    s2 = u.select_atoms("name OD2 and resname ASP")
    return s1, s2


def test_nbins(u: mda.Universe, scheduler: Client):
    s1 = u.atoms[:3]
    s2 = u.atoms[3:]
    rdf = InterRDF(s1, s2, nbins=412).run()

    assert len(rdf.results.bins) == 412


def test_domain(u: mda.Universe, scheduler: Client):
    s1 = u.atoms[:3]
    s2 = u.atoms[3:]
    rmin, rmax = 1.0, 13.0
    rdf = InterRDF(s1, s2, domain=(rmin, rmax)).run()

    assert rdf.results.edges[0] == rmin
    assert rdf.results.edges[-1] == rmax


def test_count_sum(sels: tuple[mda.AtomGroup, mda.AtomGroup], scheduler: Client):
    # OD1 vs OD2
    # should see 577 comparisons in count
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    assert rdf.results.count.sum() == 577


def test_count(sels: tuple[mda.AtomGroup, mda.AtomGroup], scheduler: Client):
    # should see two distances with 7 counts each
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    assert len(rdf.results.count[rdf.results.count == 3]) == 7


def test_double_run(sels: tuple[mda.AtomGroup, mda.AtomGroup], scheduler: Client):
    # running rdf twice should give the same result
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    rdf.run()
    assert len(rdf.results.count[rdf.results.count == 3]) == 7


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 4])
def test_same_result(
    sels: tuple[mda.AtomGroup, mda.AtomGroup], n_blocks: int, scheduler: Client
):
    # should see same results from analysis.rdf and colvar_parallel.rdf
    s1, s2 = sels
    nrdf = rdf.InterRDF(s1, s2).run()
    prdf = InterRDF(s1, s2).run(n_blocks=n_blocks)
    assert_almost_equal(nrdf.results.count, prdf.results.count)
    assert_almost_equal(nrdf.results.rdf, prdf.results.rdf)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_trj_len(
    sels: tuple[mda.AtomGroup, mda.AtomGroup], step: int, scheduler: Client
):
    # should see same results from analysis.rdf and colvar_parallel.rdf
    s1, s2 = sels
    nrdf = rdf.InterRDF(s1, s2).run(step=step)
    prdf = InterRDF(s1, s2).run(step=step)
    assert_almost_equal(nrdf.n_frames, prdf.n_frames)
    assert_almost_equal(nrdf.results.rdf, prdf.results.rdf)


def test_cdf(sels: tuple[mda.AtomGroup, mda.AtomGroup], scheduler: Client):
    s1, s2 = sels
    rdf = InterRDF(s1, s2).run()
    cdf = np.cumsum(rdf.results.count) / rdf.n_frames
    assert_almost_equal(rdf.cdf[-1], rdf.results.count.sum() / rdf.n_frames)
    assert_almost_equal(rdf.cdf, cdf)


def test_reduce(sels: tuple[mda.AtomGroup, mda.AtomGroup], scheduler: Client):
    # should see numpy.array addition
    s1, s2 = sels
    rdf = InterRDF(s1, s2)
    res = []
    single_frame = np.array([np.array([1, 2]), np.array([3])], dtype=object)
    res = rdf._reduce(res, single_frame)
    res = rdf._reduce(res, single_frame)
    assert_almost_equal(res[0], np.array([2, 4]))
    assert_almost_equal(res[1], np.array([6]))


@pytest.mark.parametrize("exclusion_block, value", [(None, 577), ((1, 1), 397)])
def test_exclusion(
    sels: tuple[mda.AtomGroup, mda.AtomGroup],
    exclusion_block: tuple[int],
    value: int,
    scheduler: Client,
):
    # should see 397 comparisons in count when given exclusion_block
    # should see 577 comparisons in count when exclusion_block is none
    s1, s2 = sels
    rdf = InterRDF(s1, s2, exclusion_block=exclusion_block).run()
    assert rdf.results.count.sum() == value
