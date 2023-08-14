# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

"""
Dihedral angles analysis --- :mod:`colvar_parallel.dihedrals`
=====================================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.dihedrals`.

This module contains classes for calculating dihedral angles for a given set of
atoms or residues. This can be done for selected frames or whole trajectories.
A list of time steps that contain angles of interest is generated and can be
easily plotted if desired.


See Also
--------
:mod:`MDAnalysis.analysis.dihedrals`
:func:`MDAnalysis.lib.distances.calc_dihedrals()`
   function to calculate dihedral angles from atom positions


Example applications
--------------------

General dihedral analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~MDAnalysis.analysis.dihedrals.Dihedral` class is useful for calculating
angles for many dihedrals of interest. For example, we can find the phi angles
for residues 5-10 of adenylate kinase (AdK). The trajectory is included within
the test data files::

   import MDAnalysis as mda
   from MDAnalysisTests.datafiles import GRO, XTC
   u = mda.Universe(GRO, XTC)

   # selection of atomgroups
   ags = [res.phi_selection() for res in u.residues[4:9]]

   from MDAnalysis.analysis.dihedrals import Dihedral
   R = Dihedral(ags).run()


.. autoclass:: dihedrals
    :members:
    :inherited-members:

"""
# Standard library
from __future__ import absolute_import
from pathlib import Path
import sys
import warnings

# External dependencies
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import calc_dihedrals
import numpy as np
import pandas as pd

# Internal dependencies
from .base import ParallelAnalysisBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class Dihedral(ParallelAnalysisBase):
    """Calculate dihedral angles for specified atomgroups.

    Dihedral angles will be calculated for each atomgroup that is given for
    each step in the trajectory. Each :class:`~MDAnalysis.core.groups.AtomGroup`
    must contain 4 atoms.

    Note
    ----
    This class takes a list as an input and is most useful for a large
    selection of atomgroups. If there is only one atomgroup of interest, then
    it must be given as a list of one atomgroup.
    """

    def __init__(
        self,
        atomgroup: AtomGroup,
        monomer: str = None,
        label: str = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Set up the analysis.

        Parameters
        ----------
        atomgroup : AtomGroup
            AtomGroup to calculate dihedral angles for.
        monomer : str, optional
            Monomer unit of chains in the system. The default is ``None``.
        label : str, optional
            Text label for system.
        superposition : bool, optional
            ``True`` perform a RMSD-superposition, ``False`` only
            calculates the RMSD. The default is ``True``.
        verbose : bool, optional
            ``True``: print progress bar to the screen; ``False``: no
            progress bar. The default is ``False``.
        """
        super().__init__(
            atomgroup.universe.trajectory,
            (atomgroup,),
            label=label,
            verbose=verbose,
            **kwargs,
        )
        self.logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")

        # set dihedral angle atomgroups
        if monomer is None:
            self._set_asp_glu_dihedrals()
        elif monomer.lower() in ["asp", "glu"]:
            self._set_asp_glu_dihedrals()
        elif monomer.lower() == "acr":
            self._set_acr_dihedrals()
        else:
            raise ValueError(f"Monomer {monomer} not recognized.")

        # update atomgroups of parent class
        self._atomgroups = (self.ag1, self.ag2, self.ag3, self.ag4, self.ag5)
        # set dihedral angle number
        self._n_dihedrals = len(self.ag3)
        self._n_phi = len(self.ag1)
        self._n_psi = len(self.ag5)

        # output data
        self._dir_out: Path = Path(f"./mdanalysis_dihedrals")
        self._df_filename = f"dihedral_{self._tag}.parquet"
        self._logger.debug(f"df_filename: {self._df_filename}")
        self._df: pd.DataFrame = None

        # set output data structures
        self._columns: list[str] = ["frame", "time"]
        for i in range(self._n_phi):
            self._columns.append(f"phi_{i}")
        for i in range(self._n_psi):
            self._columns.append(f"psi_{i}")

        self._logger.info(f"Initialized Dihedral analysis for {self._tag}.")

    def _set_asp_glu_dihedrals(self) -> None:
        """
        Copying source code of MDAnalysis.analysis.dihedrals.Ramachandran
        """
        c_name = "C"
        n_name = "N"
        ca_name = "CA"

        # pylint: disable=protected-access
        residues = self._atomgroups[0].residues
        prev = residues._get_prev_residues_by_resid()
        nxt = residues._get_next_residues_by_resid()
        # pylint: enable=protected-access

        keep = np.array([r is not None for r in prev])
        keep = keep & np.array([r is not None for r in nxt])

        if not np.all(keep):
            warnings.warn(
                "Some residues in selection do not have " "phi or psi selections"
            )
        prev = sum(prev[keep])
        nxt = sum(nxt[keep])
        residues = residues[keep]

        # find n, c, ca
        keep_prev = [sum(r.atoms.names == c_name) == 1 for r in prev]
        rnames = [n_name, c_name, ca_name]
        keep_res = [all(sum(r.atoms.names == n) == 1 for n in rnames) for r in residues]
        keep_next = [sum(r.atoms.names == n_name) == 1 for r in nxt]

        # alright we'll keep these
        keep = np.array(keep_prev) & np.array(keep_res) & np.array(keep_next)
        prev = prev[keep]
        res = residues[keep]
        nxt = nxt[keep]

        rnames = res.atoms.names
        self.ag1 = prev.atoms[prev.atoms.names == c_name]
        self.ag2 = res.atoms[rnames == n_name]
        self.ag3 = res.atoms[rnames == ca_name]
        self.ag4 = res.atoms[rnames == c_name]
        self.ag5 = nxt.atoms[nxt.atoms.names == n_name]

    def _set_acr_dihedrals(self) -> None:
        # find previous and next residues for dihedral calculations
        residues = self._atomgroups[0].residues

        # pylint: disable=protected-access
        prev = residues._get_prev_residues_by_resid()
        nxt = residues._get_next_residues_by_resid()
        # pylint: enable=protected-access

        keep = np.array([r is not None for r in prev])
        keep = keep & np.array([r is not None for r in nxt])

        if not np.all(keep):
            warnings.warn(
                "Some residues in selection do not have " "phi or psi selections"
            )

        # drop None residues
        prev = sum(prev[keep])
        nxt = sum(nxt[keep])
        residues = residues[keep]

        # atoms in dihedrals
        c_name = "C"
        ca_name = "CA"

        # find atoms for dihedrals
        keep_prev = [sum(r.atoms.names == c_name) == 1 for r in prev]
        rnames = [c_name, ca_name]
        keep_res = [all(sum(r.atoms.names == n) == 1 for n in rnames) for r in residues]
        keep_next = [sum(r.atoms.names == c_name) == 1 for r in nxt]

        # alright we'll keep these
        keep = np.array(keep_prev) & np.array(keep_res) & np.array(keep_next)
        prev = prev[keep]
        res = residues[keep]
        nxt = nxt[keep]

        # atom groups for dihedrals
        self.ag1: AtomGroup = prev.atoms[prev.atoms.names == c_name]
        self.ag2: AtomGroup = res.atoms[res.atoms.names == ca_name]
        self.ag3: AtomGroup = res.atoms[res.atoms.names == c_name]
        self.ag4: AtomGroup = nxt.atoms[nxt.atoms.names == ca_name]
        self.ag5: AtomGroup = nxt.atoms[nxt.atoms.names == c_name]

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

        # gather the atomgroups and other information
        ag1, ag2, ag3, ag4, ag5 = self._atomgroups
        dim = ag1.dimensions
        n_phi = self._n_phi
        n_psi = self._n_psi

        results = np.empty(2 + n_phi + n_psi, dtype=np.float64)
        results.fill(np.nan)

        # get the current frame and time
        results[0] = ts.frame
        results[1] = ts.time

        # calculate the dihedrals
        phis = calc_dihedrals(
            ag1.positions, ag2.positions, ag3.positions, ag4.positions, box=dim
        )
        psis = calc_dihedrals(
            ag2.positions, ag3.positions, ag4.positions, ag5.positions, box=dim
        )

        # output the dihedrals
        rad2deg = 180.0 / np.pi
        results[2 : 2 + n_phi] = phis * rad2deg
        results[2 + n_phi :] = psis * rad2deg

        # return the results
        return results
