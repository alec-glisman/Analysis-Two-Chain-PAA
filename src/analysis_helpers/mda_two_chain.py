"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-09
Description: Functions to load, parse, and analyze topology and trajectory
data from the GROMACS output files generated during the MD simulations.
"""

# Standard library
import json
import os
from pathlib import Path
import sys
import warnings

# Third party packages
from joblib import cpu_count
import MDAnalysis as mda
from MDAnalysis import transformations as trans
import numpy as np

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Internal dependencies
from utils.logs import setup_logging  # noqa: E402
from utils.parsers import parse_tags  # noqa: E402


def find_top_trj(
    d_data: Path, fname: str = None, verbose: bool = False
) -> tuple[Path, Path]:
    """
    Find the topology and trajectory files for the simulation.

    Parameters
    ----------
    d_data : Path
        Path to the data directory
    fname : str, optional
        Name of the simulation, by default None
    verbose : bool, optional
        If True, the logging level is set to DEBUG, by default False

    Returns
    -------
    tuple[Path, Path]
        Path to the topology and trajectory files
    """
    log = setup_logging(verbose=verbose, log_file="find_top_trj.log")
    log.debug(f"Data directory: {d_data}")
    log.debug(f"Filename: {fname}")
    log.debug(f"Verbose: {verbose}")

    trj = d_data / f"{fname}.xtc"

    try:
        top = d_data / f"{fname}.tpr"
        if not top.exists():
            raise FileNotFoundError
    except FileNotFoundError as e:
        top = d_data / f"{fname}.pdb"
        if not top.exists():
            raise FileNotFoundError(f"Topology file not found: {top}") from e

    log.debug(f"Data directory: {d_data}")
    log.debug(f"Topology file: {top}")
    log.debug(f"Trajectory file: {trj}")

    # verify that the data files exist
    if not top.exists():
        raise FileNotFoundError(f"Topology file not found: {top}")
    elif not trj.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trj}")

    return top, trj


def load_universe(
    working_dir: str,
    data_dir: str,
    filename: str,
    tag: str,
    unwrap: bool = False,
    refresh: bool = False,
    verbose: bool = False,
) -> tuple[mda.Universe, dict, dict]:
    """
    Load a MDAnalysis Universe from a GROMACS trajectory file.

    Parameters
    ----------
    working_dir : str
        Path to the desired working directory for data output
    data_dir : str
        Path to the directory containing the GROMACS data files
    filename : str
        Name of the GROMACS trajectory file
    tag : str
        Unique tag for the data set
    refresh : bool, optional
        If True, the data is reloaded from the original data files,
        by default False
    unwrap : bool, optional
        If True, the trajectory is unwrapped, by default False
    verbose : bool, optional
        If True, the logging level is set to DEBUG, by default False

    Returns
    -------
    tuple[mda.Universe, dict, dict]
        MDAnalysis Universe, dictionary of system information, and
        dictionary of selection strings
    """
    log = setup_logging(verbose=verbose, log_file="load_mda_universe.log")

    # create mdanalysis main directory and change to it
    dir_mda = Path(f"{working_dir}/mdanalysis")
    dir_mda.mkdir(parents=True, exist_ok=True)
    os.chdir(f"{dir_mda}")

    # create data sub-directory and change to it
    dir_mda_data = Path(f"{dir_mda}/data")
    dir_mda_data.mkdir(parents=True, exist_ok=True)
    os.chdir(f"{dir_mda_data}")
    log.debug(f"Moved to directory: {dir_mda_data}")

    # parse system information
    monomer, system, title = parse_tags(data_dir, verbose=verbose)
    info_dict = {"monomer": str(monomer), "system": str(system), "title": str(title)}
    info_dict["filename"] = str(filename)
    info_dict["data_dir"] = str(data_dir)
    if monomer == "Acr":
        info_dict["polymer"] = "PAA"
    elif monomer == "Asp":
        info_dict["polymer"] = "PASA"
    elif monomer == "Glu":
        info_dict["polymer"] = "PGA"
    else:
        raise ValueError("Monomer not recognized")

    # set selection dictionary
    chain_ids = ["A", "B"]
    sel_dict = {}
    # solvent molecules
    sel_dict["water"] = "resname SOL"
    sel_dict["water_O"] = "resname SOL and name OW"
    sel_dict["water_H"] = "resname SOL and name HW1 HW2"
    # polymer atoms
    sel_dict[
        "pe"
    ] = f"protein or (resname LAI ACI RAI AI1 LAN ACN RAN AN1 LAC ACE RAC AC1 LAL ALC RAL AL1)"
    sel_dict["chain_A"] = f"(segid {chain_ids[0]}) and ({sel_dict['pe']})"
    sel_dict["chain_B"] = f"(segid {chain_ids[1]}) and ({sel_dict['pe']})"
    sel_dict["backbone"] = f"(name CA C N) and ({sel_dict['pe']})"
    sel_dict["backbone_chain_A"] = f"(name CA C N) and ({sel_dict['chain_A']})"
    sel_dict["backbone_chain_B"] = f"(name CA C N) and ({sel_dict['chain_B']})"
    sel_dict["C_alpha"] = f"(name CA) and ({sel_dict['pe']})"
    sel_dict["C_alpha_chain_A"] = f"(name CA) and ({sel_dict['chain_A']})"
    sel_dict["C_alpha_chain_B"] = f"(name CA) and ({sel_dict['chain_B']})"
    # monatomic ions
    sel_dict["Ca"] = "resname CA"
    sel_dict["Na"] = "resname NA"
    sel_dict["Cl"] = "resname CL"
    # carboxylate groups
    if monomer == "Acr":
        sel_co = "name CB"  # carboxylate carbon
        sel_os = "name OB1 OB2"  # carboxylate oxygen
    elif monomer == "Asp":
        sel_co = "name CG"
        sel_os = "name OD1 OD2 OC1 OC2"
    elif monomer == "Glu":
        sel_co = "name CD"
        sel_os = "name OE1 OE2 OC1 OC2"
    else:
        raise ValueError(f"Invalid monomer: {monomer}")
    # functional groups
    sel_dict["carboxy_O"] = f"({sel_os}) and ({sel_dict['pe']})"
    sel_dict["carboxy_O_chain_A"] = f"({sel_os}) and ({sel_dict['chain_A']})"
    sel_dict["carboxy_O_chain_B"] = f"({sel_os}) and ({sel_dict['chain_B']})"
    sel_dict["carboxy_C"] = f"({sel_co}) and ({sel_dict['pe']})"
    sel_dict["carboxy_C_chain_A"] = f"({sel_co}) and ({sel_dict['chain_A']})"
    sel_dict["carboxy_C_chain_B"] = f"({sel_co}) and ({sel_dict['chain_B']})"
    sel_dict["acetate_C"] = f"(name CG) and ({sel_dict['pe']})"
    sel_dict["acetate_C_chain_A"] = f"(name CG) and ({sel_dict['chain_A']})"
    sel_dict["acetate_C_chain_B"] = f"(name CG) and ({sel_dict['chain_B']})"
    sel_dict["acetate_O"] = f"(name OA) and ({sel_dict['pe']})"
    sel_dict["acetate_O_chain_A"] = f"(name OA) and ({sel_dict['chain_A']})"
    sel_dict["acetate_O_chain_B"] = f"(name OA) and ({sel_dict['chain_B']})"
    sel_dict["mainchain_O"] = f"(name O) and ({sel_dict['pe']})"
    sel_dict["mainchain_N"] = f"(name N) and ({sel_dict['pe']})"
    sel_dict["pe_O"] = f"({sel_dict['mainchain_O']}) or ({sel_dict['carboxy_O']})"
    # distance cutoffs
    gr_dist_ion_c_monodentate = 0.338 * 10  # Angstrom
    gr_dist_ion_c_bidentate = 0.456 * 10
    gr_dist_ion_o = 0.34 * 10
    sel_dict["gr_dist_ion_c_monodentate_ang"] = gr_dist_ion_c_monodentate
    sel_dict["gr_dist_ion_c_bidentate_ang"] = gr_dist_ion_c_bidentate
    sel_dict["gr_dist_ion_o_ang"] = gr_dist_ion_o
    sel_dict[
        "monodentate"
    ] = f"around {gr_dist_ion_c_monodentate} {sel_dict['carboxy_C']}"
    sel_dict["bidentate"] = f"around {gr_dist_ion_c_bidentate} {sel_dict['carboxy_C']}"
    sel_dict["adsorbed"] = f"around {gr_dist_ion_o} ({sel_dict['pe_O']})"
    sel_dict["adsorbed_chain_A"] = (
        f"around {gr_dist_ion_o} ({sel_dict['pe_O']} " + f"and {sel_dict['chain_A']})"
    )
    sel_dict["adsorbed_chain_B"] = (
        f"around {gr_dist_ion_o} ({sel_dict['pe_O']} " + f"and {sel_dict['chain_B']})"
    )
    sel_dict["bridging"] = (
        f"around {gr_dist_ion_o} "
        + f"({sel_dict['pe_O']} and {sel_dict['chain_A']})"
        + f" and around {gr_dist_ion_o} "
        + f"({sel_dict['pe_O']} and {sel_dict['chain_B']})"
    )
    # ions within distance cutoffs
    sel_dict["adsorbed_Ca"] = f"({sel_dict['Ca']}) and ({sel_dict['adsorbed']})"
    sel_dict["adsorbed_Na"] = f"({sel_dict['Na']}) and ({sel_dict['adsorbed']})"

    # find gromacs files
    log.info("Loading universe")
    f_top, f_trj = find_top_trj(data_dir, fname=filename, verbose=verbose)
    uni = mda.Universe(str(f_top), str(f_trj), refresh_offsets=refresh)
    log.debug(f"Loaded universe from {f_top} and {f_trj}")
    info_dict["f_top"] = str(f_top)
    info_dict["f_trj"] = str(f_trj)
    try:
        info_dict["n_frames"] = int(uni.trajectory.n_frames)
    except AttributeError:
        info_dict["n_frames"] = 0
    info_dict["n_atoms"] = int(uni.atoms.n_atoms)
    info_dict["n_residues"] = int(len(uni.residues))
    info_dict["residues"] = list(set(uni.atoms.residues.resnames))
    info_dict["n_segments"] = int(len(uni.segments))
    info_dict["segments"] = [str(seg) for seg in uni.segments.segments]
    try:
        info_dict["n_monomers"] = len(uni.select_atoms(sel_dict["chain_A"]).residues)
        info_dict["n_Ca"] = len(uni.select_atoms(sel_dict["Ca"]))
        info_dict["n_Na"] = len(uni.select_atoms(sel_dict["Na"]))
        info_dict["n_Cl"] = len(uni.select_atoms(sel_dict["Cl"]))
        info_dict["final_time_ns"] = float(uni.trajectory[-1].time / 1000.0)
    except AttributeError:
        info_dict["n_monomers"] = 0
        info_dict["n_Ca"] = 0
        info_dict["n_Na"] = 0
        info_dict["n_Cl"] = 0
        info_dict["final_time_ns"] = np.nan

    # get box size
    try:
        info_dict["box_size_nm"] = float(uni.dimensions[0] / 10.0)
    except AttributeError:
        info_dict["box_size_nm"] = np.nan

    info_dict["original_segments"] = [str(seg) for seg in uni.segments.segments]

    log.debug("System information")
    log.debug(f"Number of frames: {info_dict['n_frames']}")
    log.debug(f"Number of atoms: {info_dict['n_atoms']}")
    log.debug(f"Number of residues: {info_dict['n_residues']}")
    log.debug(f"Name of residues: {info_dict['residues']}")
    log.debug(f"Segment IDs: {info_dict['segments']}")
    log.debug(f"Box size: {info_dict['box_size_nm']:.2f} nm")
    log.debug(f"Number of monomers: {info_dict['n_monomers']}")
    log.debug(f"Number of Ca: {info_dict['n_Ca']}")
    log.debug(f"Number of Na: {info_dict['n_Na']}")
    log.debug(f"Number of Cl: {info_dict['n_Cl']}")
    log.info(f"Final trajectory time: {info_dict['final_time_ns']:.2f} ns")

    # set topology information
    log.info("Setting topology information")
    try:
        uni.segments.segments[0].segid = chain_ids[0]
        uni.segments.segments[1].segid = chain_ids[1]
        warnings.simplefilter("ignore", UserWarning)
        guessed_elements = mda.topology.guessers.guess_types(uni.atoms.names)
        uni.add_TopologyAttr("elements", guessed_elements)
        warnings.simplefilter("default", UserWarning)
    except IndexError:
        pass

    sel_chain_a, sel_chain_b = f"segid {chain_ids[0]}", f"segid {chain_ids[1]}"
    ag_chain_a, ag_chain_b = uni.select_atoms(sel_chain_a), uni.select_atoms(
        sel_chain_b
    )
    info_dict["n_atoms_chain_A"] = int(ag_chain_a.atoms.n_atoms)
    info_dict["n_atoms_chain_B"] = int(ag_chain_b.atoms.n_atoms)

    log.debug(f"Number of atoms in Chain A: {ag_chain_a.atoms.n_atoms}")
    log.debug(f"Number of atoms in Chain B: {ag_chain_b.atoms.n_atoms}")
    log.info("Done loading universe")

    # save selection dictionary and info dictionary
    with open(f"sel_dict_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(sel_dict, f, indent=4)
    with open(f"info_dict_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=4)

    # return to original directory
    os.chdir(working_dir)

    # unwrap trajectory and center chains in box
    if unwrap:
        log.info("Unwrapping trajectory")
        protein = uni.select_atoms(sel_dict["pe"])
        not_protein = uni.select_atoms(f"not ({sel_dict['pe']})")
        ncpu = cpu_count()
        transforms = [
            trans.unwrap(uni.atoms, max_threads=ncpu),
            trans.center_in_box(protein, center="geometry", max_threads=ncpu),
            trans.wrap(not_protein, max_threads=ncpu),
            trans.fit_rot_trans(protein, protein, weights="mass", max_threads=ncpu),
        ]
        uni.trajectory.add_transformations(*transforms)

    return uni, sel_dict, info_dict
