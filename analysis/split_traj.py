"""
This script will split the trajectory into associated and disassociated states
based on the plumed.dat file.

The states will be saved in separate trajectory files. Trajectories will be
output in the mda_trajs directory. The coordinates will be unwrapped and the
chains will be centered in the box.

Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-06-26
"""

# Standard library
import logging
import os
from pathlib import Path
import sys
import warnings

# Third party packages
import MDAnalysis as mda
from MDAnalysis import transformations as trans
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# get absolute path to file's parent directory
dir_proj_base = Path(os.getcwd()).resolve().parents[1]
sys.path.insert(0, f"{dir_proj_base}/src")

# Internal dependencies
from analysis_helpers.mda_two_chain import find_top_trj  # noqa: E402
from analysis_helpers.plumed_two_chain import plumed_df  # noqa: E402
from figures.style import set_style  # noqa: E402


def split_traj(
    data_dir: str,
    filename: str,
    df_plumed: pd.DataFrame,
    dist_min: float = 0.0,
    dist_max: float = np.inf,
    refresh: bool = True,
):
    logger = logging.getLogger()
    chain_ids = ["A", "B"]

    # find gromacs files
    try:
        f_top, f_trj = find_top_trj(data_dir, fname=filename)
    except TypeError as e:
        logger.error(e)
        print(f"logger = {logger}")
        print(f"data_dir = {data_dir}")
        print(f"filename = {filename}")
        raise e

    f_path = f_top.parent
    fname = f_top.stem

    # load MDA universe from topology and trajectory files
    wd = os.getcwd()
    filename_assoc = Path(
        f"{f_path}/{fname}_d12_{dist_min:.2f}_min_{dist_max:.2f}_max_center_chainA.xtc"
    )

    if not filename_assoc.exists() or refresh:
        print(f" - Topology file: {f_top}")
        print(f" - Trajectory file: {f_trj}")
        print(f" - Output file: {filename_assoc}")
        uni = mda.Universe(str(f_top), str(f_trj), verbose=True)

        # set topology information
        uni.segments.segments[0].segid = chain_ids[0]
        uni.segments.segments[1].segid = chain_ids[1]
        warnings.simplefilter("ignore", UserWarning)
        guessed_elements = mda.topology.guessers.guess_types(uni.atoms.names)
        uni.add_TopologyAttr("elements", guessed_elements)
        warnings.simplefilter("default", UserWarning)

        # check that there are atoms in the PE chain selection
        if uni.select_atoms(f"segid {chain_ids[0]}").atoms.n_atoms == 0:
            logger.warning("No atoms found in original PE chain selection")
            chain_ids = [uni.segments.segments[0].segid, uni.segments.segments[1].segid]

        # select protein and PE atoms
        sel_pe = f"protein or (resname LAI ACI RAI AI1 LAN ACN RAN AN1 LAC ACE RAC AC1 LAL ALC RAL AL1)"

        # select non-chain A PE atoms
        ag_not_pe_a = uni.select_atoms(
            f"not ({sel_pe} and segid {chain_ids[0]}) and not resname SOL"
        )

        # select chain A and B atoms
        sel_chain_a, sel_chain_b = f"segid {chain_ids[0]}", f"segid {chain_ids[1]}"
        ag_chain_a = uni.select_atoms(sel_chain_a)
        ag_chain_b = uni.select_atoms(sel_chain_b)

        # unwrap and center chains in box
        print(" - Unwrapping and centering chains in box")
        transforms = [
            # unwrap chains and ions
            trans.unwrap(ag_chain_a, max_threads=16),
            trans.unwrap(ag_chain_b, max_threads=16),
            # center chains in box
            trans.center_in_box(ag_chain_b, wrap=True, max_threads=16),
            trans.center_in_box(ag_chain_a, wrap=True, max_threads=16),
            # rewrap non-chain A PE atoms
            trans.wrap(ag_not_pe_a, compound="residues", max_threads=16),
        ]
        uni.trajectory.add_transformations(*transforms)

        # Write trajectories (if they do not already exist) and split into
        output_dir = Path(f"{wd}/mda_trajs")
        output_dir.mkdir(parents=True, exist_ok=True)
        ag_selection = uni.atoms

        # associated states
        if (not filename_assoc.exists()) or refresh:
            n_frames = uni.trajectory.n_frames
            print(
                f" - Writing trajectories for {ag_selection.n_atoms} atoms and {n_frames} frames"
            )
            with mda.Writer(f"{filename_assoc}", ag_selection.n_atoms) as wr:
                with logging_redirect_tqdm():
                    for _ in tqdm(
                        uni.trajectory,
                        desc="Writing snapshots",
                        total=n_frames,
                        unit="frames",
                    ):
                        idx_time = np.argmin(
                            np.abs(df_plumed["time"] - uni.trajectory.time)
                        )
                        time_plumed = df_plumed["time"][idx_time]
                        time_mda = uni.trajectory.time

                        if abs(time_plumed - time_mda) > 1e-4:
                            warnings.warn(
                                f"Plumed and MDA times do not match: {time_plumed} != {time_mda}"
                            )
                            warnings.warn(
                                f"Skipping frame {uni.trajectory.frame} at time {time_mda}"
                            )

                        chain_dist = df_plumed.iloc[idx_time]["d12"]
                        if (dist_min <= chain_dist) and (chain_dist <= dist_max):
                            wr.write(ag_selection)

    # load separate trajectories for associated and disassociated states
    print(" - Loading separate trajectories for associated and disassociated states")
    uni_assoc = mda.Universe(f_top, str(filename_assoc))
    uni_assoc.segments.segments[0].segid = chain_ids[0]
    uni_assoc.segments.segments[1].segid = chain_ids[1]

    return uni_assoc


if __name__ == "__main__":
    cwd = os.getcwd()
    set_style()
    dir_out = Path(f"{cwd}/output")

    # system information
    TEMPERATURE_K: float = 300  # [K] # system temperature
    KB = 8.314462618e-3  # [kJ/mol/K]

    # MDA variables
    START = 0
    STEP = 1
    T_STATIONARY_NS = 0.0
    VERBOSE = True
    RELOAD_DATA = False

    # range of associated FES basin
    assoc_min_max_dists = np.array(
        [
            [0.00000000, 1.84892446],
            [1.18859430, 1.29364682],
            [0.33016508, 0.44722361],
            [0.83141571, 0.94847424],
            [0.50125063, 0.60930465],
            [1.23661831, 1.40770385],
        ],
        dtype=np.float64,
    )
    # range of plateau in FES basin
    dissoc_min_max_dists = np.array(
        [
            [4.9, 5.0],
            [4.9, 5.0],
            [4.9, 5.0],
            [4.9, 5.0],
            [0.97548774, 1.09854927],
            [4.9, 5.0],
        ],
        dtype=np.float64,
    )

    tags = [
        "2PAcr-16mer-0Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_10071-idx_00",
        "2PAcr-16mer-8Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_5-idx_01",
        "2PAcr-16mer-16Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_5-idx_02",
        "2PAcr-16mer-32Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_6-idx_03",
        "2PAcr-16mer-64Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_6-idx_04",
        "2PAcr-16mer-128Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_23-idx_05",
    ]

    base_dir = Path(
        Path.home() /
        "Data/1-electronic-continuum-correction/5-ECC-two-chain-PMF/2_production_completed/polyacrylate-homopolymer"
    )

    plumed_append_dir = Path("replica_00/2-trajectory-concatenation")

    # mda_filename = "no_sol_pbc_chains"
    # mda_append_dir = Path("replica_00/3-trajectory-cleaning")
    mda_filename = "nvt_hremd_prod_scaled"
    mda_append_dir = Path("replica_00/2-trajectory-concatenation")

    sub_dirs = [
        "sjobid_10071-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-0Ca-12.0nmbox-ion_pe_all_scaledEStatics-node01-2023-01-27-08:32:21.543949375/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax",
        "sjobid_5-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-8Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:49:35.421512543/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax",
        "sjobid_5-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-16Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:49:37.438178317/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax",
        "sjobid_6-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-32Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:51:09.363289554/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax",
        "sjobid_6-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-64Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:51:11.383105790/3-hremd-prod-24_replicas-100_steps-300_Kmin-440_Kmax",
        "sjobid_23-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-128Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-02-10-16:44:16.675883557/3-hremd-prod-24_replicas-100_steps-300_Kmin-440_Kmax",
    ]

    plumed_data_dirs = [base_dir / d / plumed_append_dir for d in sub_dirs]
    mda_data_dirs = [base_dir / d / mda_append_dir for d in sub_dirs]

    # check that base dir is mounted
    if not base_dir.exists():
        raise FileNotFoundError("Base directory not mounted")

    # check that all data_dirs exist
    for d in plumed_data_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Data directory not found: {d}")
    for d in mda_data_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Data directory not found: {d}")

    # set up logging
    log = logging.getLogger(__name__)
    handler = logging.FileHandler(
        "mdanalysis_split_trajs.log", mode="a", encoding="utf-8"
    )

    if VERBOSE:
        log.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)
        handler.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        fmt="%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : "
        + "%(lineno)d : Log : %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S",
    )
    handler.setFormatter(formatter)

    if not log.hasHandlers():
        log.addHandler(handler)

    # create plumed main directory and change to it
    dir_plumed = Path(f"{dir_out}/plumed")
    dir_plumed.mkdir(parents=True, exist_ok=True)

    # create plumed data sub-directory and change to it
    dir_plumed_data = Path(f"{dir_plumed}/data")
    dir_plumed_data.mkdir(parents=True, exist_ok=True)
    os.chdir(f"{dir_plumed_data}")

    # load plumed data and save in data sub-directory
    dfs_plumed = []
    for d, tag in zip(plumed_data_dirs, tags):
        df = plumed_df(
            dir_out,
            d,
            tag,
            KB * TEMPERATURE_K,
            refresh=True,
            t_ps_remove=0,
            verbose=VERBOSE,
        )
        dfs_plumed.append(df)

    # return to original directory
    os.chdir(dir_out)

    # split all trajectories into association states
    for row in tqdm(range(assoc_min_max_dists.shape[0])):
        # print current tag
        print(f"Processing {tags[row]} ({row+1}/{len(tags)})")

        # get min and max distances
        assoc_min_dist = assoc_min_max_dists[row, 0]
        assoc_max_dist = assoc_min_max_dists[row, 1]
        dissoc_min_dist = dissoc_min_max_dists[row, 0]
        dissoc_max_dist = dissoc_min_max_dists[row, 1]
        print(
            f" - Association distance range: {assoc_min_dist} to {assoc_max_dist} [nm]"
        )
        print(
            f" - Dissociation distance range: {dissoc_min_dist} to {dissoc_max_dist} [nm]"
        )

        # split trajectory into association and dissociation states
        split_traj(
            mda_data_dirs[row],
            mda_filename,
            dfs_plumed[row],
            dist_min=assoc_min_dist,
            dist_max=assoc_max_dist,
            refresh=RELOAD_DATA,
        )
        split_traj(
            mda_data_dirs[row],
            mda_filename,
            dfs_plumed[row],
            dist_min=dissoc_min_dist,
            dist_max=dissoc_max_dist,
            refresh=RELOAD_DATA,
        )
