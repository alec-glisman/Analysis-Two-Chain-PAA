"""
| Author: Alec Glisman (GitHub: @alec-glisman)
| Date: 2021-08-31
| Description: This script generates data for the 2-chain system using MDAnalysis
and Plumed.

This script is designed to be run from the command line. The following
arguments are required:
| -d, --dir: Base directory for the input data
| -f, --fname: File name for the input data
| -t, --tag: Subdirectory tag for the output data

Multiple simulations can be analyzed by running this script multiple times
and they can be analyzed in parallel by running multiple instances of this
script at the same time.
"""

# Standard library
import argparse
import logging
import os
from pathlib import Path
import time
import sys

# External dependencies
import dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from analysis_helpers.mda_two_chain import load_universe  # noqa: E402
from analysis_helpers.plumed_two_chain import plumed_df, plumed_plots  # noqa: E402
from colvar_parallel.contactmatrix import ContactMatrix  # noqa: E402
from colvar_parallel.contacts import Contacts  # noqa: E402
from colvar_parallel.dihedrals import Dihedral  # noqa: E402
from colvar_parallel.hbonds import HydrogenBondAnalysis  # noqa: E402
from colvar_parallel.rdf import InterRDF  # noqa: E402
from colvar_parallel.polymerlengths import PolymerLengths  # noqa: E402
from figures.style import set_style  # noqa: E402
from utils.logs import setup_logging  # noqa: E402
from utils.parsers import parse_tags  # noqa: E402

# ANCHOR: Script variables and setup
# system information
TEMPERATURE_K: float = 300  # [K] System temperature
KB: float = 8.314462618e-3  # [kJ/mol/K] Boltzmann constant

# MDAnalysis trajectory parameters
START: int = 0  # First frame to read
STOP: int = None  # Last frame to read
STEP: int = 1  # Step between frames to read
N_JOBS: int = 26  # Number of parallel jobs
N_BLOCKS: int = 26 * 5  # Number of blocks to split trajectory into
UNWRAP: bool = False  # Unwrap trajectory before analysis

# MDAnalysis selection parameters
CHAIN_IDS: list[str] = ["seg_0_Protein_chain_A", "seg_1_Protein_chain_B"]  # Chain IDs
CHAIN_TAGS: list[str] = ["Chain_A", "Chain_B"]  # Chain names

# Plumed analysis parameters
CVS: list[str] = ["d12", "rg1", "rg2"]  # Colvars to analyze

# Data processing parameters
VERBOSE: bool = True
RELOAD_DATA: bool = False
REFRESH_OFFSETS: bool = False
PLUMED_ANALYSIS: bool = True
MDA_ANALYSIS: bool = True
HBOND_ANALYSIS: bool = True

# File I/O
FIG_EXT: str = "png"  # Figure file extension


if __name__ == "__main__":
    t_script_start = time.time()

    # setup
    set_style()
    log = setup_logging(log_file="logs/mda_data_gen.log", verbose=VERBOSE)

    # see if dask client exists, if not, create one
    try:
        client = Client("tcp://localhost:8786", timeout=2)
        log.debug("Client found.")
    except OSError:
        log.debug("Client not found. Creating new client.")
        dask.config.set(
            {
                "distributed.worker.memory.target": 0.6,
                "distributed.worker.memory.spill": 0.7,
                "distributed.worker.memory.pause": 1,
                "distributed.worker.memory.terminate": 1,
            }
        )
        cluster = LocalCluster(
            n_workers=N_JOBS,
            threads_per_worker=1,
            processes=True,
            memory_limit="60GB",
            scheduler_port=8786,
        )
        client = Client(cluster)

    log.info(f"Client: {client}")

    # add stdout handler to log
    log.addHandler(logging.StreamHandler(sys.stdout))
    if VERBOSE:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)

    # command line input
    parser = argparse.ArgumentParser(description="Analyze MD data")
    parser.add_argument(
        "-d", "--dir", type=str, help="Base directory for the input data"
    )
    parser.add_argument("-f", "--fname", type=str, help="File name for the input data")
    parser.add_argument(
        "-t", "--tag", type=str, help="Subdirectory tag for the output data"
    )
    args = parser.parse_args()

    # create subdirectory for data output
    dir_out = Path(f"output/{args.tag}")
    dir_out.mkdir(parents=True, exist_ok=True)
    os.chdir(str(dir_out))
    log.info(f"Output directory: {dir_out}")

    # set data I/O paths
    data_dir = Path(f"{args.dir}")
    f_name = str(args.fname)
    log.info(f"Data directory: {data_dir}")
    log.info(f"Output directory: {dir_out}")
    log.info(f"Input file name: {f_name}")

    # check data directory exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # parse input path
    monomer, system, title = parse_tags(data_dir, verbose=VERBOSE)

    # find input files and data
    cwd = os.getcwd()
    df_plumed = plumed_df(
        cwd,
        data_dir,
        args.tag,
        KB * TEMPERATURE_K,
        refresh=RELOAD_DATA,
        verbose=VERBOSE,
    )

    # run plumed analysis
    if PLUMED_ANALYSIS and (RELOAD_DATA or not Path(f"plumed/figures").exists()):
        for cv in CVS:
            plumed_plots(cwd, args.tag, cv)

    if not MDA_ANALYSIS:
        sys.exit()

    # load MDA universe from topology and trajectory files
    uni, sel_dict, info_dict = load_universe(
        cwd,
        data_dir,
        args.fname,
        args.tag,
        unwrap=UNWRAP,
        refresh=REFRESH_OFFSETS,
        verbose=VERBOSE,
    )

    # ANCHOR: Rg and Ree analysis
    log.critical("Collective variable: polymer chain length")
    for chain, sel_chain in zip(["A", "B"], [sel_dict["chain_A"], sel_dict["chain_B"]]):
        log.critical(f"Collective variable: Rg for chain {chain}")
        label = f"{info_dict['system']}-chain_{chain}"

        # see if output file exists, and if so, load it
        output_df_path = Path(
            "mdanalysis_polymer_lengths/data/" + f"pl_{label}.parquet"
        )
        if output_df_path.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")

        # generate new data from input universe
        else:
            log.debug("Calculating new data")
            rg = PolymerLengths(
                atomgroup=uni.select_atoms(sel_chain), label=label, verbose=VERBOSE
            )
            start = time.time()
            rg.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            end = time.time()
            log.info(f"PL with {N_JOBS} threads took {(end - start)/60:.2f} min")
            rg.merge_external_data(df_plumed)
            rg.save()
            rg.figures(info_dict["title"], ext="png")
            log.debug("Clearing memory")
            rg = None
            plt.close("all")

    # ANCHOR: Dihedral angle analysis
    log.critical("Collective variable: backbone dihedral angles")
    for chain, sel_chain in zip(["A", "B"], [sel_dict["chain_A"], sel_dict["chain_B"]]):
        log.critical(f"Collective variable: Backbone dihedral angles for chain {chain}")
        label = f"{info_dict['system']}-chain_{chain}"
        monomer = f"{info_dict['monomer']}".lower()

        # see if output file exists, and if so, load it
        output_df_path = Path(
            "mdanalysis_dihedrals/data/" + f"dihedral_{label}.parquet"
        )
        if output_df_path.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")

        # generate new data from input universe
        else:
            log.debug("Calculating new data")
            dih = Dihedral(
                atomgroup=uni.select_atoms(sel_chain),
                monomer=monomer,
                label=label,
                verbose=VERBOSE,
            )
            start = time.time()
            dih.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            end = time.time()
            log.info(f"Dih with {N_JOBS} threads took {(end - start)/60:.2f} min")
            dih.merge_external_data(df_plumed)
            dih.save()
            log.debug("Clearing memory")
            dih = None

    # ANCHOR: Contact number analysis
    label_groups, label_references, updating = [], [], []
    kwargs_hard_cut, kwargs_rational = [], []
    # contact number of carboxylate oxygen atoms about calcium ions
    label_groups.append("carboxy_O")
    label_references.append("Ca")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 3.4})
    kwargs_rational.append({"radius": 0.98, "d0": 2.80})
    # contact number of carboxylate carbon atoms about calcium ions
    label_groups.append("carboxy_C")
    label_references.append("Ca")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.1})
    kwargs_rational.append({"radius": 1.18, "d0": 3.38})
    # contact number of carboxylate oxygen atoms about sodium ions
    label_groups.append("carboxy_O")
    label_references.append("Na")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 3.4})
    kwargs_rational.append({"radius": 0.98, "d0": 2.80})
    # contact number of carboxylate carbon atoms about sodium ions
    label_groups.append("carboxy_C")
    label_references.append("Na")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.1})
    kwargs_rational.append({"radius": 1.18, "d0": 3.38})
    # contact number of chain A c-alpha atoms about chain A c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_A")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    # contact number of chain B c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_B")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    # contact number of chain A c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    # contact number of all c-alpha atoms about all c-alpha atoms
    label_groups.append("C_alpha")
    label_references.append("C_alpha")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    if HBOND_ANALYSIS:
        # contact number of water oxygen atoms about calcium ions
        label_groups.append("water_O")
        label_references.append("Ca")
        updating.append((True, False))
        kwargs_hard_cut.append({"radius": 3.18})
        kwargs_rational.append({"radius": 1.47, "d0": 2.28})
        # contact number of water oxygen atoms about sodium ions
        label_groups.append("water_O")
        label_references.append("Na")
        updating.append((True, False))
        kwargs_hard_cut.append({"radius": 3.18})
        kwargs_rational.append({"radius": 1.47, "d0": 2.28})
        # contact number of water oxygen atoms about carboxylate oxygen atoms
        label_groups.append("water_O")
        label_references.append("carboxy_O")
        updating.append((True, False))
        kwargs_hard_cut.append({"radius": 3.5})
        kwargs_rational.append({"radius": 1.16, "d0": 2.79})
        # contact number of water oxygen atoms about carboxylate carbon atoms
        label_groups.append("water_O")
        label_references.append("carboxy_C")
        updating.append((True, False))
        kwargs_hard_cut.append({"radius": 4.2})
        kwargs_rational.append({"radius": 1.57, "d0": 3.24})

    # run contact number analysis
    for label_group, label_reference, update, kw_hc, kw_r in zip(
        label_groups, label_references, updating, kwargs_hard_cut, kwargs_rational
    ):
        methods = ["hard_cut", "6_12"]
        for method in methods:
            label = (
                f"{info_dict['system']}-{label_group}_and_{label_reference}_{method}"
            )

            log.critical(
                f"Collective variable: contact number for {label_group} about {label_reference}"
                + f" with {method} method"
            )

            # see if output file exists, and if so, load it
            output_df_path = Path(
                "mdanalysis_contacts/data/" + f"contact_{label}.parquet"
            )
            if output_df_path.exists() and not RELOAD_DATA:
                log.debug("Loading existing data")

            # generate new data from input universe
            else:
                log.debug("Calculating new data")

                # select coordinating atoms
                if "water" in label_group:
                    coord = uni.select_atoms(
                        f"{sel_dict[label_group]} and around {2*kw_hc['radius']} "
                        + f"{sel_dict[label_reference]}",
                        updating=update[0],
                    )
                else:
                    coord = uni.select_atoms(sel_dict[label_group], updating=update[0])

                # select reference atoms
                reference = uni.select_atoms(
                    sel_dict[label_reference], updating=update[1]
                )

                # if there are no Ca ions, and either group is Ca, skip
                if reference.n_atoms == 0 and "Ca" in label_reference:
                    log.warning(f"No reference Ca atoms found for {label}")
                    continue
                elif coord.n_atoms == 0 and "Ca" in label_group:
                    log.warning(f"No coordinating atoms found for {label}")
                    continue

                if method == "hard_cut":
                    cn = Contacts(
                        uni,
                        (coord, reference),
                        method=method,
                        radius=kw_hc["radius"],
                        label=label,
                        verbose=VERBOSE,
                    )

                elif method == "6_12":
                    cn = Contacts(
                        uni,
                        (coord, reference),
                        method=method,
                        radius=kw_r["radius"],
                        label=label,
                        verbose=VERBOSE,
                        kwargs={"d0": kw_r["d0"]},
                    )

                else:
                    raise ValueError(f"Method {method} not recognized")

                try:
                    start = time.time()
                    cn.run(
                        start=START,
                        stop=STOP,
                        step=STEP,
                        verbose=VERBOSE,
                        n_jobs=N_JOBS,
                        n_blocks=N_BLOCKS,
                    )
                    cn.merge_external_data(df_plumed)
                    end = time.time()
                    log.info(
                        f"CN with {N_JOBS} threads took {(end - start)/60:.2f} min"
                    )
                    cn.save()
                    cn.figures(info_dict["title"], ext="png")
                except Exception as e:
                    log.error(f"Error: {e}")
                    log.error(f"Skipping {label}")
                log.debug("Clearing memory")
                cn = None
                plt.close("all")

    # ANCHOR: Pairwise distance analysis
    label_groups, label_references = [], []
    # distance between all calpha atoms
    label_groups.append("C_alpha")
    label_references.append("C_alpha")
    # distance between chain A c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_B")
    # distance between all carboxylate carbon atoms
    label_groups.append("carboxy_C")
    label_references.append("carboxy_C")

    # distance between all sodium ions and carboxylate carbon atoms
    label_groups.append("Na")
    label_references.append("carboxy_C")
    # distance between all sodium ions and carboxylate oxygen atoms
    label_groups.append("Na")
    label_references.append("carboxy_O")
    # distance between all sodium ions and chloride ions
    label_groups.append("Na")
    label_references.append("Cl")

    # distance between all calcium ions and carboxylate carbon atoms
    label_groups.append("Ca")
    label_references.append("carboxy_C")
    # distance between all calcium ions and carboxylate oxygen atoms
    label_groups.append("Ca")
    label_references.append("carboxy_O")
    # distance between all calcium ions and chloride ions
    label_groups.append("Ca")
    label_references.append("Cl")

    # run pairwise distance analysis
    for label_group, label_reference in zip(label_groups, label_references):
        method = "linear"
        label = f"{info_dict['system']}-{label_group}_and_{label_reference}_{method}"
        log.critical(
            f"Collective variable: Pairwise distances for {label_group}"
            + f" about {label_reference} with {method} method"
        )

        # see if output file exists, and if so, load it
        output_df_path = Path(
            "mdanalysis_contact_matrix/data/" + f"contact_matrix_{label}.parquet"
        )
        if output_df_path.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")

        # generate new data from input universe
        else:
            log.debug("Calculating new data")
            coord = uni.select_atoms(sel_dict[label_group])
            reference = uni.select_atoms(sel_dict[label_reference])

            # if there are no Ca ions, and either group is Ca, skip
            if reference.n_atoms == 0 and "Ca" in label_reference:
                log.warning(f"No reference Ca atoms found for {label}")
                continue
            elif coord.n_atoms == 0 and "Ca" in label_group:
                log.warning(f"No coordinating atoms found for {label}")
                continue

            # if there are no Ca ions, and either group is Cl, skip
            if reference.n_atoms == 0 and "Cl" in label_reference:
                log.warning(f"No reference Cl atoms found for {label}")
                continue
            elif coord.n_atoms == 0 and "Cl" in label_group:
                log.warning(f"No coordinating atoms found for {label}")
                continue

            cn = ContactMatrix(
                uni,
                (coord, reference),
                method=method,
                label=label,
                verbose=VERBOSE,
            )
            try:
                start = time.time()
                cn.run(
                    start=START,
                    stop=STOP,
                    step=STEP,
                    verbose=VERBOSE,
                    n_jobs=N_JOBS,
                    n_blocks=N_BLOCKS,
                )
                cn.merge_external_data(df_plumed)
                end = time.time()
                log.info(f"CN with {N_JOBS} threads took {(end - start)/60:.2f} min")
                cn.save()
                cn.figures(info_dict["title"], ext="png")
            except Exception as e:
                log.error(f"Error: {e}")
                log.error(f"Skipping {label}")
            log.debug("Clearing memory")
            cn = None
            plt.close("all")

    # ANCHOR: Contact matrix analysis
    label_groups, label_references, updating = [], [], []
    kwargs_hard_cut, kwargs_rational = [], []
    # contact number of carboxylate oxygen atoms about calcium ions
    label_groups.append("carboxy_O")
    label_references.append("Ca")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 3.4})
    kwargs_rational.append({"radius": 0.98, "d0": 2.80})
    # contact number of carboxylate carbon atoms about calcium ions
    label_groups.append("carboxy_C")
    label_references.append("Ca")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.1})
    kwargs_rational.append({"radius": 1.18, "d0": 3.38})
    # contact number of chain A c-alpha atoms about chain A c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_A")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    # contact number of chain B c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_B")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    # contact number of chain A c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})
    # contact number of all c-alpha atoms about all c-alpha atoms
    label_groups.append("C_alpha")
    label_references.append("C_alpha")
    updating.append((False, False))
    kwargs_hard_cut.append({"radius": 4.6})
    kwargs_rational.append({"radius": 0.98, "d0": 4.0})

    # run contact number analysis
    for label_group, label_reference, update, kw_hc, kw_r in zip(
        label_groups, label_references, updating, kwargs_hard_cut, kwargs_rational
    ):
        methods = ["hard_cut", "6_12"]
        for method in methods:
            label = (
                f"{info_dict['system']}-{label_group}_and_{label_reference}_{method}"
            )

            log.critical(
                f"Collective variable: contact matrices for {label_group}"
                + f" about {label_reference} with {method} method"
            )

            # see if output file exists, and if so, load it
            output_df_path = Path(
                "mdanalysis_contact_matrix/data/" + f"contact_matrix_{label}.parquet"
            )
            if output_df_path.exists() and not RELOAD_DATA:
                log.debug("Loading existing data")

            # generate new data from input universe
            else:
                log.debug("Calculating new data")

                # select coordinating atoms
                coord = uni.select_atoms(sel_dict[label_group], updating=update[0])

                # select reference atoms
                reference = uni.select_atoms(
                    sel_dict[label_reference], updating=update[1]
                )

                # if there are no Ca ions, and either group is Ca, skip
                if reference.n_atoms == 0 and "Ca" in label_reference:
                    log.warning(f"No reference Ca atoms found for {label}")
                    continue
                elif coord.n_atoms == 0 and "Ca" in label_group:
                    log.warning(f"No coordinating atoms found for {label}")
                    continue

                # if there are no Ca ions, and either group is Cl, skip
                if reference.n_atoms == 0 and "Cl" in label_reference:
                    log.warning(f"No reference Cl atoms found for {label}")
                    continue
                elif coord.n_atoms == 0 and "Cl" in label_group:
                    log.warning(f"No coordinating atoms found for {label}")
                    continue

                if method == "hard_cut":
                    cn = ContactMatrix(
                        uni,
                        (coord, reference),
                        method=method,
                        radius=kw_hc["radius"],
                        label=label,
                        verbose=VERBOSE,
                    )

                elif method == "6_12":
                    cn = ContactMatrix(
                        uni,
                        (coord, reference),
                        method=method,
                        radius=kw_r["radius"],
                        label=label,
                        verbose=VERBOSE,
                        kwargs={"d0": kw_r["d0"]},
                    )

                else:
                    raise ValueError(f"Method {method} not recognized")

                try:
                    start = time.time()
                    cn.run(
                        start=START,
                        stop=STOP,
                        step=STEP,
                        verbose=VERBOSE,
                        n_jobs=N_JOBS,
                        n_blocks=N_BLOCKS,
                    )
                    cn.merge_external_data(df_plumed)
                    end = time.time()
                    log.info(
                        f"CN with {N_JOBS} threads took {(end - start)/60:.2f} min"
                    )
                    cn.save()
                    cn.figures(info_dict["title"], ext="png")
                except Exception as e:
                    log.error(f"Error: {e}")
                    log.error(f"Skipping {label}")
                log.debug("Clearing memory")
                cn = None
                plt.close("all")

    # ANCHOR: RDF analysis
    label_groups, label_references, updating, exclusions = [], [], [], []

    # distance between all calpha atoms
    label_groups.append("C_alpha")
    label_references.append("C_alpha")
    updating.append((False, False))
    exclusions.append((1, 1))
    # distance between chain A c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    exclusions.append(None)
    # distance between chain A c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_B")
    label_references.append("C_alpha_chain_A")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all carboxylate carbon atoms
    label_groups.append("carboxy_C")
    label_references.append("carboxy_C")
    updating.append((False, False))
    exclusions.append((1, 1))
    # distance between carboxylate carbon atoms on Chain A about carboxylate carbon atoms on Chain B
    label_groups.append("carboxy_C_Chain_A")
    label_references.append("carboxy_C_Chain_B")
    updating.append((False, False))
    exclusions.append(None)
    # distance between carboxylate carbon atoms on Chain B about carboxylate carbon atoms on Chain A
    label_groups.append("carboxy_C_Chain_B")
    label_references.append("carboxy_C_Chain_A")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all sodium ions and carboxylate carbon atoms
    label_groups.append("Na")
    label_references.append("carboxy_C")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all sodium ions and carboxylate oxygen atoms
    label_groups.append("Na")
    label_references.append("carboxy_O")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all sodium ions and chloride ions
    label_groups.append("Na")
    label_references.append("Cl")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all calcium ions and carboxylate carbon atoms
    label_groups.append("Ca")
    label_references.append("carboxy_C")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all calcium ions and carboxylate oxygen atoms
    label_groups.append("Ca")
    label_references.append("carboxy_O")
    updating.append((False, False))
    exclusions.append(None)
    # distance between all calcium ions and chloride ions
    label_groups.append("Ca")
    label_references.append("Cl")
    updating.append((False, False))
    exclusions.append(None)

    # RDF of carboxylate oxygen atoms about calcium ions
    label_groups.append("carboxy_O")
    label_references.append("Ca")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of carboxylate carbon atoms about calcium ions
    label_groups.append("carboxy_C")
    label_references.append("Ca")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of carboxylate oxygen atoms about sodium ions
    label_groups.append("carboxy_O")
    label_references.append("Na")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of carboxylate carbon atoms about sodium ions
    label_groups.append("carboxy_C")
    label_references.append("Na")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of carboxylate oxygen atoms about chlorine ions
    label_groups.append("carboxy_O")
    label_references.append("Cl")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of carboxylate carbon atoms about chlorine ions
    label_groups.append("carboxy_C")
    label_references.append("Cl")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of carboxylate carbon atoms about carboxylate carbon atoms
    label_groups.append("carboxy_C")
    label_groups.append("carboxy_C")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of chain A c-alpha atoms about chain A c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_A")
    updating.append((False, False))
    exclusions.append((1, 1))
    # RDF of chain B c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_B")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    exclusions.append((1, 1))
    # RDF of chain A c-alpha atoms about chain B c-alpha atoms
    label_groups.append("C_alpha_chain_A")
    label_references.append("C_alpha_chain_B")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of all c-alpha atoms about all c-alpha atoms
    label_groups.append("C_alpha")
    label_references.append("C_alpha")
    updating.append((False, False))
    exclusions.append((1, 1))
    # RDF of calcium ions about calcium ions
    label_groups.append("Ca")
    label_references.append("Ca")
    updating.append((False, False))
    exclusions.append((1, 1))
    # RDF of sodium ions about sodium ions
    label_groups.append("Na")
    label_references.append("Na")
    updating.append((False, False))
    exclusions.append((1, 1))
    # RDF of sodium ions about calcium ions
    label_groups.append("Na")
    label_references.append("Ca")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of chlorine ions about calcium ions
    label_groups.append("Cl")
    label_references.append("Ca")
    updating.append((False, False))
    exclusions.append(None)
    # RDF of chlorine ions about sodium ions
    label_groups.append("Cl")
    label_references.append("Na")
    updating.append((False, False))
    exclusions.append(None)
    if HBOND_ANALYSIS:
        # RDF of water oxygens about carboxylate oxygen atoms
        label_groups.append("water_O")
        label_references.append("carboxy_O")
        updating.append((False, False))
        exclusions.append(None)
        # RDF of water oxygens about carboxylate carbon atoms
        label_groups.append("water_O")
        label_references.append("carboxy_C")
        updating.append((False, False))
        exclusions.append(None)
        # RDF of water oxygens about calcium ions
        label_groups.append("water_O")
        label_references.append("Ca")
        updating.append((False, False))
        exclusions.append(None)
        # RDF of water oxygens about sodium ions
        label_groups.append("water_O")
        label_references.append("Na")
        updating.append((False, False))
        exclusions.append(None)
        # RDF of water oxygens about bound calcium ions
        label_groups.append("water_O")
        label_references.append("adsorbed_Ca")
        updating.append((False, True))
        exclusions.append(None)
        # RDF of water oxygens about bound sodium ions
        label_groups.append("water_O")
        label_references.append("adsorbed_Na")
        updating.append((False, True))
        exclusions.append(None)

    for group, reference, update, exclude in zip(
        label_groups, label_references, updating, exclusions
    ):
        log.critical(f"Collective variable: RDF({group}, {reference})")
        label = f"{info_dict['system']}-{group}_about_{reference}"

        # see if output file exists, and if so, load it
        output_path = Path("mdanalysis_rdf/data")
        file_gr = f"rdf_{label}.parquet"
        output_np = output_path / file_gr
        if output_np.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")
        else:
            log.debug("Calculating new data")
            mda_rdf = InterRDF(
                uni.select_atoms(sel_dict[reference], updating=update[0]),
                uni.select_atoms(sel_dict[group], updating=update[1]),
                nbins=1000,
                domain=(0, 50),
                exclusion_block=exclude,
                label=label,
                df_weights=df_plumed,
                verbose=VERBOSE,
            )
            try:
                start = time.time()
                mda_rdf.run(
                    start=START,
                    stop=STOP,
                    step=STEP,
                    verbose=VERBOSE,
                    n_jobs=N_JOBS,
                    n_blocks=N_BLOCKS,
                )
                end = time.time()
                log.info(f"RDF with {N_JOBS} threads took {(end - start)/60:.2f} min")
                log.info(
                    f"[frames, A atoms, B atoms] = [{mda_rdf.n_frames}, "
                    + f"{uni.select_atoms(sel_dict[reference], updating=update[0]).n_atoms}, "
                    + f"{uni.select_atoms(sel_dict[group], updating=update[1]).n_atoms}]"
                )
                mda_rdf.save()
                mda_rdf.figures(info_dict["title"], ext="png")
            except Exception as e:
                log.error(f"Error: {e}")
                log.error(f"Skipping {label}")

            log.debug("Clearing memory")
            mda_rdf = None
            plt.close("all")

    # ANCHOR: Hydrogen bond analysis
    if HBOND_ANALYSIS:
        # ANCHOR: Hydrogen bond analysis donated by water to polyelectrolyte
        log.critical(
            "Collective variable: hydrogen bonds from water to polyelectrolyte"
        )
        anl_tag = "hbond_from_water_to_chains"

        # set MDA h-bond selections
        donors_sel = f"(resname SOL and name OW) and around 10 ({sel_dict['pe']})"
        hydrogens_sel = "(resname SOL and name HW1 HW2)"
        acceptors_sel = (
            "(resname LAI ACI RAI AI1 and name OB1 OB2)"
            + " or (resname LAN ACN RAN AN1 and name OB1 OB2)"
            + " or (resname LAC ACE RAC AC1 and name OB)"
            + " or (resname LAL ALC RAL AL1 and name OA)"
            + " or (resname GLU GLH CGLU NGLU and name OE1 OE2)"
            + " or (resname ASP ASH CASP NASP and name OD1 OD2)"
            + " or (resname GLU GLH ASP ASH CGLU CASP and name N)"
        )

        # see if output file exists, and if so, load it
        output_df_path = Path(
            "mdanalysis_hbonds/data/" + f"hbond_{info_dict['system']}_{anl_tag}.parquet"
        )
        if output_df_path.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")

        # generate new data from input universe
        else:
            log.debug("Calculating new data")
            hbonds = HydrogenBondAnalysis(
                universe=uni,
                label=f"{info_dict['system']}_{anl_tag}",
                df_weights=df_plumed,
                between=[sel_dict["water"], sel_dict["pe"]],
                donors_sel=donors_sel,
                acceptors_sel=acceptors_sel,
                hydrogens_sel=hydrogens_sel,
            )

            # run analysis
            start = time.time()
            hbonds.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            end = time.time()
            log.info(
                f"Hydrogen bond analysis with {N_JOBS} threads took "
                + f"{(end - start)/60:.2f} min"
            )
            hbonds.merge_external_data(df_plumed)
            hbonds.save()
            # clear memory
            log.debug("Clearing memory")
            hbonds = None
            df_hbonds = None

    # ANCHOR: Hydrogen bond analysis
    if HBOND_ANALYSIS:
        anl_tag = "hbond_between_chains_and_water"
        log.critical("Collective variable: hydrogen bonds between chains and water")

        # see if output file exists, and if so, load it
        output_df_path = Path(
            "mdanalysis_hbonds/data/" + f"hbond_{info_dict['system']}_{anl_tag}.parquet"
        )
        if output_df_path.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")

        # generate new data from input universe
        else:
            log.debug("Calculating new data")
            hbonds = HydrogenBondAnalysis(
                universe=uni,
                label=f"{info_dict['system']}_{anl_tag}",
                df_weights=df_plumed,
                between=[sel_dict["water"], sel_dict["pe"]],
            )
            # set MDA h-bond selections
            protein_hydrogens_sel = hbonds.guess_hydrogens(sel_dict["pe"])
            protein_acceptors_sel = hbonds.guess_acceptors(sel_dict["pe"])
            water_hydrogens_sel = sel_dict["water_H"]
            water_acceptors_sel = sel_dict["water_O"]

            if protein_hydrogens_sel != "":
                hbonds.hydrogens_sel = (
                    f"({protein_hydrogens_sel}) or ({water_hydrogens_sel})"
                )
            else:
                hbonds.hydrogens_sel = water_hydrogens_sel

            if protein_acceptors_sel != "":
                hbonds.acceptors_sel = (
                    f"({protein_acceptors_sel}) or ({water_acceptors_sel})"
                )
            else:
                hbonds.acceptors_sel = water_acceptors_sel

            # run analysis
            start = time.time()
            hbonds.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            end = time.time()
            log.info(
                f"Hydrogen bond analysis with {N_JOBS} threads took "
                + f"{(end - start)/60:.2f} min"
            )
            hbonds.merge_external_data(df_plumed)
            hbonds.save()
            # clear memory
            log.debug("Clearing memory")
            hbonds = None
            df_hbonds = None

        anl_tag = "hbond_between_chains"
        log.critical("Collective variable: hydrogen bonding between chains")

        # see if output file exists, and if so, load it
        output_df_path = Path(
            "mdanalysis_hbonds/data/" + f"hbond_{info_dict['system']}_{anl_tag}.parquet"
        )
        if output_df_path.exists() and not RELOAD_DATA:
            log.debug("Loading existing data")

        # generate new data from input universe
        else:
            log.debug("Calculating new data")
            hbonds = HydrogenBondAnalysis(
                universe=uni,
                label=f"{info_dict['system']}_{anl_tag}",
                df_weights=df_plumed,
                between=[sel_dict["chain_A"], sel_dict["chain_B"]],
            )
            # set MDA h-bond selections
            hbonds.hydrogens_sel = hbonds.guess_hydrogens(sel_dict["pe"])
            hbonds.acceptors_sel = hbonds.guess_acceptors(sel_dict["pe"])
            # run analysis
            start = time.time()
            hbonds.run(
                start=START,
                stop=STOP,
                step=STEP,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                n_blocks=N_BLOCKS,
            )
            end = time.time()
            log.info(
                f"Hydrogen bond analysis with {N_JOBS} threads took "
                + f"{(end - start)/60:.2f} min"
            )
            hbonds.merge_external_data(df_plumed)
            hbonds.save()
            # clear memory
            log.debug("Clearing memory")
            hbonds = None
            df_hbonds = None

    # ANCHOR: End script
    t_script_end = time.time()
    log.info(f"Time to run script: {(t_script_end - t_script_start)/60:.2f} min")
