"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-09
Description: Functions to parse input data paths and strings
and extract relevant information.
"""

# Standard library
from pathlib import Path
import re

# Internal dependencies
from .logs import setup_logging  # noqa: E402


def parse_tags(d_data: Path, verbose: bool = False) -> tuple[str, str, str]:
    """
    Parse tags from the input data path.

    Parameters
    ----------
    d_data : Path
        Path to the input data directory
    verbose : bool, optional
        If True, the logging level is set to DEBUG, by default False

    Returns
    -------
    tuple[str, str, str]
        Tuple containing the monomer id, unique data tag, and title for plots
    """
    log = setup_logging(verbose=verbose, log_file="parse_tags.log")

    # regex parsing of input data_dir
    monomer = re.findall(r"(?<=P)[A-Z][a-z]{2}(?=-)", str(d_data))[0]
    try:
        n_monomers = int(re.findall(r"\d+(?=mer)", str(d_data))[0])
    except IndexError:
        n_monomers = 0
    log.debug(f"Monomer: {monomer}")
    log.debug(f"Number of monomers: {n_monomers}")

    try:
        n_ca = int(re.findall(r"\d+(?=Ca)", str(d_data))[0])
    except IndexError:
        n_ca = 0
    log.debug(f"Number of Ca ions: {n_ca}")

    try:
        estatic_model = re.findall(r"(?<=box-)\w*(?=EStatics)", str(d_data))[0]
    except IndexError:
        log.error("Electrostatic model tag not found in input path")
        estatic_model = None

    if estatic_model == "full":
        estatic_model = "Full"
    elif estatic_model == "ion_scaled":
        estatic_model = "ECCR"
    elif estatic_model == "ion_pe_scaled":
        estatic_model = "ECCR-P"
    elif estatic_model == "ion_pe_all_scaled":
        estatic_model = "ECCR-PA"
    elif estatic_model is not None:
        raise ValueError(f"Electrostatic model not recognized: {estatic_model}")
    log.debug(f"Electrostatic charge model: {estatic_model}")

    # unique system tags for output files
    system = f"{n_monomers}_P{monomer}-{n_ca}_Ca"
    log.debug(f"System tag: {system}")
    title = f"{n_monomers}-"

    if monomer == "Acr":
        title += "PAA"
    elif monomer == "Asp":
        title += "PASA"
    elif monomer == "Glu":
        title += "PGA"
    else:
        raise ValueError(f"Monomer not recognized: {monomer}")

    title += f", {n_ca} Ca$^{{2+}}$"

    if estatic_model is not None:
        title += f" ({estatic_model})"

    log.debug(f"System title: {title}")

    return monomer, system, title
