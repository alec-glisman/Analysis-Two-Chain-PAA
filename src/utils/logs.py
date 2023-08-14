"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-09
Description: Utility functions to use in analysis scripts.
"""

# Standard library
import logging


def setup_logging(
    verbose: bool = False, log_file: str = "default.log"
) -> logging.Logger:
    """
    Setup logging for the script.

    Parameters
    ----------
    verbose : bool, optional
        If True, the logging level is set to DEBUG, by default False
    log_file : str, optional
        Name of the log file, by default "default.log"

    Returns
    -------
    logging.Logger
        Logger object
    """
    log = logging.getLogger(__name__)

    if log.hasHandlers():
        log.debug("Logger already initialized, returning log")
        return log

    # add file handler
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

    if verbose:
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
    log.info(f"Initializing logger file with name: {log_file.split('.')[0]}")
    log.addHandler(handler)

    return log
