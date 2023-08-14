"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-14
Description: Base class for parallel analysis

This module is a simple extension of MDAnalysis.analysis.base.AnalysisBase to
allow parallelization with dask and takes inspiration from the PMDA package.
Users should specify dask options using the new kwargs `n_jobs` and `n_blocks`.
The `n_jobs` argument specifies the number of jobs to start, if `-1` use number
of logical cpu cores. The `n_blocks` argument specifies the number of blocks to
split the trajectory into. If `None` use `n_jobs`.

All derived classes must implement the `_single_frame` method. This method
should return a `np.ndarray` instead of writing to a `self.results` list.
"""

# Standard library
from __future__ import annotations
from datetime import datetime
from functools import partial
import multiprocessing
from pathlib import Path
import sys
import warnings

# External dependencies
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import dask

    FOUND_DASK = True
except:
    FOUND_DASK = False

try:
    import joblib

    FOUND_JOBLIB = True
except:
    FOUND_JOBLIB = False


# MDAnalysis interface
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.coordinates.base import ReaderBase

# add local src directory to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Local internal dependencies
from utils.logs import setup_logging  # noqa: E402


class ParallelAnalysisBase(AnalysisBase):
    """
    Simple extension of MDAnalysis.analysis.base.AnalysisBase to allow
    parallelization with dask. This class is not meant to be used
    directly, but rather as a base class for parallel analysis classes.

    Note that the `_single_frame` and `run` methods are overridden
    and require different arguments than the base class.
    """

    def __init__(
        self,
        trajectory: ReaderBase,
        atomgroups: tuple[AtomGroup],
        label: str = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the analysis object.

        Parameters
        ----------
        trajectory : :class:`~MDAnalysis.coordinates.base.ReaderBase`
            A trajectory reader.
        atomgroups : tuple[:class:`~MDAnalysis.core.groups.AtomGroup`]
            A tuple of atomgroups to be analyzed.
        label : str, optional
            A label for the data. Default: ``None``.
        verbose : bool, optional
            Turn on verbosity. Default: ``False``.
        **kwargs
            Other keyword arguments passed to the base class.
        """
        # call base class constructor
        super().__init__(trajectory, verbose, **kwargs)

        # set up logging
        self._logger = setup_logging(verbose=verbose, log_file=f"logs/{__name__}.log")
        self._verbose: bool = verbose

        # set MDA data structures
        self._atomgroups = atomgroups
        self._logger.debug(f"Number of atomgroups: {len(self._atomgroups)}")
        self._universe = self._atomgroups[0].universe
        self._trajectory = self._universe.trajectory
        self._top = self._universe.filename
        self._traj = self._universe.trajectory.filename
        self._indices = [ag.indices for ag in atomgroups]

        # external data
        self._df_weights: pd.DataFrame = None
        self._tag: str = label
        self._logger.debug(f"tag: {self._tag}")

        # output data
        # REVIEW: These should be implemented in the derived classes
        self._dir_out: Path = None
        self._df: pd.DataFrame = None
        self._results: np.array = None
        self._df_filename: str = None
        self._columns: list[str] = None

    def _prepare(self):
        """
        Prepare the analysis class for the single frame analysis.
        This is called once before the analysis is started inside
        the run() method.
        """
        pass

    def _single_frame(self, idx_frame: int) -> np.ndarray:
        """
        Perform computation on a single trajectory frame.
        Must return computed values as a list. You can only **read**
        from member variables stored in ``self``. Changing them during
        a run will result in undefined behavior. `ts` and any of the
        atomgroups can be changed (but changes will be overwritten
        when the next time step is read).

        Parameters
        ----------
        idx_frame : int
            The index of the current frame.

        Returns
        -------
        values : anything
            The output from the computation over a single frame must
            be returned. The `value` will be added to a list for each
            block and the list of blocks is stored as :attr:`_results`
            before :meth:`_conclude` is run. In order to simplify
            processing, the `values` should be "simple" shallow data
            structures such as arrays or lists of numbers. It is easiest
            if the return value is a single array or list of numbers.

        Raises
        ------
        NotImplementedError
            If this method is not implemented in a subclass.
        """
        raise NotImplementedError("Override this method in your subclass")

    def run(
        self,
        start: int = None,
        stop: int = None,
        step: int = None,
        frames: np.array = None,
        verbose: bool = None,
        n_jobs: int = 1,
        module: str = "dask",
        method: str = None,
        n_blocks: int = None,
        **kwargs,
    ) -> ParallelAnalysisBase:
        """
        Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        frames : array_like, optional
            array of integers or booleans to slice trajectory; `frames` can
            only be used *instead* of `start`, `stop`, and `step`. Setting
            *both* `frames` and at least one of `start`, `stop`, `step` to a
            non-default value will raise a :exc:`ValueError`.
            .. versionadded:: 2.2.0
        verbose : bool, optional
            Turn on verbosity
        n_jobs : int, optional
            number of jobs to start, if `-1` use number of logical cpu cores.
            This argument will be ignored when the distributed scheduler is
            used
            .. versionadded:: PERSONAL-COPY
        module : str, optional
            Parallelization module to use. Default: ``"multiprocessing"``.
            .. versionadded:: PERSONAL-COPY
        method : str, optional
            Parallelization method to use. This is the Dask scheduler,
            Joblib backend, or the multiprocessing method. Default: ``None``.
            .. versionadded:: PERSONAL-COPY
        n_blocks : int, optional
            number of blocks to divide trajectory into. If ``None`` set equal
            to n_jobs or number of available workers in scheduler.
            .. versionadded:: PERSONAL-COPY
        **kwargs
            Other keyword arguments passed to :func:`dask.base.compute`.
        .. versionchanged:: 2.2.0
            Added ability to analyze arbitrary frames by passing a list of
            frame indices in the `frames` keyword argument.

        Returns
        -------
        self : :class:`ParallelAnalysisBase` instance
            The instance itself.

        Raises
        ------
        ValueError
            If `frames` is used in conjunction with `start`, `stop`, or `step`.
        """
        self._logger.info("Starting run method")
        verbose = getattr(self, "_verbose", False) if verbose is None else verbose

        self._setup_frames(
            self._atomgroups[0].universe.trajectory,
            start=start,
            stop=stop,
            step=step,
            frames=frames,
        )

        self._prepare()

        n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        n_blocks = n_blocks if n_blocks else n_jobs
        ts_indices = frames if frames else np.arange(self.start, self.stop, self.step)
        block_indices = np.array_split(ts_indices, n_blocks)

        # issue errors
        if self.n_frames == 0:
            warnings.warn("run() analyses no frames: check start/stop/step")
        if self.n_frames < n_blocks:
            warnings.warn("run() uses more blocks than frames: " "decrease n_blocks")

        # dask scheduler
        if module == "dask" and FOUND_DASK:
            try:
                config = {"scheduler": dask.distributed.worker.get_client(), **kwargs}
                n_jobs = min(len(config["scheduler"].get_worker_logs()), n_jobs)
            except Exception as exc:
                if method is None:
                    method = "processes"
                elif method not in {
                    "distributed",
                    "processes",
                    "threading",
                    "threads",
                    "single-threaded",
                    "sync",
                    "synchronous",
                }:
                    raise ValueError("Invalid Dask scheduler.") from exc

                if method == "distributed":
                    raise RuntimeError(
                        "The Dask distributed client "
                        "(client = dask.distributed.Client(...)) "
                        "should be instantiated in the main "
                        "program (__name__ = '__main__') of "
                        "your script."
                    ) from exc
                elif method in {"threading", "threads"}:
                    raise ValueError(
                        "The threaded Dask scheduler is not "
                        "compatible with MDAnalysis."
                    ) from exc
                elif n_jobs == 1 and method not in {
                    "single-threaded",
                    "sync",
                    "synchronous",
                }:
                    method = "synchronous"
                    warnings.warn(
                        f"Since {n_jobs=}, the synchronous "
                        "Dask scheduler will be used instead."
                    )
                config = {"scheduler": method} | kwargs
                if method == "processes":
                    config["num_workers"] = n_jobs

            msg = (
                f"Starting analysis using Dask ({n_jobs=}, "
                + f"scheduler={config['scheduler']})..."
            )
            self._logger.debug(msg)
            if verbose:
                print(msg)

            jobs = []
            for indices in block_indices:
                jobs.append(dask.delayed(self._job_block)(indices))

            if verbose:
                time_start = datetime.now()

            blocks = dask.delayed(jobs)
            blocks = blocks.persist(**config)
            dask.distributed.progress(blocks, minimum=1, dt=1)
            block_results = blocks.compute(**config)

        # joblib backend
        elif module == "joblib" and FOUND_JOBLIB:
            if method is None:
                method = "processes"
            elif method not in {"processes", "threads"}:
                raise ValueError("Invalid Joblib backend.")

            msg = f"Starting analysis using Joblib ({n_jobs=}, backend={method})..."
            self._logger.debug(msg)
            if verbose:
                print(msg)
                time_start = datetime.now()

            block_results = joblib.Parallel(
                n_jobs=n_jobs,
                prefer=method,
            )(
                joblib.delayed(self._job_block)(indices)
                for indices in tqdm(
                    block_indices,
                    total=n_blocks,
                    colour="green",
                    unit="block",
                    desc="Analysis",
                    mininterval=1,
                    dynamic_ncols=True,
                    disable=(not self._verbose),
                )
            )

        # multiprocessing
        else:
            if module != "multiprocessing":
                warnings.warn(
                    "The Dask or Joblib library was not "
                    "found, so the native multiprocessing "
                    "module will be used instead."
                )

            if method is None:
                method = multiprocessing.get_start_method()
            elif method not in {"fork", "forkserver", "spawn"}:
                raise ValueError("Invalid multiprocessing start method.")

            msg = f"Starting analysis using multiprocessing ({n_jobs=}, {method=})..."
            self._logger.debug(msg)
            if verbose:
                print(msg)
                time_start = datetime.now()

            with multiprocessing.get_context(method).Pool(n_jobs) as p:
                block_results = list(
                    tqdm(
                        p.imap(partial(self._job_block), block_indices),
                        total=n_blocks,
                        colour="green",
                        unit="block",
                        desc="Analysis",
                        mininterval=1,
                        dynamic_ncols=True,
                        disable=(not self._verbose),
                    )
                )

        # combine results
        if len(block_results) > 0:
            self._results = np.concatenate(block_results)
        else:
            self._results = np.array([])
            self._logger.warning("No results returned.")

        if verbose:
            print(f"Finished! Time elapsed: {datetime.now() - time_start}.")

        # create dataframe
        if self._columns is None:
            self._logger.warning("No columns specified. Not creating dataframe.")
        else:
            try:
                self._df = pd.DataFrame(self._results, columns=self._columns)
                if self._tag is not None:
                    self._df["tag"] = self._tag

            except ValueError:
                self._logger.error(
                    "Could not create dataframe. Results have wrong shape."
                )
                warnings.warn("Could not create dataframe. Results have wrong shape.")
                self._df = None

        # conclude and return
        self._conclude()
        return self

    def merge_external_data(self, df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
        """
        Merge external data with the results of the analysis

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data to merge
        col : str, optional
            Column name to merge on, by default "time"

        Returns
        -------
        pd.DataFrame
            Dataframe with merged data

        Raises
        ------
        ValueError
            If no results are present
        ValueError
            If no data is merged
        """
        if self._df is None:
            warnings.warn("No results present. Cannot merge data.")
            return None

        nrows = len(self._df)
        df_out = pd.merge(self._df, df, on=col, how="inner")

        if len(df_out) == 0:
            warnings.warn("Could not merge dataframes. No data merged.")
        elif len(df_out) < 0.95 * nrows:
            warnings.warn(
                f"Only {len(df_out)} rows of data merged. " f"Expected {nrows} rows."
            )

        self._df = df_out
        return df_out

    def _job_block(self, absolute_frame_indices: np.array) -> np.array:
        """
        Helper function to actually setup the Dask tasks

        Parameters
        ----------
        absolute_frame_indices : np.array
            Array of frame indices to analyze. These indices are absolute
            and refer to frame numbers in the trajectory.

        Returns
        -------
        np.array
            Array of results from the analysis of the frames
        """
        res = [None] * len(absolute_frame_indices)
        for ab, idx in enumerate(absolute_frame_indices):
            res[ab] = self._single_frame(idx)

        # find maximum number of elements in the list
        max_len = 0
        for sub in res:
            try:
                max_len = max(max_len, len(sub))
            except TypeError:
                pass

        # return 2D array if each frame returns a 1D array
        if max_len > 1:
            return np.vstack(res)
        else:
            return np.array(res)

    @staticmethod
    def _reduce(res, result_single_frame):
        """'append' action for a time series"""
        res.append(result_single_frame)
        return res

    def _conclude(self) -> None:
        """
        Finalize the results of the calculation. This is called once
        after the analysis is finished inside the run() method. The method
        concatenates the results from the single frame analysis and
        input weights dataframe, if provided.

        Raises
        ------
        ValueError
            If no results are found.
        """
        pass

    def save(self, dir_out: str = None) -> None:
        """
        Save the results of the analysis to a parquet file. The results
        are saved to the `data` directory in the `dir_out` directory.

        This method should only be called after the analysis has been
        run.

        Parameters
        ----------
        dir_out : str, optional
            The directory to save the results to. If not specified, the
            results are saved to the directory specified in the
            `dir_out` attribute.

        Raises
        ------
        ValueError
            If no dataframe is found.
        ValueError
            If no filename is found.
        ValueError
            If no output directory is specified.
        """
        if self._df is None:
            raise ValueError(
                "No dataframe found. Did you run the analysis?"
                + "Did the child class set the `columns` "
                + "attribute?"
            )

        if self._df_filename is None:
            raise ValueError(
                "No filename found. Did the child class "
                + "set the `df_filename` attribute?"
            )

        if (self._dir_out is None) and (dir_out is None):
            raise ValueError(
                "No output directory specified. Please specify a directory "
                + "in the `dir_out` attribute or as an argument to the "
                + "save() method."
            )

        if dir_out is None:
            dir_out = self._dir_out / "data"
        Path(dir_out).mkdir(parents=True, exist_ok=True)

        self._logger.info(
            f"Saving {__class__.__name__} analysis for {self._tag} to {dir_out}."
        )
        self._df.to_parquet(dir_out / self._df_filename)
