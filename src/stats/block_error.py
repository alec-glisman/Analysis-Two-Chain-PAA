"""
Author: Alec Glisman (GitHub: @alec-glisman)
Date: 2023-03-14
Description: Script contains methods to block average data and estimate
error/decorrelation time. Calculations are accelerated using Dask.

A good review/explanation of the following code can be found in the Plumed
Masterclass 21.2: Statistical errors in MD, Exercise 8: Weighted averages.
https://www.plumed.org/doc-v2.7/user-doc/html/masterclass-21-2.html#masterclass-21-2-ex-8

There exists a second tutorial in the Plumed documentation as well.
https://www.plumed.org/doc-v2.7/user-doc/html/trieste-2.html

Further information on block averaging is found in the Plumed documentation
for the histogram module.
https://www.plumed.org/doc-v2.7/user-doc/html/_h_i_s_t_o_g_r_a_m.html

We estimate the overall histogram of the collective variables (CVs) by
constructing independent histograms from each block of data. We construct
the final histogram as a weighted average of the block histograms with the
input weights. If no bias is applied to the MD simulation, the weights are
all unity and the final histogram is the simple average of the block
histograms. The final histogram is not a function of the block size.

However, the error of the final histogram is a function of the block size.
This is caused by correlations between data points, which invalidates the
central limit theorem. The error of each histogram is calculated as the
the sample standard deviation of the weighted input data.

At large enough block sizes, the variance on the histogram estimates should
plateau. This is because the histogram estimates from each block become
uncorrelated and independent. This allows us to use the central limit theorem
to calculate the true error estimates. Using the raw data before block
averaging would underestimate the error, as the raw data is not independent.
"""

# Standard library
import logging
import warnings

# Third party packages
import dask
import numpy as np
from scipy import integrate
from scipy import stats


@dask.delayed
def kde(
    block_idx: int,
    block_size: int,
    bandwidth: float,
    data: np.array,
    x_pts: np.array,
    weights: np.array = None,
) -> tuple[np.array, float]:
    """
    Calculate the kernel density estimate (KDE) of the input data.
    This function is used to parallelize the KDE calculation.

    Parameters
    ----------
    block_idx : int
        Index of the block to calculate the KDE for.
    block_size : int
        Size of the block to calculate the KDE for.
    bandwidth : float
        Bandwidth of the KDE.
    data : np.array
        Array of data to calculate the KDE for.
    x_pts : np.array
        Array of x points to evaluate the KDE at.
    weights : np.array, optional
        Array of weights for each data point. This should be a 1D array.
        If not specified, the data will be assumed to be unweighted.

    Returns
    -------
    hist : np.array
        Array of the KDE evaluated at the x points.
    weight_norm : float
        Sum of the weights for the block.
    """
    # index of data to use for this block
    idxs = slice(block_idx * block_size, (block_idx + 1) * block_size)

    # calculate bandwidth for this block
    if bandwidth is not None:
        try:
            bandwidth_block = bandwidth / np.std(data[idxs])
        except TypeError as exc:
            print(f"Bandwidth: {bandwidth}")
            print(f"Block Index: {block_idx}")
            print(f"Block Size: {block_size}")
            print(f"Indices: {idxs}")
            raise TypeError(exc) from exc
    else:
        bandwidth_block = None

    # create gaussian kde object and fit data
    if weights is not None:
        gkde_obj = stats.gaussian_kde(
            data[idxs], weights=weights[idxs], bw_method=bandwidth_block
        )
        hist = gkde_obj.evaluate(x_pts)
        norm = np.sum(weights[idxs])
    else:
        gkde_obj = stats.gaussian_kde(data[idxs], bw_method=bandwidth_block)
        hist = gkde_obj.evaluate(x_pts)
        norm = 1

    # normalize kde
    integral = integrate.simps(hist, x=x_pts)
    hist /= integral

    return hist, norm


@dask.delayed
def histogram(
    block_idx: int,
    block_size: int,
    data: np.array,
    bins: np.array,
    weights: np.array = None,
) -> tuple[np.array, float]:
    """
    Calculate the histogram of the input data. This function is used to
    parallelize the histogram calculation. The histogram is normalized to be a
    probability density.

    Parameters
    ----------
    block_idx : int
        Index of the block to calculate the histogram for.
    block_size : int
        Size of the block to calculate the histogram for.
    data : np.array
        Array of data to calculate the histogram for.
    bins : np.array
        Array of bin edges to use for the histogram.
    weights : np.array, optional
        Array of weights for each data point. This should be a 1D array.
        If not specified, the data will be assumed to be unweighted.

    Returns
    -------
    hist : np.array
        Array of the histogram evaluated at the x points.
    weight_norm : float
        Sum of the weights for the block.
    """
    # index of data to use for this block
    idxs = slice(block_idx * block_size, (block_idx + 1) * block_size)

    # calculate normalized histogram
    if weights is not None:
        hist, _ = np.histogram(
            data[idxs], bins=bins, weights=weights[idxs], density=True
        )
        norm = np.sum(weights[idxs])
    else:
        hist, _ = np.histogram(data[idxs], bins=bins, density=True)
        norm = 1

    return hist, norm


class BlockError:
    """
    Methods to perform block average analysis on input data. This allows for
    better estimation of error bars on correlated data, which is extremely
    common in MD simulations.

    The initializer of this class takes in a pandas DataFrame containing the
    collective variables and the user specified labels for the columns
    containing the data and respective weights. The class can be extended to
    take in numpy arrays of data directly, but the current implementation takes
    advantage of the additional tools in pandas.

    NOTE: If using biased simulation data, the weights should be the exponential
    of the bias. For better numerical stability, before taking the exponential,
    subtract the maximum bias from all the biases. This results in the
    exponential of the bias being less than 1.

    The static block averaging methods could be generalized to work for
    arbitrary dimensions. However, the methods are currently only implemented
    for 1D.

    Output histograms are properly normalized to be probability density and
    energy calculations are in units of thermal energy (kT). Distributions
    can be calculated using discrete histogram bins or a kernel density
    estimator (KDE). The KDE is calculated using the Gaussian kernel.

    FUTURE: Implement 2D block average error analysis.
    """

    def __init__(
        self,
        block_sizes: np.array,
        x_data: np.array,
        y_data: np.array = None,
        weights: np.array = None,
        verbose: bool = False,
    ):
        """
        Initialize BlockAverage object. Input data is a pandas DataFrame. Users
        should specify the names of the columns corresponding to CVs to perform
        the block average analysis on them as well as the column containing
        the weights of each data point.

        For Plumed bias output, the weights column should be the exponential of
        the energetic bias divided by thermal energy. If data points are
        unweighted, add a column of ones to the DataFrame.

        Parameters
        ----------
        block_sizes : np.array
            Array of block sizes to use for block averaging. Each block size
            should be an integer. The block size should be less than the
            length of the input data.
        x_data : np.array
            Array of data to perform block averaging on. This should be a 1D
            array.
        y_data : np.array, optional
            Array of data to perform block averaging on. This should be a 1D
            array. If not specified, the data will be assumed to be univariate.
        weights : np.array, optional
            Array of weights for each data point. This should be a 1D array.
            If not specified, the data will be assumed to be unweighted.
        verbose : bool, optional
            If True, print debug messages. Default is False.
        """

        # class logger
        self._logger: logging.Logger = logging.getLogger("BlockError")
        if not verbose:
            self._logger.setLevel(logging.WARN)

        self._logger.info("Initializing BlockError class")
        self._logger.debug(f"verbose : {verbose}")

        if len(x_data) == 0:
            raise ValueError("x_data must have at least one element")
        if max(block_sizes) > len(x_data):
            raise ValueError("block_sizes must be less than length of x_data")

        # save input data
        self.block_sizes: np.array = block_sizes
        self.x_data: np.array = x_data
        self.y_data: np.array = y_data
        self.weights: np.array = weights
        self.weighted_data: bool = False if weights is None else True

        # output data structures
        self.results: dict = {}

        # ignore divide by zero warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def x_val(self) -> tuple[np.array, np.array]:
        """
        Calculate the block average and error of the value of the input
        values of the x data.

        Returns
        -------
        x_mean : np.array
            Array of block average values of the input x data.
        x_err : np.array
            Array of block average standard errors of the input x data.
        """

        # output data structures
        x_mean = np.zeros([len(self.block_sizes)])
        x_err = np.zeros_like(x_mean)

        # iterate over all block sizes
        for i, size in enumerate(self.block_sizes):
            n_block = int(len(self.x_data) / size)

            # initialize variables for block average
            vals = np.zeros([n_block])
            norms = np.zeros([n_block])

            # iterate over all data blocks to calculate statistics
            for j in range(n_block):
                idxs = slice(j * size, (j + 1) * size)

                if self.weighted_data:
                    vals[j] = np.average(self.x_data[idxs], weights=self.weights[idxs])
                    norms[j] = np.sum(self.weights[idxs])
                else:
                    vals[j] = np.average(self.x_data[idxs])
                    norms[j] = 1

            # effective degrees of freedom prefactor
            if self.weighted_data:
                dof_eff = np.square(np.sum(norms)) / np.sum(np.square(norms))
            else:
                dof_eff = n_block

            # bessel correction to sample data variance
            bessel_corr = dof_eff / (dof_eff - 1.0)

            # histogram weighted statistics
            val_wavg = np.average(vals, axis=0, weights=norms)
            val_wvar = bessel_corr * np.average(
                np.square(vals - val_wavg), axis=0, weights=norms
            )
            val_werr = np.sqrt(val_wvar / n_block)

            # save data
            x_mean[i] = val_wavg
            x_err[i] = val_werr

        # return data
        return x_mean, x_err

    def x_kde(
        self,
        x_min: float = None,
        x_max: float = None,
        n_pts: int = 100,
        bandwidth: float = None,
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Calculate the block average and error of the value of the input
        distribution of the x data using a kernel density estimator (KDE).

        Parameters
        ----------
        x_min : float, optional
            Minimum value of the x data to use for the KDE. If not specified,
            the minimum value of the input data will be used.
        x_max : float, optional
            Maximum value of the x data to use for the KDE. If not specified,
            the maximum value of the input data will be used.
        n_pts : int, optional
            Number of points to use for the KDE. Default is 100.
        bandwidth : float, optional
            Bandwidth to use for the KDE. If not specified, the bandwidth will
            be estimated using the Scott's rule of thumb.

        Returns
        -------
        x_pts : np.array
            Array of x values used for the KDE.
        kde_mean : np.array
            Array of block average values of the KDE.
        kde_var : np.array
            Array of block average variances of the KDE.
        kde_err : np.array
            Array of block average standard errors of the KDE.
        """

        # find min, max, and range of values for gaussian kde
        if x_min is None:
            x_min = np.nanmin(self.x_data)
        if x_max is None:
            x_max = np.nanmax(self.x_data)
        x_pts = np.linspace(x_min, x_max, n_pts)

        # drop nan values
        idxs = np.where(np.isfinite(self.x_data))[0]
        self.x_data = self.x_data[idxs]
        if self.weighted_data:
            self.weights = self.weights[idxs]

        # output data structures
        kde_mean = np.zeros([len(self.block_sizes), len(x_pts)])
        kde_var = np.zeros_like(kde_mean)
        kde_err = np.zeros_like(kde_mean)

        # iterate over all block sizes
        for i, size in enumerate(self.block_sizes):
            n_block = int(len(self.x_data) / size)

            # initialize variables for block average
            hists = np.zeros([n_block, len(x_pts)])
            norms = np.zeros([n_block])

            # iterate over all data blocks to calculate kdes/norms
            data = dask.delayed(self.x_data)
            weights = dask.delayed(self.weights)
            pts = dask.delayed(x_pts)
            blocks = []
            for j in range(n_block):
                task = dask.delayed(kde)(j, size, bandwidth, data, pts, weights=weights)
                blocks.append(task)
            blocks = dask.delayed(blocks)

            # compute results
            results = blocks.compute()
            for j, (hist, norm) in enumerate(results):
                hists[j] = hist
                norms[j] = norm

            # check that all norms are finite
            if not np.all(np.isfinite(norms)):
                raise ValueError("Norms are not finite.")

            # effective degrees of freedom prefactor
            if self.weighted_data:
                dof_eff = np.square(np.sum(norms)) / np.sum(np.square(norms))
            else:
                dof_eff = n_block

            # check that dof_eff is finite
            if not np.isfinite(dof_eff):
                print(f"Weight Norms: {norms}")
                print(f"Numerator: {np.square(np.sum(norms))}")
                print(f"Denominator: {np.sum(np.square(norms))}")
                print(f"10 largest weights:\n{np.sort(self.weights)[-10:][::-1]}")
                raise ValueError("Effective degrees of freedom is not finite.")

            # bessel correction to sample data variance
            bessel_corr = dof_eff / (dof_eff - 1.0)

            # histogram weighted statistics
            kde_mean[i] = np.average(hists, axis=0, weights=norms)
            kde_var[i] = bessel_corr * np.average(
                np.square(hists - kde_mean[i]), axis=0, weights=norms
            )
            kde_err[i] = np.sqrt(kde_var[i] / n_block)

        # return data
        return x_pts, kde_mean, kde_var, kde_err

    def x_kde_fes(
        self,
        x_min: float = None,
        x_max: float = None,
        n_pts: int = 100,
        bandwidth: float = None,
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Calculate the block average and error of the free energy surface
        of the input distribution of the x data using a kernel density
        estimator (KDE). This function calls the x_kde function to calculate
        the block average and error of the KDE and then transforms the KDE
        distribution into a free energy surface.

        Parameters
        ----------
        x_min : float, optional
            Minimum value of the x data to use for the KDE. If not specified,
            the minimum value of the input data will be used.
        x_max : float, optional
            Maximum value of the x data to use for the KDE. If not specified,
            the maximum value of the input data will be used.
        n_pts : int, optional
            Number of points to use for the KDE. Default is 100.
        bandwidth : float, optional
            Bandwidth to use for the KDE. If not specified, the bandwidth will
            be estimated using the Scott's rule of thumb.

        Returns
        -------
        x_pts : np.array
            Array of x values used for the KDE.
        fes_mean : np.array
            Array of block average values of the free energy surface.
        fes_var : np.array
            Array of block average variances of the free energy surface.
        fes_err : np.array
            Array of block average standard errors of the free energy surface.
        """

        # calculate the KDE distribution
        x_pts, kde_mean, kde_var, _ = self.x_kde(
            x_min=x_min, x_max=x_max, n_pts=n_pts, bandwidth=bandwidth
        )

        # output data structures
        fes_mean = np.zeros_like(kde_mean)
        fes_var = np.zeros_like(kde_mean)
        fes_err = np.zeros_like(kde_mean)

        # iterate over all block sizes
        for i, size in enumerate(self.block_sizes):
            n_block = int(len(self.x_data) / size)

            # free energy surface weighted statistics
            # simply propagate the error in the histogram by leading-order
            # Taylor series expansion of the natural logarithm
            # :SOURCE: https://physics.stackexchange.com/a/527268
            fes_mean[i] = -np.log(kde_mean[i])
            fes_mean[i] -= np.nanmin(fes_mean[i])  # shift to zero minimum
            fes_var[i] = kde_var[i] / np.square(kde_mean[i])
            fes_err[i] = np.sqrt(fes_var[i] / n_block)

        # return data
        return x_pts, fes_mean, fes_var, fes_err

    def x_histo(self, bins: int = 100, return_edges: bool = False):
        """
        Calculate the block average and error of the histogram of the input
        distribution of the x data.

        Parameters
        ----------
        bins : int or list or np.array, optional
            Number of bins to use for the histogram. If a list or np.array is
            provided, the values will be used as the bin edges.
        return_edges : bool, optional
            If True, the bin edges will be returned.

        Returns
        -------
        centers : np.array
            Array of histogram bin centers.
        histo_mean : np.array
            Array of block average values of the histogram.
        histo_var : np.array
            Array of block average variances of the histogram.
        histo_err : np.array
            Array of block average standard errors of the histogram.
        edges : np.array, optional
            Array of histogram bin edges. Only returned if return_edges is
            True.
        """

        # find min, max, and range of values for histogram bin
        x_range = np.max(self.x_data) - np.min(self.x_data)
        h_min = np.min(self.x_data) - 0.01 * x_range
        h_max = np.max(self.x_data) + 0.01 * x_range

        # check if bins is a list or np.array and set histogram values
        if isinstance(bins, (list, np.ndarray)):
            h_edges = np.array(bins)
        else:
            h_edges = np.linspace(h_min, h_max, bins + 1)

        h_centers = (h_edges[:-1] + h_edges[1:]) / 2.0

        # output data structures
        histo_mean = np.zeros([len(self.block_sizes), len(h_edges) - 1])
        histo_var = np.zeros_like(histo_mean)
        histo_err = np.zeros_like(histo_mean)

        # iterate over all block sizes
        for i, size in enumerate(self.block_sizes):
            n_block = int(len(self.x_data) / size)

            # initialize variables for block average
            hists = np.zeros([n_block, len(h_edges) - 1])
            norms = np.zeros([n_block])

            # iterate over all data blocks to calculate histograms/norms
            data = dask.delayed(self.x_data)
            weights = dask.delayed(self.weights)
            bins = dask.delayed(h_edges)
            blocks = []
            for j in range(n_block):
                task = dask.delayed(histogram)(j, size, data, bins, weights=weights)
                blocks.append(task)
            blocks = dask.delayed(blocks)

            # compute results
            results = blocks.compute()
            for j, (hist, norm) in enumerate(results):
                hists[j] = hist
                norms[j] = norm

            # effective degrees of freedom prefactor
            if self.weighted_data:
                dof_eff = np.square(np.sum(norms)) / np.sum(np.square(norms))
            else:
                dof_eff = n_block

            # bessel correction to sample data variance
            bessel_corr = dof_eff / (dof_eff - 1.0)

            # histogram weighted statistics
            histo_wavg = np.average(hists, axis=0, weights=norms)
            histo_wvar = bessel_corr * np.average(
                np.square(hists - histo_wavg), axis=0, weights=norms
            )
            histo_werr = np.sqrt(histo_wvar / n_block)

            # save data
            histo_mean[i] = histo_wavg
            histo_var[i] = histo_wvar
            histo_err[i] = histo_werr

        # return data
        if return_edges:
            return h_centers, histo_mean, histo_var, histo_err, h_edges
        else:
            return h_centers, histo_mean, histo_var, histo_err

    def x_histo_fes(
        self, bins: int = 100, return_edges: bool = False
    ) -> tuple[np.array, np.array, np.array, np.array, np.array]:
        """
        Calculate the block average and error of the free energy surface
        of the input distribution of the x data using discrete histograms.

        Parameters
        ----------
        bins : int or list or np.array, optional
            Number of bins to use for the histogram. If a list or np.array is
            provided, the values will be used as the bin edges.
        return_edges : bool, optional
            If True, the bin edges will be returned.

        Returns
        -------
        centers : np.array
            Array of histogram bin centers.
        fes_mean : np.array
            Array of block average values of the free energy surface.
        fes_var : np.array
            Array of block average variances of the free energy surface.
        fes_err : np.array
            Array of block average standard errors of the free energy surface.
        edges : np.array, optional
            Array of histogram bin edges. Only returned if return_edges is
            True.
        """

        # calculate histogram of x data
        h_centers, histo_mean, histo_var, _, h_edges = self.x_histo(
            bins=bins, return_edges=True
        )

        # output data structures
        fes_mean = np.zeros_like(histo_mean)
        fes_var = np.zeros_like(histo_mean)
        fes_err = np.zeros_like(histo_mean)

        # iterate over all block sizes
        for i, size in enumerate(self.block_sizes):
            n_block = int(len(self.x_data) / size)

            # free energy surface weighted statistics
            # simply propagate the error in the histogram by leading-order
            # Taylor series expansion of the natural logarithm
            # :SOURCE: https://physics.stackexchange.com/a/527268
            fes_mean[i] = -np.log(histo_mean[i])
            fes_mean[i] -= np.nanmin(histo_mean[i])  # shift to zero minimum
            fes_var[i] = histo_var[i] / np.square(histo_mean[i])
            fes_err[i] = np.sqrt(histo_var[i] / n_block)

        # return data
        if return_edges:
            return h_centers, fes_mean, fes_var, fes_err, h_edges
        else:
            return h_centers, fes_mean, fes_var, fes_err
