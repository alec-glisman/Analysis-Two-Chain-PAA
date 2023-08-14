.. Gromacs Data Analysis documentation master file, created by
   sphinx-quickstart on Wed Mar 15 11:12:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Gromacs Data Analysis's documentation!
=================================================


This is the documentation for the Gromacs Data Analysis package. It is
designed to be used with outputs from Gromacs simulations, but can be used
with any data that is formatted in a similar way and allows for Plumed 
integration into the MD engine.

The analysis_helpers module contains a number of helper functions to
load and process simulation trajectory files. The mda module will 
load a trajectory into a MDAnalysis Universe object and unwrap the
periodic boundary conditions as well as return two dictionaries
with useful selection commands and information about the system.

The colvar_parallel module contains updated versions of the functions
in the colvar module. These analyses use Dask to parallelize the
calculations and are therefore much faster than the original functions.
It also includes an inherited version of MDAnalysis analysis base class
to implement the Dask wrapping.

The figures module contains a function that sets the default figure
plotting formatting for matplotlib. These settings are publication
quality.

The stats module contains a class to block average data in order
to remove the effects of autocorrelation inherent in MD simulations.

The utilities module contains a few small functions that are used
throughout the package.

The rest of the documentation is fairly sparse and relies on Sphinx
autodoc to generate the documentation from the docstrings in the code.
Please feel free to contact me if you have any questions or suggestions.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   analysis_helpers
   colvar_parallel
   figures
   stats
   utils

.. toctree::
   :maxdepth: 1
   :caption: Key files:

   colvar_parallel.base
   stats.block_error

.. toctree::
   :maxdepth: 1
   :caption: External links:

   Dask <https://www.dask.org/get-started>
   MDAnalysis Quick Start Guide <https://userguide.mdanalysis.org/stable/examples/quickstart.html>
   MDAnalysis Examples <https://userguide.mdanalysis.org/stable/examples/README.html>
   MDAnalysis Analysis Modules <https://docs.mdanalysis.org/stable/documentation_pages/analysis_modules.html>
   Plumed Tutorials <https://www.plumed.org/doc-v2.8/user-doc/html/tutorials.html>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
