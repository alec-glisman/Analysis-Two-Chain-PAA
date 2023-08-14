# Sphinx Documentation for this Project

Sphinx is used to generate the documentation for this project. The
documentation is generated using the `make html` command. If you want
to update the underlying documentation (i.e. the `rst` files), you
will need to install the `sphinx` package and the `sphinx_rtd_theme`.
Update the documentation by changing to the `docs` directory and
running the following command:

    `sphinx-apidoc --force --separate -o  ./source ../src`

The important files to update are the `conf.py` and the `index.rst`.