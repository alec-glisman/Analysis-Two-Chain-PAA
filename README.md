# Data Analysis Repository

**Summary:** Data analysis scripts and notebooks for my MD research projects in the Wang Group.  
**Authors:** [Alec Glisman](https://github.com/alec-glisman)  

[![CodeFactor](https://www.codefactor.io/repository/github/alec-glisman/data-analysis/badge?s=80921ba60816cb7eea4040f846dafcbbc123c96c)](https://www.codefactor.io/repository/github/alec-glisman/data-analysis)
[![CodeCov](https://codecov.io/gh/alec-glisman/Analysis-Two-Chain-PAa/branch/main/graph/badge.svg?token=5ALJXCQZ61)](https://codecov.io/gh/alec-glisman/Analysis-Two-Chain-PAa)
[![WakaTime](https://wakatime.com/badge/github/alec-glisman/data-analysis.svg)](https://wakatime.com/badge/github/alec-glisman/data-analysis)

[![Pytest](https://github.com/alec-glisman/Analysis-Two-Chain-PAa/actions/workflows/conda-pytest.yml/badge.svg)](https://github.com/alec-glisman/Analysis-Two-Chain-PAa/actions/workflows/conda-pytest.yml)
[![Linting](https://github.com/alec-glisman/Analysis-Two-Chain-PAa/actions/workflows/code-linting.yml/badge.svg)](https://github.com/alec-glisman/Analysis-Two-Chain-PAa/actions/workflows/code-linting.yml)

## Project structure

### Directories

- `.github`: GitHub workflows and issue templates  
- `.vscode`: Visual Studio Code settings and preferences
- `analysis`: Python scripts for analyzing MD simulation data
- `docs`: Sphinx documentation for Python modules
- `requirements`: Conda environment files
- `src`: Python modules for loading MD simulation data and calculating collective variables.
Unit tests are performed using Pytest.
- `tutorials`: External repositories with tutorials for using some of the analysis tools I employ.

### Documentation

Documentation is generated using Sphinx.
The HTML index file is found [here](docs/build/html/index.html) and is best viewed in a web browser.

### Continuous Integration (CI) and Continuous Deployment (CD)

On each pull request, GitHub Actions will check the code for linting errors and run unit tests.
The code coverage results are uploaded to CodeCov, and code quality is checked using CodeFactor.

### Python

Install Conda virtual environment with  `conda env create -f requirements/environment.yml`.
Update currently activated Conda virtual environment (and remove unneeded dependencies) with `conda env update -f requirements/environment.yml --prune`.

### Visual Studio Code

Recommended extensions are located in `.vscode/extensions.json`.
Python path and other editor variables are located in `.vscode/settings.json`.
